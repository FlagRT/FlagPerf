"""Complete generation requests, with optional attachment-owned observations."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

from .onerec import GreedyExecution


@contextmanager
def interval(handle, name, category, attrs):
    if not handle or handle.state['effective_config'].get('probes') == 'none':
        yield
        return
    from flagprof.probes.base import clock
    parent = handle.context.current_scope()
    start = clock()
    with handle.context.scope(name) as token:
        try:
            yield
        finally:
            event = dict(t='span', ph='x', cat=category, name=name, corr=token.corr,
                         ts=start, dur=clock()-start, attrs=attrs)
            if parent:
                event['parent'] = parent.corr
            handle.writer.emit(event)


def wrapper(model):
    import torch
    class StepModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model.model
            self.lm_head = model.lm_head
            self.config = model.config

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def forward(self, session):
            return session.advance()
    return StepModel()


def run_request(model, step_model, inputs, sample_ids, *, mode, sync, handle=None,
                request_id='0', parallel=None, observe_cache=False):
    import torch
    session = GreedyExecution(model, inputs, mode)
    rows, tokens, cache = [], [], []
    attrs = dict(kind='generation_request', request_id=request_id, samples=sample_ids, mode=mode)
    if parallel:
        parallel.barrier()
    sync()
    begin = time.perf_counter_ns()
    try:
        with torch.inference_mode(), interval(handle, 'request', 'user', attrs):
            with interval(handle, 'kv_created', 'lifecycle', dict(request_id=request_id, initialized=False)):
                pass
            while not session.done:
                context = dict(session.context)
                step_attrs = dict(kind='generation_step', request_id=request_id, samples=sample_ids, **context)
                if handle:
                    mask = (session.mask if session.index == 0 else
                            torch.tensor(session.active, device=session.mask.device,
                                         dtype=session.mask.dtype).unsqueeze(1))
                    handle.bind_generation_step(sample_ids, dict(attention_mask=mask), step_attrs)
                sync()
                start = time.perf_counter_ns()
                with interval(handle, context['phase'], 'step', step_attrs):
                    step = step_model(session)
                    sync()
                end = time.perf_counter_ns()
                selected = step.token_ids.tolist()
                rows.append(dict(**context, start_ns=start, end_ns=end, latency_ns=end-start))
                tokens.append(selected)
                step_attrs.update(tokens=selected, stop_reasons=list(session.stop_reasons), done=session.done)
                if handle:
                    handle.record_tensor('hidden', step.outputs['hidden'], step=context['generation_step'])
                    handle.record_tensor('token', step.token_ids, step=context['generation_step'])
                    # Events are immutable at emit; record the completion as its own lifecycle span.
                    with interval(handle, 'generation_step_complete', 'lifecycle', step_attrs):
                        pass
                if observe_cache:
                    snapshot=dict(generation_step=context['generation_step'], **session.cache_snapshot())
                    cache.append(snapshot)
                    with interval(handle, 'kv_observation', 'lifecycle', dict(request_id=request_id, **snapshot)):
                        pass
                if parallel and mode == 'accuracy':
                    parallel.check_step(context, selected)
            sync()
            end = time.perf_counter_ns()
            last = step.outputs['hidden']
            stops = list(session.stop_reasons)
    finally:
        session.close()
        if handle:
            with interval(handle, 'kv_released', 'lifecycle', dict(request_id=request_id, released=session.cache is None)):
                pass
            handle.context.probe_state().pop('generation', None)
    if parallel:
        parallel.check_request(sample_ids, tokens, stops)
    return last, dict(request_id=request_id, sample_ids=sample_ids, mode=mode,
                     start_ns=begin, end_ns=end, latency_ns=end-begin, steps=rows,
                     tokens=tokens, stop_reasons=stops, cache=cache,
                     kv_released=session.cache is None, input_tokens=int(inputs['attention_mask'].sum()),
                     generated_tokens=sum(sum(r['active_samples']) for r in rows))


def execute(args):
    import torch
    from flagprof import current_attachment
    from .fixed import load, numerics, software
    if args.batch_size not in (None, 1) or args.max_new_tokens < 1:
        raise ValueError('OneRec generation requires batch 1 and positive max-new-tokens')
    if args.generation_mode == 'accuracy' and (args.warmup or args.forwards != 1 or args.repeats != 1):
        raise ValueError('accuracy uses one request per sample without warmup or repeats')
    if not args.inputs_from:
        raise ValueError('OneRec generation requires the sealed --inputs-from package')
    if args.flaggems not in (None, 'off') or args.flaggems_include:
        raise ValueError('OneRec registered execution has FlagGems off')
    handle = current_attachment()
    if handle and handle.state['effective_config']['flaggems'] != 'off':
        raise ValueError('OneRec requires attachment FlagGems off')
    if args.device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(args.device)
        sync = torch.npu.synchronize
    elif args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        sync = torch.cuda.synchronize
    else:
        sync = lambda: None
    # Initialize the vendor extension before Transformers probes optional backends.
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    numeric = numerics(args.device, args.numerics)
    model_path = args.model[3:]
    package = json.loads((Path(args.inputs_from)/'manifest.json').read_text())
    contract = package['contract']
    expected = dict(batch_size=1, max_length=args.max_length, padding_side=args.padding_side)
    if any(contract['inputs'][k] != v for k, v in expected.items()):
        raise ValueError('generation input contract mismatch')
    if (contract['model']['dtype'] != args.dtype or contract['model']['attention'] != args.attention or
            contract['model']['options']['max_new_tokens'] != args.max_new_tokens):
        raise ValueError('generation numerical or step contract mismatch')
    began = time.perf_counter()
    batches, identity = load(args.inputs_from, samples=args.samples.split(',') if args.samples else None,
                             batch_size=1, model=model_path)
    parallel = None
    try:
        world = int(os.environ.get('WORLD_SIZE', '1'))
        plan = None
        if world > 1:
            from .parallel import Parallel
            parallel = Parallel(args.device, os.environ['FLAGPROF_COMM_BACKEND'])
            architecture = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
            plan = {**{'model.'+k:v for k,v in architecture.base_model_tp_plan.items()},
                    'lm_head':'colwise_gather_output'}
        options = dict(local_files_only=True, trust_remote_code=False,
                       dtype=getattr(torch, args.dtype), attn_implementation=args.attention)
        if parallel:
            options.update(tp_plan=plan, device_mesh=parallel.mesh())
        model = AutoModelForCausalLM.from_pretrained(model_path, **options).eval().requires_grad_(False)
        if parallel:
            if list(model.lm_head.weight.shape) != [176384 // 2, 4096]:
                raise ValueError('OneRec vocabulary head was not sharded as declared')
        else:
            model.to(args.device)
        if (model.config.model_type, model.config.hidden_size, model.config.num_hidden_layers,
                model.config.vocab_size) != ('qwen3', 4096, 36, 176384):
            raise ValueError('OneRec architecture mismatch')
        model._flagperf_generation = dict(max_new_tokens=args.max_new_tokens)
        step_model = wrapper(model)
        boundaries = ['<root>.model.layers.'+str(i) for i in range(36)] + ['<root>.model.norm', '<root>.lm_head']
        step_model._flagprof_tensor_boundaries = boundaries
        step_model._flagprof_output_boundaries = ['hidden', 'token']
        if handle:
            handle.register_model(step_model)
        identity.update(task='generate', dtype=args.dtype, attention=args.attention,
                        generation=dict(mode=args.generation_mode, max_new_tokens=args.max_new_tokens,
                                        decoding='greedy', eos_token_id=model.generation_config.eos_token_id,
                                        stopping='EOS_or_max_new_tokens' if args.generation_mode=='accuracy' else 'fixed_steps'),
                        device=args.device, software=software(), numerics=numeric, flaggems='off', include=None,
                        parallelism=dict(world_size=world, plan=plan),
                        comm_backend=parallel.backend if parallel else 'none')
        if handle:
            handle.bind_tensor_inputs([], {}, identity=identity)
        out = Path(args.outputs)
        out.mkdir(parents=True, exist_ok=False)
        requests = []
        load_s = time.perf_counter()-began
        for repeat in range(args.repeats):
            for index, (sample_ids, values) in enumerate(batches):
                inputs = {k:v.to(args.device) for k,v in values.items()}
                for cycle in range(-args.warmup, args.forwards):
                    request_id = '%s:%s:%s' % (repeat, index, cycle)
                    last, row = run_request(model, step_model, inputs, sample_ids, mode=args.generation_mode,
                                            sync=sync, handle=handle, request_id=request_id, parallel=parallel,
                                            observe_cache=args.generation_mode=='accuracy')
                    row.update(repeat=repeat, cycle=cycle, batch_index=index, warmup=cycle<0)
                    if cycle >= 0:
                        # Output transfer and sealing are outside the synchronized execution envelope.
                        cpu = last.detach().cpu().contiguous()
                        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
                        row['final_hidden'] = dict(shape=list(cpu.shape), dtype=str(cpu.dtype),
                                                   sha256=hashlib.sha256(raw).hexdigest())
                        if args.generation_mode == 'accuracy' or cycle == args.forwards-1:
                            torch.save(cpu, out/('hidden-'+request_id.replace(':','-')+'.pt'))
                    requests.append(row)
                    del last
        result = dict(schema='generation-runner/1', oracle_identity=identity, requests=requests,
                      load_s=load_s, instrumentation='attachment' if handle else 'none',
                      rank=int(os.environ.get('RANK','0')), status='complete')
        (out/'runner.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(output=str(out), status='complete', requests=len(requests))))
        return result
    finally:
        if parallel:
            parallel.close()
