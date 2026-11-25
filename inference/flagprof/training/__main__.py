# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Standard PyTorch training examples, observed by the existing flagprof record."""
import argparse
import atexit
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('classification','causal-lm','ctr','embedding','rerank'),default=None,
                   help='Training task; optional for adapted models in targets/model_registry.json (otherwise required); defaults to the classification demo when no model is given')
    p.add_argument('--sharding',choices=('none','fsdp'),default='none',
                   help='Multi-rank strategy: none=DistributedDataParallel full replica per rank; fsdp=fully_shard shards parameters/gradients/optimizer state so models too large for one card can train')
    p.add_argument('--model',help='Trusted local HF model directory; no network download')
    p.add_argument('--inputs',help='JSONL of token IDs (causal-lm) or triplet groups (embedding)')
    p.add_argument('--ctr-arch',help='CTR architecture (dlrm din widedeep deepfm dcn sasrec lightgcn bst dien dcnmix esmm mmoe ple dssm); required')
    p.add_argument('--package',help='Sealed package directory (ctr/rerank task); digests verified, corpus unchanged; ctr may omit it to generate the deterministic default corpus')
    p.add_argument('--pooling',choices=('last','cls','mean'),default=None,help='Embedding pooling; optional for adapted models in targets/model_registry.json, otherwise required')
    p.add_argument('--temperature',type=float,default=.05,help='InfoNCE temperature (embedding task)')
    p.add_argument('--rerank-score',choices=('causal_yesno','sequence_logit'),default=None,help='Rerank scoring semantics; optional for adapted models in targets/model_registry.json, otherwise required; must match the package manifest')
    p.add_argument('--outputs',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,default=3,help='Total optimizer steps, including resumed steps')
    p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=4,help='Per-rank microbatch size')
    p.add_argument('--seed',type=int,default=531)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--optimizer',choices=('sgd','adamw'),default='sgd')
    p.add_argument('--amp',choices=('off','bfloat16','float16'),default='off')
    p.add_argument('--mode',choices=('accuracy','performance'),default='accuracy')
    p.add_argument('--checkpoint',help='Explicit checkpoint output; rank suffix added for DDP')
    p.add_argument('--resume',help='Trusted local pickle checkpoint; rank suffix added for DDP')
    p.add_argument('--stop-after',type=int,help='Stop after this absolute optimizer step, at a checkpoint boundary')
    p.add_argument('--ddp',action='store_true',help='Use torchrun-provided ranks and DistributedDataParallel')
    p.add_argument('--backend',choices=('gloo','nccl','hccl'),default=None)
    return p


def local_files(model):
    h=hashlib.sha256(); root=Path(model)
    for path in sorted(root.rglob('*')):
        if path.is_file():
            h.update(path.relative_to(root).as_posix().encode())
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def ensure_generated(kind, outputs, seed, vocab=None, rows=None):
    """Generate the deterministic default corpus or package next to --outputs.

    The generators are the sealed, seeded CLI tools under inference/targets/;
    an existing result is reused untouched. Single-process only: DDP ranks
    must share an explicit --inputs/--package instead of racing on creation.
    kind: 'corpus' (embedding triplets), 'lm-corpus' (pretokenized LM rows),
    or 'ctr-package'. rows defaults so a full batch grid is always possible.
    """
    base=Path(outputs)
    suffix, marker, script_name = {
        'corpus': ('-default-corpus','groups.jsonl','make_embedding_corpus.py'),
        'lm-corpus': ('-default-corpus','tokens.jsonl','make_lm_corpus.py'),
        'ctr-package': ('-default-package','manifest.json','make_ctr_package.py'),
    }[kind]
    target=base.with_name(base.name+suffix)
    if (target/marker).exists(): return str(target)
    script=Path(__file__).resolve().parents[2]/'targets'/script_name
    if not script.is_file(): raise ValueError('generator script missing: %s' % script)
    target.mkdir(parents=True, exist_ok=True)
    import subprocess, sys
    cmd=[sys.executable,'-B',str(script),'--seed',str(seed)]
    if kind=='corpus':
        cmd += ['--groups','32','--negs','3','--out',str(target/'groups.jsonl')]
    elif kind=='lm-corpus':
        cmd += ['--rows',str(rows or 16),'--seq-len','64','--vocab',str(vocab or 65536),
                '--out',str(target/'tokens.jsonl')]
    else:
        cmd += ['--samples','1024','--out',str(target)]
    done=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if done.returncode!=0:
        raise ValueError('default %s generation failed: %s' % (
            kind, done.stderr.decode(errors='replace')[-400:]))
    return str(target)


def prepared_lm(path, *, batch_size, rank=0, world_size=1):
    import torch
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r=json.loads(line); ids=r['input_ids']; mask=r.get('attention_mask',[1]*len(ids))
        labels=r.get('labels',[v if m else -100 for v,m in zip(ids,mask)])
        if len(ids)<2 or len(ids)!=len(mask) or len(ids)!=len(labels):
            raise ValueError('LM token, mask and label lengths must match and contain at least two tokens')
        if any(type(v) is not int or v<0 for v in ids) or any(v not in (0,1) for v in mask):
            raise ValueError('invalid token IDs or attention mask')
        if any(type(v) is not int or v<0 and v!=-100 for v in labels): raise ValueError('invalid labels')
        if not any(v!=-100 for v in labels[1:]): raise ValueError('LM batch has no supervised next token')
        rows.append(dict(input_ids=ids,attention_mask=mask,labels=labels))
    group=batch_size*world_size
    if not rows or len(rows)%group:
        raise ValueError('input row count must be a positive multiple of global microbatch size; no silently dropped data')
    batches=[]
    for start in range(0,len(rows),group):
        selected=rows[start+rank*batch_size:start+(rank+1)*batch_size]
        lengths={len(r['input_ids']) for r in selected}
        if len(lengths)!=1: raise ValueError('pretokenized rows within a microbatch require equal length; supply explicit padding')
        batches.append({k:torch.tensor([r[k] for r in selected],dtype=torch.long) for k in selected[0]})
    return batches


def block_layers(model):
    """Repeated transformer blocks of a HF model, best effort.

    Qwen3-style ``model.model.layers``, Roberta-style ``model.model.encoder.layer``,
    and host-mounted variants (``model.<host>.encoder.layer`` as used by
    AutoModel wrappers around XLMRoberta/BERT backbones, e.g. bge-m3).
    Sharding per block keeps the forward all-gather peak at one layer instead
    of the whole model; models without such repetition fall back to
    root-only wrapping.
    """
    inner=getattr(model,'model',None)
    layers=getattr(inner,'layers',None) if inner is not None else None
    if not layers:
        encoder=getattr(inner,'encoder',None) if inner is not None else None
        layers=getattr(encoder,'layer',None) if encoder is not None else None
    if not layers:
        for host_name in ('roberta','bert','xlm-roberta','xlmroberta'):
            host=getattr(model,host_name,None)
            if host is None: continue
            encoder=getattr(host,'encoder',None)
            layers=getattr(encoder,'layer',None) if encoder is not None else None
            if layers: break
    if not layers:
        # Backbone models themselves (XLMRobertaModel from AutoModel, as used
        # for bge-m3) expose the encoder directly: model.encoder.layer.
        encoder=getattr(model,'encoder',None)
        layers=getattr(encoder,'layer',None) if encoder is not None else None
    if not layers:
        layers=getattr(model,'layers',None)
    return list(layers) if layers else []


def apply_sharding(model, device_type, world_size):
    """Wrap with FSDP2 fully_shard: repeated blocks first, then the root.

    Returns (model, wrapping description, device mesh) so the protocol
    records the structure that was actually used and observers can bind
    the mesh's collective group.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    mesh=init_device_mesh(device_type,(world_size,))
    wrapped=block_layers(model)
    for layer in wrapped: fully_shard(layer,mesh=mesh)
    fully_shard(model,mesh=mesh)
    return model, ('per-block+root' if wrapped else 'root-only'), mesh


def monitor_start(root,vendor,world,rank, *, control=None, reference=None):
    """Opt-in node monitoring and online diagnosis; FLAGPROF_TRAIN_MONITOR=on.

    Mirrors the inference semantics per platform. When the vendor has device
    telemetry (ascend, nvidia) the session uses the independent agent
    topology: global rank 0 reuses or starts a detached `flagprof watch`
    supervisor under FLAGPROF_TRAIN_MONITOR_ROOT (default: the store root),
    the reference is broadcast over a gloo handshake group, and every rank
    relocates a per-run reference, registers as a target and attaches the
    diagnostic probe (FLAGPROF_TRAIN_DIAGNOSE, default on) so heartbeat,
    progress, communication and OOM findings share one timeline. On vendors
    without device telemetry (e.g. thead) the session uses the local sidecar
    topology — the same as inference today — and diagnosis stays off; the
    chosen topology and diagnose state are returned for the attachment and
    the run protocol. Best-effort by construction: every failure is recorded
    and never fails the training run.
    """
    if os.environ.get('FLAGPROF_TRAIN_MONITOR')!='on': return None
    from flagprof.monitor.lifecycle import Monitor
    try:
        interval=int(os.environ.get('FLAGPROF_TRAIN_MONITOR_INTERVAL_MS','1000'))
    except ValueError:
        interval=1000
    interval=max(100,min(5000,interval))
    monitor_root=str(Path(os.environ.get('FLAGPROF_TRAIN_MONITOR_ROOT') or root).resolve())
    reference = str(Path(reference or Path(root)/'monitor-ref.json').resolve())
    owner = supervisor = None

    def stop_child(child):
        if child is None or child.poll() is not None:
            return
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    def start_agent():
        nonlocal supervisor, owner
        from flagprof.monitor.subscriptions import request
        try:
            request(monitor_root, dict(action='hello'))
        except (OSError, ValueError):
            supervisor = subprocess.Popen(
                [sys.executable, '-B', '-S', '-m', 'flagprof', 'watch', 'start',
                 '--root', monitor_root, '--vendor', vendor],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + 10
            while True:
                try:
                    request(monitor_root, dict(action='hello'))
                    break
                except (OSError, ValueError) as exc:
                    if supervisor.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError('watch supervisor failed to become ready') from exc
                    time.sleep(.2)
        owner = Monitor(root, reference, vendor, dict(monitor='on', monitor_topology='agent',
                        monitor_root=monitor_root, monitor_interval_ms=interval))
        receipt = json.loads(Path(reference).read_text())
        if receipt.get('status') != 'running' or not receipt.get('session_id'):
            raise RuntimeError('watch session unavailable: ' + str(receipt.get('errors')))
        return dict(topology='agent', reference=reference,
                    diagnose='on' if os.environ.get('FLAGPROF_TRAIN_DIAGNOSE','on')=='on' else 'off')

    def start_local(path):
        nonlocal owner
        owner = Monitor(root, path, vendor, dict(monitor='on', monitor_topology='local',
                        monitor_interval_ms=interval))
        receipt = json.loads(Path(path).read_text())
        if not receipt.get('monitor_run_id') or receipt.get('status') == 'failed':
            raise RuntimeError('local monitor unavailable: ' + str(receipt.get('errors')))
        if owner.child is not None and owner.child.poll() is not None:
            raise RuntimeError('local monitor exited during startup')
        return dict(topology='local', diagnose='off', reference=path)

    decision = None
    if rank == 0:
        errors = []
        if vendor in ('ascend', 'nvidia'):
            try:
                decision = start_agent()
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(str(error))
                if owner:
                    owner.close()
                stop_child(supervisor)
                owner = supervisor = None
        if decision is None:
            try:
                local_reference = str(Path(reference).with_suffix('.local.json')) if errors else reference
                decision = start_local(local_reference)
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(str(error))
                if owner:
                    owner.close()
                    owner = None
                decision = dict(topology='unavailable', diagnose='off', reference=None)
        decision['errors'] = errors
    def close():
        try:
            if owner is not None:
                owner.close()
        finally:
            stop_child(supervisor)

    if world > 1:
        import torch.distributed as dist
        try:
            if control is None:
                control = dist.new_group(backend='gloo')
            message = [decision]
            dist.broadcast_object_list(message, src=0, group=control)
            decision = message[0]
        except BaseException:
            close()
            raise
    if supervisor is not None:
        atexit.register(stop_child, supervisor)
    topology = decision['topology']
    if topology == 'agent':
        config = dict(monitor='on', diagnose=decision['diagnose'], monitor_topology='agent',
                      monitor_root=monitor_root)
    elif topology == 'local':
        config = dict(monitor='on', monitor_topology='local')
    else:
        config = dict(monitor='off', diagnose='off')
    return dict(decision, owner=owner, supervisor=supervisor, handshake=control, config=config, close=close)


def record_monitor_status(handle, monitoring, device):
    topology = monitoring['topology']
    diagnose = monitoring['diagnose']
    errors = list(monitoring['errors'])
    if handle.monitor_enabled and topology != 'unavailable':
        handle.register_device(device)
        try:
            receipt = json.loads(handle.monitor_reference.read_text())
            if receipt.get('worker_proc_key') != handle.state['process']:
                raise ValueError('monitor registration missing this rank')
            if topology == 'agent':
                if receipt.get('subscription_status') != 'registered' or not receipt.get('target_id'):
                    raise ValueError(receipt.get('reason') or 'agent target registration unavailable')
            elif not receipt.get('monitor_run_id') or handle.state.get('monitor_errors'):
                raise ValueError('local target registration unavailable')
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            topology, diagnose = 'unavailable', 'off'
        if diagnose == 'on' and (handle.diagnostic is None or handle.state.get('diagnostic_status')):
            errors.append(handle.state.get('diagnostic_status') or 'diagnostic probe unavailable')
            diagnose = 'off'
    if errors:
        handle.state['monitor_start_errors'] = errors
    handle.state.update(monitor_topology=topology, monitor_diagnose=diagnose)
    handle._save()


def observation_start(args, *, default_vendor=None, diagnostic='flagprof-train-observe/1'):
    """Opt-in diagnostic collector attach; FLAGPROF_TRAIN_OBSERVE names a store root.

    Unset keeps plain runs byte-identical to production. Attaches the existing
    flagprof official profiler session (perf preset) to this rank so the
    training phases (flagprof/training/* record_function markers) and the
    device operators land in one trace; observation never changes execution
    semantics, and the run's protocol carries the diagnostic mark. DDP ranks
    install the CommunicationProbe on a gloo control group afterwards.

    Customer-facing conveniences: the vendor profiler tracing switch is set
    when absent (XPU stacks refuse to start their runtime profiler without
    it), a missing identity file is generated from the live torch device
    stack (real framework UUIDs) instead of failing the attach, and the same
    node monitoring / online diagnosis / result-report reading experience as
    inference is available: FLAGPROF_TRAIN_MONITOR=on starts one monitor
    session per store root with per-rank targets (FLAGPROF_TRAIN_DIAGNOSE,
    FLAGPROF_TRAIN_MONITOR_INTERVAL_MS tune it), and after a successful run
    each rank renders its own report (report.html/report.md next to the run
    dir; FLAGPROF_TRAIN_REPORT=off skips it). Monitoring and report
    generation are best-effort and never change execution semantics.
    """
    root=os.environ.get('FLAGPROF_TRAIN_OBSERVE')
    if not root: return None
    from flagprof.runtime import attach
    from flagprof.store import create_run
    # Vendor auto-detection: npu devices always run the built-in ascend
    # session; everything else keeps the historical thead default. The env
    # variable still overrides explicitly.
    device_hint=str(getattr(args,'device',''))
    default_vendor = default_vendor or ('ascend' if device_hint.startswith('npu') else 'thead')
    vendor = os.environ.get('FLAGPROF_TRAIN_OBSERVE_VENDOR', default_vendor)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    from . import coordinated_start, observation_group
    group_id, control = observation_group(rank, world)
    cleanup = ExitStack()
    handle = None
    try:
        with coordinated_start(control, rank, world):
            preset = os.environ.get('FLAGPROF_TRAIN_PRESET', 'perf')
            if preset not in ('perf', 'perf-lite', 'coverage', 'modules'):
                raise ValueError('FLAGPROF_TRAIN_PRESET must be one of perf/perf-lite/coverage/modules')
            capture_mode = os.environ.get('FLAGPROF_TRAIN_CAPTURE', 'final')
            if capture_mode not in ('final', 'windowed'):
                raise ValueError('FLAGPROF_TRAIN_CAPTURE must be final or windowed')
            capture_config = json.loads(os.environ.get('FLAGPROF_TRAIN_CAPTURE_CONFIG') or '{}')
            identity_note = 'vendor-session'
            if vendor == 'thead':
                identity_note = 'sourced' if os.environ.get('FLAGPROF_THEAD_IDENTITY') else 'autogenerated'
            run = create_run(Path(root)/('rank-'+str(rank)), identity=dict(
                diagnostic=diagnostic, vendor=vendor, preset=preset, task=args.task, group_id=group_id,
                world_size=world, identity_source=identity_note))
            if 'XPU_ENABLE_PROFILER_TRACING' not in os.environ:
                os.environ['XPU_ENABLE_PROFILER_TRACING'] = '1'
                cleanup.callback(os.environ.pop, 'XPU_ENABLE_PROFILER_TRACING', None)
            if identity_note == 'autogenerated':
                from flagprof.vendors.thead.session import generate_identity_file
                identity_path = run/'identity-autogenerated.json'
                generate_identity_file(identity_path, rank=rank)
                os.environ['FLAGPROF_THEAD_IDENTITY'] = str(identity_path)
                cleanup.callback(os.environ.pop, 'FLAGPROF_THEAD_IDENTITY', None)
        monitoring = monitor_start(root, vendor, world, rank, control=control, reference=run.parent/('monitor-ref-'+run.name+'.json'))
        if monitoring:
            cleanup.callback(monitoring['close'])
        with coordinated_start(control, rank, world):
            handle = attach(run, vendor, preset=preset, rank=rank, capture_mode=capture_mode,
                            group_id=group_id, monitor_parent=monitoring['reference'] if monitoring else None,
                            config=dict(capture_config, **(monitoring['config'] if monitoring else {})) or None)
            if handle.state['status'] != 'running':
                errors = list(handle.state.get('errors', []))
                for producer in sorted(run.glob('capture-staging/*/producer.json')):
                    try:
                        errors.extend(json.loads(producer.read_text(encoding='utf-8')).get('errors', []))
                    except (OSError, ValueError):
                        pass
                raise RuntimeError('train observation attachment did not start: ' + str(errors))
            if monitoring:
                record_monitor_status(handle, monitoring, device_hint)
            from .oracle import start_from_env
            handle._oracle_pending = start_from_env(handle)
        probe = None
        with coordinated_start(control, rank, world):
            if control is not None and handle.state['effective_config']['probes'] != 'none':
                from flagprof.probes.communication import CommunicationProbe
                probe = CommunicationProbe(handle.probe_manager, control)
                handle.probe_manager.communication = probe
                probe.install()
        return dict(handle=handle, probe=probe, control=control, rank=rank, world=world, run=run,
                    monitoring=monitoring, cleanup=cleanup)
    except BaseException:
        try:
            if handle is not None:
                try:
                    capture = getattr(handle, '_oracle_pending', None)
                    if capture is not None:
                        capture.close()
                finally:
                    handle.close(normal=False, returncode=1)
        finally:
            cleanup.close()
            if control is not None:
                import torch.distributed as dist
                dist.destroy_process_group(control)
        raise


def observation_close(observe, *, failed=False, report=True):
    from contextlib import nullcontext
    from . import coordinated_start
    handle = observe['handle']
    control = observe['control']
    def finish_phase():
        if failed or not observe['monitoring']:
            return nullcontext()
        return coordinated_start(control, observe['rank'], observe['world'])
    try:
        try:
            with finish_phase():
                try:
                    capture = getattr(handle, '_oracle_pending', None)
                    if capture is not None:
                        capture.close()
                finally:
                    handle.close(normal=not failed, returncode=1 if failed else 0)
        finally:
            # Rank 0 stops its monitor after target handles finish. Peers read
            # the monitor inventory after the owner completes reclamation.
            with finish_phase():
                observe['cleanup'].close()
    finally:
        if control is not None:
            import torch.distributed as dist
            dist.destroy_process_group(control)
    if failed:
        return
    from flagprof.store.manifest import finalize
    try:
        observe['manifest'] = finalize(observe['run'])
    except Exception as error:
        handle.state['finalize_error'] = type(error).__name__ + ': ' + str(error)
        handle._save()
    if report and os.environ.get('FLAGPROF_TRAIN_REPORT') != 'off':
        from flagprof.report import generate
        try:
            observe['report'] = generate(observe['run'])
        except Exception as error:
            handle.state['report_error'] = type(error).__name__ + ': ' + str(error)
            handle._save()


def execute(args):
    import torch
    from .engine import train
    from . import phase as training_phase
    import torch.distributed as dist
    if args.task is None:
        if args.model:
            from .registry import resolve_task
            args.task=resolve_task(args.model)
            task_source='registry'
            if not args.task:
                raise ValueError('--task is required: model %r is not in inference/targets/model_registry.json; '
                                 'pass one of: classification, causal-lm, ctr, embedding, rerank' % args.model)
        else:
            args.task='classification'
            task_source='default'
    else:
        task_source='explicit'
    if args.steps<=0 or args.batch_size<=0 or args.accumulation<=0:
        raise ValueError('steps, batch-size and accumulation must be positive')
    if not 0<args.lr<float('inf'): raise ValueError('lr must be positive and finite')
    if args.ddp and args.task=='embedding' and not args.inputs:
        raise ValueError('DDP ranks must share an explicit --inputs; default corpus generation is single-process only')
    if args.ddp and args.task=='ctr' and not args.package:
        raise ValueError('DDP ranks must share an explicit --package; default corpus generation is single-process only')
    if args.ddp and args.task=='causal-lm' and not args.inputs:
        raise ValueError('DDP ranks must share an explicit --inputs; default corpus generation is single-process only')
    if args.sharding!='none' and not args.ddp:
        raise ValueError('--sharding fsdp requires --ddp (torchrun); a single process has nothing to shard across')
    if args.sharding!='none' and (args.checkpoint or args.resume):
        raise ValueError('checkpoint/resume is not yet supported with FSDP sharding; per-rank shard checkpoints are a recorded limitation, not silently dropped')
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if world!=1 and not args.ddp: raise ValueError('torchrun requires explicit --ddp')
    if args.ddp and world<2: raise ValueError('DDP requires at least two torchrun workers')
    device=args.device
    if args.backend=='hccl' and not device.startswith('npu'):
        raise ValueError('--backend hccl requires a npu device (got %r); use nccl for cuda accelerators' % device)
    if args.backend=='nccl' and device.startswith('npu'):
        raise ValueError('npu devices use the hccl backend; drop --backend nccl or pass --backend hccl')
    if device in ('cuda','npu'): device=device+':0'
    if args.ddp and device.startswith(('cuda','npu')): device=device.split(':')[0]+':'+str(local_rank)
    if device.startswith('cuda'): torch.cuda.set_device(device)
    elif device.startswith('npu'): torch.npu.set_device(device)
    torch.manual_seed(args.seed);random.seed(args.seed)
    owned=False
    observe=None
    failed=False
    try:
        if args.ddp:
            from datetime import timedelta
            # The store timeout covers the whole process-group bootstrap,
            # including the vendor stack's first-collective handshake (e.g.
            # the FlagCX device-key exchange on XPU), so slow first-compile
            # platforms need headroom. FLAGPROF_TRAIN_DIST_TIMEOUT seconds,
            # default 120 unchanged.
            try:
                dist_timeout=float(os.environ.get('FLAGPROF_TRAIN_DIST_TIMEOUT','120'))
            except ValueError:
                dist_timeout=120.0
            dist.init_process_group(args.backend or ('nccl' if device.startswith('cuda') else 'hccl' if device.startswith('npu') else 'gloo'),timeout=timedelta(seconds=dist_timeout))
            owned=True
        args.device=device
        observe=observation_start(args)
        if args.ddp and observe:
            observe['handle'].state['parallel']=dict(rank=observe['rank'],world_size=observe['world'],
                                                     backend=dist.get_backend())
            observe['handle']._save()
        protocol=dict(task=args.task,seed=args.seed,lr=args.lr,batch_size_per_rank=args.batch_size,
                      data='synthetic labelled classification' if args.task=='classification' else 'explicit pretokenized local input',
                      task_source=task_source,
                      numerical_policy='framework defaults; no TF32/determinism/reduction override',
                      dtype='float32 parameters; optional explicit AMP',backend=dist.get_backend() if args.ddp else None)
        if observe:
            protocol['observation']=dict(diagnostic='flagprof-train-observe/1',
                                         vendor=observe['handle'].state.get('vendor'),
                                         preset=observe['handle'].state['preset'])
            oracle_at=observe['handle'].state.get('oracle_at')
            if oracle_at:
                protocol['observation']['oracle']=dict(diagnostic='training-oracle/1',
                                                       boundaries=oracle_at,
                                                       scope='activation capture; purely observational')
        if args.task=='classification':
            if args.model or args.inputs: raise ValueError('classification demo does not accept model or inputs')
            model=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.ReLU(),torch.nn.Linear(32,4))
            generator=torch.Generator().manual_seed(args.seed+1)
            batches=[]
            with training_phase('data', step=0):
                for _ in range(4):
                    x=torch.randn(args.batch_size*world,16,generator=generator)
                    y=torch.randint(0,4,(args.batch_size*world,),generator=generator)
                    sl=slice(rank*args.batch_size,(rank+1)*args.batch_size)
                    batches.append(dict(features=x[sl],labels=y[sl]))
            def loss_fn(m,b):return torch.nn.functional.cross_entropy(m(b['features']),b['labels'])
        elif args.task=='ctr':
            if args.model or args.inputs: raise ValueError('ctr task does not accept model or inputs')
            from .ctr import TASK_ARCHS
            if not args.ctr_arch:
                raise ValueError('ctr task requires --ctr-arch with one of: '+', '.join(TASK_ARCHS))
            if not args.package and args.ddp:
                raise ValueError('DDP ranks must share an explicit --package')
            package=args.package
            if not package:
                package=ensure_generated('ctr-package',args.outputs,args.seed)
                protocol['data']='deterministic sealed CTR corpus generated next to --outputs (make_ctr_package, 1024 samples)'
                protocol['default_package_generated']=True
            from .ctr import prepare_ctr
            with training_phase('data', step=0):
                model,batches,loss_fn,ctr_protocol=prepare_ctr(arch=args.ctr_arch,package=package,
                    seed=args.seed,batch_size=args.batch_size,rank=rank,world_size=world)
            protocol.update(ctr_protocol)
        elif args.task=='embedding':
            if args.package or args.ctr_arch: raise ValueError('embedding task does not accept package or ctr-arch')
            if not args.model: raise ValueError('embedding requires --model')
            from .registry import resolve
            pooling,pooling_source=resolve(args.pooling,args.model,kind='pooling')
            if not pooling:
                raise ValueError('embedding requires --pooling (last|cls|mean); model %r is not in '
                                 'inference/targets/model_registry.json, so no default pooling is known' % args.model)
            protocol['pooling_source']=pooling_source
            inputs=args.inputs
            if not inputs:
                inputs=ensure_generated('corpus',args.outputs,args.seed)
                protocol['data']='deterministic synthetic triplet corpus generated next to --outputs (make_embedding_corpus, 32 groups, 3 negatives)'
            from transformers import AutoModel,AutoModelForCausalLM
            source=args.model.removeprefix('hf:')
            if not Path(source).is_dir(): raise ValueError('local model directory missing')
            protocol['model_files_sha256']=local_files(source)
            protocol['attention']='eager'
            try:
                model=AutoModel.from_pretrained(source,local_files_only=True,trust_remote_code=False,
                                                torch_dtype=torch.float32,attn_implementation='eager')
                protocol['model_class']='AutoModel'
            except Exception:
                model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=False,
                                                           torch_dtype=torch.float32,attn_implementation='eager').model
                protocol['model_class']='AutoModelForCausalLM.base'
            from .embedding import prepare_embedding
            with training_phase('data', step=0):
                model,batches,loss_fn,emb_protocol=prepare_embedding(model=model,inputs=inputs,
                    pooling=pooling,temperature=args.temperature,batch_size=args.batch_size,
                    rank=rank,world_size=world)
            protocol.update(emb_protocol)
        elif args.task=='rerank':
            if args.inputs: raise ValueError('rerank task does not accept inputs')
            if not args.package or not args.model: raise ValueError('rerank task requires --package and --model')
            from .registry import resolve
            score,score_source=resolve(args.rerank_score,args.model,kind='rerank_score')
            if not score:
                raise ValueError('rerank requires --rerank-score (causal_yesno|sequence_logit); model %r is not in '
                                 'inference/targets/model_registry.json, so no default scoring semantics is known' % args.model)
            protocol['rerank_score_source']=score_source
            from transformers import AutoTokenizer
            source=args.model.removeprefix('hf:')
            if not Path(source).is_dir(): raise ValueError('local model directory missing')
            protocol['model_files_sha256']=local_files(source)
            protocol['attention']='eager'
            tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
            yes_id=no_id=None
            if score=='causal_yesno':
                yes_id=tokenizer.convert_tokens_to_ids('yes')
                no_id=tokenizer.convert_tokens_to_ids('no')
                if yes_id==tokenizer.unk_token_id or no_id==tokenizer.unk_token_id:
                    raise ValueError('tokenizer lacks literal yes/no tokens')
            cls_name='AutoModelForCausalLM' if score=='causal_yesno' else 'AutoModelForSequenceClassification'
            import transformers
            model=getattr(transformers,cls_name).from_pretrained(source,local_files_only=True,trust_remote_code=False,
                                                                torch_dtype=torch.float32,attn_implementation='eager')
            protocol['model_class']=cls_name
            if hasattr(model.config,'use_cache'): model.config.use_cache=False
            from .rerank import prepare_rerank
            with training_phase('data', step=0):
                model,batches,loss_fn,rerank_protocol=prepare_rerank(model=model,package=args.package,
                    score=score,batch_size=args.batch_size,yes_id=yes_id,no_id=no_id,
                    rank=rank,world_size=world)
            protocol.update(rerank_protocol)
        else:
            if not args.model: raise ValueError('causal-lm requires --model')
            from transformers import AutoModelForCausalLM
            source=args.model.removeprefix('hf:')
            if not Path(source).is_dir(): raise ValueError('local model directory missing')
            protocol['model_files_sha256']=local_files(source)
            inputs=args.inputs
            if not inputs:
                if args.ddp: raise ValueError('DDP ranks must share an explicit --inputs; default corpus generation is single-process only')
                config=json.loads((Path(source)/'config.json').read_text(encoding='utf-8'))
                vocab=int(config.get('vocab_size') or 65536)
                # Keep at least four full microbatches and sixteen sample rows.
                batch_count=max(4, (16 + args.batch_size - 1) // args.batch_size)
                corpus=ensure_generated('lm-corpus',args.outputs,args.seed,vocab=vocab,
                                        rows=batch_count*args.batch_size)
                inputs=Path(corpus)/'tokens.jsonl'
                protocol['data']='deterministic synthetic LM tokens generated next to --outputs (make_lm_corpus)'
                protocol['default_corpus_generated']=True
            model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=False,
                                                       torch_dtype=torch.float32,attn_implementation='eager')
            protocol['attention']='eager'
            with training_phase('data', step=0):
                batches=prepared_lm(inputs,batch_size=args.batch_size,rank=rank,world_size=world)
            def loss_fn(m,b):return m(**b).loss
        model=model.to(device)
        if args.ddp:
            if args.sharding=='fsdp':
                model,wrapping,mesh=apply_sharding(model,torch.device(device).type,world)
                protocol['sharding']='fsdp'
                protocol['fsdp_wrapping']=wrapping
                if observe and observe['probe'] is not None:
                    observe['probe'].model_groups.append(mesh.get_group())
            else:
                protocol['sharding']='ddp-replica'
                model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank] if device.startswith('cuda') else None)
        else:
            protocol['sharding']='none'
        optimizer=(torch.optim.SGD(model.parameters(),lr=args.lr) if args.optimizer=='sgd' else
                   torch.optim.AdamW(model.parameters(),lr=args.lr))
        def rank_path(path):return str(path)+'.rank-'+str(rank) if path and args.ddp else path
        outputs=Path(args.outputs)/('rank-'+str(rank)) if args.ddp else Path(args.outputs)
        result=train(model,optimizer,batches,loss_fn,outputs=outputs,steps=args.steps,
                     accumulation=args.accumulation,device=device,amp=args.amp,mode=args.mode,protocol=protocol,
                     checkpoint=rank_path(args.checkpoint),resume=rank_path(args.resume),stop_after=args.stop_after,
                     rank=rank,world_size=world)
        print(json.dumps(dict(output=str(outputs),status=result['status'],next_step=result['next_step'],
                              final_model_sha256=result['final_model_sha256'])),flush=True)
        return 0
    except BaseException:
        failed=True
        raise
    finally:
        try:
            if observe:
                observation_close(observe, failed=failed)
        finally:
            if owned:
                dist.destroy_process_group()


def main(argv=None):
    args=parser().parse_args(argv)
    return execute(args)


if __name__=='__main__':
    raise SystemExit(main())
