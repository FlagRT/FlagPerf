# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""A small explicit training loop; arbitrary applications can use phase() only."""
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from . import attachment, phase, register_model
from .state import cpu_state, fingerprint, raw_model, restore, save


def synchronize(device):
    import torch
    dev=torch.device(device)
    if dev.type=='cuda': torch.cuda.synchronize(dev)
    elif dev.type!='cpu': raise ValueError('unsupported synchronization device: '+str(dev))


def atomic_json(path, value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def train(model, optimizer, batches, loss_fn, *, outputs, steps=3, accumulation=1,
          device='cpu', amp='off', mode='accuracy', protocol=None, checkpoint=None,
          resume=None, stop_after=None, world_size=1, rank=0):
    """batches: fixed CPU dicts; loss_fn(model, moved_batch) returns scalar loss.

    An optimizer step consumes exactly `accumulation` equally weighted loss
    microbatches; each microbatch's loss reduction is defined by the task.
    Inputs cycle by absolute microstep, making optimizer-boundary resume exact.
    No automatic clipping, scheduler, TF32 changes, or backend substitutions.
    """
    import torch
    from torch.nn.parallel import DistributedDataParallel
    for name,v in [('steps',steps),('accumulation',accumulation),('world_size',world_size)]:
        if type(v) is not int or v<=0: raise ValueError(name+' must be a positive integer')
    if not batches: raise ValueError('training batches are empty')
    if mode not in ('accuracy','performance'): raise ValueError('unknown training mode')
    if amp not in ('off','bfloat16','float16'): raise ValueError('unknown AMP policy')
    if amp=='float16' and not str(device).startswith('cuda'):
        raise ValueError('FP16 AMP requires CUDA-compatible device with GradScaler')
    if rank<0 or rank>=world_size: raise ValueError('invalid rank')
    if world_size>1 and not isinstance(model,DistributedDataParallel):
        raise ValueError('world_size requires DistributedDataParallel')
    if stop_after is not None and (type(stop_after) is not int or not 0<stop_after<=steps):
        raise ValueError('stop_after must be within planned optimizer steps')
    destination=Path(outputs)
    destination.mkdir(parents=True,exist_ok=False)
    register_model(raw_model(model))
    model.train()
    batch_digest=fingerprint(batches)
    protocol=dict(protocol or {}, engine='training-loop/1', steps=steps, accumulation=accumulation,
                  amp=amp, mode=mode, input_sha256=batch_digest, world_size=world_size, rank=rank,
                  optimizer_class=type(optimizer).__module__+'.'+type(optimizer).__qualname__,
                  initial_optimizer_sha256=fingerprint(optimizer.state_dict()),
                  initial_model_sha256=fingerprint(raw_model(model).state_dict()))
    scaler=torch.amp.GradScaler('cuda') if amp=='float16' else None
    result=dict(schema='training-result/1',status='running',protocol=protocol,
                device=str(device),torch_version=torch.__version__,steps=[],next_step=0,
                observed_numerics=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                    cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
                attachment=bool(attachment()),snapshot_mode=mode,
                timing_scope='synchronized optimizer-step wall time; accuracy includes pre-update gradient capture; excludes post-update snapshots/checkpoint/output',
                qualification='not_qualified',reason='functional training workload; no convergence or exclusive performance qualification')
    handle=attachment()
    if handle:
        handle.writer.emit(dict(t='meta',kind='env',data=dict(training=protocol)))
    start_step=0
    try:
        if resume: start_step=restore(resume,model,optimizer,protocol=protocol,device=device,scaler=scaler)
        if start_step>steps: raise ValueError('checkpoint is beyond planned steps')
        if stop_after is not None and stop_after<=start_step:
            raise ValueError('stop_after must follow the resumed checkpoint position')
        result.update(start_step=start_step,next_step=start_step)
        atomic_json(destination/'training-result.json',result)
        for index in range(start_step,stop_after or steps):
            losses=[]; samples=0; tokens=0
            synchronize(device); started=time.perf_counter_ns()
            with phase('step',step=index,accumulation=accumulation):
                with phase('zero_grad',step=index): optimizer.zero_grad(set_to_none=True)
                for micro in range(accumulation):
                    batch=batches[(index*accumulation+micro)%len(batches)]
                    moved={k:v.to(device) for k,v in batch.items()}
                    if handle and handle.probe_manager and handle.probe_manager.tensor:
                        size=next(iter(batch.values())).shape[0]
                        handle.bind_tensor_inputs(
                            ['rank-'+str(rank)+'/step-'+str(index)+'/micro-'+str(micro)+'/sample-'+str(i) for i in range(size)],
                            moved)
                    samples+=next(iter(batch.values())).shape[0]
                    if 'attention_mask' in batch: tokens+=int(batch['attention_mask'].sum())
                    # no_sync wraps both forward and backward, as required by DDP.
                    ddp=model.no_sync() if isinstance(model,DistributedDataParallel) and micro+1<accumulation else nullcontext()
                    autocast=torch.autocast(torch.device(device).type,dtype=getattr(torch,amp)) if amp!='off' else nullcontext()
                    with ddp:
                        with phase('forward',step=index,microstep=micro):
                            with autocast: loss=loss_fn(model,moved)
                            if loss.ndim!=0: raise ValueError('training loss must be scalar')
                            if handle and handle.probe_manager and handle.probe_manager.tensor:
                                # The generic tensor probe expects a batch axis.
                                # The scalar objective remains in training-result.
                                handle.record_tensor('training_loss',loss.detach().expand(size))
                            scaled=loss/accumulation
                        losses.append(loss.detach())
                        with phase('backward',step=index,microstep=micro):
                            (scaler.scale(scaled) if scaler else scaled).backward()
                if scaler: scaler.unscale_(optimizer)
                if mode=='accuracy':
                    # Explicit diagnostic overhead: timing is labelled accordingly.
                    synchronize(device)
                    grad={n:p.grad.detach().cpu().clone() for n,p in raw_model(model).named_parameters() if p.grad is not None}
                    if not grad: raise RuntimeError('no parameter gradients produced')
                    gradients_finite=all(bool(torch.isfinite(t).all()) for t in grad.values())
                    if not gradients_finite and scaler is None:
                        raise FloatingPointError('nonfinite gradients')
                scale_before=scaler.get_scale() if scaler else None
                with phase('optimizer',step=index):
                    if scaler: scaler.step(optimizer);scaler.update()
                    else: optimizer.step()
                synchronize(device)
            elapsed=time.perf_counter_ns()-started
            loss_values=[float(v.cpu()) for v in losses]
            if not all(math.isfinite(v) for v in loss_values): raise FloatingPointError('nonfinite loss')
            row=dict(step=index,loss=sum(loss_values)/len(loss_values),microbatch_losses=loss_values,
                     synchronized_step_ns=elapsed,samples=samples,valid_tokens=tokens,
                     global_samples=samples*world_size,
                     samples_per_second_local=samples*1e9/elapsed,
                     timing_includes_accuracy_observation=mode=='accuracy')
            if scaler:
                row.update(scaler_before=scale_before,scaler_after=scaler.get_scale(),
                           optimizer_step_skipped=scaler.get_scale()<scale_before)
            if mode=='accuracy':
                state=cpu_state(raw_model(model).state_dict())
                snapshot=dict(model=state,gradients=grad,optimizer=cpu_state(optimizer.state_dict()),
                              scaler=scaler.state_dict() if scaler else None)
                filename='step-'+str(index)+'.pt'; torch.save(snapshot,destination/filename)
                row.update(snapshot=filename,gradients_finite=gradients_finite,
                           model_sha256=fingerprint(state),gradient_sha256=fingerprint(grad),
                           optimizer_sha256=fingerprint(snapshot['optimizer']))
            result['steps'].append(row);result['next_step']=index+1
            if handle: handle.step()
            if checkpoint: save(checkpoint,model,optimizer,next_step=index+1,protocol=protocol,device=device,scaler=scaler)
            atomic_json(destination/'training-result.json',result)
        final=cpu_state(raw_model(model).state_dict())
        torch.save(final,destination/'model-final.pt')
        if any(not bool(torch.isfinite(v).all()) for v in final.values() if v.is_floating_point()):
            raise FloatingPointError('nonfinite final model parameters or buffers')
        result.update(status='complete' if result['next_step']==steps else 'paused',
                      final_model_sha256=fingerprint(final),final_optimizer_sha256=fingerprint(optimizer.state_dict()))
    except BaseException as error:
        result.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:
        atomic_json(destination/'training-result.json',result)
    return result
