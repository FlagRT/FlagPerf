# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Explicit trusted checkpoints and device-independent training fingerprints."""
import hashlib
import json
import os
from pathlib import Path
import random


def local_shard(value):
    """FSDP2 parameters/gradients arrive as DTensor; keep the local shard.

    Plain tensors pass through untouched, so DDP and single-device paths
    behave exactly as before. Shard placement is intentionally NOT collapsed:
    saved states and fingerprints are per-rank shard states under sharding.
    """
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        return value
    if isinstance(value, DTensor):
        return value.to_local()
    return value


def fingerprint(value):
    import torch
    h = hashlib.sha256()
    def visit(v):
        placements = getattr(v, 'placements', None)
        if placements is not None:
            h.update(json.dumps(['DTensor', sorted(str(p) for p in placements)]).encode())
            v = local_shard(v)
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(json.dumps([str(t.dtype), list(t.shape)]).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b'{')
            for key in sorted(v, key=lambda k:(type(k).__name__, repr(k))):
                visit(key); visit(v[key])
            h.update(b'}')
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for item in v: visit(item)
        else:
            h.update(json.dumps([type(v).__name__, v], sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def cpu_state(value):
    import torch
    if isinstance(value, torch.Tensor): return local_shard(value).detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_state(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_state(v) for v in value)
    return value


def raw_model(model):
    from torch.nn.parallel import DistributedDataParallel
    return model.module if isinstance(model, DistributedDataParallel) else model


def save(path, model, optimizer, *, next_step, protocol, device, scaler=None):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(schema='training-checkpoint/1', next_step=next_step, protocol=protocol,
                 model=cpu_state(raw_model(model).state_dict()), optimizer=cpu_state(optimizer.state_dict()),
                 torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                 accelerator_rng=None, scaler=scaler.state_dict() if scaler else None)
    if str(device).startswith('cuda'):
        state['accelerator_rng'] = torch.cuda.get_rng_state(device)
    elif not str(device).startswith('cpu'):
        raise ValueError('checkpoint RNG unsupported for device: '+str(device))
    temporary=path.with_name(path.name+'.tmp')
    try:
        torch.save(state, temporary)
        with temporary.open('rb+') as stream: os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def restore(path, model, optimizer, *, protocol, device, scaler=None):
    """Only call on a user-selected, trusted local checkpoint (pickle)."""
    import torch
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('schema')!='training-checkpoint/1' or state['protocol']!=protocol:
        raise ValueError('checkpoint training protocol differs')
    if bool(state['scaler'] is not None)!=bool(scaler is not None):
        raise ValueError('checkpoint scaler differs')
    if type(state['next_step']) is not int or state['next_step']<0:
        raise ValueError('invalid checkpoint position')
    raw_model(model).load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    if scaler: scaler.load_state_dict(state['scaler'])
    torch.set_rng_state(state['torch_rng'])
    random.setstate(state['python_rng'])
    if str(device).startswith('cuda'):
        if state['accelerator_rng'] is None: raise ValueError('checkpoint missing accelerator RNG')
        torch.cuda.set_rng_state(state['accelerator_rng'],device)
    elif state['accelerator_rng'] is not None:
        raise ValueError('checkpoint accelerator differs')
    return state['next_step']
