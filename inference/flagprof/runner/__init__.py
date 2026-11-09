# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Single-model inference runner; imports torch only when executing."""
import hashlib
from collections.abc import Mapping
import json
from pathlib import Path
import time


def pool_embedding(hidden, mask, *, mean=False, cls=False, normalize=True):
    """CLS token / last valid token / token mean, then FP32 L2, matching the registered workload semantics."""
    import torch
    if cls:
        pooled=hidden[:,0]
    elif mean:
        weights=mask.unsqueeze(-1).to(torch.float32)
        pooled=(hidden.float()*weights).sum(1)/weights.sum(1).clamp_min(1)
    else:
        positions=torch.arange(mask.shape[1],device=mask.device).expand_as(mask)
        indices=positions.masked_fill(mask==0,-1).max(dim=1).values
        pooled=hidden[torch.arange(hidden.shape[0],device=hidden.device),indices]
    return torch.nn.functional.normalize(pooled.float(),p=2,dim=1) if normalize else pooled


def run_prepared(model, inputs, *, warmup=0, forwards=3, outputs=None, sync=None, transform=None):
    """Run already prepared inputs through the same production registration path."""
    import torch
    from flagprof import current_attachment
    handle = current_attachment()
    if handle:
        handle.register_model(model)
    sync = sync or (lambda: None)
    records, times = [], []
    destination = Path(outputs) if outputs else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i in range(warmup + forwards):
            sync()
            start = time.perf_counter_ns()
            if handle:
                handle.context.probe_state()['forward_phase'] = 'warmup' if i < warmup else 'measure'
            value = model(**inputs) if isinstance(inputs, Mapping) else model(inputs)
            if transform:
                value = transform(value)
            sync()
            duration = time.perf_counter_ns()-start
            if handle:
                # Windowed collection advances once per completed forward.
                # Keep export/validation outside the measured envelope.
                handle.step()
            if i < warmup:
                continue
            times.append(duration)
            tensor = value.detach().cpu().contiguous()
            raw = tensor.view(torch.uint8).numpy().tobytes()
            row = dict(index=i, shape=list(tensor.shape), dtype=str(tensor.dtype),
                       finite=bool(torch.isfinite(tensor).all()), sha256=hashlib.sha256(raw).hexdigest())
            if destination:
                path = destination/(str(i)+'.bin')
                path.write_bytes(raw)
                row['file'] = path.name
            records.append(row)
    result = dict(warmup=warmup, forwards=forwards, latencies_ns=times, outputs=records,
                  attachment=handle.id if handle else None)
    if destination:
        (destination/'result.json').write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    return result
