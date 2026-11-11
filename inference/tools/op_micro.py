# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Per-op microbenchmark for accelerator comparison (diagnostic tool).

Adapted from FlagScale examples/retrieval/profile_op_micro.py (Apache License
2.0, https://github.com/FlagOpen/FlagScale). This is a standalone diagnostic:
it is not part of the record/collect protocol and must not be merged into
production capture runs. Shapes follow the retrieval trio's dominant ops at
B=8 S=512 H=1024 FFN=4096 vocab=151669, bf16.
"""
import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--iters', type=int, default=50)
    args = ap.parse_args()

    dev = torch.device(args.device)
    dt = torch.bfloat16
    B, S, H, I, V = 8, 512, 1024, 4096, 151669
    x = torch.randn(B, S, H, device=dev, dtype=dt)
    w_h = torch.randn(H, H, device=dev, dtype=dt)
    w_i = torch.randn(H, I * 2, device=dev, dtype=dt)
    w_v = torch.randn(H, V, device=dev, dtype=dt)
    w_n = torch.randn(H, device=dev, dtype=dt)
    q = torch.randn(B * S, 32, 128, device=dev, dtype=dt)
    k = torch.randn(B * S, 8, 128, device=dev, dtype=dt)

    ops = {}
    ops['mm_hidden'] = lambda: torch.mm(x.view(-1, H), w_h)
    ops['mm_ffn'] = lambda: torch.mm(x.view(-1, H), w_i)
    ops['mm_lm_head'] = lambda: torch.mm(x.view(-1, H), w_v)
    ops['flash_attn_fwd'] = lambda: F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0), enable_gqa=True)
    ops['rms_norm'] = lambda: F.rms_norm(x, (H,), w_n, 1e-6)
    ops['layer_norm'] = lambda: F.layer_norm(x, (H,), w_n, w_n, 1e-6)
    ops['silu'] = lambda: F.silu(x)
    ops['gelu'] = lambda: F.gelu(x)
    ops['add'] = lambda: x + x
    ops['mul'] = lambda: x * x
    ops['sum_last'] = lambda: x.sum(-1)
    ops['softmax'] = lambda: torch.softmax(x, dim=-1)
    ops['bmm_scores'] = lambda: torch.bmm(x[:, :128], x[:, :128].transpose(1, 2))
    ops['cross_entropy'] = lambda: F.cross_entropy(
        x.view(-1, H)[:4096].float(), torch.randint(0, H, (4096,), device=dev))

    out = {}
    for name, fn in ops.items():
        try:
            out[name] = bench(fn, iters=args.iters)
        except Exception as exc:
            out[name] = None
            print(f'  {name}: SKIP ({type(exc).__name__}: {str(exc)[:60]})')
    print(f'=== {args.label} ===')
    for name, ms in out.items():
        print(f'  {name:20} {ms:8.3f} ms' if ms else f'  {name:20} skip')
    with open(args.out, 'w') as stream:
        json.dump({'label': args.label, 'device': args.device,
                   'shape': {'B': B, 'S': S, 'H': H, 'I': I, 'V': V, 'dtype': 'bfloat16'},
                   'iters': args.iters, 'ops_ms': out}, stream, indent=1)


if __name__ == '__main__':
    main()
