# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Explicit loss/gradient/update comparison of completed training runs."""
import argparse
import json
import math
from pathlib import Path


def compare(left, right, *, atol=0., rtol=0., declare=()):
    import torch
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError('finite nonnegative tolerances required')
    roots = (Path(left), Path(right))
    results = [json.loads((p/'training-result.json').read_text()) for p in roots]
    if any(r.get('status') != 'complete' for r in results):
        raise ValueError('comparison requires completed training runs')
    a, b = [r['protocol'] for r in results]
    if results[0].get('observed_numerics') != results[1].get('observed_numerics'):
        raise ValueError('observed numerical backend settings differ')
    differences = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    allowed = {'backend', 'world_size', 'rank'}
    if set(declare) - allowed:
        raise ValueError('only backend/world_size/rank differences may be declared')
    if set(differences) - set(declare):
        raise ValueError('training protocol differs: '+', '.join(differences))
    if any(r.get('snapshot_mode') != 'accuracy' for r in results):
        raise ValueError('full gradient/update comparison requires accuracy snapshots')
    trajectories = [{row['step']: row for row in r['steps']} for r in results]
    if set(trajectories[0]) != set(trajectories[1]) or not trajectories[0]:
        raise ValueError('training step coverage differs or is empty')
    rows = []

    def inspect(a, b, path, index):
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise ValueError('tensor structure differs at '+path)
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError('tensor shape/dtype differs at '+path)
            exact = a.reshape(-1).view(torch.uint8).equal(b.reshape(-1).view(torch.uint8))
            if not (a.is_floating_point() or a.is_complex()):
                mismatch = a != b
                rows.append(dict(step=index, path=path, elements=a.numel(), exact=exact,
                                 mismatches=int(mismatch.sum()), nonfinite=0, max_abs=None,
                                 comparison='integer/bool exact; tolerance not applied',
                                 status='different' if bool(mismatch.any()) else 'pass'))
                return
            dtype = torch.complex128 if a.is_complex() else torch.float64
            x, y = a.to(dtype), b.to(dtype)
            finite = torch.isfinite(x) & torch.isfinite(y)
            delta = (x-y).abs()
            mismatch = ~(finite & (delta <= atol + rtol*x.abs()))
            rows.append(dict(step=index, path=path, elements=a.numel(), exact=exact,
                             mismatches=int(mismatch.sum()), nonfinite=int((~finite).sum()),
                             max_abs=float(delta[finite].max()) if bool(finite.any()) else None,
                             reference='left', status='pass' if not bool(mismatch.any()) else 'different'))
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys(): raise ValueError('mapping keys differ at '+path)
            for key in a: inspect(a[key], b[key], path+'/'+str(key), index)
        elif isinstance(a, (list, tuple)) and isinstance(b, type(a)):
            if len(a) != len(b): raise ValueError('sequence length differs at '+path)
            for k, (x, y) in enumerate(zip(a, b)): inspect(x, y, path+'/'+str(k), index)
        elif a != b:
            rows.append(dict(step=index, path=path, status='different', ref=a, cmp=b))

    for index in sorted(trajectories[0]):
        pair = [table[index] for table in trajectories]
        inspect(torch.tensor(pair[0]['microbatch_losses'], dtype=torch.float64),
                torch.tensor(pair[1]['microbatch_losses'], dtype=torch.float64), 'loss', index)
        snapshots = []
        for root, row in zip(roots, pair):
            path = (root/row['snapshot']).resolve()
            if not path.is_relative_to(root.resolve()): raise ValueError('snapshot escapes run')
            snapshots.append(torch.load(path, map_location='cpu', weights_only=True))
        inspect(*snapshots, 'snapshot', index)
    return dict(schema='training-comparison/1', status='pass' if all(r['status']=='pass' for r in rows) else 'different',
                atol=atol, rtol=rtol, declared_protocol_differences=differences, rows=rows,
                environments=[dict(device=r['device'],torch_version=r['torch_version'],
                                   observed_numerics=r.get('observed_numerics')) for r in results],
                scope='observed loss, gradients, model buffers/parameters and optimizer state; not convergence certification')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ref'); parser.add_argument('cmp'); parser.add_argument('--out', required=True)
    parser.add_argument('--atol', type=float, default=0.); parser.add_argument('--rtol', type=float, default=0.)
    parser.add_argument('--declare', action='append', default=[], choices=('backend','world_size','rank'))
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    for source in (args.ref, args.cmp):
        root = Path(source).resolve()
        if out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError('comparison output must be separate from source runs')
    result = compare(args.ref, args.cmp, atol=args.atol, rtol=args.rtol, declare=args.declare)
    out.mkdir(parents=True, exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# 训练精度对比', '', '| 状态 | atol | rtol | 比较记录 |', '|---|---:|---:|---:|',
             '| '+result['status']+' | '+str(args.atol)+' | '+str(args.rtol)+' | '+str(len(result['rows']))+' |',
             '', result['scope'], '', '| 步骤 | 张量/状态 | 结果 | 不匹配元素 | 最大绝对误差 |', '|---:|---|---|---:|---:|']
    lines += ['| '+str(row['step'])+' | '+row['path']+' | '+row['status']+' | '+str(row.get('mismatches',''))+' | '+str(row.get('max_abs',''))+' |' for row in result['rows']]
    (out/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(out), status=result['status'])))
    return 0 if result['status']=='pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
