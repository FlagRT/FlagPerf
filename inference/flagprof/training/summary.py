# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Read training JSON only; never infer cross-rank timing or convergence."""
import argparse
import json
from pathlib import Path


def summarize(path):
    root = Path(path)
    paths = [root/'training-result.json'] if (root/'training-result.json').is_file() else sorted(root.glob('rank-*/training-result.json'))
    if not paths:
        raise ValueError('no training results found')
    runs = [json.loads(p.read_text()) for p in paths]
    worlds = {r['protocol']['world_size'] for r in runs}
    if len(worlds) != 1:
        raise ValueError('rank world sizes differ')
    world = worlds.pop()
    rank_ids = [r['protocol']['rank'] for r in runs]
    complete = set(rank_ids) == set(range(world)) and len(rank_ids) == world
    # CLI runs record protocol['sharding']; the tree-external diagnostic
    # script form records check_mode (fsdp_cpu_check.py), treated the same.
    shardings = {r['protocol'].get('sharding') or r['protocol'].get('check_mode') or 'none' for r in runs}
    sharded = shardings == {'fsdp'}
    identities = []
    for run in runs:
        skip = {'rank', 'input_sha256'}
        if sharded:
            skip.add('initial_model_sha256')  # rank-local shard bytes, not a knob
        identities.append({k:v for k,v in run['protocol'].items() if k not in skip})
    protocol_equal = all(p == identities[0] for p in identities)
    finals = [r.get('final_model_sha256') for r in runs]
    final_equal = bool(finals[0]) and len(set(finals)) == 1
    next_equal = len({r['next_step'] for r in runs}) == 1
    # Sharded ranks keep distinct parameter shards by design, so the only
    # cross-rank evidence available here is step coverage: divergent step
    # sequences mean the ranks did not advance the same schedule.
    step_sequences = [tuple(step['step'] for step in r['steps']) for r in runs]
    steps_aligned = len(set(step_sequences)) == 1
    states_complete = all(r['status']=='complete' for r in runs)
    # Under FSDP each rank persists its own parameter shard, so rank-final shas
    # differ by design; full-model equality is only meaningful for replicas.
    status = 'complete' if complete and protocol_equal and (steps_aligned if sharded else final_equal) and next_equal and states_complete else 'partial'
    return dict(schema='training-summary/2', status=status,
                expected_ranks=world, observed_ranks=rank_ids, rank_coverage_complete=complete,
                common_protocol_equal=protocol_equal,
                sharding='fsdp' if sharded else (shardings.pop() if len(shardings) == 1 else 'mixed'),
                rank_final_models_equal=None if sharded else final_equal,
                rank_steps_aligned=None if not sharded else steps_aligned,
                next_step_equal=next_equal, qualification='not_qualified',
                scope='rank-local training results; no cross-rank synchronized throughput, convergence or collective coverage claim',
                runs=[dict(path=str(p),rank=r['protocol']['rank'],status=r['status'],next_step=r['next_step'],
                           final_model_sha256=r.get('final_model_sha256'),steps=r['steps']) for p,r in zip(paths,runs)])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run');parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    source=Path(args.run).resolve();out=Path(args.out).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('summary must be outside source result tree')
    result=summarize(source)
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# 训练运行摘要','','| 项目 | 结果 |','|---|---|',
           '| 完整性状态 | '+result['status']+' |',
           '| rank 覆盖 | '+str(result['observed_ranks'])+' / '+str(result['expected_ranks'])+' |',
           '| 公共协议一致 | '+str(result['common_protocol_equal'])+' |',
           '| rank 最终参数相同 | '+('不适用（分片按 rank 留证）' if result['rank_final_models_equal'] is None else str(result['rank_final_models_equal']))+' |',
           '| rank 步骤覆盖一致 | '+('不适用（非分片）' if result['rank_steps_aligned'] is None else str(result['rank_steps_aligned']))+' |',
           '| 资格 | not_qualified |','',result['scope'],'',
           '| rank | step | 本地 loss | 本地样本数 | 同步整步 ms | 本地 samples/s |','|---:|---:|---:|---:|---:|---:|']
    for run in result['runs']:
        for step in run['steps']:
            lines.append('| '+str(run['rank'])+' | '+str(step['step'])+' | '+str(step['loss'])+' | '+str(step['samples'])+' | '+format(step['synchronized_step_ns']/1e6,'.3f')+' | '+format(step['samples_per_second_local'],'.3f')+' |')
    lines+=['','accuracy 模式包含梯度观察开销；首次初始化保留。这里没有整组同步计时，不将本地吞吐相加作为全局吞吐。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(out),status=result['status'])))
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
