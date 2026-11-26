# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Multi-rank training report index: per-rank reports plus a group index page.

Scans a training observation store root (rank-*/<run-id>), renders each
rank's report when missing (same generate() the finalize path uses), and
writes a Markdown index and a JSON inventory linking every rank report, the run manifests and
any operator-placed oracle.json. Deliberately NOT the inference TP2
delivery group format: training ranks are arbitrary world_size and their
reports are read per rank.
"""
import argparse
import json
from os.path import relpath
from urllib.parse import quote
from pathlib import Path


def build(store_root, out=None):
    root = Path(store_root)
    ranks = []
    for rank_dir in sorted(root.glob('rank-*')):
        if not rank_dir.is_dir():
            continue
        for run in sorted(p for p in rank_dir.iterdir() if (p/'manifest.json').exists()):
            ranks.append(dict(rank=rank_dir.name, run=run, manifest=run/'manifest.json'))
    if not ranks:
        raise ValueError('no rank runs found under '+str(root))
    from flagprof.report import generate
    for row in ranks:
        run = row['run']
        report = run.parent/(run.name+'-report')
        if not (report/'report.html').exists():
            generate(run, out=report)
        row['report'] = report
        oracle = run/'oracle.json'
        row['oracle'] = oracle if oracle.exists() else None
        result = report/'result.json'
        row['status'] = None
        if result.exists():
            try:
                row['status'] = json.loads(result.read_text()).get('run_status') or 'rendered'
            except ValueError:
                row['status'] = 'rendered'
    out = Path(out) if out else root.parent/(root.name+'-group')
    out.mkdir(parents=True, exist_ok=True)
    lines = ['# 训练多卡报告索引', '', f"- Store：`{relpath(root, out)}`", f"- Rank 数：{len(ranks)}", '']
    lines += ['| rank | run | 状态 | 报告 | oracle |', '|---|---|---|---|---|']
    for row in ranks:
        report = quote(relpath(row['report']/'report.html', out))
        manifest = quote(relpath(row['manifest'], out))
        oracle = f"[oracle.json]({quote(relpath(row['oracle'], out))})" if row['oracle'] else ''
        lines.append('| {} | [`{}`]({}) | {} | [report]({}) | {} |'.format(
            row['rank'].removeprefix('rank-'), row['run'].name, manifest, row['status'] or '-', report, oracle))
    lines += ['', '逐 rank 的 loss/吞吐/协议见各 rank 目录 `training-result.json`；'
              '跨 rank 汇总见 store 旁 `summary.json`（`flagprof.training.summary` 生成）。'
              '本页不合并数值：跨 rank 数值不相加，一致性判据见 summary 的字段语义。']
    (out/'group.md').write_text('\n'.join(lines)+'\n')
    (out/'group.json').write_text(json.dumps(dict(
        schema='training-group-index/1', store=str(root),
        ranks=[dict(rank=r['rank'], run=str(r['run']), report=str(r['report']),
                    status=r['status'], oracle=str(r['oracle']) if r['oracle'] else None)
               for r in ranks]), ensure_ascii=False, indent=1)+'\n')
    return dict(ranks=len(ranks), out=str(out))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('store')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.store, args.out), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
