# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Batch adaptation driver: models.yaml -> acceptance suite -> status table.

Prototype for the 500-model scaleout (plan section 8.2). Each enabled row
runs the suite twice through the canonical runner surfaces (A/A protocol)
and the driver compares ``score_batches_sha256`` / ``oracle_identity`` for
the A/A verdict, then emits a per-model status table (passed/failed + shas
+ artifact paths). Artifact filenames keep the historical ``battery-*``
prefix; the protocol is called the acceptance suite (A/A x2 per model).

Run inside the target container:

    python3 -B tools/batch_driver.py --config tools/models.yaml --out /w/runs/batch-<date>

Rows reference the three runner tasks (embedding / rerank / ctr) directly;
``task: attach`` rows are executed via their attach target script (legacy
surface kept for families outside the three runner tasks).
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml ships with transformers envs
    yaml = None

# The interpreter running this driver is the right one for battery runs
# on every host (containers expose python3 == sys.executable; Windows
# local scaffolds have no python3 and the Store stub is unusable).
_PY = os.environ.get('BATCH_PYTHON') or sys.executable


def load_config(path):
    if yaml is None:
        raise SystemExit('pyyaml is required for batch_driver (pip install pyyaml)')
    with open(path, encoding='utf-8') as f:
        doc = yaml.safe_load(f)
    rows = doc.get('models') or []
    if not rows:
        raise ValueError('config has no models list')
    seen = set()
    for r in rows:
        for key in ('id', 'task'):
            if not r.get(key):
                raise ValueError(f'row missing {key}: {r}')
        if r['id'] in seen:
            raise ValueError(f'duplicate model id: {r["id"]}')
        seen.add(r['id'])
    return rows


def build_cmd(row, run_idx, out_dir, device='cuda:0', dtype=None, batch_size=None):
    """One battery run as an argv list WITHOUT the interpreter (run_battery
    prepends sys.executable). Module-level for testability."""
    out = str(Path(out_dir) / f'{row["id"]}-run{run_idx}')
    task = row['task']
    dtype = dtype or row.get('dtype', 'bfloat16')
    common = ['--device', device, '--dtype', dtype, '--outputs', out]
    if task == 'attach':
        script = row['target']
        cmd = ['-B', script]
        for k, v in row.get('args', {}).items():
            cmd += [f'--{k}', str(v)]
        cmd += common
        return cmd, out
    cmd = ['-B', '-m', 'flagprof.runner', '--task', task]
    if row.get('model'):
        cmd += [row['model']]
    for k in ('package', 'score', 'arch', 'pooling', 'inputs', 'seed', 'padding_side'):
        if row.get(k) is not None:
            cmd += [f'--{k.replace("_", "-")}', str(row[k])]
    for flag in row.get('flags') or []:
        cmd += ['--' + flag]
    if batch_size or row.get('batch_size'):
        cmd += ['--batch-size', str(batch_size or row['batch_size'])]
    if row.get('forwards'):
        cmd += ['--forwards', str(row['forwards'])]
    cmd += common
    return cmd, out


def _score_digest(doc):
    """Top-level score_batches_sha256 (ctr/rerank/attach), or a digest over
    the per-forward output sha256 list for legacy embedding runner.json
    (whose score lives at run level, not top level)."""
    top = doc.get('score_batches_sha256')
    if top:
        return top
    runs = doc.get('runs') or []
    if runs and isinstance(runs[0], dict) and 'run' in runs[0]:
        runs = [r['run'] for r in runs]
    digests = [o['sha256'] for r in runs for o in (r.get('outputs') or [])
               if 'sha256' in o]
    if not digests:
        return None
    return hashlib.sha256('|'.join(digests).encode()).hexdigest()


def _result_sha(path):
    p = Path(path)
    if not p.exists():
        return None, None
    with open(p, encoding='utf-8') as f:
        doc = json.load(f)
    return _score_digest(doc), doc.get('oracle_identity')


def run_battery(row, out_dir, device='cuda:0', runner_cwd=None):
    """Two identical runs; returns A/A verdict plus artifact evidence."""
    entry = dict(id=row['id'], task=row['task'], runs=[])
    shas, idents, outs = [], [], []
    env_note = dict(cwd=runner_cwd or '.')
    for i in (1, 2):
        cmd, out = build_cmd(row, i, out_dir, device=device)
        proc = subprocess.run([_PY] + cmd, cwd=runner_cwd or '.', capture_output=True,
                              text=True)
        result_file = Path(out) / (
            'runner.json' if row['task'] != 'attach' else 'target-result.json')
        sha, ident = _result_sha(result_file)
        entry['runs'].append(dict(argv=cmd, exit_code=proc.returncode,
                                  result=str(result_file),
                                  tail=(proc.stdout + proc.stderr)[-400:]))
        shas.append(sha)
        idents.append(ident)
        outs.append(str(result_file))
    entry['aa_sha_equal'] = (shas[0] is not None and shas[0] == shas[1])
    entry['aa_ident_equal'] = (idents[0] is not None and idents[0] == idents[1])
    entry['score_batches_sha256'] = shas[0]
    entry['oracle_identity'] = idents[0]
    entry['env'] = env_note
    exit_ok = all(r['exit_code'] == 0 for r in entry['runs'])
    if row.get('status') == 'blocked':
        entry['status'] = 'blocked'
    elif exit_ok and entry['aa_sha_equal'] and entry['aa_ident_equal']:
        entry['status'] = 'passed'
    else:
        entry['status'] = 'failed'
    return entry


def summarize(entries):
    ok = [e for e in entries if e['status'] == 'passed']
    return dict(
        total=len(entries),
        passed=len(ok),
        failed=sum(1 for e in entries if e['status'] == 'failed'),
        blocked=sum(1 for e in entries if e['status'] == 'blocked'),
        models=entries,
    )


def render_markdown(summary, generated_at):
    lines = [f'# batch adaptation status ({generated_at})', '',
             f'total={summary["total"]} passed={summary["passed"]} '
             f'failed={summary["failed"]} blocked={summary["blocked"]}', '',
             '| model | task | status | AA | score_batches_sha256 |',
             '|---|---|---|---|---|']
    for e in summary['models']:
        sha = (e.get('score_batches_sha256') or '-')[:16]
        aa = e.get('aa_sha_equal')
        lines.append(f'| {e["id"]} | {e["task"]} | {e["status"]} | {aa} | {sha} |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--filter', default=None, help='substring on model id')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)

    rows = load_config(args.config)
    if args.filter:
        rows = [r for r in rows if args.filter in r['id']]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    if args.dry_run:
        for r in rows:
            cmd, out = build_cmd(r, 1, args.out, device=args.device)
            print(r['id'], '->', ' '.join(cmd))
        return 0

    entries = []
    for r in rows:
        print(f'=== battery {r["id"]} ({r["task"]})', flush=True)
        try:
            entries.append(run_battery(r, args.out, device=args.device))
        except Exception as exc:  # keep the batch going, mark the row failed
            entries.append(dict(id=r['id'], task=r['task'], status='failed',
                                error=repr(exc)))
        json.dump({'generated_at': stamp, 'models': entries},
                  open(Path(args.out) / 'battery-state.json', 'w',
                       encoding='utf-8'), ensure_ascii=False, indent=1)
    summary = summarize(entries)
    json.dump(summary, open(Path(args.out) / 'battery-summary.json', 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=1)
    md = render_markdown(summary, stamp)
    (Path(args.out) / 'battery-summary.md').write_text(md, encoding='utf-8')
    print(md)
    return 0 if summary['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
