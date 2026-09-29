# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Portable JSON artifacts and deterministic, offline reports."""
import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seal(root):
    write(root / 'artifacts.json', {'schema_version': 2, 'files': {
        str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
        if p.is_file() and p.name not in ('artifacts.json', 'report.md') and '__pycache__' not in p.parts}})


def verify(root):
    root = Path(root).resolve()
    manifest = read(root / 'artifacts.json')
    if manifest.get('schema_version') not in (1,2):
        raise ValueError('unsupported artifact version')
    for rel,digest in manifest['files'].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != digest:
            raise ValueError(f'missing/corrupt artifact: {rel}')
    if 'summary.json' not in manifest['files']:
        raise ValueError('summary.json must be covered by the manifest')
    return manifest


def merge_runs(sources, destination):
    """Latest attempted row wins for identical runtime, device, dtype and workload.

    Inputs are verified, never rewritten. Missing/interrupted rows cannot erase
    completed attempts. Different sizes remain separate rows.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('aggregate output must be a new directory')
    combined = {}; records = []; source_comparisons = []; identity = None; elapsed = 0
    for source in sources:
        source = Path(source).resolve()
        manifest = verify(source)
        data = read(source / 'summary.json')
        if manifest['schema_version'] != 2 or data.get('source_runs') or data.get('protocol') in ('operation-attribution-v1', 'operation-diagnosis-v1'):
            raise ValueError('aggregate inputs must be original operation-v2 runs')
        current = (data.get('actual_image_id'), data.get('vendor'), data.get('execution'), data.get('preflight',{}).get('host'))
        if not current[0] or current[2] != 'docker' or not current[3]:
            raise ValueError('aggregation requires recorded Docker image identity')
        if identity is not None and current != identity:
            raise ValueError('cannot aggregate different hosts, runtime images, vendors or execution modes')
        if data.get('cleanup_error') or data.get('postflight_error'):
            raise ValueError('source has unresolved cleanup or device health failure')
        identity = current
        relative = os.path.relpath(source,destination)
        records.append({'directory':relative,'manifest_sha256':sha(source/'artifacts.json'),
                        'status':data['status'], 'summary_sha256':sha(source/'summary.json')})
        if data.get('comparisons'):
            source_comparisons.append({'directory': relative, 'comparisons': data['comparisons']})
        elapsed += data.get('elapsed_s',0)
        for item in data.get('tasks',[]):
            key = (item['case'], item['dtype'], item['oplib'], item.get('device_id'), item.get('seed', 2026),
                   item.get('diagnostics_mode', 'legacy'), item.get('diagnostic_protocol', 'legacy'), item.get('performance_protocol', 'legacy'), item.get('profiling_mode', 'off'),
                   json.dumps(item.get('runtime_config', {}), sort_keys=True),
                   json.dumps(item.get('case_config',{}),sort_keys=True))
            if key in combined and item['status'] in ('not-run','interrupted'):
                continue
            row = dict(item)
            if row.get('directory'):
                row['directory'] = relative + '/' + row['directory']
            row['source_run'] = relative
            combined[key] = row
    if not records: raise ValueError('no source runs')
    tasks = [combined[k] for k in sorted(combined)]
    statuses = [t['status'] for t in tasks if t['status'] != 'not-applicable']
    status = 'failed' if 'failed' in statuses else 'passed' if statuses and all(s=='passed' for s in statuses) else 'partial'
    summary = {'schema_version':2,'protocol':'operation-v2','status':status,'tasks':tasks,
               'actual_image_id':identity[0],'vendor':identity[1],'execution':identity[2],'host':identity[3],
               'source_runs':records,'source_comparisons':source_comparisons,'comparisons':[],'elapsed_s':elapsed,
               'aggregation':'latest attempted identical-workload row in supplied source order; source identities and raw evidence retained',
               'elapsed_scope':'sum of source-run elapsed times, including repeated and interrupted work',
               'counts':{s:sum(t['status']==s for t in tasks) for s in sorted({t['status'] for t in tasks})}}
    destination.mkdir(parents=True)
    write(destination/'summary.json',summary);seal(destination)
    return report(destination)


def report(root):
    root = Path(root).resolve()
    manifest = verify(root)
    if manifest['schema_version'] == 1:
        from vendors import legacy_reporter
        return legacy_reporter(manifest['schema_version'])(root)
    data = read(root / 'summary.json')
    if data.get('protocol') == 'operation-attribution-v1':
        raise ValueError('experimental attribution reports require their original source version')
    if data.get('protocol') == 'operation-diagnosis-v1':
        from runtime.diagnosis import render
        return render(root, data)
    for source in data.get('source_runs',[]):
        directory = (root / source['directory']).resolve()
        if sha(directory/'artifacts.json') != source['manifest_sha256'] or sha(directory/'summary.json') != source['summary_sha256']:
            raise ValueError('aggregate source identity changed')
        verify(directory)
    from runtime.reporting import run_report
    lines = run_report(root, data)
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return root / 'report.md'
