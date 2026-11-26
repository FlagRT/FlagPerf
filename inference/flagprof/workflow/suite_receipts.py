# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Project the exporter's manifest or the batch driver's A/A verdicts into a workflow summary."""
import json
from pathlib import Path

from .executor import artifact, evidence


def graph_receipt(summary, job, config, code):
    from flagprof.graph.data import read_manifest

    root = job/'export-bundle'
    manifest = root/'manifest.json'
    artifact(summary, 'graph-bundle', root)
    artifact(summary, 'export-log', root/'export.log')
    if not artifact(summary, 'graph-manifest', manifest):
        raise ValueError('导出未生成 manifest.json')
    data = read_manifest(root)
    options = config['export']
    evidence(summary, 'export', data['status'], [data['error']] if data.get('error') else [], manifest,
             bundle=data['bundle'], device=data.get('device'), inputs=data.get('inputs'), outputs=data.get('outputs'))
    for name, row in data['formats'].items():
        reasons = [row['error']] if row.get('error') else []
        evidence(summary, 'export.'+name, row['status'], reasons, manifest, graph_id=row.get('graph_id'))
        for check in ('roundtrip','structural_check','runtime'):
            if check in row:
                value = row[check]
                status = value['status'] if isinstance(value,dict) else value
                evidence(summary, f'export.{name}.{check}', status, [], manifest,
                         **({'checks':value} if isinstance(value,dict) else {}))
    for name in data.get('files', {}):
        path = (root/name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f'图包文件无效 {name}')
        artifact(summary, 'graph-viewer' if name=='index.html' else 'graph-file', path)
    if code:
        return
    if (data['status'] != 'completed' or data['bundle'] != options['bundle']
            or set(data['formats']) != set(options['formats'])):
        raise ValueError('图包完成状态、格式或 bundle 与请求不一致')
    required = {'index.html'}
    for fmt in options['formats']:
        row = data['formats'][fmt]
        required.add(fmt+'.json')
        if row['status'] != 'completed':
            raise ValueError(f'{fmt} 导出未完成')
        if fmt == 'fx':
            required.add('model.fx.txt')
            if row.get('roundtrip', {}).get('status') != 'passed':
                raise ValueError('FX 回读校验未通过')
        else:
            expected = 'passed' if options['verify_onnx'] else 'not_run'
            if row.get('structural_check') != 'passed' or row.get('runtime', {}).get('status') != expected:
                raise ValueError('ONNX 结构或 Runtime 校验状态与请求不一致')
        if options['bundle'] == 'full':
            required.update(('example.pt', 'model.pt2' if fmt=='fx' else 'model.onnx'))
            required.update(row.get('external_data', []))
    missing = required - data.get('files', {}).keys()
    if missing:
        raise ValueError('图包清单缺少 ' + ', '.join(sorted(missing)))


def batch_receipt(summary, job, plan, code):
    from flagprof.batch.batch_driver import summarize

    root = job/'batch-results'
    final = root/'battery-summary.json'
    state = root/'battery-state.json'
    complete = artifact(summary, 'batch-summary', final)
    has_state = artifact(summary, 'batch-state', state)
    has_report = artifact(summary, 'batch-markdown', root/'battery-summary.md')
    if not complete and not has_state:
        raise ValueError('批量驱动未生成汇总或逐模型状态')
    path = final if complete else state
    doc = json.loads(path.read_bytes())
    rows = doc['models']
    counts = {key:value for key,value in summarize(rows).items() if key != 'models'}
    if complete and any(doc.get(k) != v for k,v in counts.items()):
        raise ValueError('battery-summary 计数与逐模型结果不一致')
    evidence(summary, 'batch', 'completed' if complete else 'partial', [], path, **counts)
    for row in rows:
        reasons = [row['error']] if row.get('error') else []
        runs = row.get('runs', [])
        if row['status'] == 'passed' and (len(runs) != 2 or not row.get('aa_sha_equal')
                or not row.get('aa_ident_equal') or not row.get('score_batches_sha256')
                or row.get('oracle_identity') is None or any(r['exit_code'] for r in runs)):
            raise ValueError(f"通过行缺少两次一致的 A/A 回执 {row['id']}")
        for run in runs:
            dest = Path(run['result']).resolve()
            if root not in dest.parents:
                raise ValueError('批量回执指向结果目录之外 ' + str(dest))
            exists = artifact(summary, 'model-result', dest, model_id=row['id'])
            if not exists and row['status']=='passed':
                raise ValueError('通过行缺少 A/A 结果 ' + str(dest))
            if run['exit_code']:
                reasons.append(f"退出码 {run['exit_code']}  {run.get('tail','').strip()}")
        if row['status']=='failed' and not reasons:
            reasons.extend(name+'=false' for name in ('aa_sha_equal','aa_ident_equal') if row.get(name) is False)
        evidence(summary, 'batch.model', row['status'], reasons, path, model_id=row['id'], task=row['task'],
                 score_batches_sha256=row.get('score_batches_sha256'), oracle_identity=row.get('oracle_identity'),
                 aa_sha_equal=row.get('aa_sha_equal'), aa_ident_equal=row.get('aa_ident_equal'),
                 result_paths=[r['result'] for r in runs], exit_codes=[r['exit_code'] for r in runs])
    if complete and [row['id'] for row in rows] != plan['model_ids']:
        raise ValueError('批量汇总的模型与所选名册不一致')
    if not code and (not complete or not has_report or counts['failed']):
        raise ValueError('批量驱动返回 0，但汇总、Markdown 或失败计数不一致')
