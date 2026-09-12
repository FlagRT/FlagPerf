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
    combined = {}; records = []; identity = None; elapsed = 0
    for source in sources:
        source = Path(source).resolve()
        manifest = verify(source)
        data = read(source / 'summary.json')
        if manifest['schema_version'] != 2 or data.get('source_runs'):
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
        elapsed += data.get('elapsed_s',0)
        for item in data.get('tasks',[]):
            key = (item['case'], item['dtype'], item['oplib'], item.get('device_id'), item.get('seed', 2026),
                   item.get('diagnostics_mode', 'legacy'), item.get('diagnostic_protocol', 'legacy'),
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
               'source_runs':records,'elapsed_s':elapsed,
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
    for source in data.get('source_runs',[]):
        directory = (root / source['directory']).resolve()
        if sha(directory/'artifacts.json') != source['manifest_sha256'] or sha(directory/'summary.json') != source['summary_sha256']:
            raise ValueError('aggregate source identity changed')
        verify(directory)
    lines = ['# Operation 测试报告', '', f"状态：{data['status']}；总耗时：{data.get('elapsed_s', '—')} s", '',
             '状态计数：' + ', '.join(f'{k}={v}' for k,v in data.get('counts', {}).items()), '',
             '| Case | dtype | 路径 | Device | 状态 | 正确性 | 路由 | 平均调用 μs | 端到端 s | 证据 |',
             '|---|---|---|---|---|---|---|---:|---:|---|']
    for item in data.get('tasks', []):
        if item['status'] == 'not-applicable': continue
        m = item.get('measurement', {})
        lines.append('| ' + ' | '.join(str(v) for v in [item['case'], item['dtype'], item['oplib'],
            item.get('device_id', '—'), item['status'], item.get('correctness', {}).get('status', 'not-run'),
            item.get('routing', {}).get('status', 'not-run'), m.get('median_us', '—'),
            round(item.get('elapsed_s', 0), 3), f"[详情]({item['directory']}/result.json)" if item.get('directory') else item.get('reason', '—')]) + ' |')
    skipped = [t for t in data.get('tasks', []) if t['status']=='not-applicable']
    if skipped:
        lines += ['', f'<details><summary>不适用组合（{len(skipped)} 项，不计入适配通过数）</summary>', '',
                  '| Case | dtype | 路径 | 原因 |', '|---|---|---|---|']
        lines += [f"| {t['case']} | {t['dtype']} | {t['oplib']} | {t.get('reason','类型不适用')} |" for t in skipped]
        lines += ['', '</details>']
    measured = [t for t in data.get('tasks', []) if t.get('measurement')]
    if measured:
        lines += ['', '## 性能与输入规模', '',
                  '耗时单位为 μs；N/A 表示未提供峰值、类型不适用或计时能力未实现。正确性失败时保留原始测量供诊断，不计为有效性能通过。', '',
                  '| Case / dtype / 路径 / Device | 配置维度 | cold | warm | 批次中位数 | kernel | 调用/s | 等效 TFLOPS | FU % |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        def number(value):
            return 'N/A' if value is None else f'{value:.6g}'
        for item in measured:
            m = item['measurement']
            dims = f"seed={item.get('seed', 2026)}, " + ', '.join(f'{k}={v}' for k,v in item.get('case_config',{}).items()
                             if k not in {'WARMUP','ITERS','rounds','KERNELWARMUP','KERNELITERS'})
            label = f"{item['case']} / {item['dtype']} / {item['oplib']} / {item.get('device_id', '—')}"
            lines.append('| ' + ' | '.join([label, dims] + [number(m.get(k)) for k in
                         ('cold_us','warm_us','median_us','kernel_us','throughput_op_s','equivalent_tflops','fu_percent')]) + ' |')
    errors = [t for t in data.get('tasks', []) if t.get('status') in ('failed', 'blocked', 'partial')]
    if errors:
        lines += ['', '## 失败与底层缺口', '']
        for item in errors:
            reason = item.get('diagnosis') or item.get('error') or ('numerical-mismatch' if item.get('correctness', {}).get('status') == 'failed' else 'routing-evidence insufficient')
            reason = reason.replace('\n',' ')[:500]
            lines.append(f"- {item['case']} / {item['dtype']} / {item['oplib']}：{item['status']} / {item.get('failure_stage', '未记录')} / {reason}；[原始证据]({item['directory']}/result.json)")
            diagnostic = item.get('diagnostics', {})
            if diagnostic:
                lines.append(f"  - 附加诊断耗时：{diagnostic.get('elapsed_s', '未记录')} s（不在性能计时内）。")
            for mode, entry in diagnostic.get('results', {}).items():
                if entry.get('evidence'):
                    lines.append(f"  - {mode}: {entry.get('status')} / [诊断证据]({item['directory']}/{entry['evidence']})；不改变原正确性或路由门禁。")
                else:
                    lines.append(f"  - {mode}: {entry.get('status')} / {entry.get('error', '未记录')}")
        from collections import Counter
        groups = Counter(t.get('diagnosis', 'legacy: inspect raw evidence') for t in errors)
        lines += ['', '根因族计数（组合数；blocked 仍未解决）：']
        lines += [f'- {reason}: {count}' for reason, count in sorted(groups.items())]
    lines += ['', '## 测量边界', '',
              '- 默认 PyTorch 路径与显式 FlagGems 路径是注册方式，不保证对应不同底层实现。',
              '- 主机计时包含提交、分配和同步等待；输入搬运、CPU 参考和路由跟踪不在计时区间。',
              '- cold/warm latency 保留原前向语义；带反向的批次包含前向、sum 和原协议零上游梯度反向。正确性另用非零上游梯度。',
              '- 纯设备时间仅在计时器受支持时提供。等效 TFLOPS 沿用 case 公式；非浮点算子不声称浮点吞吐。',
              '- 两分钟是每个组合的软目标，超过该时间不自动失败；总耗时包含多个组合。',
              '', '## 运行身份及错误', '', '```json', json.dumps({k:v for k,v in data.items() if k != 'tasks'}, ensure_ascii=False, indent=2), '```',
              '', '全部原始文件的 SHA-256 见 [artifacts.json](artifacts.json)。']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return root / 'report.md'
