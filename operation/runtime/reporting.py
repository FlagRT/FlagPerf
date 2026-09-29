# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Reader-oriented, offline views. No inference of successful device execution."""
from collections import Counter
from pathlib import Path
from urllib.parse import quote
import html
import json
import math
import shlex
import statistics

from vendors import report_metadata


def cell(value):
    return html.escape(str(value), quote=False).replace('|', '&#124;').replace('\n', ' ')


def number(value):
    return '未记录' if value is None else f'{value:.6g}' if isinstance(value, (int, float)) else cell(value)


def status(value):
    names = {'passed': '通过', 'failed': '失败', 'blocked': '受阻', 'partial': '证据或覆盖不完整',
             'not-run': '未执行', 'interrupted': '中断', 'not-applicable': '不适用',
             'completed': '已完成', 'missing': '缺失', 'recorded': '已记录', 'unsupported': '不支持', 'not-supported': '不支持', 'measured': '已采集', 'off': '主动关闭', 'skipped': '已跳过'}
    return f'{names.get(value, "未知状态")}（{cell(value)}）' if value else '未记录'


def link(root, path, label=None):
    if not path or not (root / path).is_file():
        return '证据文件未保存'
    return f'[{cell(label or Path(path).name)}]({quote(str(path), safe="/.-_")})'


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |',
            *['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]]


def label(task):
    return cell(' / '.join(str(task.get(k, '未记录')) for k in ('case', 'dtype', 'oplib', 'device_id')))


def valid_performance(task):
    m = task.get('measurement', {})
    return (task.get('status') == 'passed' and task.get('correctness', {}).get('status') == 'passed'
            and m.get('correctness', {}).get('status') == 'passed'
            and task.get('routing', {}).get('status') == 'passed'
            and not task.get('measurement_fallback_observed') and m.get('median_us') is not None)


def probe_failed(task):
    return task.get('correctness', {}).get('status') == 'failed'


def correctness_phases(task):
    """Collapse a later measurement failure after probe already failed."""
    phases = (
        ('probe', '探测', task.get('correctness', {})),
        ('measure', '测量', task.get('measurement', {}).get('correctness', {})),
    )
    return phases[:1] if probe_failed(task) else phases


def expected_gate_skip(task, record):
    """A gated post-check is expected, not an acquisition failure."""
    return (probe_failed(task) and record.get('status') == 'skipped'
            and record.get('reason') == 'requires passed correctness and routing')


def displayed_measurement_status(task):
    if probe_failed(task):
        return '未展开（probe 已失败）'
    return status(task.get('measurement', {}).get('correctness', {}).get('status'))


def interpretation(task):
    gates = [task.get('correctness', {}), task.get('measurement', {}).get('correctness', {})]
    if any(g.get('status') == 'failed' for g in gates):
        return ('正确性门禁失败：实际输出或梯度未满足原比较契约。已采集的耗时用于诊断。',
                '先核对失败张量、原容差及参考校验；有保存输入时再做同输入复放，通过受控对照定位误差原因。')
    if task.get('error') or task.get('status') == 'blocked':
        return ('执行受阻或报错，现有记录不足以完成本组合验收。错误分类是定位线索。',
                '先查看失败阶段的原始异常及日志，修复对应依赖或运行条件后重跑原组合。')
    if task.get('status') in ('not-run', 'interrupted'):
        return ('该组合未完成，正确性和性能待验证。', '检查运行级错误及中断记录，再补跑该组合。')
    if task.get('routing', {}).get('status') != 'passed' or task.get('measurement_fallback_observed'):
        return ('目标执行路径待确认，或测量观察到回退；目标实现性能验收待完成。',
                '核对路由记录和目标调用链；需要定位设备 kernel 时另采设备时间线。')
    return ('当前记录尚不足以形成完整验收结论。', '查看 result.json 中未完成的门禁和测量字段。')


def checks_view(root, checks, prefix=''):
    rows = []
    scopes = {'saved-source': '原运行封存证据', 'new-replay': '本次复放', 'automatic': '运行时附加检查'}
    for c in checks:
        rows.append([cell(scopes.get(c.get('scope'), c.get('scope', '未记录'))), cell(c.get('check', '未记录')),
                     status(c.get('status')), status(c.get('execution_status', 'recorded')),
                     cell(c.get('observation', '未记录')),
                     cell(c.get('next_step') or '无额外建议'),
                     ' '.join(link(root, prefix + p) for p in c.get('evidence', [])) or '无文件证据'])
    return table(['证据所属运行', '检查项', '观察结果', '检查执行', '事实与解释', '下一步', '证据'], rows)


def diagnostic_findings(root, checks, prefix='', collapse_measurement=False):
    lines = []
    for c in checks:
        if c.get('check') in ('reference-check', 'diagnose', 'numeric', 'trace'):
            observation = cell(c.get('observation', '未记录'))
            if collapse_measurement and c.get('check') in ('diagnose', 'numeric'):
                observation = 'probe 已形成失败结论；数值补充对照已保存，逐阶段结果保留在折叠证据中。'
            lines += [f'- **补充发现（{cell(c.get("scope", "未记录"))} / {cell(c["check"])}）：** '
                      + observation + ' '
                      + ' '.join(link(root, prefix + p) for p in c.get('evidence', []))]
    if lines:
        lines += ['', '这些对照帮助区分“参考是否复算一致”和“设备输出偏差”，但没有控制数学模式或底层实现变量，尚不足以确认原因。', '']
    return lines


def runtime_errors(data):
    return [(name, data[key]) for key, name in [('error', '运行异常'), ('cleanup_error', '资源清理失败'),
                                               ('postflight_error', '运行后设备检查失败')] if data.get(key)]


def appendix(root, data):
    identity = data.get('source_identity', data)
    rows = [(cell(k), cell(identity[k])) for k in ('vendor', 'execution', 'actual_image_id', 'host', 'device_ids') if k in identity]
    lines = ['', '## 附录：运行条件与证据', '', *table(['条件', '记录值'], rows), '',
             '完整配置、设备映射和状态：' + link(root, 'summary.json') + '；文件完整性索引：' + link(root, 'artifacts.json') + '。', '',
             'SHA-256 校验用于核对文件完整性；测量资格见关键结果，诊断观察见对应证据。', '',
             '离线重建（在 FlagPerf 仓库根目录执行）：', '',
             '```bash', 'python3 operation/run.py report --run-dir <报告所在目录>', '```']
    if data.get('source_runs'):
        lines += ['', '本报告按输入顺序保留同运行环境、设备和工作负载的最近有效尝试；总耗时为所有源运行耗时之和，包含重复工作。', '',
                  *table(['源运行', '原状态'], [(link(root, s['directory'] + '/summary.json', s['directory']), status(s.get('status'))) for s in data['source_runs']])]
    return lines


def issue_details(root, data):
    active = data.get('tasks', [])
    issues = [t for t in active if t.get('status') != 'not-applicable' and not valid_performance(t)]
    lines = []
    if issues:
        lines += ['', '## 异常与下一步', '']
        for i, t in enumerate(issues, 1):
            reason, action = interpretation(t)
            prefix = t.get('directory', '') + '/'
            lines += [f'### {i}. {label(t)}', '', f'**判断：** {reason}', '',
                      f'**记录事实：** 阶段 {cell(t.get("failure_stage", "未记录"))}；'
                      f'{cell(t.get("diagnosis") or t.get("error") or "无异常分类记录")}。'
                      + ' ' + link(root, prefix + 'result.json'), '']
            for _, name, g in correctness_phases(t):
                if g:
                    lines += [f'- {name}比较：{status(g.get("status"))}；atol={number(g.get("atol"))}，rtol={number(g.get("rtol"))}。']
                    for j, c in enumerate(g.get('checks', [])):
                        if c.get('passed') is False:
                            lines += [f'  - 张量 {j}：不匹配元素 {number(c.get("mismatch_count"))}；最大绝对误差 {number(c.get("max_absolute_error"))}；精确比较 {cell(c.get("exact", False))}。']
            lines += ['', '**下一步：** ' + action, '']
            if t.get('diagnostics_mode') == 'off':
                lines += ['附加诊断主动关闭；原正确性与路径门禁仍有效。', '']
            d = t.get('diagnostics', {})
            if d:
                lines += [f'附加诊断：{status(d.get("status"))}，耗时 {number(d.get("elapsed_s"))} s；检查完成不改变原任务结论。', '']
                if d.get('checks'):
                    lines += diagnostic_findings(root, d['checks'], prefix, collapse_measurement=probe_failed(t))
                    lines += ['<details><summary>展开附加检查与证据</summary>', '', *checks_view(root, d['checks'], prefix), '', '</details>', '']
                else:
                    lines += [f'- {cell(k)}：{status(v.get("status"))}；' + (link(root, prefix + v['evidence']) if v.get('evidence') else cell(v.get('error', '未保存证据'))) for k, v in d.get('results', {}).items()]
    return lines


def replay_source(root, task):
    """Return the sealed task directory used by a same-input replay."""
    directory = task.get('directory')
    if not directory:
        return None
    source_root = (root / task['source_run']).resolve() if task.get('source_run') else root.resolve()
    # Aggregate rows carry the source-relative prefix in ``directory`` while
    # ``source_run`` identifies the same source root.  Resolve from ``root``
    # once, then only use ``source_root`` for the containment check.
    source = (root / directory).resolve()
    if not source.is_relative_to(source_root) or not source.is_dir():
        return None
    return source


def replay_command(root, task):
    """Build a runnable replay command only when source prerequisites are sealed."""
    source = replay_source(root, task)
    if source is None:
        return None, '失败任务目录未保存'

    required = ['task.json', 'inputs.pt']
    if task.get('case') not in ('dropout', 'native_dropout'):
        required.append('reference.pt')
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        return None, '缺少封存文件：' + '、'.join(missing)

    source_root = (root / task['source_run']).resolve() if task.get('source_run') else root.resolve()
    try:
        source_summary = json.loads((source_root / 'summary.json').read_text())
    except (OSError, ValueError, TypeError):
        return None, '源运行 summary.json 不可读取'
    if not source_summary.get('actual_image_id'):
        return None, '源运行未记录 Docker 镜像身份'

    operation_root = Path(__file__).resolve().parents[2]
    command = [
        'cd', shlex.quote(str(operation_root)), '&&', 'python3', 'operation/run.py',
        'diagnose', '--source-task', shlex.quote(str(source)), '--replay',
    ]
    if task.get('device_id') is not None:
        command += ['--device-ids', shlex.quote(str(task['device_id']))]
    command += ['--allow-privileged-root']
    return ' '.join(command), None


def replay_diagnostic_lines(root, names, tasks):
    """Promote actionable replay commands into the report conclusion."""
    candidates = []
    for task in tasks:
        if task.get('status') in ('failed', 'blocked', 'interrupted'):
            candidates.append(task)
            continue
        if task.get('status') == 'partial' and (
                task.get('correctness', {}).get('status') == 'failed'
                or task.get('measurement', {}).get('correctness', {}).get('status') == 'failed'
                or task.get('routing', {}).get('status') in ('failed', 'partial')):
            candidates.append(task)
    if not candidates:
        return []

    lines = ['', '### 失败任务的 replay 诊断命令', '',
             '以下命令复用对应任务封存的输入、参考和 Docker 镜像，执行一次同输入诊断复放；'
             '请直接复制命令执行，新的诊断结果会写入 `operation/result/diagnosis-*`。', '']
    for task in candidates:
        lines += [f'**{names[id(task)]}**', '']
        command, reason = replay_command(root, task)
        if command:
            lines += ['```bash', command, '```', '']
        else:
            lines += [f'未生成可执行命令：{cell(reason)}。先补齐封存证据后再使用 `diagnose --replay`。', '']
    return lines


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def folded(title, lines):
    return ['<details><summary>' + cell(title) + '</summary>', '', *lines, '', '</details>', '']


def task_name(task, index):
    return f'C{index + 1:03d} · {label(task)}'



def reason_text(reason):
    translations = {
        'run has unresolved execution, cleanup or device health errors': '存在未解决的运行、清理或设备健康异常',
        'missing or duplicate path': '配对缺少一条路径，或同一路径重复',
        'both paths must pass correctness, routing and measurement gates': '两侧必须同时通过正确性、路径和测量门禁',
        'runtime, saved input/module state, source or measurement contract mismatch': '两侧环境、保存输入/模块状态、源码或测量契约不一致',
        'invalid timing': '耗时无效，不能计算倍率',
        'requires passed correctness and routing': '正确性或路径门禁未通过，跳过附加采集',
        'msprof not found': '环境中未找到 msprof 采集工具',
        'capture watchdog exceeded': '采集超过 watchdog 时限',
        'no target-correlated hardware metric values': '未取得与目标任务关联的硬件计数值',
        'memory execution error; no extra device replay': '显存采集执行异常，未继续设备复放',
    }
    return cell(translations.get(reason, reason or '未记录原因'))

def comparison_views(data):
    """Use sealed pair records only; never manufacture cross-run comparisons."""
    tasks = {t.get('directory'): t for t in data.get('tasks', [])}
    result = []
    if data.get('source_runs') or data.get('oplib_selection') != 'both':
        return result
    for pair in data.get('comparisons', []):
        members = [tasks[p] for p in pair.get('directories', []) if p in tasks]
        libs = {t.get('oplib'): t for t in members}
        ok = (pair.get('status') == 'comparable' and len(members) == 2
              and set(libs) == {'nativetorch', 'flaggems'}
              and all(valid_performance(t) for t in members)
              and not runtime_errors(data))
        ratio = pair.get('speedup')
        ok = ok and finite(ratio) and ratio > 0
        if ok:
            explanation = (f'FlagGems 耗时为 native 的 {1 / ratio:.4g} 倍' if ratio < 1
                           else f'FlagGems 耗时为 native 的 {1 / ratio:.4g} 倍（更快）' if ratio > 1
                           else '两侧记录的主机耗时相同')
        else:
            explanation = '不可比较：' + reason_text(pair.get('reason') or '配对记录或有效测量不完整')
        result.append((pair, libs, ok, explanation))
    return result


def run_report(root, data):
    tasks = data.get('tasks', [])
    active = [t for t in tasks if t.get('status') != 'not-applicable']
    eligible = [t for t in active if valid_performance(t)]
    names = {id(t): task_name(t, i) for i, t in enumerate(tasks)}
    counts = Counter(t.get('status', 'unknown') for t in active)
    pairs = comparison_views(data)
    profiles, profile_appendix, gaps = profiling_view(root, data, names)
    references = {id(t): saved_json(root, t, 'reference.json') or {} for t in tasks}
    lines = ['# Operation 运行结果报告', '', '## 本次结论', '']
    if data.get('evidence_kind') == 'synthetic-fixture':
        lines += ['**合成测试样例：数值由测试夹具构造，用于审阅报告。**', '']
    lines += [f'**运行状态：{status(data.get("status"))}。**', '',
              f'工作负载：{cell(data.get("workload", "未记录"))}；算子库选择：{cell(data.get("oplib_selection", "未记录"))}；'
              f'Profiling：{cell(data.get("profiling", "未记录"))}。', '',
              f'记录 {len(tasks)} 个组合，其中适用 {len(active)} 个；具备完整通过证据和主机耗时的组合：**{len(eligible)} 个**。',
              '；'.join(f'{status(k)} {v} 个' for k, v in sorted(counts.items())) if active else '没有适用组合，无法给出通过结论。', '',
              f'总耗时 {number(data.get("elapsed_s"))} s，包含准备、检查及附加采集。', '']
    if data.get('workload') == 'smoke':
        lines += ['本次为 smoke 小负载检查；耗时对应附录所列输入与配置。', '']
    for name, value in runtime_errors(data):
        lines += [f'- **{name}：** {cell(value)}；运行级异常单独记录并处理。']
    lines += replay_diagnostic_lines(root, names, active)
    highlights = [(names[id(t)] + '：' + interpretation(t)[0])
                  for t in active if not valid_performance(t) and not probe_failed(t)]
    highlights += gaps
    if len(eligible) == 1 and data.get('oplib_selection') != 'both':
        t = eligible[0]
        highlights.append(names[id(t)] + '：主机每调用耗时中位数 ' + number(t['measurement']['median_us']) + ' μs；结论仅覆盖本次输入、精度和设备。')
    for pair, libs, ok, explanation in sorted(pairs, key=lambda x: abs(math.log(x[0]['speedup'])) if x[2] else 0, reverse=True):
        if ok:
            highlights.append(' ↔ '.join(names[id(t)] for t in libs.values()) + '：' + explanation + '。')
    lines += ['', *['- ' + h for h in highlights[:5]]]
    if len(highlights) > 5:
        lines += [f'- 其余 {len(highlights)-5} 项见下方完整结果和异常说明。']
    if not eligible:
        lines += ['本次没有可作为完整性能通过依据的测量。']
    lines += ['', '## 关键结果及解读', '',
              '主指标是每轮“同步主机批次耗时 ÷ 调用次数”的中位数，包含提交、分配和同步等待。', '']
    lines += correctness_view(root, active, names)
    if data.get('oplib_selection') == 'both' and not data.get('source_runs'):
        rows = []
        for pair, libs, ok, explanation in pairs:
            rows.append([' ↔ '.join(names[id(t)] for t in libs.values()) or cell(pair.get('pair_id')),
                         number(libs.get('nativetorch', {}).get('measurement', {}).get('median_us')) if ok else '见单项结果',
                         number(libs.get('flaggems', {}).get('measurement', {}).get('median_us')) if ok else '见单项结果', explanation])
        lines += ['### 同次双库对比', '', *table(['组合', 'native μs', 'FlagGems μs', '结论'], rows), '',
                  '> 比较使用同次运行已通过输入、环境和测量契约检查的配对。加速比＝native / FlagGems 主机耗时，大于 1 表示 FlagGems 更快；执行归属见路径证据，测量波动见轮间统计。', '']
        if not pairs:
            lines += ['本次请求双库比较，但没有保存配对记录；不从单项结果补造倍率。', '']
    lines += measurement_view(root, active, names)
    input_lines = ['### 本次输入与测量范围', '', *table(['组合', '输入形状 / 执行范围', '规模配置', '预热 / 每轮调用 / 轮数', '证据'],
        [(names[id(t)], input_scope(t, references[id(t)]), cell(json.dumps({k:v for k,v in t.get('case_config', {}).items() if k not in ('WARMUP','ITERS','KERNELWARMUP','KERNELITERS','rounds')}, ensure_ascii=False, sort_keys=True)),
          ' / '.join(number(t.get('case_config', {}).get(k)) for k in ('WARMUP','ITERS','rounds')),
          link(root, t.get('directory','') + '/result.json')) for t in active]), '']
    for t in eligible:
        st = t['measurement'].get('statistics', {})
        n = st.get('rounds', t.get('case_config', {}).get('rounds'))
        if n == 1:
            lines += [f'- {names[id(t)]}：仅 1 轮，无法评估轮间波动或置信区间。']
        elif st.get('median_ci95_us') is None:
            lines += [f'- {names[id(t)]}：未提供中位数置信区间；' + ('当前少于 10 轮。' if finite(n) and n < 10 else cell(st.get('ci_reason') or '旧记录未保存统计依据。'))]
    if data.get('profiling') == 'off' and not profiles:
        lines += ['', '本次主动关闭设备 Profiling；已记录的测量范围为主机耗时、轮间统计和独立显存采集。']
    lines += profiles
    qualification_note = ('探测和测量正确性描述两个阶段的比较结果；路径表示执行归属检查，回退表示测量期间是否发生实现切换。'
                          '两次正确性、路径门禁及无测量回退共同决定有效性能资格。')
    if any(probe_failed(t) for t in tasks):
        qualification_note += ' probe 已失败的组合不在正文重复展开 measurement 失败；原始 measurement 证据仍保留。'
    lines += ['', '### 执行路径与测量资格', '', *table(['组合','探测正确性','测量正确性','路径','回退','最终状态'],
        [(names[id(t)], status(t.get('correctness',{}).get('status')), displayed_measurement_status(t),
          status(t.get('routing',{}).get('status')), cell(t.get('measurement_fallback_observed','未记录')), status(t.get('status'))) for t in tasks]) +
        ['', qualification_note]]
    lines += issue_details(root, data)
    lines += ['', '## 附录：输入、配置与证据', '', *input_lines]
    skipped = [t for t in tasks if t.get('status') == 'not-applicable']
    if skipped:
        lines += folded('不适用组合及原因', table(['组合','原因'], [(names[id(t)],cell(t.get('reason','未记录'))) for t in skipped]))
    config_lines = table(['组合','seed','完整配置','计时边界'],
        [(names[id(t)],number(t.get('seed')),cell(json.dumps(t.get('case_config',{}),sort_keys=True)),
          cell(t.get('measurement',{}).get('timing_boundary','未记录'))) for t in tasks])
    config_lines += ['', 'cold/warm 为前向首次／预热调用；带反向批次的范围以保存契约为准。输入搬运和参考计算不在正常计时区间。']
    if pairs:
        config_lines += ['', '加速比定义为 native / FlagGems 主机耗时，仅保留已有比较记录：',
            *table(['配对','加速比','状态'],[(cell(p.get('pair_id')),number(p.get('speedup')),cell(p.get('status'))) for p,_,_,_ in pairs])]
    lines += folded('配置、计时与比较契约', config_lines)
    lines += profile_appendix
    if data.get('source_comparisons'):
        lines += ['', '### 源运行对比', '', '仅保留源运行原有对比，不跨运行重新配对。']
        for source in data['source_comparisons']:
            lines += [link(root, source['directory'] + '/report.md', source['directory']),
                      *table(['原配对','状态','加速比','原因'],[(cell(p.get('pair_id')),cell(p.get('status')),number(p.get('speedup')),cell(p.get('reason') or '原配对通过')) for p in source.get('comparisons',[])])]
    return lines + appendix(root, data)


def mib(value):
    return number(value / 1048576) if finite(value) else '未记录'


def correctness_view(root, tasks, names):
    """Summarize saved checks without loading tensors or changing verdicts."""
    rows, extra = [], []
    for task in tasks:
        diagnostic = saved_json(root, task, 'numeric-diagnostic.json') or {}
        for phase, title, gate in correctness_phases(task):
            checks = list(gate.get('checks', []))
            checks += gate.get('gradient', {}).get('checks', [])
            errors = [c['max_absolute_error'] for c in checks if finite(c.get('max_absolute_error'))]
            counts = [c['mismatch_count'] for c in checks if finite(c.get('mismatch_count'))]
            coverage = f'{len(counts)}/{len(checks)} 项检查有元素计数' if checks else '未记录检查统计'
            rows.append((names[id(task)], title, status(gate.get('status')),
                         number(max(errors) if errors else None),
                         number(sum(counts) if counts else None),
                         number(gate.get('atol')), number(gate.get('rtol')), coverage))
            if 'keep_fraction' in gate:
                extra.append(f'- {names[id(task)]} / {title}：保留率 {number(gate.get("keep_fraction"))}，目标 0.8，允许偏差 {number(gate.get("keep_fraction_tolerance"))}；保留率为非零输入中输出保留的元素比例。')
            values = [i.get('device_vs_fp64', {}) for i in diagnostic.get('comparisons', {}).get(phase, [])]
            complete = values and all(finite(v.get('elements')) and finite(v.get('mismatch_count')) for v in values)
            total = sum(v['elements'] for v in values) if complete else 0
            ratios = [v['max_tolerance_ratio'] for v in values if finite(v.get('max_tolerance_ratio'))]
            if total or ratios:
                fraction = number(100 * sum(v['mismatch_count'] for v in values) / total) + '%' if total else '未记录'
                extra.append(f'- {names[id(task)]} / {title}附加对照：不匹配比例 {fraction}，最大容差倍数 {number(max(ratios) if ratios else None)}。 ' + link(root, task.get('directory', '') + '/numeric-diagnostic.json'))
    return ['### 正确性与误差统计', '', *table(
        ['组合','阶段','正确性','最大绝对误差','不匹配元素数','atol','rtol','统计覆盖'], rows), '',
        '> 最大绝对误差为已记录检查中的最大 |实际值−参考值|，原检查对有限值计算该误差；不匹配元素数为已记录检查计数之和。atol 是绝对容差，rtol 是相对容差，浮点匹配条件为 |实际值−参考值| ≤ atol＋rtol×|参考值|。整数和布尔使用精确比较；随机算子使用保存的结构、保留率及梯度契约。', '',
        *extra, *(['', '附加对照的不匹配比例按对应总元素数计算；最大容差倍数为绝对误差与允许误差之比，大于 1 表示超出容差。'] if extra else []), '']


def measurement_view(root, tasks, names):
    lines = []
    diagnostic = [t for t in tasks if t.get('measurement') and not valid_performance(t)]
    if diagnostic:
        lines += ['### 仅供诊断的测量（未满足完整通过条件）', '', *table(['组合','主机 μs','说明'],
            [(names[id(t)],number(t['measurement'].get('median_us')),interpretation(t)[0]) for t in diagnostic]), '']
    measured = [t for t in tasks if t.get('measurement')]
    fields = [('median_us','主机 μs/调用'),('cold_us','cold μs'),('warm_us','warm μs'),('kernel_us','原测量设备 μs'),('throughput_op_s','调用/s'),
              ('effective_bandwidth_gb_s','逻辑 GB/s'),('equivalent_tflops','等效 T运算/s'),('fu_percent','FU %')]
    fields = [(k,n) for k,n in fields if any(t['measurement'].get(k) is not None for t in measured)]
    meanings = dict(median_us='各轮同步主机批次平均耗时的中位数；越低表示每调用耗时越少',
                    cold_us='前向首次调用耗时，覆盖首次执行的初始化等开销',
                    warm_us='完成预热后的单次前向调用耗时',
                    kernel_us='原测量记录的设备计时值；计时方法见附录中的保存契约',
                    throughput_op_s='每秒调用数，等于 1,000,000 ÷ 主机每调用微秒数',
                    effective_bandwidth_gb_s='语义读写字节数 ÷ 主机每调用耗时，按输入及参数各读一次、实体输出写一次估算',
                    equivalent_tflops='Case 公式估算的每调用运算量 ÷ 主机耗时，以每秒万亿次运算表示；反向使用估计乘数',
                    fu_percent='等效运算率 ÷ 配置峰值 × 100%，表示相对配置峰值的比例')
    body = ['### 性能', '', *table(['组合','用途',*[n for _,n in fields]],[(names[id(t)],'有效性能' if valid_performance(t) else '仅供诊断',*[number(t['measurement'].get(k)) for k,_ in fields]) for t in measured]), '',
            *table(['指标','含义'], [(n, meanings[k]) for k,n in fields])]
    body += ['', '### 轮间统计', '', *table(['组合','轮数','最小 μs','最大 μs','标准差 μs','CV %','中位数 95% CI μs'],[(names[id(t)],number(t['measurement'].get('statistics',{}).get('rounds',t.get('case_config',{}).get('rounds'))),*[number(t['measurement'].get('statistics',{}).get(k)) for k in ('min_us','max_us','stddev_us','cv_percent')],interval(t['measurement'].get('statistics',{}).get('median_ci95_us'))) for t in measured]),
             '', '> 每轮得到一个批次平均耗时样本。轮数为样本数量；CV表示相对波动。中位数 95% CI 使用同进程轮次样本进行 bootstrap 重采样。', '',
             '### 显存', '',
             *table(['组合','显存状态','allocated 峰值/增量 MiB','reserved 峰值/增量 MiB','原因'],
                    [(names[id(t)],status(t.get('memory',{}).get('status')),
                      ' / '.join(mib(t.get('memory',{}).get(k)) for k in ('peak_allocated_bytes','incremental_allocated_bytes')),
                      ' / '.join(mib(t.get('memory',{}).get(k)) for k in ('peak_reserved_bytes','incremental_reserved_bytes')),
                      reason_text(t.get('memory',{}).get('reason') or '; '.join(t.get('memory',{}).get('missing',[])))
                      if t.get('memory',{}).get('reason') or t.get('memory',{}).get('missing') else '') for t in tasks])]
    return lines + body + ['', '> allocated 为分配器已分配给张量等对象的内存，reserved 为分配器保留的内存池。峰值为独立 worker 采集窗口内最高值，增量为峰值相对预热后基线的增长；MiB＝1,048,576 字节。显存状态与原因说明该次采集的完成情况。', '']


def diagnosis_stage(root, directory, phase, gate, check=None):
    """A correctness verdict is an observation, never a process exit status."""
    check = check or {}
    task = {'directory': directory}
    exit_record = saved_json(root, task, phase + '-exit.json') or {}
    error = saved_json(root, task, phase + '-error.json') or {}
    observed = gate.get('status') or (check.get('status') if check.get('execution_status') == 'completed' else None)
    failed = (check.get('execution_status') == 'failed' or bool(error)
              or exit_record.get('returncode') not in (None, 0))
    if failed:
        execution = '执行异常；阶段状态为未完成'
    elif observed in ('passed', 'failed'):
        execution = '已完成计算并产生结果' if exit_record.get('returncode') == 0 else '已有正确性判定；进程退出记录未确认'
    elif exit_record.get('returncode') == 0:
        execution = '进程正常退出；正确性证据缺失'
    else:
        execution = '未取得完成计算的证据'
    evidence = [link(root, directory + '/' + phase + '-exit.json'),
                link(root, directory + '/' + ('correctness.json' if phase == 'probe' else 'measurement.json'))]
    return execution, observed, '；'.join(evidence)


def diagnosis_numeric(root, replay=False):
    rows, meanings = [], {}
    scopes = [('source-task', '原运行')] + ([('replay', '本次复放')] if replay else [])
    for directory, scope in scopes:
        value = saved_json(root, {'directory': directory}, 'numeric-diagnostic.json') or {}
        for phase in ('probe', 'measure'):
            for item in value.get('comparisons', {}).get(phase, []):
                a, c = item.get('device_vs_fp64', {}), item.get('cpu_fp32_vs_fp64', {})
                count, total = a.get('mismatch_count'), a.get('elements')
                fraction = f'（{100 * count / total:.2f}%）' if finite(count) and finite(total) and total > 0 else ''
                name = f'{scope} {phase} / 张量 {item.get("tensor", "?")} / ' + ('输入梯度' if item.get('role') == 'input-gradient' else '输出')
                rows.append((cell(name), f'{number(count)} / {number(total)}{fraction}',
                             number(a.get('absolute_error_quantiles', {}).get('max')),
                             number(a.get('max_tolerance_ratio')), number(c.get('mismatch_count')),
                             link(root, directory + '/numeric-diagnostic.json')))
                if finite(count) and count > 0 and c.get('mismatch_count') == 0:
                    meanings.setdefault("设备结果超阈值，而 CPU FP32 对照全部满足同一容差；优先检查设备计算路径及精度行为，尚不能据此指定某个 kernel 或数学模式为根因。", []).append(cell(name))
                elif finite(count) and count > 0 and finite(c.get('mismatch_count')) and c['mismatch_count'] > 0:
                    meanings.setdefault("设备与 CPU FP32 都有超阈值元素；应同时检查算子数值敏感性、参考和容差依据，不能仅凭失败归责设备实现。", []).append(cell(name))
    if not rows:
        return ['未保存可用的逐张量数值对照；不能量化偏差或比较 CPU FP32 与设备误差。', '']
    return [*table(['范围 / 阶段 / 张量', '设备超阈值 / 总元素', '最大绝对误差', '最大容差倍数', 'CPU FP32 超阈值元素', '证据'], rows), '',
            '浮点误差按 |设备值−参考值| ≤ atol + rtol×|参考值| 判断；容差倍数大于 1 表示超阈值。整数/布尔按精确比较，不适用的浮点指标显示“未记录”。参考值由 CPU FP64 实现计算。', '',
            *(['**这些数据意味着什么：**', '', *['- ' + text + ' 对应：' + '；'.join(names) + '。' for text, names in meanings.items()], ''] if meanings else [])]


def diagnosis_report(root, data):
    source, checks = data.get('source_result', {}), data.get('checks', [])
    replay = data.get('mode') == 'replay'
    def check(name, scope):
        return next((c for c in checks if c.get('check') == name and c.get('scope') == scope), {})
    original_gate = source.get('correctness') or {'status': check('correctness', 'saved-source').get('status')}
    measured_gate = source.get('measurement', {}).get('correctness') or {'status': check('measurement-correctness', 'saved-source').get('status')}
    replay_check = check('probe', 'new-replay')
    replay_gate = saved_json(root, {'directory': 'replay'}, 'correctness.json') or {}
    original_failure = any(g.get('status') == 'failed' for g in (original_gate, measured_gate))
    execution_error = bool(source.get('error')) or source.get('status') in ('blocked', 'interrupted')
    if execution_error:
        conclusion = '原任务执行受阻或中断，未完成完整测试。'
        if original_failure: conclusion += '此前还记录了正确性失败，两类问题需分别处理。'
    elif original_failure:
        conclusion = '原任务已有计算结果，但正确性未通过。失败原因是已产出的结果不满足比较契约。'
    elif source.get('status') == 'passed':
        conclusion = '原任务记录为通过；本报告检查其证据，不扩大原验收范围。'
    else:
        conclusion = '现有记录不足以确认原任务完整通过；需要区分执行证据与路径证据的缺口。'
    lines = ['# Operation 事后诊断报告', '', '## 先看结论', '',
             f'对象：**{label(data.get("source_task", {}))}**。', '', '**' + conclusion + '**', '']
    if replay:
        replay_status = replay_gate.get('status') or replay_check.get('status')
        if replay_check.get('execution_status') == 'completed' and replay_status == 'failed':
            result = '本次复放完成了 probe 计算，正确性仍未通过。' if original_failure else '本次复放完成了 probe 计算，发现正确性失败。'
        elif replay_check.get('execution_status') == 'completed' and replay_status == 'passed':
            result = '本次复放 probe 正确性通过；本次未复现原数值失败，但单次 probe 不能推翻原结果或证明问题已经修复。' if original_failure else '本次复放 probe 正确性通过；不构成完整 benchmark 或重复性验收。'
        elif replay_check.get('execution_status') == 'failed':
            result = '本次复放 probe 执行异常，无法据此判断原数值问题是否再次出现。'
        else:
            result = '本次请求了复放，但未取得 probe 完成证据，不能声称已重新执行成功。'
        lines += ['**复放结果：** ' + result + ' 这是本次复放的观察，与原运行分别判读。', '']
    else:
        lines += ['**本次为离线诊断：** 没有重新执行算子。下面的结果和补采对照均来自原运行，不能证明当前环境仍然复现。', '']
    for name, value in runtime_errors(data):
        lines += [f'**本次诊断{name}：** {cell(value)}。本次检查未完整结束，已取得的原证据仍单独保留。', '']
    failed_checks = [c for c in checks if c.get('execution_status') == 'failed']
    for c in failed_checks:
        lines += [f'**检查执行异常（{cell(c.get("scope"))} / {cell(c.get("check"))}）：** {cell(c.get("observation"))}', '']
    # Promote the decisive comparison, keeping its run/phase/tensor scope explicit.
    for directory, scope, phases in [('replay', '本次复放', ('probe',)), ('source-task', '原运行', ('measure', 'probe'))]:
        if directory == 'replay' and not replay:
            continue
        numeric = saved_json(root, {'directory': directory}, 'numeric-diagnostic.json') or {}
        candidates = [(phase, row) for phase in phases for row in numeric.get('comparisons', {}).get(phase, [])
                      if finite(row.get('device_vs_fp64', {}).get('mismatch_count')) and row['device_vs_fp64']['mismatch_count'] > 0]
        if candidates:
            phase, row = candidates[0]
            a, c = row['device_vs_fp64'], row.get('cpu_fp32_vs_fp64', {})
            role = '输入梯度' if row.get('role') == 'input-gradient' else '输出'
            lines += [f'**关键数值证据（{scope} {phase}，{role}张量 {cell(row.get("tensor", "?"))}）：** '
                      f'设备超阈值 {number(a.get("mismatch_count"))}/{number(a.get("elements"))} 个元素；'
                      f'CPU FP32 对照超阈值 {number(c.get("mismatch_count"))} 个。'
                      '其余张量与阶段见下表。 ' + link(root, directory + '/numeric-diagnostic.json'), '']
            break
    lines += ['**根因确认程度：** 错误分类和数值对照用于缩小排查范围；故障根因等待受控实验确认。', '']
    if original_failure:
        lines += ['**影响：** 原任务的耗时仅供诊断，不能作为正确性通过的性能结论。', '']
    if execution_error or source.get('diagnosis'):
        lines += [f'原失败阶段：{cell(source.get("failure_stage", "未记录"))}；'
                  f'原异常：{cell(source.get("error") or "未记录执行异常")}；'
                  f'已有分类提示：{cell(source.get("diagnosis") or "未分类")}。 ' + link(root, 'source-result.json'), '']
    lines += ['## 为什么作出这个判断', '', '### 计算是否完成，与结果是否正确分开看', '']
    stages = [('原运行 probe', *diagnosis_stage(root, 'source-task', 'probe', original_gate))]
    if source.get('measurement') or measured_gate.get('status'):
        stages.append(('原运行 measure', *diagnosis_stage(root, 'source-task', 'measure', measured_gate)))
    if replay:
        stages.append(('本次复放 probe', *diagnosis_stage(root, 'replay', 'probe', replay_gate, replay_check)))
    lines += table(['范围 / 阶段', '执行情况', '正确性判定', '证据'],
                   [(name, execution, status(gate), evidence) for name, execution, gate, evidence in stages])
    lines += ['', '进程退出码 0 只表示该阶段正常返回；正确性 failed 表示产出的结果未满足契约。这两种状态可以同时出现。', '']
    if original_failure or replay_gate.get('status') == 'failed' or any(c.get('check') in ('diagnose', 'numeric') for c in checks):
        stochastic = data.get('source_task', {}).get('case') in ('dropout', 'native_dropout')
        lines += ['### 偏差在哪里，CPU 对照说明什么', '']
        if stochastic:
            lines += ['随机算子按结构、mask 与梯度契约检查；不使用跨设备逐元素 FP64 输出对照。具体失败子项见原正确性记录。', '']
        else:
            lines += diagnosis_numeric(root, replay)
        gates = [('原 probe', original_gate), ('原 measure', measured_gate), ('复放 probe', replay_gate)]
        lines += ['；'.join(f'{name}：atol={number(g.get("atol"))}，rtol={number(g.get("rtol"))}' for name, g in gates if g.get('status')) + '。', '']
    reference_checks = [c for c in checks if c.get('check') == 'reference-check']
    if reference_checks:
        lines += ['### 输入与参考是否可信', '']
        for c in reference_checks:
            text = ('CPU 参考重算一致；支持保存参考可复现，但不证明 Case 数学语义正确。' if c.get('status') == 'passed'
                    else '参考未通过校验或未完成校验；应先解决输入/参考问题，不能直接归责设备计算。' if c.get('status') != 'not-applicable'
                    else '随机算子只核对输入契约，沿用结构/mask/梯度规则。')
            lines += [f'- {"本次复放" if c.get("scope") == "new-replay" else "原运行"}：{text} ' + ' '.join(link(root, p) for p in c.get('evidence', []))]
        lines += ['']
    roundtrip = saved_json(root, {'directory': 'replay'}, 'input-roundtrip.json') if replay else None
    if roundtrip:
        lines += ['本次复放输入往返校验：' + ('通过，保存输入与设备往返值一致。' if roundtrip.get('passed') is True
                  else '未通过，应先检查输入传输与表示，不能直接归责计算实现。')
                  + ' ' + link(root, 'replay/input-roundtrip.json'), '']
    routes = [('原运行', source.get('routing') or {'status': check('routing', 'saved-source').get('status')}, 'source-result.json')]
    if replay:
        routes.append(('本次复放', saved_json(root, {'directory': 'replay'}, 'routing.json') or check('routing', 'new-replay'), 'replay/routing.json'))
    route_gap = any(r.get('status') != 'passed' for _, r, _ in routes)
    if route_gap:
        lines += ['## 独立的证据缺口：计算路径尚未完整确认', '',
                  'routing partial 表示路径证据不足，不能据此判断算子没有运行，也不能把它当作数值偏差的原因。', '']
        for name, route, path in routes:
            lines += [f'- {name}：{status(route.get("status"))}。已记录派发：{cell("；".join(route.get("dispatch", [])) or "未记录具体派发")}。 ' + link(root, path)]
        lines += ['', 'CPU 调用链仅提供 API/ATen 路径线索；主体计算归属与 kernel 耗时仍需设备执行证据。', '']
    lines += ['## 下一步：按当前证据选择验证动作', '']
    reference_bad = any(c.get('status') not in ('passed', 'not-applicable') for c in reference_checks)
    if runtime_errors(data) or failed_checks:
        lines += ['1. 先解决本次诊断的执行、清理或健康异常；查看上面的失败阶段及附录日志，确认设备可用后再补查。', '']
    if reference_bad:
        lines += ['- 优先核对保存输入、模块状态和 Case 契约，并重算参考；参考一致后再解释设备误差。当前证据不能用于锁定设备实现缺陷。']
    elif execution_error:
        lines += ['- 从原失败阶段的异常和日志检查运行依赖/资源条件；错误分类只作线索。恢复条件后重跑原组合，取得实际输出才能开展数值对照。']
    elif original_failure or replay_gate.get('status') == 'failed':
        if replay_check.get('execution_status') == 'completed' and replay_gate.get('status') == 'passed':
            lines += ['- 对比原测量进程与本次 probe 的源码、设备、状态和调用方式；原 measure 失败不能由一次 probe 通过消除。以相同输入做重复验证，观察失败是否与阶段或状态相关。']
        else:
            lines += ['- 保持保存输入、权重与容差不变，先确认失败张量和主体计算路径，再针对该实现做单变量精度/实现对照。只有误差随受控变量变化，才能进一步支持具体原因；当前公共诊断不自动执行这些专项实验。']
        if not replay:
            lines += ['- 如需判断当前环境是否复现，且输入与必要参考齐全，可显式使用 --replay；离线结果本身不回答当前是否复现。']
        elif replay_check.get('execution_status') == 'completed' and replay_gate.get('status') == 'failed':
            lines += ['- 已取得的复放结果见首段；重复性验证可另做多次运行，根因确认仍需要受控对照。']
    elif route_gap:
        lines += ['- 优先补充主体计算的执行路径证据；路径未确认前，不将已有耗时作为目标实现的通过性能。']
    else:
        lines += ['- 核对附录中的 missing/failed/skipped 项；没有新增异常时，无需仅为生成报告再次占用设备。']
    missing = [c for c in checks if c.get('status') in ('missing', 'unsupported', 'not-run')]
    if missing:
        lines += ['', '**仍缺少的证据：** ' + '；'.join(f'{cell(c.get("check"))}：{cell(c.get("observation"))}' for c in missing) + '。']
    lines += ['', '## 附录：原状态、完整检查与证据', '',
              f'原任务：**{status(source.get("status"))}**；本次检查执行：{status(data.get("execution_status"))}；'
              f'证据检查完整性（原 status 字段）：{status(data.get("status"))}。', '',
              '这些状态分别描述任务验收、诊断流程和证据覆盖。下表 execution 项沿用原任务总状态；算子正确性由对应比较契约判定。', '',
              *folded('展开完整检查记录（保留原字段与观察）', checks_view(root, checks)),
              '源结果：' + link(root, 'source-result.json') + '；源运行身份：' + link(root, 'source-identity.json') + '。']
    return lines + appendix(root, data)


def read_counter_evidence(root, task, group):
    """Only consume manifest-covered files, including verified aggregate sources."""
    source_root = (root / task['source_run']).resolve() if task.get('source_run') else root
    path = root / task.get('directory', '') / group.get('directory', '') / 'counter-rows.json'
    try:
        rel = str(path.resolve().relative_to(source_root.resolve()))
        manifest = json.loads((source_root / 'artifacts.json').read_text())
        if rel not in manifest.get('files', {}):
            return [], '计数器文件未封存或未保存'
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get('rows'), list):
            return [], '计数器文件 rows 格式无效'
        return data['rows'], None
    except (OSError, ValueError, TypeError):
        return [], '计数器文件不可读取或格式无效'


def metric_rows(root, task, group):
    rows, error = read_counter_evidence(root, task, group)
    values = {}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get('values', {})
        if not isinstance(raw, dict) or not isinstance(row.get('metrics', {}), dict):
            continue
        # Duplicated export rows must not increase the sample count.
        signature = json.dumps(row, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        kernel = raw.get('Op Name', '未记录 kernel 名称')
        for field, value in row.get('metrics', {}).items():
            if finite(value):
                values.setdefault((kernel, field), []).append(value)
    output = []
    metadata = report_metadata(task.get('vendor'))
    describe = metadata.describe_metric if metadata else lambda group, field: None
    for (kernel, field), samples in sorted(values.items()):
        description = describe(group.get('group'), field) or {
            'explanation': '当前报告未收录该字段定义；该数值仅用于回查 counter-rows.json，不参与正文瓶颈判断。',
            'scale': 1.0,
            'suffix': '',
            'key': False,
        }
        output.append(dict(kernel=kernel, field=field, samples=len(samples), median=statistics.median(samples),
                           low=min(samples), high=max(samples), **description))
    if not output and not error:
        error = '没有可用的有限数值计数器记录'
    return output, error


def metric_value(record, value):
    if not finite(value):
        return '未记录'
    return f"{value * record.get('scale', 1.0):.6g}{record.get('suffix', '')}"


def metric_table(entries, fields=None):
    """Render one observation per path/kernel/field without merging kernel populations."""
    all_records = [record for _, _, records in entries for record in records]
    selected_fields = fields or sorted({record['field'] for record in all_records})
    rows = []
    for field in selected_fields:
        meaning = next(record['explanation'] for record in all_records if record['field'] == field)
        for task, _, records in entries:
            selected = [record for record in records if record['field'] == field]
            if not selected:
                rows.append((cell(field), cell(task.get('oplib', '未记录')), '未记录',
                             '未记录', '未记录', meaning))
                continue
            for record in selected:
                rows.append((cell(field), cell(task.get('oplib', '未记录')), cell(record['kernel']),
                             metric_value(record, record['median']), record['samples'], record['explanation']))
    return table(['原始字段', '路径', 'kernel', '中位数', '样本数', '含义'], rows)


def duration_summary(calls, field):
    values = [c.get(field) for c in calls]
    if not values or not all(finite(v) and v >= 0 for v in values):
        return '未记录'
    return f'{statistics.median(values):.6g} [{min(values):.6g}, {max(values):.6g}]'


def profiling_view(root, data, names):
    body, details, gaps = [], [], []
    captures, counters, timeline_rows = [], [], []
    task_timelines = {}
    requested = attempted = False
    for task in data.get('tasks', []):
        name = names[id(task)]
        mem = task.get('memory', {})
        if mem.get('status') not in ('measured', 'off', None) and not expected_gate_skip(task, mem):
            gaps.append(name + '：显存' + status(mem.get('status')) + '；' + cell(mem.get('reason') or '; '.join(mem.get('missing', [])) or '见附录。'))
        prof = task.get('profiling', {})
        gated_skip = expected_gate_skip(task, prof)
        mode = task.get('profiling_mode', data.get('profiling'))
        groups = prof.get('groups', [])
        if mode == 'off' and not groups:
            continue
        if mode not in (None, 'off') or groups or prof.get('status') not in (None, 'off'):
            requested = True
        else:
            continue
        if not gated_skip:
            attempted = True
        captures.append((name, '采集整体', status(prof.get('status')),
                         reason_text(prof.get('reason')) if prof.get('reason') else ''))
        if prof.get('status') not in ('measured', 'off') and not gated_skip:
            gaps.append(name + '：Profiling ' + status(prof.get('status')) + '；' + cell(prof.get('reason') or '没有完整采集记录。'))
        if not gated_skip and mode in ('timeline', 'full') and not any(g.get('group') == 'timeline' for g in groups):
            gaps.append(name + '：未取得独立 timeline 组，不以其他复放时间代替。')
        group_names = {g.get('group') for g in groups}
        metadata = report_metadata(task.get('vendor', data.get('vendor')))
        if not gated_skip and mode == 'full' and metadata:
            absent = sorted(set(metadata.GROUP_DESCRIPTIONS) - group_names)
            if absent:
                gaps.append(name + '：未记录采集组 ' + ', '.join(absent))
        for group in groups:
            group_name = group.get('group', '未记录')
            timeline = group.get('timeline', {})
            calls = timeline.get('calls', [])
            qualified = group.get('status') == 'measured' and timeline.get('status') == 'measured'
            prefix = task.get('directory', '') + '/' + group.get('directory', '') + '/'
            explanation = group.get('reason') or timeline.get('reason') or '已取得采集数据，供结合执行时间分析瓶颈'
            captures.append((name, cell(group_name), status(group.get('status')), reason_text(explanation)))
            if not qualified:
                gaps.append(name + ' / ' + cell(group_name) + '：' + reason_text(explanation))
            kernel_names = sorted({str(k.get('name')) for k in timeline.get('kernels', [])})
            if group_name == 'timeline' and qualified and calls:
                counts = [c.get('kernel_count') for c in calls]
                count_text = (str(min(counts)) if min(counts) == max(counts) else str(min(counts)) + '～' + str(max(counts))) if all(finite(c) for c in counts) else '未记录'
                timeline_rows.append((name, cell(', '.join(kernel_names)), len(calls), count_text,
                                      duration_summary(calls, 'device_union_us')))
                task_timelines[task.get('directory')] = calls
            trace = timeline.get('trace')
            evidence = [link(root, prefix + 'capture.log')]
            if trace:
                evidence.append(link(root, prefix + trace, '原始设备时间线'))
                if not (root / (prefix + trace)).is_file():
                    gaps.append(name + ' / ' + cell(group_name) + '：摘要引用的原始时间线文件缺失。')
            local = ['；'.join(evidence), '', 'kernel：' + cell(', '.join(kernel_names) or '未记录'), '']
            if calls:
                local += table(['调用','kernel 数','累计 μs','并集 μs','首尾跨度 μs'],
                               [(i+1,c.get('kernel_count','未记录'),number(c.get('kernel_sum_us')),number(c.get('device_union_us')),number(c.get('device_span_us'))) for i,c in enumerate(calls)])
            if group_name != 'timeline':
                records, error = metric_rows(root, {**task, 'vendor':task.get('vendor',data.get('vendor'))}, group)
                if error:
                    gaps.append(name + ' / ' + cell(group_name) + '：' + error + '；摘要状态保留为 ' + cell(group.get('status')))
                    local += ['', error]
                else:
                    local += ['', link(root, prefix + 'counter-rows.json'), '',
                              *metric_table([(task, name, records)])]
                    if qualified:
                        counters.append((task, name, group_name, records))
            else:
                local += ['', '此组仅采集时间线，不适用硬件计数器文件。']
            details += folded(name + ' / ' + group_name + '：逐次调用及原始指标', local)
    if not requested:
        return [], [], gaps
    body += ['', '### 设备执行与硬件观察', '',
             'kernel 是设备实际执行的计算程序；每调用 kernel 数表示一次算子调用关联的执行事件数量，调用样本数为本组记录的目标调用数量。', '',
             '', '']
    if timeline_rows:
        body += table(['组合','目标 kernel','调用样本数','每调用 kernel 数','并集 μs：中位数 [范围]'], timeline_rows)
        body += ['', '并集只累计至少一个关联 kernel 在执行的时间，重叠只计一次', '']
        multiple = [(directory,calls) for directory,calls in task_timelines.items() if any(finite(c.get('kernel_count')) and c['kernel_count']>1 for c in calls)]
        if multiple:
            body += ['累计时间把 kernel 时长相加（重叠重复计）；首尾跨度从最早开始到最晚结束（包含间隙）。', '',
                     *table(['组合','累计 μs：中位数 [范围]','首尾跨度 μs：中位数 [范围]'],
                            [(next(names[id(t)] for t in data['tasks'] if t.get('directory')==d),duration_summary(c,'kernel_sum_us'),duration_summary(c,'device_span_us')) for d,c in multiple]), '']
        elif all(c.get('kernel_count') == 1 for calls in task_timelines.values() for c in calls):
            body += ['', '']
        for pair, libs, ok, _ in comparison_views(data):
            if ok and all(t.get('directory') in task_timelines for t in libs.values()):
                a = [c.get('device_union_us') for c in task_timelines[libs['nativetorch']['directory']]]
                b = [c.get('device_union_us') for c in task_timelines[libs['flaggems']['directory']]]
                if all(finite(v) and v>0 for v in a+b):
                    body += [f'- {names[id(libs["nativetorch"])]} / {names[id(libs["flaggems"])]}：独立 timeline 中 FlagGems 每调用设备并集时间中位数为 native 的 {statistics.median(b)/statistics.median(a):.4g} 倍；此倍率描述独立复放的设备执行时间。']
    elif attempted:
        body += ['没有完整的独立 timeline，当前无法汇总目标设备执行时间。', '']
    if counters:
        body += ['', '### 硬件指标如何理解', '',
                 '每行对应一个采集组中的“路径 × kernel × 字段”；中位数来自去重后的重复调用样本。'
                 '已识别的周期占比统一显示为百分比，带宽单位保留在字段名 `(GB/s)` 中，运算次数保持计数值。', '',
                 '六个硬件指标组来自相互独立的复放，不能把不同行直接拼成同一次执行；'
                 '应结合独立 timeline 的设备时间，分别判断计算、搬运和资源冲突。', '']
        body += hardware_view(data, counters, names)
    coverage = []
    for task in data.get('tasks', []):
        prof = task.get('profiling', {})
        groups = prof.get('groups', [])
        if groups or task.get('profiling_mode', data.get('profiling')) in ('timeline', 'full'):
            measured = sum(g.get('status') == 'measured' for g in groups)
            coverage.append((names[id(task)], status(prof.get('status')), f'{measured} / {len(groups)}',
                             reason_text(prof.get('reason')) if prof.get('reason')
                             else '按实际记录组数统计采集完成情况'))
    body += ['### 采集覆盖与缺口', '', *table(['组合','整体状态','已采集 / 已记录组数','解释'], coverage), '']
    details = folded('完整采集组状态', table(['组合','采集项','状态','解释'], captures)) + details
    if gaps:
        body += ['- ' + g for g in dict.fromkeys(gaps)] + ['', '下一步：先核对对应采集日志与原始文件；离线重建不会补采数据。']
    return body, ['', '## 附录：Profiling 逐次证据', '', *details], list(dict.fromkeys(gaps))


def hardware_view(data, counters, names):
    """Align fields only within existing qualified pairs; keep kernel populations separate."""
    paired = {}
    pair_members = {}
    for pair, libs, ok, _ in comparison_views(data):
        if ok:
            pair_members[pair['pair_id']] = libs
            for task in libs.values():
                paired[task.get('directory')] = pair['pair_id']
    blocks = {}
    for task, name, group, records in counters:
        identity = paired.get(task.get('directory'), task.get('directory'))
        block = blocks.setdefault((identity, group), [])
        block.append((task, name, records))
    lines = []
    for (identity, group), entries in blocks.items():
        present = {t.get('directory') for t,_,_ in entries}
        for task in pair_members.get(identity, {}).values():
            if task.get('directory') not in present:
                entries.append((task, names[id(task)], []))
        entries.sort(key=lambda e: e[0].get('oplib') != 'nativetorch')
        description = group
        metadata = report_metadata(entries[0][0].get('vendor', data.get('vendor')))
        if metadata:
            description = metadata.GROUP_DESCRIPTIONS.get(group, group)
        lines += ['**' + cell(description) + '（' + cell(group) + '）** — ' + ' / '.join(e[1] for e in entries), '']
        all_records = [r for _,_,records in entries for r in records]
        if all(records for _,_,records in entries) and all(r['low'] == r['high'] == 0 for r in all_records):
            lines += ['已采集，当前记录值均为零；逐字段定义和样本数见附录。'
                      '零值只表示这些样本未记录到对应活动或阻塞，不代表所有执行都不存在。', '']
            continue
        fields = sorted({r['field'] for r in all_records if r['key'] and (r['low'] != 0 or r['high'] != 0)})
        if not fields:
            lines += ['当前没有已识别且非零的摘要指标；完整字段、数值和已收录解释见附录。', '']
            continue
        lines += metric_table(entries, fields) + ['']
    return lines


def saved_json(root, task, relative):
    source_root = (root / task['source_run']).resolve() if task.get('source_run') else root.resolve()
    path = root / task.get('directory', '') / relative
    try:
        rel = str(path.resolve().relative_to(source_root))
        manifest = json.loads((source_root / 'artifacts.json').read_text())
        if rel in manifest.get('files', {}):
            value = json.loads(path.read_text())
            return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        pass
    return None


def input_scope(task, reference):
    inputs = reference.get('inputs') or task.get('measurement', {}).get('logical_traffic', {}).get('inputs', [])
    shapes = '; '.join(str(t.get('shape', '未记录')) for t in inputs) or '形状未记录'
    backward = reference.get('backward')
    execution = '前向＋sum＋反向批次' if backward is True else '前向' if backward is False else '前向/反向范围未记录'
    return cell(shapes + ' / ' + execution)


def interval(values):
    return '[' + ', '.join(number(v) for v in values) + ']' if isinstance(values, (list, tuple)) else '未记录'
