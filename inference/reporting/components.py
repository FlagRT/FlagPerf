# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Component-aware presentation; numbers remain owned by existing report generators."""
NAMES={'flaggems':'FlagGems','flagtree':'FlagTree','flagcx':'FlagCX'}


def description(result):
    axis=result.get('comparison_component')
    if result.get('command') == 'preview':
        environment = '按 '+NAMES[axis]+' off/on 分别探测' if axis else '在固定编译器和通信环境中探测'
        return ('本次主动探测 FlagGems，'+environment+'。表中的 FlagGems 是背景配置值；'
                '实际探测从原生基线开始，逐步启用候选，再完整复验最终集合。这里不生成精度差分或性能比。')
    if axis:
        return f'本次只比较 {NAMES[axis]}：off 为参照，on 为待比较路径；其他组件保持固定，完整配置见下表。off 提供本次组件对照的参照结果。'
    return '本次为固定组件组合的单路径执行，展示该路径的观测指标。路径名沿用 FlagGems 状态，完整配置见下表。'


def add_evidence(root,result):
    if 'component_profiles' not in result: return
    lines=['## 三组件配置与生效证据','']
    lines += ['', 'FlagTree off/on 分别选择厂商 Triton-Ascend/镜像内 FlagTree；FlagCX off/on 分别选择 HCCL/FlagCX 模型通信组。Gloo 仅用于测试协调。FlagCX 在 Ascend 下通过底层 HCCL 执行通信，框架选择见模型通信后端列。', '',
              '编译器 import 版本、包元数据版本可能不同；以实际模块路径和源码摘要核验选择。JIT 调用列记录本次实际参与；结合配对计时查看编译器路径差异。', '',
              '| 路径/rank | 编译器 | JIT 执行调用 | launch hook | 模型通信后端 | collective 调用 | 通信状态 | 证据 |',
              '|---|---|---:|---:|---|---:|---|---|']
    sources=result.get('component_audits',{}) if result.get('command')=='performance' else result.get('paths',{})
    for side,state in sources.items():
        for rank,row in state.get('ranks',{'single':state}).items():
            evidence=row.get('components',{})
            compiler=evidence.get('compiler',{}); comm=evidence.get('communication',{})
            status={'observed':'已观测','not_applicable':'不适用','not_observed':'未观测','mismatch':'后端不匹配'}.get(comm.get('status'),'未采集')
            base=('audit/'+side if result.get('command')=='performance' else side)
            if rank!='single': base+='/rank-'+rank
            link=f'[组件证据]({base}/components.json)' if (root/base/'components.json').is_file() else '未生成'
            lines.append(f"| {side}/{rank} | {compiler.get('provider','未采集')} | {compiler.get('jit_run_calls','未采集')} | {compiler.get('launch_hook_calls') if compiler.get('launch_hook_calls') is not None else '未采集'} | {', '.join(comm.get('observed_backends',[])) or '未观测'} | {comm.get('model_collective_calls','未采集')} | {status} | {link} |")
    lines += ['', '上表来自精度执行或独立性能诊断。JIT/launch 计数覆盖实际调用和可能的自动调优；ATen 路由与硬件事件各自计数。单卡集合通信标为不适用；缺少设备通信时间时保持未知，已采集项展示实测值。', '',
              '[两侧配置与身份](comparison-context.json) · [完整执行结果](result.json)', '']
    if result.get('policy_profiles'):
        lines += ['### 联合策略各环境记录','',
                  '以下链接对应共同集合在各环境的完整输入复验；覆盖分类见首页。','',
                  '| 环境身份 | 复验 | 证据 |','|---|---|---|']
        for key,entry in result['policy_profiles'].items():
            verification=entry.get('verification')
            if verification:
                link=verification if verification.endswith('.json') else verification.rstrip('/')+'/result.json'
                evidence=f'[完整输入复验]({link})'
            else:
                evidence='未完成复验，未发布策略'
            lines.append(f"| {key[:12]} | {entry['status']} | {evidence} |")
        lines += ['']
    summaries=result.get('route_summaries',{})
    for side,summary in summaries.items():
        lines += [f'### {side} 的 FlagGems 策略调用分类','',
                  '| 类别 | 顶层 ATen 调用数 |','|---|---:|']
        labels={'excluded':'策略已排除','unverified':'未纳入本次策略','uncovered':'无候选覆盖','allowed_function_observed':'候选函数已命中','ambiguous':'无法唯一归因'}
        for name,count in summary['counts'].items(): lines.append(f'| {labels.get(name,name)} | {count} |')
        lines += ['', '分类由该环境独立原生取证与 FlagGems 取证生成；硬件 kernel fallback 未采集。辅助原生取证提供路由分类参照。','']
    path=root/'report.md'
    text=path.read_text()
    title,_,body=text.partition('\n')
    path.write_text(title+'\n'+body+'\n'+'\n'.join(lines))


STATUS_NAMES = {'observed':'已观测参与', 'not_observed':'未观测到参与',
                'not_applicable':'不适用', 'incomplete':'证据不完整',
                'single_path':'单路径', 'not_evaluated':'不评价组件收益'}


def overview(root, result, assessment):
    from runtime.common import read_json
    lines = ['## 本次结论', '']
    identity_path = root/'prepared/identity.json'
    if identity_path.is_file():
        identity = read_json(identity_path)
        model, inputs = identity.get('model', {}), identity.get('inputs', {})
        devices = identity.get('parallelism', {}).get('devices', identity.get('physical_device', '未记录'))
        lines += [f"模型：{model.get('name','未记录')}；dtype：{model.get('dtype','未记录')}；物理设备：{devices}；batch size：{inputs.get('batch_size','未记录')}；max length：{inputs.get('max_length','未记录')}。", '']
    if result.get('component_profiles'):
        lines += [description(result), '', '| 路径 | FlagGems | FlagTree | FlagCX |', '|---|---|---|---|']
        for side, p in result['component_profiles'].items():
            lines.append(f"| {side} | {p['flaggems']} | {p['flagtree']} | {p['flagcx']} |")
        lines += ['']
    observation = assessment['comparison']
    lines += [f"执行状态：**{result.get('status','未记录')}**；比较证据：**{STATUS_NAMES[observation['status']]}**。", '', observation['reason']+'。', '']
    if result.get('error'):
        lines += ['失败原因：'+result['error'], '']
    if (root/'policy-mismatch.json').is_file():
        lines += ['[身份失配详情](policy-mismatch.json)：请按当前配置重新 preview；不修改旧策略的身份。', '']
    if result.get('command') == 'performance':
        lines += [f"主计时：{assessment.get('measurement_status','未记录')}；独立通信证据：{assessment.get('communication_status','未记录')}。", '',
                  '主计时从输入驻留设备开始，覆盖 forward、pooling、归一化至最终同步；不含加载、传输、预热与取证。', '',
                  '| 路径 | 平均批次时延 ms | 样本/s | 批次数 |', '|---|---:|---:|---:|']
        for side, row in result.get('summary', {}).items():
            lines.append(f"| {side} | {row['latency_mean_ms']:.3f} | {row['samples_per_second']:.3f} | {row['batches']} |")
        comparison = result.get('comparison')
        if comparison:
            ratio = comparison['off_over_on_measured_time']
            throughput = comparison['on_over_off_samples_per_second']
            lines += ['', f"两路径测量比：on 耗时为 off 的 **{1/ratio:.3f} 倍**（变化 {(1/ratio-1)*100:+.2f}%）；on/off 吞吐比 **{throughput:.4f}**（变化 {(throughput-1)*100:+.2f}%）。"]
        lines += ['', '这些数值只描述本次固定输入和策略；未建立独占环境或稳定收益结论。', '']
    elif result.get('command') == 'accuracy':
        lines += ['精度仅报告差异，不设置达标阈值；同设备 off 不是绝对真值。', '']
        if result.get('comparison'):
            diff = read_json(root/result['comparison'])
            rows = diff.get('ranks', {'single':diff})
            lines += ['| rank | 模型输出 | MSE | MAE | 最低样本余弦 |', '|---|---|---:|---:|---:|']
            for rank, row in rows.items():
                for name in ('pooled','embedding'):
                    m = row.get('aggregate',{}).get(name)
                    if m:
                        fmt = lambda x: '未定义' if x is None else f'{x:.7g}'
                        lines.append(f"| {rank} | {name} | {fmt(m['mse'])} | {fmt(m['mae'])} | {fmt(m['sample_cosine_min'])} |")
            lines += ['', '异常记录数：'+str(sum(len(row.get('anomalies',[])) for row in rows.values()))+'；逐层和最差样本见后文。', '']
    policy = assessment.get('policy')
    if policy:
        lines += ['**策略验证范围：当前输入下执行完成、选择函数命中、检查边界输出有限；不代表精度达标。**', '',
                  '共同函数：'+(', '.join(next(iter(policy['environments'].values()))['common_include']) or '无')+'。',
                  '选择摘要：`'+str(policy.get('selection_sha256') or '未记录')+'`。比较历史性能前先核对函数集合及执行身份。', '',
                  '| 环境 | 候选数 | 各自接受 | 共同选择 | 选择限制 | 明确排除 | 未知 |', '|---|---:|---:|---:|---:|---:|---:|']
        for key, row in policy['environments'].items():
            lines.append(f"| {key[:12]} | {row['candidate_count'] if row['candidate_count'] is not None else '未记录'} | {len(row['accepted_include'])} | {len(row['common_include'])} | {len(row['not_in_joint_selection'])} | {len(row['excluded'])} | {len(row['unknown'])} |")
        lines += ['', '以上按环境统计函数集合，跨环境不相加；不是调用覆盖率或硬件 kernel 覆盖率。共同集合对照只评价相同函数配置，未评价各环境独立最大能力。', '']
    groups = assessment.get('performance_groups', {})
    recommendations = []
    if observation['status'] in ('not_observed','not_applicable','incomplete'):
        recommendations.append('先核对组件证据与适用条件；当前数值不能用于评价该组件收益。')
    if policy:
        reasons = {reason for row in policy['environments'].values() for reason in row['unknown_reasons']}
        if reasons & {'budget_exhausted','failure_not_confirmed_within_budget','not_scheduled',
                      'estimate_does_not_fit','global_budget_exhausted','search_allowance_exhausted','confirmation_incomplete'}:
            recommendations.append('先查看实际剩余额度、成本估计及未完成阶段，再安排足够的 --budget-seconds 在新目录续探；尚未执行不表示算子失败。')
        if 'worker_timeout' in reasons:
            recommendations.append('查看 worker 最后阶段及 runtime.timeout_seconds；单纯增加总预算不会提高单 worker 上限，超时不能直接定位为算子问题。')
        if reasons & {'unresolved_or_resource_failure','resource_failure','unresolved_failure'}:
            recommendations.append('先查看失败日志及资源占用，排除资源问题后重新 preview；未知不能当作不支持。')
        if 'evidence_incomplete' in reasons:
            recommendations.append('核对 forward-checks.json 的摘要、样本和检查边界；缺失检查证据不能发布策略。')
        if 'not_observed' in reasons:
            recommendations.append('查看未命中函数的候选签名和路由记录；当前输入没有提供足够调用证据。')
        if any(row['excluded'] for row in policy['environments'].values()):
            recommendations.append('明确排除项可沿策略中的 trial/repeat/control 记录查看失败复现和原生对照；结论限于该累积函数配置。')
    repeats = [r['on_over_off_latency'] for r in groups.get('repeat',[]) if 'on_over_off_latency' in r]
    if repeats:
        directions = {1 if r>1 else -1 if r<1 else 0 for r in repeats}
        text = '各 repeat 的变化方向一致' if len(directions)==1 else '各 repeat 的变化方向不一致'
        lines += [text+f'（{len(repeats)} 组）；描述性观察，不是稳定收益判定。', '']
    shapes = [r for r in groups.get('shape',[]) if r.get('on_over_off_latency',0)>1]
    if shapes:
        worst = max(shapes,key=lambda r:r['on_over_off_latency'])
        recommendations.append('优先复查输入形状 '+worst['label']+'：本次 on/off 平均时延比最大；不能据此认定某个函数是瓶颈。')
    if result.get('command') == 'performance' and result.get('summary'):
        recommendations.append('如需判断稳定收益，先确认设备及同板资源负载，再在相同策略下增加独立重复；单次或短轮次不作稳定性结论。')
    if recommendations:
        lines += ['### 下一步建议', ''] + ['- '+r for r in recommendations] + ['']
    if policy and (root/'effective.yaml').is_file():
        import shlex
        import yaml
        cfg = yaml.safe_load((root/'effective.yaml').read_text())
        section = cfg.get(result.get('command'), {})
        argv = ['python', 'run.py', 'preview', '--config', str(root/'effective.yaml'), '--flaggems', 'off']
        for component in ('flagtree', 'flagcx'):
            argv += ['--'+component, section.get(component,'off')]
        if (root/'preview/checkpoint.json').is_file():
            argv += ['--resume-from', str(root), '--budget-seconds', str(cfg.get('preview',{}).get('budget_seconds',3600))]
            argv += ['--preview-search', cfg.get('preview',{}).get('search_strategy','sequential'),
                     '--preview-budget', cfg.get('preview',{}).get('budget_mode','fixed')]
        argv += ['--output', str(root)+'-preview-next']
        lines += ['重新探测的命令（在 inference 目录执行，输出目录必须不存在）：', '',
                  '```bash', shlex.join(argv), '```', '',
                  '可用 --budget-seconds 指定本次预算；原始配置与证据保持只读。无新版检查点的历史运行需要全新 preview。', '']
    if assessment.get('environment'):
        lines += ['实际队列设置：'+', '.join(k+'='+str(v) for k,v in assessment['environment'].items())+'。', '']
    if 'worker_process' in assessment.get('native_backend_lifetimes',[]):
        lines += ['FlagCX 兼容条件：原生组所有者保留至一次性 worker 退出；未证明长驻进程反复建组/销毁安全。', '']
    return lines


def detail_lines(root, assessment):
    from analysis.assessment import REASONS
    lines = []
    groups = assessment.get('performance_groups')
    if groups:
        lines += ['## 输入形状与重复对照', '', groups['reason']+'。', '']
        for key, title in [('shape','按实际输入形状'),('repeat','按独立 repeat')]:
            if not groups.get(key):
                continue
            lines += ['### '+title, '', '| 分组 | 路径 | 批次数 | 平均批次 ms | 样本/s | on/off 时延比 | on/off 吞吐比 |', '|---|---|---:|---:|---:|---:|---:|']
            for group in groups[key]:
                for side, row in group['paths'].items():
                    ratio = f"{group['on_over_off_latency']:.4f}" if side=='on' and 'on_over_off_latency' in group else '—'
                    speed = f"{group['on_over_off_throughput']:.4f}" if side=='on' and 'on_over_off_throughput' in group else '—'
                    lines.append(f"| {group['label']} | {side} | {row['batches']} | {row['latency_mean_ms']:.3f} | {row['samples_per_second']:.3f} | {ratio} | {speed} |")
            lines += ['']
        if groups.get('shape_note'): lines += [groups['shape_note']+'。', '']
        lines += ['分组来自同一批正式计时记录，属于总 level 的输入负载对照；不是逐层或算子计时。TP 使用全局批次窗口，样本和 token 不按 rank 重复累计。', '']
    policy = assessment.get('policy')
    if policy:
        from pathlib import Path
        import yaml
        policy_path = root/'preview/policy.yaml' if (root/'preview/policy.yaml').is_file() else root/'policy.yaml'
        evidence_policy = policy_path
        if (root/'effective.yaml').is_file() and policy_path.name == 'policy.yaml' and policy_path.parent == root:
            cfg = yaml.safe_load((root/'effective.yaml').read_text())
            original = cfg.get('policy', {}).get('path')
            if original: evidence_policy = Path(original)
        lines += ['## 策略覆盖与未解决项', '', f'[完整策略与签名](<{policy_path}>)', '']
        for key, row in policy['environments'].items():
            background = ', '.join(NAMES.get(k,k)+'='+str(v) for k,v in (row.get('stack') or {}).items())
            lines += ['### 环境 '+key[:12], '', '背景开关：'+(background or '旧策略未记录')+'。', '', '各自接受：'+(', '.join(row['accepted_include']) or '无')+'。',
                      '仅因共同选择而未使用：'+(', '.join(row['not_in_joint_selection']) or '无')+'。', '',
                      '| 原因 | 数量 |', '|---|---:|']
            for reason,count in row['unknown_reasons'].items():
                lines.append(f'| {REASONS.get(reason,reason)} | {count} |')
            lines += ['', '| 函数 | 分类/原因 | 观测签名与证据 |', '|---|---|---|']
            for status, records in [('明确排除',row['excluded']),('未知',row['unknown'])]:
                for record in records:
                    signatures = set()
                    def tensors(value):
                        if isinstance(value, dict):
                            if 'dtype' in value and 'shape' in value:
                                signatures.add(str(value['dtype'])+' '+str(value['shape']))
                            for child in value.values(): tensors(child)
                        elif isinstance(value, list):
                            for child in value: tensors(child)
                    tensors(record.get('signatures', []))
                    text = '; '.join(sorted(signatures)[:3]) or '签名未记录'
                    if len(signatures)>3: text += f'；另 {len(signatures)-3} 种 dtype/shape，完整参数见策略'
                    evidence_base = evidence_policy.parent / Path(row.get('candidate_policy') or 'policy.yaml').parent
                    attempts = []
                    for name in ('trial','repeat','control'):
                        if name not in record: continue
                        reference = Path(record[name])
                        if reference.parts and reference.parts[0] == 'preview':
                            run_root = evidence_policy.parent.parent if evidence_policy.parent.name == 'preview' else evidence_policy.parent
                            target = run_root/reference
                        else:
                            target = evidence_base/reference
                        if target.suffix != '.json': target = target/'result.json'
                        attempts.append(f'[{name}](<{target}>)' if target.is_file() else name+'='+str(record[name])+'（原记录当前不可读）')
                    if attempts: text += '；' + ', '.join(attempts)
                    lines.append(f"| {record.get('function','未记录')} | {status}：{REASONS.get(record.get('reason'),record.get('reason','未记录'))} | {text} |")
            lines += ['']
        lines += ['观测 dtype/shape 只是本次签名；排除和回退范围仍为当前配置中的整个函数。预算和固定候选顺序会影响最后选择，策略不是最大可用集合。', '']
    return lines


def add_overview(root, result):
    from analysis.assessment import assess
    assessment = result.get('assessment') or assess(root,result)
    path = root/'report.md'
    title, _, body = path.read_text().partition('\n')
    if result.get('component_profiles'):
        body = '\n'.join(line for line in body.splitlines()
                         if not line.startswith(('执行状态：', '首个失败：', '**参照边界：'))
                         and line != description(result))
    if result.get('command') == 'preview': title = '# Inference Preview：函数执行验证'
    path.write_text(title+'\n\n'+'\n'.join(overview(root,result,assessment))+'\n'+
                    '\n'.join(detail_lines(root,assessment))+'\n'+body)
