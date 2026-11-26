# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Shared vocabulary and display units for production result reports."""
import math

LABELS = {
    'execution_summary':'执行结果','process_returncode':'进程退出码','exit_code':'退出码',
    'measure':'正式测量','unknown':'阶段未知','valid_tokens':'有效输入 token 数',
    'forward_phases':'各阶段前向次数','model_count_matches':'模型通信次数符合预期',
    'allreduce':'AllReduce 次数','allgather':'AllGather 次数',
    'differences':'条件差异','left':'基准','right':'候选','input_tokens':'输入 token 数','batch':'批量','category_share':'同类别工作量占比','comm_backend':'通信后端','workload_identity':'负载身份',
    'source_label':'比较来源','left_label':'基准来源','right_label':'比较来源','report':'对应报告',
    'event_time_ns':'原始事件时刻','event_clock':'原始事件时钟','event_detail':'原始事件内容',
    'allocation_summary':'分配失败概况','logical_device':'逻辑设备','phyid':'物理设备编号',
    'occurrence_task_relative_ns':'原始事件任务内时刻','returncode':'进程返回码',
    'observed_since_onset_ns':'本次判定距登记起点','configured_timeout_ns':'封存超时阈值','configured_heartbeat_misses':'封存心跳缺失阈值（次数）',
    'ref_backend':'参考通信后端','cmp_backend':'比较通信后端','overhead_ratio':'采集扰动比例',
    'baseline_mean_ns':'无插桩平均时延','instrumented_mean_ns':'采集条件平均时延',
    'group_execution':'组执行标识','pair_id':'配对实验标识',
    'native_owned_time_ratio':'原生归属时长占比','project_resolved_time_ratio':'项目归属时长占比','linked_time_ratio':'关联事件时长占比',
    'native_owned_ratio':'原生归属计数占比','project_resolved_ratio':'项目归属计数占比','linked_ratio':'关联事件计数占比',
    'count_denominator':'计数分母','time_denominator':'时长分母','warmup_excluded':'已排除预热观测',
    'process_repeat_distributions':'进程内重复分布','timing_boundary':'计时边界',
    'before':'前一边界','after':'后一边界','ratio':'变化倍数','change':'变化类型',
    'restricted_ns':'累计受限时间','observed_window':'有依据的观测时长','conclusion':'区间观察',
    'frequency_min_hz':'最低频率（Hz）','frequency_max_hz':'最高频率（Hz）',
    'joules_per_token':'每 token 能耗（J）','token_reason':'能耗分母范围',
    'input_sources_ref':'参考侧输入生产者','input_sources_cmp':'比较侧输入生产者',
    'input_side':'固定输入来自','self_reproduction':'各侧原输出复现',
    'metrics':'同输入误差','routes':'两侧实际实现','performance_links':'独立性能证据',
    'trigger':'触发依据','location':'已记录活动位置','next_step':'接手方向',
    'reference_to_baseline':'参考→基线','reference_to_candidate':'参考→候选','baseline_to_candidate':'基线→候选',
    'rmse':'RMSE',
    'near_zero_reference_fraction':'近零参考占比','near_zero_reference_elements':'近零参考元素数',
    'weighting':'样本与元素权重对照','pair':'对照路径','parent_names':'已记录父层',
    'sample_equal_mae':'每样本等权 MAE','element_equal_mae':'每元素等权 MAE',
    'sample_equal_mse':'每样本等权 MSE','element_equal_mse':'每元素等权 MSE',
    'cv_screening':'重复波动筛查','spike_screening':'尖峰筛查',
    'minimum_ns':'最短时延','maximum_ns':'最长时延',
    'semantics':'输出语义','load_s':'加载及准备（秒）',
    'members':'层实例数','op_calls':'层内调用数','dispatch':'逐调用采集','forwards':'正式前向次数',
    'warmup':'预热次数','repeats':'进程内重复次数','profiler':'设备采集','window_kind':'窗口类型',
    'profile_memory':'官方显存采集','task':'任务类型','max_new_tokens':'生成 token 上限',
    'allocations':'分配次数','releases':'释放次数','released_bytes':'观测释放量',
    'release_not_observed':'未观测到释放的记录','record':'观测记录','identity':'证据标识',
    'name':'名称', 'scope':'层 / 范围', 'boundary':'数值边界', 'sample':'样本',
    'samples':'样本数 / 标识', 'dev':'设备', 'rank':'rank', 'sid':'调用签名',
    'calls':'调用次数', 'count':'记录数', 'events':'设备记录数', 'category':'事件类别',
    'host_ns':'主机调用耗时', 'device_work_ns':'设备工作量', 'inclusive_ns':'主机含子层耗时',
    'exclusive_ns':'主机独占耗时', 'mean_ns':'平均耗时', 'min_ns':'最短耗时', 'max_ns':'最长耗时',
    'median_ns':'中位耗时', 'p90_ns':'p90', 'p95_ns':'p95', 'device_events':'设备事件数',
    'host_duration_ns':'采集中的推进耗时', 'window_ns':'窗口长度', 'blank_ns':'空白时间',
    'all_activity_union_ns':'全部活动并集', 'compute_union_ns':'计算活动并集',
    'comm_union_ns':'通信活动并集', 'memcpy_union_ns':'搬运活动并集', 'wait_union_ns':'等待活动并集',
    'association_label':'项目调用关联', 'link_status':'关联状态', 'native_owner':'原生调用归属',
    'route_decision':'实际路由', 'expected_route':'预期实现', 'impl':'实现', 'reason':'原因',
    'reasons':'原因', 'non_hit_calls':'非命中次数', 'non_hit_ratio':'非命中比例',
    'forward_non_hit_calls':'前向内非命中次数', 'host_share':'主机耗时占比',
    'impls':'涉及实现', 'sids':'涉及签名', 'scopes':'涉及层', 'occurrences':'出现次数',
    'kind':'类别', 'phase':'阶段', 'generation_step':'生成步', 'step':'生成步',
    'request_id':'请求', 'complete':'请求完成', 'steps':'生成步数', 'tokens':'生成 token',
    'stop_reasons':'停止原因', 'mode':'运行模式', 'kv_length_before':'推进前 KV 长度',
    'input_shape':'输入形状', 'shape':'形状', 'dtype':'数值类型', 'strides':'步长',
    'backend':'后端', 'ver':'版本', 'total_ns':'总耗时', 'cache_hit_ratio':'缓存命中率',
    'field':'条件', 'ref':'参考侧', 'cmp':'比较侧', 'equal':'是否相同',
    'requirement':'比较要求', 'declared':'已声明变化', 'wording':'首差异判定',
    'max_abs':'最大绝对误差', 'max_abs_ulp':'该位置 ULP 距离', 'relative_l2':'相对 L2 误差',
    'cosine_similarity':'余弦相似度', 'digest_equal':'数值摘要相同', 'differing_fraction':'不同元素占比',
    'valid_elements':'有效配对元素', 'ulp_histogram':'ULP 分布', 'ulp_max':'最大 ULP 距离',
    'input_status':'输入来源', 'call_config_status':'调用配置', 'layout_differs':'布局不同',
    'error_in':'输入相对 L2', 'error_out':'输出相对 L2', 'amplification':'误差放大倍数',
    'introduced':'同输入引入差异', 'level':'数值证据', 'levels':'数值证据',
    'side':'比较侧', 'matched_rows':'匹配记录数', 'route_agrees':'实际与预期路由一致',
    'call_key':'逻辑调用', 'first_token_divergence':'首次不同 token 的步', 'compare_steps':'可比步数',
    'ref_steps':'参考步数', 'cmp_steps':'比较步数', 'lengths_equal':'长度相同',
    'group':'通信域', 'group_seq':'调用序号', 'op':'通信操作', 'role':'通信用途',
    'device_sum_ns':'设备表示时长和', 'device_union_ns':'设备活动并集',
    'device_linked_calls':'已关联逻辑调用', 'device_unlinked_calls':'未关联逻辑调用',
    'logical_input_bytes':'逻辑输入量', 'logical_output_bytes':'逻辑输出量',
    'cross_rank_min_observed_ns':'跨 rank 最小观测占用',
    'cross_rank_max_observed_ns':'跨 rank 最大观测占用', 'cross_rank_difference_ns':'跨 rank 观测差值',
    'rank_device_union_ns':'各 rank 活动并集', 'device_link_status':'设备关联',
    'metric':'指标', 'entity':'观测对象', 'unit':'单位', 'component':'来源组件',
    'peak_bytes':'峰值容量', 'allocated_bytes':'已分配', 'reserved_bytes':'已保留',
    'active_bytes':'活跃容量', 'device_used_bytes':'设备已用容量', 'bytes':'数据量',
    'peak_source_time_ns':'峰值来源时刻', 'source_time_ns':'来源时刻',
    'minimum':'最小观测值', 'maximum':'最大观测值', 'mean':'均值', 'p95':'p95',
    'valid':'有效样本', 'missing':'缺失样本', 'coverage':'覆盖信息',
    'coverage_ratio':'有效覆盖率', 'coverage_status':'覆盖状态', 'status':'状态',
    'joules':'能耗', 'power_domain':'功率域', 'value':'观测值', 'start_ns':'起点', 'end_ns':'终点',
    'reference_rms':'参考 RMS', 'normalized_rmse':'归一化 RMSE',
    'absolute_error_quantiles':'绝对误差分位数', 'near_zero_reference':'参考近零处误差',
    'element_weighted_mse':'元素加权 MSE', 'sample_equal_weight_mse':'样本等权 MSE',
    'repeat_cv':'独立重复均值 CV', 'independent_repeats':'独立重复数',
    'observations':'有效观测数', 'spike_count':'尖峰次数', 'timing_scope':'计时边界',
    'generated_tokens_per_second':'生成 token/s', 'samples_per_second':'样本/s',
    'instrumentation':'采集模式', 'rule':'异常类型', 'state':'事件状态',
    'task_relative_ns':'任务内发生时刻', 'first_detected_relative_ns':'任务内首次检出',
    'run_id':'运行标识', 'threshold_ns':'观测阈值', 'detail':'观测详情',
    'registered_execution':'有效执行观测', 'source_gap':'来源缺口', 'probe_gap':'探针缺口',
    'load_or_warmup_excluded':'加载 / 预热观测', 'no_active_interval':'无活动区间观测',
    'baseline':'基准值', 'candidate':'比较值', 'delta':'差值', 'relative_change':'观测变化',
    'comparison_status':'可比性', 'source':'统计来源', 'evidence':'证据', 'notes':'说明',
    'forward':'前向序号', 'root_corr':'前向关联', 'selected':'进入前向摘要',
    'compile_count':'已归属编译次数', 'compile_ns':'已归属编译耗时', 'phase_source':'阶段依据',
}

PANEL_LABELS = {'group.counts': {'forwards':'前向总次数（含预热）'},
                'group.execution': {'status':'执行状态','returncode':'worker 自报退出码'},
                'oracle.replay': {'valid':'输入与执行有效'}, 'monitor.series': {'valid':'有效样本'},
                'training': {'step':'训练步'}, 'training.phases': {'step':'训练步','phase':'训练阶段'}}
REASON_TEXT = {'reference_missing':'缺少参考记录','no_tensor_ops':'未记录张量算子',
               'diagnostic monitor unavailable: []':'诊断监控未提供可用观测'}
VALUES = {
    'failed':'失败','running':'运行中','missing':'缺少 worker 回执，无法确认执行状态',
    'unreadable':'worker 回执无法读取，无法确认执行状态',
    **REASON_TEXT,
    'forward':'同步前向','request':'完整请求','group_request':'组请求','request_decode':'整次请求 decode',
    'source_window':'完整来源窗口','runner_synchronized':'runner 同步边界',
    'identity_mismatch':'身份不同','exact_signature_missing':'未找到精确签名',
    'performance_read_partial':'独立性能记录只读到一部分','source_gap':'来源读取缺口',
    'missing_rank_device_link':'缺少某个 rank 的设备关联','identity_not_comparable':'身份或窗口不满足比较条件',
    'root':'根层','recorded_parent':'父层关系已记录','parent_unavailable':'父层关系未记录',
    '0_to_nonzero':'误差由零变为非零','ratio':'已有误差的倍数变化',
    'incomplete_energy_coverage':'能耗覆盖未达到完整窗口','token_denominator_unavailable':'缺少 token 分母',
    'power_domain_includes_other_chips':'功率域包含任务之外的芯片','preexisting_usage':'存在启动前占用',
    'framework_unobserved':'未观测到分配器数据','no_pre_sample_or_baseline':'缺少启动前样本或空闲基线',
    'not_comparable_or_missing':'口径不满足或缺少指标',
    'sum_of_valid_device_record_durations_including_overlap':'有效设备记录时长和，包含重叠',
    'all_recorded_device_events':'全部已记录设备事件',
    'zero_baseline':'基准为零，只给差值','missing_pair_or_different_evidence_population':'缺少对照或有效元素范围不同',
    'zero_reference_scale':'参考尺度为零','finite_pairs_only':'仅有限有效元素',
    'insufficient_independent_repeats':'独立重复不足，未启用 CV 筛查','insufficient_observations':'有效观测不足，未启用尖峰筛查',
    'unstable':'重复波动超过阈值','stable':'重复波动未超阈值','clear':'未发现超过阈值的尖峰','triggered':'存在超过阈值的尖峰',
    'on':'开启','off':'关闭','flaggems_off':'未启用 FlagGems',
    'ok':'可用', 'available':'可读取', 'complete':'完成', 'completed':'完成',
    'partial':'部分可用', 'unavailable':'未提供', 'unknown':'未确定', 'not_recorded':'未记录',
    'not_comparable':'条件不满足比较', 'comparable':'条件可比', 'declared_comparison':'已声明对照条件',
    'linked':'已关联项目调用', 'unlinked':'未关联项目调用', 'ambiguous':'关联不唯一',
    'hit':'命中', 'fallback':'回退', 'excluded':'策略 / 设备排除', 'error':'执行错误',
    'not_registered':'实现未注册', 'device_unsupported':'设备不支持', 'policy_excluded':'策略排除',
    'compute':'计算', 'memcpy':'数据搬运', 'comm':'通信', 'wait':'等待',
    'warmup':'预热', 'measure':'正式测量', 'measurement':'正式测量', 'load':'加载',
    'prefill':'prefill', 'decode':'decode', 'model':'模型通信', 'control':'控制通信',
    'full':'完整张量', 'sampled':'抽样位置', 'aggregate':'摘要', 'native':'原生框架归属',
    'each':'逐调用', 'summary_only':'仅汇总记录', 'none':'无', 'attachment':'已启用采集',
    'insufficient_resolution':'观测分辨率不足', 'source_unknown':'输入生产者未确定',
    'same':'相同', 'different':'不同', 'equal':'相同', 'unequal':'不同',
    'must_equal':'必须一致', 'declarable':'可声明对照因素',
    'oom.device':'设备 OOM', 'hang.heartbeat':'目标心跳超时',
    'hang.no_progress':'执行进度停滞', 'comm.timeout':'通信调用超时', 'proc.exit':'任务异常退出',
    'persisted_device_events':'全采集窗口内已保存的设备事件',
    'official_capture':'官方采集窗口', 'project_request':'完整请求',
    'needs_review':'存在待复查波动', 'no_trigger':'本次未触发筛查条件',
    'insufficient_evidence':'重复 / 观测数量不足', 'invalid_evidence':'存在无效记录',
}

PANELS = {
    'training':('训练步汇总',('step','owner','host_duration_ns','phase_counts','phase_host_ns','status','issues')),
    'oracle.samples':('逐样本误差与尺度',('scope','boundary','sample','level','max_abs','absolute_error_quantiles','reference_rms','normalized_rmse','near_zero_reference_fraction')),
    'perf.op':('算子热点', ('name','calls','host_ns','device_work_ns','source','scope','sid')),
    'perf.layer':('逐层耗时', ('name','members','inclusive_ns','exclusive_ns','device_work_ns','op_calls')),
    'perf.device':('设备 kernel 与活动热点', ('name','dev','category','events','missing_durations','device_work_ns','native_owner','association_label','sid')),
    'perf.gap':('请求与设备窗口', ('kind','request_id','step','dev','window_ns','all_activity_union_ns','blank_ns','coverage_status')),
    'route.coverage':('组件路由与回退', ('name','sid','route_decision','reason','calls','host_ns','impl','scopes')),
    'compile.health':('编译观测', ('backend','ver','count','total_ns','median_ns','cache_hit_ratio')),
    'oracle.identity_diff':('比较条件', ('field','ref','cmp','equal','requirement','declared')),
    'oracle.first_divergence':('首个已观测差异', ('wording','scope','boundary','sample','max_abs','max_abs_ulp','differing_fraction')),
    'oracle.layer':('逐层数值差异', ('scope','boundary','samples','digest_equal','relative_l2','cosine_similarity','max_abs','differing_fraction','ulp_histogram')),
    'oracle.op_sensitivity':('算子输入与输出差异', ('boundary','input_status','call_config_status','error_in','error_out','amplification','introduced','expected_route')),
    'oracle.perf_links':('同签名的独立性能证据', ('side','call_key','sid','matched_rows','route_agrees','reasons')),
    'oracle.replay':('同输入回放', ('call_key','input_side','valid','self_reproduction','metrics','routes','reasons')),
    'oracle.generation':('生成轨迹差异', ('sample','first_token_divergence','compare_steps','ref_steps','cmp_steps','lengths_equal')),
    'memory':('显存峰值', ('dev','component','metric','peak_bytes','peak_source_time_ns')),
    'resource':('资源观测', ('entity','metric','unit','mean','maximum','p95','coverage_ratio','status')),
    'generation':('生成请求', ('request_id','samples','mode','steps','complete','stop_reasons')),
    'communication':('通信调用', ('group','op','backend','role','calls','device_linked_calls','device_unlinked_calls','device_sum_ns')),
    'diagnosis':('业务故障诊断与退出', ('run_id','rank','rule','state','task_relative_ns','first_detected_relative_ns','trigger','location','next_step')),
}


def number(value):
    if not math.isfinite(value):
        return '无有限值'
    if value == 0:
        return '0'
    return format(value, '.6g')


# Shared by static HTML, the interactive tables and Markdown.
MISSING_TEXT = {
    'report':'缺少该 rank 的子报告，无法下钻到采集明细',
    'scope':'缺少统计范围记录，无法确定该项覆盖哪些调用',
    'source':'缺少统计来源，无法核对该指标的采集口径',
    'dev':'缺少设备编号，无法将该项归属到具体设备',
    'status':'缺少状态记录，无法确认该项结果',
    'process_returncode':'缺少进程退出码，无法核实进程退出结果',
    'returncode':'缺少 worker 返回码，无法核实 worker 的执行结果',
    'repeat_cv':'缺少独立进程均值的变异系数，无法量化跨进程时延波动',
    'device_sum_ns':'缺少已关联通信调用的设备时长，无法计算通信设备工作量',
    'device_union_ns':'缺少调用对应的设备区间，无法计算设备占用时长',
    'cross_rank_difference_ns':'缺少两侧可比的设备占用时长，无法计算 rank 间差值',
    'device_work_ns':'缺少可归属的设备时长，无法计算该项设备工作量',
    'host_ns':'缺少主机调用计时，无法判断主机侧耗时',
    'native_owner':'缺少框架归属记录，无法确定发起该活动的框架调用',
    'sid':'缺少项目调用签名，无法按签名定位具体调用',
    'generated_tokens_per_second':'缺少可用于计算吞吐的生成 token 数或耗时，无法给出生成速率',
    'generation_step':'缺少步序号，无法定位到具体生成步',
    'median_ns':'缺少有效时延观测，无法计算中位数',
    'p95_ns':'缺少有效时延观测，无法计算 p95',
    'task_relative_ns':'缺少事件时刻与任务起点的时钟对应关系，无法定位到任务时间线',
    'first_detected_relative_ns':'缺少可映射的检出时刻，无法确定任务启动后多久检出',
    'last_equal_boundary':'此前未观测到一致的边界，无法进一步缩小差异出现的范围',
    'p95':'缺少足够的有效采样，无法判断高分位水平',
    'observed_bytes':'缺少显存观测值，无法核算该部分用量',
    'preexisting_usage':'缺少启动前占用依据，无法区分已有显存和本次新增用量',
    'baseline':'缺少基准侧指标值，无法计算两侧变化',
    'candidate':'缺少候选侧指标值，无法计算两侧变化',
}


def display(value, key=''):
    if value is None:
        return MISSING_TEXT.get(key, '缺少'+LABELS.get(key,key or '该字段')+'记录')
    if isinstance(value, bool):
        return '是' if value else '否'
    if isinstance(value, (int, float)):
        if key.endswith('_ns'):
            if abs(value) >= 1e9: return number(value/1e9)+' s'
            if abs(value) >= 1e6: return number(value/1e6)+' ms'
            if abs(value) >= 1e3: return number(value/1e3)+' μs'
            return number(value)+' ns'
        if key.endswith('_bytes') or key == 'bytes':
            for scale, unit in ((1<<30,'GiB'),(1<<20,'MiB'),(1<<10,'KiB')):
                if abs(value) >= scale: return number(value/scale)+' '+unit
            return number(value)+' B'
        if key.endswith(('_ratio','_share','_fraction')) or key in ('relative_change','repeat_cv'):
            return number(value*100)+'%'
        return number(value)
    if isinstance(value, str):
        if key=='instrumentation' and value=='none':return '无插桩'
        if key in ('field','metric'):
            return ' / '.join(LABELS.get(part,VALUES.get(part,part)) for part in value.split('.'))
        return VALUES.get(value, value)
    if isinstance(value, dict):
        if 'value' in value:
            text = display(value['value'], key)
            for name in ('sample','index','ulp'):
                if value.get(name) is not None:
                    text += ' · '+{'sample':'样本','index':'位置','ulp':'ULP'}[name]+' '+str(value[name])
            return text
        return '；'.join(LABELS.get(k,k)+'：'+display(v,k) for k,v in value.items())
    if isinstance(value, (list, tuple)):
        if 'shape' in key and all(isinstance(v,int) for v in value):
            return ' × '.join(map(str,value))
        return '、'.join(display(v) for v in value) if value else '无'
    return str(value)


def cell(row, key):
    if key in ('max_abs','relative_l2','cosine_similarity') and key not in row:
        return row.get('worst',{}).get(key)
    return row.get(key)


def prepare_fields(ident, row, forward_only):
    """Project applicable fields into the reader view; preserve machine results."""
    scope = row.get('scope')
    if ident.startswith('performance.') or ident == 'comparison.metrics':
        if scope == 'forward':
            for key in ('generation_step','generated_tokens','generated_tokens_per_second'):
                row.pop(key, None)
        elif scope in ('request','request_decode','group_request') and row.get('generation_step') is None:
            row.pop('generation_step', None)
        if ident == 'performance.spikes' and row.get('phase') == 'measure' and row.get('generation_step') is None:
            row.pop('generation_step', None)
    if ident == 'group.counts' and (row.get('count_basis') == 'root_module_spans' or
                                  not row.get('count_basis') and forward_only):
        row.pop('steps', None)
    if ident == 'resource.energy' and forward_only:
        for key in ('generated_tokens','joules_per_token','token_reason'):
            row.pop(key, None)
    elif ident == 'resource.energy' and row.get('token_reason') and row.get('joules_per_token') is None:
        row['_field_notes'] = dict(joules_per_token=display(row['token_reason'])+'，无法计算本任务的每生成 token 能耗')
    if ident in ('resource','resource.stages') and row.get('p95') is None and isinstance(row.get('samples'),int) and row['samples']<20:
        row['_field_notes'] = dict(p95=f"有效采样 {row['samples']} 个，少于 20 个，无法估计 p95")
    if ident == 'monitor.findings' and row.get('entity') == 'node' and not row.get('run_id'):
        row['_field_notes'] = {}
        for key in ('rank','run_id','task_relative_ns'):
            if row.get(key) is None:
                row.pop(key, None)
                row['_field_notes'][key] = '节点级观测'
    if ident == 'oracle.identity_diff':
        field = display(row.get('field'),'field')
        if row.get('equal') is None:
            row['_field_notes'] = {key:f'缺少{side}的{field}记录，无法核对该条件'
                                   for key,side in (('ref','参考侧'),('cmp','比较侧')) if row.get(key) is None}
            row['_field_notes']['equal'] = '比较条件记录不完整，无法判定两侧是否一致'
        elif row.get('field')=='include' and row.get('equal') and row.get('ref') is None and row.get('cmp') is None:
            row['_field_notes'] = dict(ref='未指定算子清单',cmp='未指定算子清单')
    if ident in ('perf.gap','perf.gap.streams') and row.get('kind') in ('official_capture','project_request') and row.get('step') is None:
        row.pop('step', None)
    if (ident == 'group.control_calls' and row.get('backend') == 'gloo' and row.get('op') == 'barrier'
            and row.get('host_duration_ns') is not None and row.get('device_union_ns') is None):
        row['_field_notes'] = dict(device_union_ns='此 Gloo 屏障仅记录主机耗时')
    if ident == 'comparison.metrics' and row.get('delta') is None:
        missing = '、'.join(label for key,label in (('baseline','基准侧'),('candidate','候选侧')) if row.get(key) is None)
        reason = ('缺少'+missing+'指标值' if missing else '两侧指标口径未通过可比性核对')
        row['_field_notes'] = dict(delta=reason+'，无法计算差值',relative_change=reason+'，无法计算变化率')
    elif ident == 'comparison.metrics' and row.get('reason') == 'zero_baseline':
        row['_field_notes'] = dict(relative_change='基准值为 0，无法计算相对变化率')
    if ident == 'performance.baseline' and row.get('repeat_cv') is None:
        count = row.get('independent_repeats')
        if count == 0 and row.get('observations'):
            row['_field_notes'] = dict(repeat_cv='有效观测缺少可区分独立进程的身份，无法量化跨进程时延波动')
        elif count in (0,1):
            row['_field_notes'] = dict(repeat_cv=f'仅能识别 {count} 次独立进程测量，至少需要 2 次才能量化跨进程时延波动')


def field_text(row, key):
    value = cell(row,key)
    if value is None:
        note = row.get('_field_notes',{}).get(key)
        if note:
            return note
        if key == 'generation_step' and key not in row:
            return '按完整计时范围汇总'
    unit = key
    if row.get('unit')=='ns' and key in ('baseline','candidate','delta') and value is not None:
        unit = 'duration_ns'
    return display(value,unit)
