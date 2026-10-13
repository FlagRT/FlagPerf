"""Generation views from sealed steps; works after tensor compaction, without a device."""
from runtime.common import read_json


def number(value, scale=1):
    return '未记录/未定义' if value is None else f'{value/scale:.6g}'


def append_report(root, result):
    path=root/'prepared/workload.json'
    if not path.is_file(): return
    workload=read_json(path)
    if workload.get('execution_kind')!='autoregressive': return
    maximum=workload['generation']['max_new_tokens']
    lines=['', '## 生成执行与比较边界', '',
           f'每次请求从空 KV cache 开始，greedy 生成。精度遇 EOS 或 {maximum} 个新 token 停止；'
           f'性能固定 {maximum} 个新 token，其中 prefill 产生第一个 token，随后执行 {maximum-1} 次 decode。',
           '模型级数值边界：'+workload.get('numeric_boundary_description',
               ', '.join(s['name'] for s in workload['boundaries'] if s['level']=='model'))+'。'
           '选定层级边界及各阶段有效区域以封存 workload 为准。',
           '每对参照与被测输出按样本独立对齐，数值比较包含首次选 token 分歧的那一步，随后停止数值配对。'
           '剩余生成步仍执行有限值检查；不同比较可能有不同的数值范围。', '']
    comparisons=[]
    if result.get('comparison') and isinstance(result['comparison'],str) and (root/result['comparison']).is_file():
        value=read_json(root/result['comparison'])
        comparisons.extend(('同设备 off→on',rank,row) for rank,row in value.get('ranks',{'single':value}).items() if row.get('generation'))
    if (root/'oracle-comparison.json').is_file():
        value=read_json(root/'oracle-comparison.json')
        comparisons.extend(('A100→'+side,rank,row) for side,target in value.get('candidates',{}).items()
                           for rank,row in target.get('ranks',{}).items() if row.get('generation'))
    if comparisons:
        lines+=['| 比较 | rank | 样本 ID | 首次分歧 step | 数值配对步数 | 参照/被测生成 token | 参照/被测停止原因 |',
                '|---|---|---|---:|---:|---:|---|']
        for label,rank,value in comparisons:
            for row in value['generation']['trajectories']:
                lines.append(f"| {label} | {rank} | {row['sample_id']} | {row['first_divergence_step'] if row['first_divergence_step'] is not None else '无'} | "
                             f"{row['compared_steps']} | {row['reference_generated_tokens']}/{row['candidate_generated_tokens']} | {row['reference_stop']}/{row['candidate_stop']} |")
        lines+=['', '| 比较 | rank | 阶段 | hidden MSE | MAE | 最大绝对误差 | 最低样本余弦 | 配对元素 |',
                '|---|---|---|---:|---:|---:|---:|---:|']
        for label,rank,value in comparisons:
            for phase,state in value['generation']['phase_comparisons'].items():
                m=state['aggregate'].get('hidden')
                if m:
                    lines.append(f"| {label} | {rank} | {phase} | {number(m['mse'])} | {number(m['mae'])} | {number(m['max_abs'])} | "
                                 f"{number(m['sample_cosine_min'])} | {m['valid_elements']}/{m['total_elements']} |")
        lines+=['', '逐步误差、完整停止原因及分歧 token 保存在 comparison.json / oracle-comparison.json；step 从 0 开始。', '']
    if result.get('command') in ('performance','performance-baseline'):
        lines+=['### 生成性能', '',
                '总体计时包含同步后的完整生成主机窗口、greedy 选择及 token 控制回传，输入在计时前驻留设备。'
                '首 token 指标是同步 prefill 窗口，不含排队、分词、加载、输入传输或网络。生成 token/s 使用完整生成窗口，输入 token/s 使用 prefill 窗口。', '',
                '| 路径 | 完整生成均值 ms | 首 token/prefill 均值 ms | 输入 token/s | decode 均值/p50/p90/p99 ms | 生成 token/s |',
                '|---|---:|---:|---:|---|---:|']
        for side,value in result.get('summary',{}).items():
            if 'generation' not in value: continue
            g=value['generation']
            lines.append(f"| {side} | {number(value.get('latency_mean_ms'))} | {number(g.get('prefill_mean_ms'))} | {number(g.get('input_tokens_per_prefill_second'))} | "
                         f"{' / '.join(number(g.get(k)) for k in ('decode_mean_ms','decode_p50_ms','decode_p90_ms','decode_p99_ms'))} | {number(g.get('generated_tokens_per_second'))} |")
        lines+=['', '### KV cache 与独立阶段显存', '',
                '下表来自每个 worker 首批封存输入的额外独立重放；KV 字节按实际 storage 去重，含 batch 的 padding 位置。allocator 峰值还包含权重、临时张量和工作区，'
                'KV 已在 allocator 中，不能相加。rank 的本地字节和峰值分别展示，缓存不跨请求复用。', '',
                '| 路径/repeat/rank | 阶段/step | KV 长度 | KV storage MiB | allocated/reserved 峰值 MiB | allocated 峰值增量 MiB |',
                '|---|---|---:|---:|---|---:|']
        for side,repeats in result.get('paths',{}).items():
            for repeat,state in repeats.items():
                for rank,row in state.get('ranks',{'single':state}).items():
                    cache=row.get('generation_cache',{})
                    for stage in ('prefill','final_decode'):
                        obs=cache.get(stage)
                        if not obs: continue
                        peak=obs.get('allocator_peak',{})
                        lines.append(f"| {side}/{repeat}/{rank} | {obs['phase']}/{obs['generation_step']} | {obs['cache'].get('sequence_length')} | "
                                     f"{number(obs['cache'].get('storage_bytes'),2**20)} | {number(peak.get('allocated_bytes'),2**20)}/{number(peak.get('reserved_bytes'),2**20)} | "
                                     f"{number(obs.get('allocator_peak_increment',{}).get('allocated_bytes'),2**20)} |")
        lines+=['', '逐层结果按 phase、step、KV 长度分组；离线报告可用 --phases 与 --generation-steps 筛选，不重新执行模型。', '']
        if (root/'performance-comparison.json').is_file():
            comparison=read_json(root/'performance-comparison.json')
            lines+=['### A100 生成阶段性能', '', '时延倍率为被测/A100，越低越快；不同服务器的独立重复不作配对检验。', '',
                    '| 路径 | 阶段 | A100 均值 ms | 被测均值 ms | 被测/A100 |', '|---|---|---:|---:|---:|']
            for side,value in comparison.get('paths',{}).items():
                for phase,row in value.get('generation',{}).items():
                    lines.append(f"| {side} | {phase} | {number(row['a100']['mean_ms'])} | {number(row['candidate']['mean_ms'])} | {number(row['candidate_over_a100']['mean_ms'])} |")
            lines+=['', '| 路径/rank | 阶段 | 字段 | A100 MiB | 被测 MiB | 被测/A100 |', '|---|---|---|---:|---:|---:|']
            for side,value in comparison.get('paths',{}).items():
                for row in value.get('generation_cache',[]):
                    lines.append(f"| {side}/{row['rank']} | {row['stage']} | {row['field']} | {number(row['a100'],2**20)} | {number(row['candidate'],2**20)} | {number(row['candidate_over_a100'])} |")
    with (root/'report.md').open('a') as stream:
        stream.write('\n'.join(lines)+'\n')
