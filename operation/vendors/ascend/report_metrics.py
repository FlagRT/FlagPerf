# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Ascend profiling presentation vocabulary; never imports a device runtime."""

GROUP_DESCRIPTIONS = {
    'ArithmeticUtilization': '计算活动：观察 Cube/Vector 浮点运算量及不同数据类型指令所占周期',
    'PipeUtilization': '流水活动：观察矩阵、向量计算及 MTE 数据搬运指令所占周期',
    'Memory': '数据搬运：观察主存、L1/L2 通道在任务总周期内的平均读写带宽',
    'MemoryL0': 'Cube 片上缓存：观察 L0A/L0B 输入缓存与 L0C 累加/结果缓存的读写带宽',
    'MemoryUB': 'Vector 片上缓存：观察 Scalar/Vector 对统一缓冲区 UB 的读写带宽',
    'ResourceConflictRatio': '资源冲突：观察 Vector 指令因 UB bank、bank group 或执行单元竞争而阻塞的周期占比',
}


# Deliberate allowlist: only reviewed fields are promoted into the report body.
RATIOS = {
    'ArithmeticUtilization': {
        'aic_mac_fp16_ratio': (
            'FP16 矩阵乘加（MAC/Cube）指令执行周期占 AI Cube Core 总周期的比例；'
            '反映本次任务用于 FP16 矩阵计算的周期份额，不等同于理论峰值算力利用率。'
        ),
        'aic_mac_int8_ratio': (
            'INT8 矩阵乘加（MAC/Cube）指令执行周期占 AI Cube Core 总周期的比例；'
            '反映本次任务用于 INT8 矩阵计算的周期份额，不等同于理论峰值算力利用率。'
        ),
        'aiv_vec_fp16_ratio': (
            'FP16 Vector 指令执行周期占 AI Vector Core 总周期的比例；'
            '表示 FP16 向量计算在本次任务中的时间份额，不直接代表向量峰值吞吐。'
        ),
        'aiv_vec_fp32_ratio': (
            'FP32 Vector 指令执行周期占 AI Vector Core 总周期的比例；'
            '表示 FP32 向量计算在本次任务中的时间份额，不直接代表向量峰值吞吐。'
        ),
    },
    'PipeUtilization': {
        'aic_mac_ratio': (
            '矩阵乘加（MAC/Cube）指令执行周期占 AI Cube Core 总周期的比例；'
            '用于观察矩阵计算流水所占时间，判断瓶颈时还需与 MTE 周期占比和设备执行时间对照。'
        ),
        'aic_mte1_ratio': (
            'MTE1 执行 L1→L0A/L0B 片上搬运的周期占 AI Cube Core 总周期的比例；'
            '用于观察 Cube 输入数据从 L1 送入两路输入缓存所占的时间。'
        ),
        'aic_mte2_ratio': (
            'MTE2 执行主存（GM）→Cube 侧存储层级搬运的周期占 AI Cube Core 总周期的比例；'
            '用于观察矩阵计算输入加载所占的时间。'
        ),
        'aiv_mte2_ratio': (
            'MTE2 执行主存（GM）→Vector/UB 侧搬运的周期占 AI Vector Core 总周期的比例；'
            '用于观察向量计算输入加载所占的时间。'
        ),
        'aiv_vec_ratio': (
            'Vector 指令执行周期占 AI Vector Core 总周期的比例；'
            '用于观察向量计算流水所占时间，不能单独解释为算子整体利用率。'
        ),
    },
    'ResourceConflictRatio': {
        'aiv_vec_bank_cflt_ratio': (
            'Vector 指令因多个访问竞争同一 UB bank 而阻塞的周期占全部指令周期的比例；'
            '数值升高通常提示操作数读写地址布局可能造成 bank 冲突。'
        ),
        'aiv_vec_bankgroup_cflt_ratio': (
            'Vector 指令因 UB bank group 冲突而阻塞的周期占全部指令周期的比例；'
            '数值升高通常提示 Vector 指令的 block stride 需要检查。'
        ),
        'aiv_vec_resc_cflt_ratio': (
            'Vector 指令因执行单元资源竞争而阻塞的周期占全部指令周期的比例；'
            '数值升高说明指令可能持续投递到忙碌单元，需要检查多计算单元的并发调度。'
        ),
    },
}

FLOPS = {
    'aic_cube_fops': (
        'Cube 矩阵浮点运算次数，表示设备实际执行的矩阵计算量；它是工作量而非速度，'
        '需结合设备执行时间计算吞吐，也不等同于上层公式估算的逻辑运算量。'
    ),
    'aiv_vector_fops': (
        'Vector 浮点运算次数，表示设备实际执行的向量计算量；它是工作量而非速度，'
        '需结合设备执行时间计算吞吐，也不等同于上层公式估算的逻辑运算量。'
    ),
}

BANDWIDTH_PARTS = {
    'Memory': {
        'main_mem_read_bw': '从主存（GM）读取数据到计算核心侧',
        'main_mem_write_bw': '从计算核心侧向主存（GM）写回数据',
        'l1_read_bw': 'L1 从其他单元读取数据',
        'l1_write_bw': 'L1 向其他单元写出数据',
        'l2_read_bw': '经 L2 缓存读取数据',
        'l2_write_bw': '经 L2 缓存写出数据',
    },
    'MemoryL0': {
        'l0a_read_bw': 'Cube 输入缓存 L0A 从其他片上单元读取数据',
        'l0a_write_bw': 'Cube 输入缓存 L0A 向其他片上单元写出数据',
        'l0b_read_bw': 'Cube 输入缓存 L0B 从其他片上单元读取数据',
        'l0b_write_bw': 'Cube 输入缓存 L0B 向其他片上单元写出数据',
        'l0c_read_bw': 'Vector 从 Cube 累加/结果缓存 L0C 读取数据',
        'l0c_write_bw': 'Vector 向 Cube 累加/结果缓存 L0C 写入数据',
        'l0c_read_bw_cube': 'Cube 从累加/结果缓存 L0C 读取数据',
        'l0c_write_bw_cube': 'Cube 向累加/结果缓存 L0C 写入数据',
    },
    'MemoryUB': {
        'ub_read_bw_scalar': 'Scalar 从统一缓冲区 UB 读取数据',
        'ub_write_bw_scalar': 'Scalar 向统一缓冲区 UB 写入数据',
        'ub_read_bw_vector': 'Vector 从统一缓冲区 UB 读取数据',
        'ub_write_bw_vector': 'Vector 向统一缓冲区 UB 写入数据',
    },
}


def _metric(explanation, *, scale=1.0, suffix='', key=True):
    return {'explanation': explanation, 'scale': scale, 'suffix': suffix, 'key': key}


def describe_metric(group, field):
    ratio = RATIOS.get(group, {}).get(field)
    if ratio:
        return _metric(ratio, scale=100.0, suffix='%')
    if group == 'ArithmeticUtilization' and field in FLOPS:
        return _metric(FLOPS[field])
    for prefix, core in (('aic_', 'AI Cube Core'), ('aiv_', 'AI Vector Core')):
        if field.startswith(prefix) and field.endswith('(GB/s)'):
            action = BANDWIDTH_PARTS.get(group, {}).get(field[len(prefix):-len('(GB/s)')])
            if action:
                return _metric(
                    f'{core} 侧，{action}时，以任务总周期计算的平均带宽；'
                    '字段名中的 GB/s 表示每秒传输的数据量，需结合设备时间和数据规模判断是否形成带宽瓶颈。'
                )
    return None
