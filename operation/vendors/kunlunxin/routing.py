# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Observe the locked Triton XPU launcher without wrapping backend functions."""
from pathlib import Path
import time

DRIVER = '/triton/backends/xpu/driver.py'
# Reviewed simple forward targets. Compound/backward cases stay partial until
# all their semantic components have an equally specific evidence rule.
REVIEWED = {'abs': {'abs', 'abs_wrapper'}, 'add': {'add', 'add_tensor', 'add_tensor_'}, 'mm': {'mm'}}
# Exact target kernels observed in the locked runtime. A target ancestor alone
# does not distinguish its computation from nested copies or autotune helpers.
# mm remains partial until a successful target launch has been reviewed.
REVIEWED_KERNELS = {'abs': {'abs_func_kernel_rank_1'}, 'add': {'add_func_kernel_rank_1'}}


class Observer:
    def __init__(self, task, backward):
        self.case, self.oplib, self.backward = task['case'], task['oplib'], backward
        self.windows = {};self.window = None;self.pending = {};self.compiling = {}

    def begin(self, name):
        self.window = {'launches':[], 'compilation_calls':0, 'compilation_attempt_s':0, 'complete':False}
        self.windows[name] = self.window

    def event(self, frame, event, arg):
        path = frame.f_code.co_filename
        if path.endswith('/triton/compiler/compiler.py') and frame.f_code.co_name == 'compile':
            if event == 'call':
                self.window['compilation_calls'] += 1;self.compiling[id(frame)] = time.monotonic()
            elif event == 'return' and id(frame) in self.compiling:
                self.window['compilation_attempt_s'] += time.monotonic() - self.compiling.pop(id(frame))
        if not path.endswith(DRIVER) or frame.f_code.co_name != '__call__': return
        obj = frame.f_locals.get('self')
        if type(obj).__name__ != 'XPULauncher' or arg is not getattr(obj, 'launch', None): return
        key = id(frame)
        if event == 'c_call':
            ancestry = [];parent = frame.f_back
            while parent is not None:
                p = parent.f_code.co_filename
                if '/flag_gems/' in p and ('/ops/' in p or '/fused/' in p):
                    ancestry.append({'path':p, 'function':parent.f_code.co_name})
                parent = parent.f_back
            self.pending[key] = {'kernel':str(getattr(obj, '_fn_name', 'unknown')),
                                 'driver':path, 'class':'XPULauncher', 'implementation':ancestry,
                                 'actual_c_return':False}
        elif event in ('c_return','c_exception') and key in self.pending:
            row = self.pending.pop(key);row['actual_c_return'] = event == 'c_return'
            self.window['launches'].append(row)

    def end(self, elapsed):
        self.window.update(complete=not self.pending, elapsed_s=elapsed)

    def record(self):
        return {'protocol':'p800-xpu-launch-v1', 'case':self.case, 'oplib':self.oplib,
                'backward':self.backward, 'windows':self.windows}


def meaningful(launch, case):
    implementation = launch.get('implementation', [])
    target = implementation[0] if implementation else {}
    return (launch.get('actual_c_return') is True and launch.get('class') == 'XPULauncher'
            and launch.get('driver','').endswith(DRIVER)
            and launch.get('kernel') in REVIEWED_KERNELS.get(case, set())
            and Path(target.get('path', '')).stem == case
            and target.get('function') in REVIEWED.get(case, set())
            and '/flag_gems/' in target.get('path', '')
            and '/ops/' in target.get('path', ''))


def route(probe, log):
    data = probe.get('p800_route',{});case = data.get('case')
    devices = probe.get('output_devices',[])
    on_device = bool(devices) and all(d == probe.get('device') and d.startswith('cuda:') for d in devices)
    complete = (log.count('OPERATION_TARGET_BEGIN') == log.count('OPERATION_TARGET_END') == 1
                and log.index('OPERATION_TARGET_BEGIN') < log.index('OPERATION_TARGET_END'))
    windows = data.get('windows',{})
    proven = {name:bool(windows.get(name,{}).get('complete')) and
              any(meaningful(row, case) for row in windows.get(name,{}).get('launches',[]))
              for name in ('first','cached')}
    passed = (complete and on_device and not data.get('backward',True) and data.get('oplib') == 'flaggems'
              and data.get('protocol') == 'p800-xpu-launch-v1' and all(proven.values()))
    return {'status':'failed' if devices and not on_device else 'passed' if passed else 'partial',
            'output_devices':devices, 'target_interval_complete':complete, 'windows':windows,
            'cached_launch_proven':all(proven.values()),
            'target_cpu_fallback':True if devices and all(d == 'cpu' for d in devices) else None,
            'fallback_detection':'CPU target output fails; no universal native fallback detector established',
            'boundary':'reviewed simple forward Gems implementation ancestry + actual XPULauncher C return + synchronization; not a device timeline',
            'reason':None if passed else 'native/compound/backward target dispatch or required launcher evidence is not proven'}
