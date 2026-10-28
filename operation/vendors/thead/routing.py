"""First/cached PPU routing requires correlated semantic device computations."""


class Observer:
    def __init__(self, task, backward):
        self.case, self.dtype = task['case'], task['dtype']
        self.oplib, self.backward = task['oplib'], backward
        self.dispatch = {}
        self.windows = {}

    def begin(self, name):
        self.current = name
        self.windows[name] = {'complete': False}

    def event(self, frame, event, arg):
        pass

    def end(self, elapsed):
        self.windows[self.current].update(complete=True, elapsed_s=elapsed)

    def record(self):
        return {'protocol': 'ppu-routing-v1', 'case': self.case, 'dtype': self.dtype,
                'oplib': self.oplib, 'backward': self.backward,
                'dispatch': self.dispatch, 'windows': self.windows}


def route(probe, log):
    data = probe.get('vendor_route', {})
    devices = probe.get('output_devices', [])
    on_device = bool(devices) and all(d == probe.get('device') and d.startswith('cuda:') for d in devices)
    complete = (log.count('OPERATION_TARGET_BEGIN') == log.count('OPERATION_TARGET_END') == 1
                and log.index('OPERATION_TARGET_BEGIN') < log.index('OPERATION_TARGET_END'))
    rows = [data.get('dispatch', {}).get(name, {}) for name in ('first', 'cached')]
    fallback = any(row.get('target_cpu_fallback') is True for row in rows)
    calls = probe.get('calls', [])
    gems = [row for row in calls if '/flag_gems/' in row.get('path', '')
            and ('/ops/' in row['path'] or '/fused/' in row['path'])]
    registration = probe.get('registration', {})
    path_consistent = (registration.get('oplib') == data.get('oplib')
                       and registration.get('public_enable_called') is (data.get('oplib') == 'flaggems')
                       and (bool(gems) if data.get('oplib') == 'flaggems' else not gems))
    proven = (data.get('protocol') == 'ppu-routing-v1' and complete and on_device
              and all(data.get('windows', {}).get(name, {}).get('complete') for name in ('first', 'cached'))
              and all(row.get('protocol') == 'ppu-dispatch-correlation-v1' and row.get('status') == 'passed' for row in rows)
              and path_consistent)
    return {'status': 'failed' if fallback or devices and not on_device else 'passed' if proven else 'partial',
            'output_devices': devices, 'target_interval_complete': complete,
            'target_cpu_fallback': fallback if rows and all(rows) else None,
            'dispatch': data.get('dispatch', {}), 'cached_launch_proven': proven,
            'path_registration_consistent': path_consistent, 'implementation': gems,
            'reason': None if proven else 'first/cached semantic computation lacks complete host/device correlation',
            'boundary': 'ATen computation requires matching host launch, ac2g flow and device kernel; output placement is not proof; opaque internal backend fallback is not fully instrumented'}
