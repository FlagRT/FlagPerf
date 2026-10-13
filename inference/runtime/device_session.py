"""Normalized device measurements. SDK conventions belong to vendor adapters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    total_memory: object = None


class DeviceSession:
    source = 'vendor device session'
    capabilities = frozenset()

    def require(self, *names):
        missing = set(names)-set(self.capabilities)
        if missing:
            raise ValueError('device session lacks required capabilities: '+', '.join(sorted(missing)))

    def metadata(self):
        return {'source': self.source, 'capabilities': sorted(self.capabilities),
                'unmeasured': {n: 'vendor session does not expose '+n for n in
                               ('memory', 'peak_memory', 'events', 'stream') if n not in self.capabilities}}

    def synchronize(self):
        raise ValueError('device session does not implement synchronization')

    def get_device_name(self, rank=0):
        return self.get_device_properties(rank).name

    def get_device_properties(self, rank=0):
        raise ValueError('device session does not implement device metadata')

    def memory_allocated(self, rank=0): return None
    def memory_reserved(self, rank=0): return None
    def max_memory_allocated(self, rank=0): return None
    def max_memory_reserved(self, rank=0): return None
    def reset_peak_memory_stats(self, rank=0): pass
    def stream_id(self): return None
    def Event(self, enable_timing=True):
        raise ValueError('device session does not implement timing events')


class TorchDeviceSession(DeviceSession):
    """Compatibility adapter for torch.cuda/torch.npu; never used as a universal SDK contract."""
    def __init__(self, backend, source, stream=None):
        self.backend, self.source, self._stream = backend, source, stream
        names = {'synchronize', 'device_info'}
        for cap, methods in {'memory': ('memory_allocated','memory_reserved'),
                             'peak_memory': ('max_memory_allocated','max_memory_reserved','reset_peak_memory_stats'),
                             'events': ('Event',)}.items():
            if all(callable(getattr(backend, m, None)) for m in methods): names.add(cap)
        if stream is not None: names.add('stream')
        self.capabilities = frozenset(names)
    def synchronize(self): self.backend.synchronize()
    def get_device_properties(self, rank=0):
        return DeviceInfo(self.backend.get_device_name(rank), self.backend.get_device_properties(rank).total_memory)
    def memory_allocated(self, rank=0): return self.backend.memory_allocated(rank) if 'memory' in self.capabilities else None
    def memory_reserved(self, rank=0): return self.backend.memory_reserved(rank) if 'memory' in self.capabilities else None
    def max_memory_allocated(self, rank=0): return self.backend.max_memory_allocated(rank) if 'peak_memory' in self.capabilities else None
    def max_memory_reserved(self, rank=0): return self.backend.max_memory_reserved(rank) if 'peak_memory' in self.capabilities else None
    def reset_peak_memory_stats(self, rank=0):
        if 'peak_memory' in self.capabilities: self.backend.reset_peak_memory_stats(rank)
    def Event(self, enable_timing=True): return self.backend.Event(enable_timing=enable_timing)
    def stream_id(self): return self._stream(self.backend) if self._stream else None


def increment(peak, baseline):
    return {k: peak[k]-baseline[k] if peak[k] is not None and baseline[k] is not None else None for k in peak}
