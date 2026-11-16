"""Small typed boundary between vendor reads, sampling and persistence."""
from dataclasses import dataclass, field, asdict
import math
import time

REASONS = frozenset(('unsupported', 'read_error', 'timeout', 'isolated',
                     'profiler_conflict', 'fallback_unparsable', 'invalid'))


def raw_clock():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


@dataclass(frozen=True)
class Metric:
    name: str
    unit: str
    scope: str = 'chip'
    source_api: str = ''
    kind: str = 'instant'
    group: str = 'fast'
    original_unit: str = ''
    factor: float = 1.0
    window_ns: object = None
    profiler_conflict: bool = False
    busy_indicator: bool = False
    roles: tuple = ()
    restriction_masks: dict = field(default_factory=dict)

    def snapshot(self):
        return asdict(self)


@dataclass
class Observation:
    entity: str
    values: dict
    start_ns: int
    end_ns: int
    miss: dict = field(default_factory=dict)
    source: str = ''
    sample_times: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)
    health: list = field(default_factory=list)

    def __post_init__(self):
        for key, value in list(self.values.items()):
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
                self.values[key] = None
                self.miss[key] = 'invalid'
        if set(self.miss) != {k for k, v in self.values.items() if v is None}:
            raise ValueError('miss must describe exactly the null metrics')
        if not set(self.miss.values()) <= REASONS:
            raise ValueError('unknown observation missing reason')


@dataclass
class Call:
    key: str
    device: str
    entity: str
    metrics: tuple
    read: object
    group: str = 'fast'
    source_api: str = ''


class ReadError(Exception):
    def __init__(self, reason, detail=None):
        self.reason, self.detail = reason, detail
        super().__init__(str(detail or reason))
