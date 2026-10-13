"""Device-management telemetry contract; independent of execution frameworks."""
import math
import time


class TelemetryProvider:
    def discover(self, devices):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def close(self):
        pass


def metric(name, unit, meaning, scope='chip', **extra):
    return dict(name=name, unit=unit, meaning=meaning, scope=scope,
                source_window_ns=None, refresh_period_ns=None, **extra)


def observation(entity, definition, value, start_ns, *, raw=None, error=None, source=None):
    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
        error, value = 'invalid_value', None
    if definition['unit'] == '%' and value is not None and not 0 <= value <= 100:
        error, value = 'invalid_percentage', None
    return dict(entity=entity, metric=definition['name'], value=value, raw=raw,
                read_start_ns=start_ns, read_end_ns=time.perf_counter_ns(),
                source_timestamp_ns=None, status='ok' if error is None else 'unavailable',
                reason=error, source=source)
