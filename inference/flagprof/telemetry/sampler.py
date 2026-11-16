"""Per-device sequential readers with bounded isolation of hung APIs."""
import math
import queue
import threading
import time
from .contract import Observation, ReadError, raw_clock


class Sampler:
    def __init__(self, calls, interval_ms=1000, timeout=3.0, queue_capacity=512):
        if not 100 <= interval_ms <= 5000:
            raise ValueError('monitor interval must be 100..5000 ms')
        self.calls = list(calls)
        self.interval = interval_ms/1000
        self.slow = self.interval*math.ceil(2/self.interval)
        self.timeout = timeout
        self.epoch_ns = raw_clock()
        self.results = queue.Queue(maxsize=queue_capacity)
        self.queue_overflow = False
        self.lock = threading.Lock()
        self.active = {}
        self.isolated = set()
        self.hangs = {}
        self.generations = {}
        self.next = {}
        self.previous = {}
        self.pending = {}
        self.closed = False

    def _read(self, device, generation, calls):
        for call in calls:
            with self.lock:
                if self.closed or self.generations.get(device) != generation:
                    return
                start = raw_clock()
                self.active[device] = (generation, call, start)
            try:
                observation = call.read()
            except Exception as error:
                reason = error.reason if isinstance(error, ReadError) else 'read_error'
                observation = Observation(call.entity, {m: None for m in call.metrics}, start, raw_clock(),
                                          {m: reason for m in call.metrics}, detail={'error': str(error)})
            with self.lock:
                if self.generations.get(device) != generation:
                    return
                self.active.pop(device, None)
                self.pending[device].extend((call, point) for point in (observation if isinstance(observation, list) else [observation]))
        with self.lock:
            if self.generations.get(device) == generation:
                self.generations.pop(device)
                self._publish(self.pending.pop(device, []))

    def tick(self, *, schedule=True):
        """Called by the writer loop; returns observations and hang events."""
        now = raw_clock()
        health = []
        with self.lock:
            if self.queue_overflow:
                raise RuntimeError('telemetry result queue capacity')
            for device, (generation, call, start) in list(self.active.items()):
                if now-start < self.timeout*1e9:
                    continue
                self.isolated.add(call.key)
                count = self.hangs.get(device, 0)+1
                self.hangs[device] = count
                self.generations.pop(device, None)
                pending = self.pending.pop(device, [])
                pending.append((call, Observation(call.entity, {m:None for m in call.metrics}, start, now,
                                                  {m:'timeout' for m in call.metrics})))
                self._publish(pending)
                self.active.pop(device)
                health.append(dict(t='health', ts=now, kind='hang', sev='error' if count >= 2 else 'warn',
                                   detail=dict(scope='telemetry_call', entity=call.entity, api=call.key,
                                               metrics=list(call.metrics),
                                               source_api=call.source_api,
                                               started_raw_ns=start, device_isolated=count >= 2)))
            for device in (dict.fromkeys(c.device for c in self.calls) if schedule else ()):
                if device in self.generations:
                    continue
                due = []
                missing = []
                for call in (c for c in self.calls if c.device == device):
                    if now < self.next.get(call.key, 0):
                        continue
                    period = self.slow if call.group == 'slow' else self.interval
                    period_ns=int(period*1e9)
                    scheduled=self.next.get(call.key,self.epoch_ns)
                    self.next[call.key] = scheduled+((now-scheduled)//period_ns+1)*period_ns
                    if call.key in self.isolated or self.hangs.get(device, 0) >= 2:
                        reason = 'isolated'
                        missing.append((call, Observation(call.entity, {m: None for m in call.metrics},
                                             now, now, {m: reason for m in call.metrics})))
                    else:
                        due.append(call)
                if not due:
                    self._publish(missing)
                if due:
                    self.pending[device] = missing
                    generation = object()
                    self.generations[device] = generation
                    threading.Thread(target=self._read, args=(device, generation, due), daemon=True).start()
        observations = []
        while True:
            try:
                key, period, observation = self.results.get_nowait()
            except queue.Empty:
                break
            if observation.entity.startswith('proc:') and not any(c.entity==observation.entity.split('/',1)[0] for c in self.calls):
                continue
            last = self.previous.get(key)
            observation.detail['interval_ns'] = (observation.start_ns-last if last is not None else
                int(period*1e9))
            self.previous[key] = observation.start_ns
            observations.append(observation)
        return observations, health

    def _publish(self, readings):
        groups = {}
        for call, point in readings:
            key = (point.entity, call.group)
            if key not in groups:
                groups[key] = Observation(point.entity, {}, point.start_ns, point.end_ns,
                                           detail={'group': call.group, 'reads': []})
            merged = groups[key]
            merged.start_ns = min(merged.start_ns, point.start_ns)
            merged.end_ns = max(merged.end_ns, point.end_ns)
            merged.values.update(point.values); merged.miss.update(point.miss)
            merged.sample_times.update(point.sample_times)
            merged.health.extend(point.health)
            merged.detail['reads'].append(dict(api=point.source, ts=point.start_ns,
                dur=point.end_ns-point.start_ns, **point.detail))
        for key, point in groups.items():
            try:self.results.put_nowait((key, self.slow if key[1]=='slow' else self.interval, point))
            except queue.Full:self.queue_overflow=True

    def remove(self, keys):
        """Retire process calls and their scheduling state, including late results."""
        keys = set(keys)
        with self.lock:
            removed = [c for c in self.calls if c.key in keys]
            self.calls[:] = [c for c in self.calls if c.key not in keys]
            live_devices = {c.device for c in self.calls}
            for call in removed:
                self.next.pop(call.key, None)
                self.isolated.discard(call.key)
                for key in list(self.previous):
                    if key[0]==call.entity or key[0].startswith(call.entity+'/'):
                        self.previous.pop(key,None)
                if call.device not in live_devices:
                    self.generations.pop(call.device, None)
                    self.pending.pop(call.device, None)
                    self.active.pop(call.device, None)
                    self.hangs.pop(call.device, None)

    def close(self):
        with self.lock:
            self.closed = True

    def finish(self):
        """One final read after target exit brackets the last energy interval."""
        deadline=time.monotonic()+self.timeout
        while self.generations and time.monotonic()<deadline:
            yield self.tick(schedule=False)
            time.sleep(.01)
        self.next.clear()
        yield self.tick()
        while self.generations and time.monotonic()<deadline:
            yield self.tick(schedule=False)
            time.sleep(.01)
        yield self.tick(schedule=False)
