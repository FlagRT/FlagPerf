# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Device-free vendor backend for the CPU smoke test.

It follows the official profiler protocol used by ProfilerSession and exports
one Chrome trace with a single kernel event per window.
"""
import json

from flagprof.vendors.session import validate_chrome


class FakeBackend:
    activity_source = 'smoke-activity'

    def __init__(self):
        self.destinations = []
        self.stops = 0

    def conflict(self):
        return 'not_detectable'

    def identity(self, config):
        return dict(collector_prerequisites=[])

    def set_destination(self, path):
        self.destinations.append(path)

    def create(self, config, callback):
        self.config = config
        self.callback = callback
        self.steps = 0
        return self

    def start(self):
        pass

    def step(self):
        self.steps += 1
        if self.steps % self.config.active == 0:
            self.callback(self)

    def stop(self):
        self.stops += 1

    def export(self, profiler, path):
        trace = dict(traceEvents=[dict(ph='X', cat='kernel', ts=1, dur=2)], window_step=self.steps)
        (path/'trace.json').write_text(json.dumps(trace))

    def validate(self, path):
        return validate_chrome(path, 'trace.json', lambda event, _: event.get('cat') == 'kernel')
