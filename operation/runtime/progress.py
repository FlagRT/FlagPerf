# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Best-effort host progress; never instrument the device timing loop."""
from functools import wraps
import json
import sys
import threading
import time


HEARTBEAT_SECONDS = 5


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class Progress:
    def __init__(self, clock=time.monotonic, stream=None):
        self.clock = clock
        self.stream = sys.stderr if stream is None else stream
        self.started = clock()
        self.task_started = None
        self.label, self.phase, self.completed = '', 'preparing', 0
        self.profile_path = None
        self.profile_sample = None
        self.profile_seen = self.started
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        self.emit()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True,
                                       name='operation-progress')
        self.thread.start()

    def _heartbeat(self):
        # Poll only the small profiling status, never raw captures or tensors.
        previous = ''
        last = self.clock()
        while not self.stop.wait(0.25):
            with self.lock:
                detail = self._profile()
                now = self.clock()
                identity = detail.split(' group_elapsed=')[0]
                if identity != previous or now - last >= HEARTBEAT_SECONDS:
                    self.emit(detail)
                    previous, last = identity, now

    def _profile(self):
        if self.profile_path is None:
            return ''
        try:
            data = json.loads(self.profile_path.read_text())
            sample = (data['group'], data['index'], data['state'], data['elapsed_s'])
            if sample != self.profile_sample:
                self.profile_sample, self.profile_seen = sample, self.clock()
            elapsed = data['elapsed_s']
            if data['state'] in ('preparing', 'capturing', 'parsing'):
                elapsed += self.clock() - self.profile_seen
            return (f' group={data["group"]}({data["index"]}/{data["total"]})'
                    f' capture={data["state"]} group_elapsed={duration(elapsed)}')
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            return ''

    def emit(self, detail=None):
        with self.lock:
            now = self.clock()
            text = f'[progress] {self.label} completed={self.completed} phase={self.phase}'
            text += self._profile() if detail is None else detail
            if self.task_started is not None:
                text += f' task_elapsed={duration(now-self.task_started)}'
            text += f' total_elapsed={duration(now-self.started)}'
            try:
                print(text, file=self.stream, flush=True)
            except (OSError, ValueError):
                pass

    def stage(self, name, profile_path=None):
        with self.lock:
            self.phase = name
            self.profile_path = profile_path
            self.profile_sample = None
            self.emit()

    def task(self, task, index=None, total=None):
        with self.lock:
            prefix = f'[{index}/{total}] ' if index is not None else ''
            self.label = prefix + ' '.join(f'{k}={task.get(k, "?")}' for k in
                                         ('case', 'dtype', 'oplib', 'device_id'))
            dimensions = {k: v for k, v in task.get('case_config', {}).items()
                          if k not in ('WARMUP', 'ITERS', 'KERNELWARMUP', 'KERNELITERS', 'rounds')}
            self.label += ' config=' + json.dumps(dimensions, sort_keys=True, separators=(',', ':'))
            self.task_started = self.clock()
            self.stage('task-start')

    def done(self, status):
        with self.lock:
            if status not in ('interrupted', 'not-run'):
                self.completed += 1
            self.stage('task-' + status)
            self.task_started = None

    def finish(self, status):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self.stage(status)


def tracked(function):
    """Keep heartbeat alive through blocking setup, cleanup and report rendering."""
    @wraps(function)
    def wrapper(args, *positional, **kwargs):
        if getattr(args, 'dry_run', False):
            return function(args, *positional, **kwargs)
        progress = Progress()
        args._progress = progress
        status = 'failed'
        progress.start()
        try:
            result = function(args, *positional, **kwargs)
            status = {0: 'passed', 1: 'failed', 2: 'partial'}.get(result, 'finished')
            return result
        except (KeyboardInterrupt, SystemExit):
            status = 'interrupted'
            raise
        finally:
            progress.finish(status)
            del args._progress
    return wrapper


def stage(args, name, profile_path=None):
    progress = getattr(args, '_progress', None)
    if progress is not None:
        progress.stage(name, profile_path)


def publish_profile(root, group, index, total, state, elapsed_s):
    from runtime.evidence import write
    try:
        write(root / 'profiling-progress.json', dict(group=group, index=index,
              total=total, state=state, elapsed_s=elapsed_s))
    except OSError:
        pass
