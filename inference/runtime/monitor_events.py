"""Best-effort, append-only lifecycle events outside timed device work."""
import json
import os
from pathlib import Path
import time


ENV = 'FLAGPERF_MONITOR_ROOT'


def emit(path, event, **fields):
    root = os.environ.get(ENV)
    if not root:
        return
    try:
        root, path = Path(root).resolve(), Path(path).resolve()
        relative = path.relative_to(root).as_posix()
        timestamp = time.perf_counter_ns()
        row = dict(timestamp_ns=timestamp, utc_ns=time.time_ns(), event=event, worker=relative,
                   pid=os.getpid(), rank=int(os.environ.get('RANK', '0')),
                   process_start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
                   pid_namespace=os.stat('/proc/self/ns/pid').st_ino,
                   clock=time.get_clock_info('perf_counter').implementation, **fields)
        with (root/'monitor'/'events'/f'{os.getpid()}.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    except (OSError, ValueError, TypeError, IndexError):
        # Monitoring cannot change execution or measured results.
        pass
