# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Best-effort host progress; elapsed times are not benchmark measurements."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time


class RunProgress:
    def __init__(self, kind: str, *, case: str | None = None,
                 events: Path | None = None, interval: float = 10.0):
        self.kind = kind
        self.events = events
        self.interval = interval
        self.started = time.monotonic()
        self.state = {"phase": "container-startup", "completed": 0}
        if case is not None:
            self.state.update(case=case, index=1, total=1,
                              phase="running", case_started=self.started)
        self.offset = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _emit(self):
        now = time.monotonic()
        state = self.state
        parts = [f"[{self.kind}] {state['phase']}"]
        if state.get("case"):
            parts.append(f"case {state['index']}/{state['total']}: {state['case']}")
            parts.append(f"case elapsed={max(0, now - state['case_started']):.1f}s")
        total = state.get("total", "?")
        parts.append(f"completed={state['completed']}/{total}")
        parts.append(f"elapsed={now - self.started:.1f}s")
        try:
            print(" | ".join(parts), file=sys.stderr, flush=True)
        except OSError:
            pass

    def _poll(self):
        if self.events is None:
            return
        try:
            with self.events.open(encoding="utf-8") as stream:
                stream.seek(self.offset)
                while True:
                    line = stream.readline()
                    if not line.endswith("\n"):
                        break
                    self.offset = stream.tell()
                    self.state = json.loads(line)
                    self._emit()
        except (OSError, ValueError):
            pass

    def _watch(self):
        next_update = time.monotonic() + self.interval
        while not self.stop.wait(0.2):
            self._poll()
            if time.monotonic() >= next_update:
                self._emit()
                next_update = time.monotonic() + self.interval

    def __enter__(self):
        self._emit()
        self.thread.start()
        return self

    def finished(self, returncode: int):
        self.stop.set()
        self.thread.join()
        self._poll()
        if self.events is None:
            self.state["completed"] = 1
        self.state["phase"] = f"container-exited (code={returncode})"

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        self._poll()
        if exc_type is not None:
            self.state["phase"] = f"container-wait-aborted ({exc_type.__name__})"
        self._emit()
