# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Transient, stderr-only progress shared by a CLI invocation and its workers."""
import contextlib
import contextvars
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unicodedata
import uuid


_CHANNEL = '_FLAGPROF_PROGRESS_DIR'
_PARENT = '_FLAGPROF_PROGRESS_PARENT'
_RANK = '_FLAGPROF_PROGRESS_RANK'
_VARIANT = '_FLAGPROF_PROGRESS_VARIANT'
_invocation = contextvars.ContextVar('flagprof_progress_channel', default=None)
_display = contextvars.ContextVar('flagprof_progress_display', default=None)
_current = contextvars.ContextVar('flagprof_phase', default=None)
_labels = contextvars.ContextVar('flagprof_progress_labels', default={})


class _Status:
    code = 0

    def finish(self, code):
        self.code = code or 0

    def update(self, completed, *, total=None, unit=None):
        pass


def _channel():
    return _invocation.get() or os.environ.get(_CHANNEL)


def _exception_code(exc):
    if isinstance(exc, KeyboardInterrupt):
        return 130
    if isinstance(exc, SystemExit):
        return exc.code if isinstance(exc.code, int) else int(exc.code is not None)
    return 1


@contextlib.contextmanager
def scope(*, rank=None, variant=None):
    labels = dict(_labels.get())
    if rank is not None:
        labels['rank'] = str(rank)
    if variant is not None:
        labels['variant'] = str(variant)
    token = _labels.set(labels)
    try:
        yield
    finally:
        _labels.reset(token)


def child_env(env=None):
    """Carry the active phase and labels, including into a prebuilt launch env."""
    child = dict(os.environ if env is None else env)
    channel = _channel()
    if channel:
        child[_CHANNEL] = channel
        current = _current.get()
        if current:
            child[_PARENT] = current.id
        elif _PARENT in os.environ:
            child[_PARENT] = os.environ[_PARENT]
        for name, key in (('rank', _RANK), ('variant', _VARIANT)):
            value = _labels.get().get(name, os.environ.get(key))
            if value is not None:
                child[key] = value
    return child


class _Phase(_Status):
    def __init__(self, channel, name, total, unit, detail):
        self.journal = Path(channel) / (str(os.getpid()) + '.jsonl')
        self.id = uuid.uuid4().hex
        self.total, self.unit = total, unit
        self.completed = 0 if total is not None else None
        self.last_update = 0.
        parent = _current.get()
        self.send('start', name=name, parent=parent.id if parent else os.environ.get(_PARENT),
                  rank=_labels.get().get('rank', os.environ.get(_RANK, os.environ.get('RANK'))),
                  variant=_labels.get().get('variant', os.environ.get(_VARIANT)), detail=detail)

    def send(self, kind, **fields):
        fields.update(kind=kind, id=self.id, at=time.monotonic(), completed=self.completed,
                      total=self.total, unit=self.unit)
        try:
            with self.journal.open('ab') as out:
                out.write((json.dumps(fields) + '\n').encode('ascii'))
        except OSError:
            pass

    def update(self, completed, *, total=None, unit=None):
        changed_total = total is not None and total != self.total
        changed_count = self.completed != completed
        self.completed = completed
        if total is not None:
            self.total = total
        if unit is not None:
            self.unit = unit
        now = time.monotonic()
        if (changed_total or (changed_count and self.total is not None and self.total <= 100)
                or now - self.last_update >= .4):
            self.send('update')
            self.last_update = now


@contextlib.contextmanager
def phase(name, *, total=None, unit=None, rank=None, variant=None, detail=None):
    channel = _channel()
    if not channel or channel == 'off':
        yield _Status()
        return
    with scope(rank=rank, variant=variant):
        step = _Phase(channel, name, total, unit, detail)
        token = _current.set(step)
        try:
            yield step
        except BaseException as exc:
            step.finish(_exception_code(exc))
            raise
        finally:
            step.send('end', code=step.code)
            _current.reset(token)


def _meter(item, now):
    done, total, unit = item['completed'], item['total'], item['unit'] or ''
    if total is not None and total > 0:
        percent = 100 * done // total
        filled = min(10, max(0, percent // 10))
        return f"[{'#' * filled}{'-' * (10 - filled)}] {done}/{total} {unit} ({percent}%)"
    if done is not None and total == 0:
        return f'{done}/0 {unit}'
    spinner = '|/-\\'[int(now * 3) % 4]
    return f'[{spinner}]' + (f' {done} {unit}' if done is not None else '')


def _char_width(char):
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in 'WF' else 1


def _fit_tail(text, columns):
    """Fit a redraw on one terminal row, retaining the innermost phase."""
    width = 0
    for i in range(len(text) - 1, -1, -1):
        width += _char_width(text[i])
        if width > columns:
            return '…' + text[i + 2:]
    return text


class _Display:
    def __init__(self, channel, stream, tty):
        self.channel, self.stream, self.tty = Path(channel), stream, tty
        self.started = self.last_line = time.monotonic()
        self.active, self.offsets = {}, {}
        self.overlay = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name='flagprof-progress', daemon=True)

    def line(self, message, *, permanent=False):
        try:
            end = '\n' if permanent or not self.tty else ''
            self.stream.write(('\r\033[2K' if self.tty else '') + '[flagprof] ' + message + end)
            self.overlay = self.tty and not permanent
            self.stream.flush()
        except (ValueError, OSError, UnicodeError):
            pass

    def show(self, item, now, *, status='', detail=False):
        chain = [item]
        while chain[-1].get('parent') in self.active:
            chain.append(self.active[chain[-1]['parent']])
        parents = []
        for parent in reversed(chain[1:]):
            label = parent['name']
            if parent['total'] is not None:
                label += ' ' + _meter(parent, now)
            parents.append(label)
        path = ' > '.join(parents + [item['name']])
        labels = ' '.join(f'{key}={item[key]}' for key in ('variant', 'rank') if item.get(key) is not None)
        info = f" {item['detail']}" if detail and item.get('detail') else ''
        stage_time = now - item['start']
        total_time = now - self.started
        meter = _meter(item, now)
        suffix = f'{labels} {meter}  阶段 {stage_time:.1f}s / 总计 {total_time:.1f}s {status}'.strip()
        if self.tty and not info:
            try:
                columns = os.get_terminal_size(self.stream.fileno()).columns - len('[flagprof] ') - 1
            except (ValueError, OSError, AttributeError):
                columns = 0
            if columns > 0 and sum(map(_char_width, path + ' ' + suffix)) > columns:
                suffix = f'{labels} {meter} {stage_time:.0f}s/{total_time:.0f}s {status}'.strip()
                path = _fit_tail(path, max(1, columns - sum(map(_char_width, suffix)) - 1))
        self.line(f'{path} {suffix}{info}', permanent=bool(info))
        self.last_line = now

    def poll(self):
        events = []
        for journal in self.channel.glob('*.jsonl'):
            with journal.open('rb') as rows:
                rows.seek(self.offsets.get(journal, 0))
                for raw in rows:
                    if not raw.endswith(b'\n'):
                        break
                    self.offsets[journal] = rows.tell()
                    try:
                        events.append(json.loads(raw))
                    except (ValueError, UnicodeError):
                        continue
        for event in sorted(events, key=lambda e: e['at']):
            kind, key, now = event['kind'], event['id'], event['at']
            if kind == 'start':
                event['start'] = now
                self.active[key] = event
                self.show(event, now, detail=True)
            elif key in self.active:
                item = self.active[key]
                item.update(event)
                if kind == 'end':
                    self.show(item, now, status='完成' if not event['code'] else f"退出 {event['code']}")
                    self.close_phase(key)
                elif self.tty or not item.get('announced') or now - self.last_line >= 10:
                    self.show(item, now)
                    item['announced'] = True
        now = time.monotonic()
        if self.active and now - self.last_line >= (.4 if self.tty else 10):
            self.show(max(self.active.values(), key=lambda e: e['start']), now)

    def close_phase(self, key):
        del self.active[key]
        # A terminated worker may never send its own end event.
        for child in [e['id'] for e in self.active.values() if e.get('parent') == key]:
            self.close_phase(child)

    def run(self):
        while True:
            stopping = self.stop.is_set()
            try:
                self.poll()
            except OSError:
                # Progress output is best effort, including during temp cleanup.
                pass
            if stopping:
                return
            self.stop.wait(.2)


def finish_display():
    """Release the terminal line before a CLI prints its result or diagnostic."""
    display = _display.get()
    if display and display.tty and not display.stop.is_set():
        display.stop.set()
        display.thread.join()
        if display.overlay:
            try:
                display.stream.write('\r\033[2K')
                display.stream.flush()
            except (ValueError, OSError):
                pass


@contextlib.contextmanager
def session(mode='auto', *, json_output=False):
    """Own one display at the outermost CLI and propagate its policy to workers."""
    status = _Status()
    if _channel():
        yield status
        return
    enabled = mode != 'off' and not (mode == 'auto' and json_output)
    with contextlib.ExitStack() as stack:
        channel = 'off'
        if enabled:
            try:
                channel = stack.enter_context(tempfile.TemporaryDirectory(prefix='flagprof-progress-'))
            except OSError:
                pass
        token = _invocation.set(channel)
        stack.callback(_invocation.reset, token)
        display = None
        if channel != 'off':
            stream = sys.stderr
            display = _Display(channel, stream, mode == 'auto' and stream.isatty())
            token = _display.set(display)
            stack.callback(_display.reset, token)
            display.thread.start()
        try:
            yield status
        except BaseException as exc:
            status.finish(_exception_code(exc))
            raise
        finally:
            if display:
                display.stop.set()
                display.thread.join()
                elapsed = time.monotonic() - display.started
                result = '完成' if status.code == 0 else f'退出 {status.code}'
                display.line(f'{result}  总计 {elapsed:.1f}s', permanent=True)
