# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Production process Writer with cached wire encoding and bounded admission."""
from collections import Counter, deque
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import asdict
import json
import marshal
import mmap
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import platform
import threading
import time
import uuid
import zlib

from .writer import Writer
from flagprof.store.manifest import atomic_json
from flagprof.store.schema import is_control

RING_BYTES = 8 << 20
BATCH_EVENTS = 512
BATCH_SECONDS = .01
FLUSHER_PERIOD = .05
HDR = 4096
PROD = struct.Struct("<QQ")          # head bytes, head frame id            @0
CONS = struct.Struct("<QQQQQ")       # tail, consumed events, written, durable, heartbeat @64
CONS_OFF = 64
FRAME = struct.Struct("<QQ")        # frame id, length, crc32
PAD = 0xFFFFFFFF
MIN_FREE = 64 << 10
INFLIGHT_BYTES = 16 << 20
CONTROL_BYTES = 1 << 20


def frame_size(payload):
    return (FRAME.size+len(payload)+7)//8*8

def prefault_ring(words):
    # Commit the fixed reservation in both processes before admission. PSS
    # then attributes shared pages consistently during pause and drain.
    # The startup handshake has not published any frame yet. A same-value
    # write commits actual shmem pages instead of only faulting a zero page.
    for index in range(0,len(words),4096//8):
        words[index] = words[index]



@contextmanager
def _held(lock, deadline):
    if not lock.acquire(timeout=max(0., deadline-time.monotonic())):
        raise TimeoutError('proxy lock deadline')
    try: yield
    finally: lock.release()


class ProxyWriter(Writer):
    def __init__(self, run_dir, *, config=None, attachment_dir=None, handshake_callback=None, **kwargs):
        if sys.platform != 'linux' or platform.machine() not in ('aarch64', 'x86_64') or platform.python_implementation() != 'CPython' or not hasattr(os, 'pidfd_open'):
            raise RuntimeError('proxy unsupported platform or pidfd unavailable')
        ring_bytes = RING_BYTES
        self.attachment_dir = Path(attachment_dir) if attachment_dir else Path(run_dir)/'attachments'/uuid.uuid4().hex
        self.attachment_dir.mkdir(parents=True, exist_ok=True)
        self._wire_total = self._wire_peak = self._max_frame = self._control_total = 0
        self._wire_consumed_cache = self._control_consumed_cache = 0
        self.errors = []
        self._batch, self._batch_t0 = [], 0.
        self._pub_lock = threading.Lock()
        self._management = threading.RLock()
        self._ack_lock = threading.RLock()
        self._reply_event = threading.Event()
        self._failure_lock = threading.RLock()
        self._replies = deque()
        self._ack_eof = False
        self._closed_ack = False
        self._closing_request = None
        self._failure_monotonic = None
        self._terminal_monotonic = None
        self._reaped_monotonic = None
        self._closing = False
        self._cleanup_state = "not_started"
        self._cleanup_errors = []
        self._transport_removed = False
        self._accepted_since_live = 0
        self._cons_cache = [0]*5
        self._torn_reads = 0
        self._buffered_peak = 0
        self._pidfd = None
        self._disposed = False
        self._tids = {}
        self._clock_raw = time.CLOCK_MONOTONIC_RAW
        if ring_bytes < 256 or ring_bytes % 8:
            raise ValueError("ring_bytes must be >=256 and aligned to 8")
        self._size = ring_bytes
        self._head = self._frame = 0
        self._consumed_cache = 0
        self._failure = None
        self._consumer = None
        self._ipc_drops = Counter()
        self._request = 0
        self._barriers = {}
        self._retry_at = 0.
        self._acks = b""
        self.consumer_cpu_seconds = 0.
        self._stop = threading.Event()
        path = self.attachment_dir/'transport'
        self._file = os.fdopen(os.memfd_create('flagprof-ring', os.MFD_CLOEXEC), 'w+b')
        self._transport_path = path
        try:
            self._file.truncate(HDR+ring_bytes)
            self._map = mmap.mmap(self._file.fileno(), HDR+ring_bytes)
            self._words = memoryview(self._map).cast("Q")
            prefault_ring(self._words)
        except BaseException:
            try:
                path.unlink(missing_ok=True)
            finally:
                self._file.close()
            raise
        ctrl_r = ack_w = None
        try:
            self._instance = uuid.uuid4().int & ((1 << 63)-1)
            super().__init__(run_dir, config=config, **kwargs)   # run_start lands in _batch
            self._thread.join(10)
            self._compressor.shutdown(wait=True)
            ctrl_r, self._ctrl_w = os.pipe()
            self._ack_r, ack_w = os.pipe()
            identity = dict(run_dir=str(self.run_dir), config=asdict(self.config), fragment=self.fragment,
                            proc=self.proc, defaults=self.defaults, instance=self._instance,
                            ring=str(path), ring_fd=self._file.fileno(),
                            size=ring_bytes, attachment_dir=str(self.attachment_dir))
            ident_path = self.attachment_dir/"proxy-identity.json"
            atomic_json(ident_path, identity)
            inherited = (ctrl_r, ack_w, self._file.fileno())
            environment = dict(os.environ)
            root = str(Path(__file__).resolve().parents[2])
            environment['PYTHONPATH'] = os.pathsep.join([root]+([environment['PYTHONPATH']] if environment.get('PYTHONPATH') else []))
            self._consumer = subprocess.Popen([sys.executable, '-B', '-S', '-m', 'flagprof.store._proxy_consumer', str(ident_path), str(ctrl_r), str(ack_w)], pass_fds=inherited, env=environment)
            os.close(ctrl_r)
            ctrl_r = None
            os.close(ack_w)
            ack_w = None
            os.set_blocking(self._ack_r, False)
            self._pidfd = os.pidfd_open(self._consumer.pid)
            ready = self._recv(10)
            if ready.get("op") != "ready" or ready.get("instance") != self._instance:
                raise RuntimeError("consumer startup: %r" % ready)
            self.consumer_key = ready["proc_key"]
            if handshake_callback is None:
                atomic_json(self.attachment_dir/'state.json', dict(id=self.attachment_dir.name, process=self.proc, status='running', writer_mode='process_proxy', writer_proxy=self.registration()))
            if handshake_callback is not None:
                handshake_callback(self)
            self._send({"op": "start"})
            if self._recv(10).get("op") != "started":
                raise RuntimeError("consumer handshake")
            self._flusher = threading.Thread(target=self._flusher_main, daemon=True, name="b2-flusher")
            self._flusher.start()
        except BaseException:
            # Popen may fail before it takes ownership of inherited endpoints.
            for fd in (ctrl_r, ack_w):
                if fd is not None:
                    os.close(fd)
            self._kill()
            raise

    def registration(self):
        from .manifest import proc_key
        return dict(process=proc_key(self._consumer.pid), instance=self._instance, writer_id=self.writer_id)

    # -- plumbing -------------------------------------------------------
    def _disk_main(self):
        self._ready.set()

    def _send(self, message):
        os.write(self._ctrl_w, json.dumps(message).encode()+b"\n")

    def _drain_ready(self):
        # All pipe reads and reply routing have one owner at a time. No blocking
        # receive while holding this lock; management callers use a separate lock.
        if not self._ack_lock.acquire(False):
            return
        try:
            if self._ack_r is None:
                return
            while not self._ack_eof:
                try:
                    chunk = os.read(self._ack_r, 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    self._ack_eof = True
                    break
                self._acks += chunk
            while b"\n" in self._acks:
                line, self._acks = self._acks.split(b"\n", 1)
                reply = json.loads(line)
                if not isinstance(reply, dict):
                    raise ValueError("ack must be an object")
                if reply.get("op") == "failed":
                    if reply.get("instance") != self._instance:
                        self._fail("invalid_failure_ack")
                    else:
                        self._fail("consumer_failure:"+str(reply.get("error")))
                if reply.get("op") == "closed":
                    state = self._barriers.get(self._closing_request)
                    if (state is not None and reply.get("request") == self._closing_request and
                            reply.get("instance") == self._instance and reply.get("consumed") == state["covered"]):
                        self._closed_ack = True
                self._replies.append(reply)
            if self._replies or self._ack_eof:
                self._reply_event.set()
        finally:
            self._ack_lock.release()

    def _wait_for_reply(self, timeout):
        flusher = getattr(self, "_flusher", None)
        if flusher is None:
            # Startup has never had a competing pipe reader.
            select.select([self._ack_r], [], [], timeout)
        else:
            # The flusher may already have consumed the pipe bytes. Its sticky
            # notification represents the in-memory reply, not fd readability.
            # It can be blocked in publishing, or can exit. Management keeps
            # draining bytes itself at the registered 50 ms cadence as well.
            timeout = min(timeout, FLUSHER_PERIOD)
            self._reply_event.wait(timeout)

    def _recv(self, timeout):
        deadline = time.monotonic()+timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("consumer ack timeout")
            self._drain_ready()
            with _held(self._ack_lock, deadline):
                if time.monotonic() >= deadline:
                    raise TimeoutError("consumer ack timeout")
                if self._replies:
                    return self._replies.popleft()
                eof = self._ack_eof
                # Empty-test and reset share the publisher's lock. A reply
                # arriving after this reset cannot lose its notification.
                self._reply_event.clear()
            if eof:
                self._check_alive()
                if not self._failure:
                    self._fail("consumer_lost")
                raise EOFError("consumer ack pipe closed")
            if self._failure:
                raise EOFError("consumer failed while awaiting ack")
            left = deadline-time.monotonic()
            if left <= 0:
                raise TimeoutError("consumer ack timeout")
            self._wait_for_reply(left)

    def _cleanup_error(self, stage, exc):
        row = dict(stage=stage, error=repr(exc))
        if row not in self._cleanup_errors and len(self._cleanup_errors) < 16:
            self._cleanup_errors.append(row)
        self._fail("writer_internal_error:cleanup_%s:%r" % (stage, exc))

    def _kill(self, deadline=None):
        """One management owner; never destroy memory still used by a thread.

        A pending cleanup is resumable by a later close. The outer case owner
        must isolate/reap a worker stuck in an uninterruptible filesystem call.
        """
        deadline = time.monotonic()+1 if deadline is None else deadline
        self._cleanup_state = "in_progress"
        self._stop.set()

        def step(name, action):
            if time.monotonic() >= deadline:
                return False
            try:
                action()
                return True
            except Exception as exc:
                self._cleanup_error(name, exc)
                return False

        if self._consumer is not None and self._consumer.returncode is None:
            step("signal", self._consumer.kill)
            if step("reap", lambda: self._consumer.wait(max(0., deadline-time.monotonic()))):
                self._reaped_monotonic = time.monotonic()
        reaped = self._consumer is None or self._consumer.returncode is not None
        flusher = getattr(self, "_flusher", None)
        if flusher is not None and flusher is not threading.current_thread():
            step("join", lambda: flusher.join(max(0., deadline-time.monotonic())))
        stopped = flusher is None or not flusher.is_alive()
        locks = []
        if stopped:
            for lock in (self._pub_lock, getattr(self, "_lock", None), self._ack_lock):
                if lock is None:
                    continue
                if not lock.acquire(timeout=max(0., deadline-time.monotonic())):
                    break
                locks.append(lock)
            else:
                try:
                    if not self._map.closed:
                        self._final = tuple(self._cons_cache)
                        step("snapshot", lambda: setattr(self, "_final", self._cons()))
                        step("view_release", self._words.release)
                        step("mmap_close", self._map.close)
                    # Remove an inspect-only /proc link before its descriptor
                    # could be recycled. A failed unlink keeps the fd for retry.
                    if self._map.closed and not self._transport_removed:
                        if step("transport_unlink", lambda: self._transport_path.unlink(missing_ok=True)):
                            self._transport_removed = True
                    if self._transport_removed:
                        step("file_close", self._file.close)
                    for name in ("_ctrl_w", "_ack_r", "_pidfd"):
                        # Serialize pidfd lifetime with nonblocking probes.
                        with self._failure_lock:
                            fd = getattr(self, name, None)
                            if fd is not None and step(name, lambda fd=fd: os.close(fd)):
                                setattr(self, name, None)
                finally:
                    for lock in reversed(locks):
                        lock.release()
                    locks.clear()
            for lock in reversed(locks):
                lock.release()
        self._disposed = (stopped and reaped and self._map.closed and self._file.closed and
                          all(getattr(self, n, None) is None for n in ("_ctrl_w", "_ack_r", "_pidfd")) and
                          self._transport_removed)
        self._cleanup_state = "complete" if self._disposed else "pending"
        if not self._disposed:
            self._cleanup_error("incomplete", TimeoutError("cleanup pending at management deadline"))
        return self._disposed

    def _fail(self, reason):
        with self._failure_lock:
            if self._failure is None:
                self._failure_monotonic = time.monotonic()
                self._failure = reason
                self.errors.append(reason)
                self._reply_event.set()
            elif self._failure == "consumer_lost" and reason.startswith("consumer_failure:"):
                self.errors[self.errors.index(self._failure)] = reason
                self._failure = reason

    def _read_cons(self):
        return tuple(self._words[8+i] for i in range(5))

    def _cons(self):
        if not self._ack_lock.acquire(False):
            return tuple(self._cons_cache)
        try:
            return self._read_cons_monotonic()
        finally:
            self._ack_lock.release()

    def _read_cons_monotonic(self):
        if self._map.closed:
            return self._final
        values = list(self._read_cons())
        if any(v < old for v, old in zip(values, self._cons_cache)):
            self._torn_reads += 1
            values = list(self._read_cons())
        self._cons_cache = [max(v, old) for v, old in zip(values, self._cons_cache)]
        self._wire_consumed_cache = max(self._wire_consumed_cache, self._words[15])
        self._control_consumed_cache = max(self._control_consumed_cache, self._words[16])
        self._buffered_peak = max(self._buffered_peak, self._words[13])
        self._consumer_torn_reads = self._words[14]
        return tuple(self._cons_cache)

    def _check_alive(self, now=None):
        with self._failure_lock:
            fd = self._pidfd
            if fd is not None and select.select([fd], [], [], 0)[0]:
                if self._terminal_monotonic is None:
                    self._terminal_monotonic = time.monotonic()
                # Closing already stopped admission. Its management owner must
                # still adjudicate a matching closed ack AND a zero exit status.
                if not self._closing:
                    self._fail("consumer_lost")

    def _finish_failure_messages(self):
        try:
            self._check_alive()
            if self._ack_lock.acquire(False):
                try:
                    self._drain_ready()
                finally:
                    self._ack_lock.release()
        except Exception as exc:
            self._fail("writer_internal_error:"+repr(exc))

    # -- hot path -------------------------------------------------------
    def emit(self, record, *, _receipt=None):
        if os.getpid() != self._owner_pid:
            return False
        if not isinstance(record, Mapping):
            raise TypeError("emit requires a Mapping")
        try:
            event = dict(record)
            return self._emit_event(event)
        except Exception as exc:
            self._fail("writer_internal_error:"+repr(exc))
            return False

    def _emit_event(self, event):
        t = event.get("t")
        if t == "sig" or (t == "meta" and is_control(event)) or "seq" in event:
            self.errors.append("emit rejected writer-owned control record/seq")
            return False
        if "ts" not in event:
            event["ts"] = time.clock_gettime_ns(self._clock_raw)
        if "tid" not in event:
            ident = threading.get_ident()
            tid = self._tids.get(ident)
            if tid is None:
                tid = self._tids[ident] = threading.get_native_id()
            event["tid"] = tid
        sigs = self._signatures
        sid = event.get("sid")
        missing = sid is not None and sid not in sigs
        if not missing:
            osid = event.get("osid")
            missing = osid is not None and osid not in sigs
        lock = self._lock
        lock.acquire()
        try:
            if not self._accepting:
                return False
            event["seq"] = self._next_seq
            self._next_seq += 1
            reason = "writer_failure" if self._failure else ("missing_signature" if missing else None)
            if reason is None and self.enqueued-self._consumed_cache >= self.config.queue_capacity:
                self._consumed_cache = self._cons()[1]
                if self.enqueued-self._consumed_cache >= self.config.queue_capacity:
                    reason = "capacity"
            if reason:
                self._dropped[str(t if t is not None else "unknown")] += 1
                self._ipc_drops[reason] += 1
                return False
            batch = self._batch
            try:
                payload = marshal.dumps((0, event))
            except (ValueError, TypeError):
                self._dropped[str(t if t is not None else 'unknown')] += 1
                self._ipc_drops['serialization_rejected'] += 1
                return False
            if not self._reserve(payload):
                self._dropped[str(t if t is not None else 'unknown')] += 1
                self._ipc_drops['frame_too_large' if frame_size(payload) > self._size//2 else 'byte_capacity'] += 1
                return False
            batch.append(payload)
            self.enqueued += 1
            self._accepted_since_live += 1
            check_live = self._accepted_since_live >= 64
            if check_live:
                self._accepted_since_live = 0
            n = len(batch)
            if n == 1:
                self._batch_t0 = time.monotonic()
            if check_live:
                self._check_alive()
        finally:
            lock.release()
        if n >= BATCH_EVENTS or (n & 63 == 0 and time.monotonic()-self._batch_t0 >= BATCH_SECONDS):
            self._publish(None)
        return True

    def _flusher_main(self):
        poller = select.poll()
        poller.register(self._pidfd, select.POLLIN)
        poller.register(self._ack_r, select.POLLIN | select.POLLHUP)
        try:
            while not self._stop.is_set():
                poller.poll(int(FLUSHER_PERIOD*1000))
                if self._stop.is_set():
                    break
                self._check_alive()
                self._drain_ready()
                if self._failure:
                    break
                if self._batch and time.monotonic()-self._batch_t0 >= BATCH_SECONDS:
                    self._publish(None)
        except Exception as exc:
            self._fail("writer_internal_error:"+repr(exc))
        finally:
            # A waiter must also wake when the competing reader goes away.
            self._reply_event.set()

    def _publish(self, deadline):
        """Publish ordered prefixes, bounded both by count and serialized bytes."""
        try:
            return self._publish_impl(deadline)
        except Exception as exc:
            self._fail("writer_internal_error:"+repr(exc))
            return False

    def _reserve(self, payload, *, control=False):
        charge = frame_size(payload)
        if charge > self._size//2:
            return False
        maximum = charge if charge > self._max_frame else self._max_frame
        outstanding = self._wire_total-self._wire_consumed_cache
        # Cached credits are lower bounds on completed work. They may retain
        # charges longer, but can never admit extra bytes. Refresh only when
        # that conservative proof cannot establish both independent limits.
        if (outstanding+charge+3*maximum > INFLIGHT_BYTES or
                (control and self._control_total-self._control_consumed_cache+charge > CONTROL_BYTES)):
            if not self._map.closed:
                consumed, control_done = self._words[15], self._words[16]
                if consumed > self._wire_consumed_cache:self._wire_consumed_cache = consumed
                if control_done > self._control_consumed_cache:self._control_consumed_cache = control_done
            outstanding = self._wire_total-self._wire_consumed_cache
            if outstanding+charge+3*maximum > INFLIGHT_BYTES:
                return False
            if control and self._control_total-self._control_consumed_cache+charge > CONTROL_BYTES:
                return False
        self._wire_total += charge
        if control:self._control_total += charge
        self._max_frame = maximum
        reserved = outstanding+charge+3*maximum
        if reserved > self._wire_peak:self._wire_peak = reserved
        return True

    def _publish_impl(self, deadline):
        self._check_alive()
        if self._failure:
            return False
        if deadline is None and time.monotonic() < self._retry_at:
            return False
        acquired = self._pub_lock.acquire(False) if deadline is None else self._pub_lock.acquire(timeout=max(0., deadline-time.monotonic()))
        if not acquired:
            return False
        try:
            with self._lock:
                items, self._batch = self._batch, []
            i = 0
            try:
                while i < len(items):
                    if not self._write_frame(items[i], deadline):
                        self._retry_at = time.monotonic()+.001
                        return False
                    items[i] = None
                    i += 1
                return True
            finally:
                if i < len(items):
                    with self._lock:
                        self._batch = items[i:]+self._batch
        finally:
            self._pub_lock.release()

    def _write_frame(self, payload, deadline):
        size, need = self._size, (FRAME.size+len(payload)+7)//8*8
        if need > size // 2:
            raise ValueError("frame larger than half ring")
        while True:
            # A cached monotonic tail is a conservative lower bound. Refresh
            # only when it cannot prove space; never infer extra free bytes.
            tail = self._cons_cache[0]
            pos = self._head % size
            to_end = size-pos
            pad = to_end if to_end < need else 0
            if size-(self._head-tail) >= pad+need:
                break
            tail = self._cons()[0]
            if size-(self._head-tail) >= pad+need:
                break
            if deadline is None or time.monotonic() >= deadline or self._failure:
                self._ipc_drops["ring_deferred"] += 1   # not a drop: items retained
                return False
            time.sleep(.0005)
        m = self._map
        if pad:
            if to_end >= FRAME.size:
                self._words[(HDR+pos)//8+1] = PAD << 32
                self._words[(HDR+pos)//8] = self._frame+1
                self._frame += 1
            self._head += pad
            pos = 0
        m[HDR+pos+FRAME.size:HDR+pos+FRAME.size+len(payload)] = payload
        self._words[(HDR+pos)//8+1] = (len(payload) << 32) | zlib.crc32(payload)
        self._words[(HDR+pos)//8] = self._frame+1
        self._frame += 1
        self._head += need
        self._words[1] = self._frame
        self._words[0] = self._head
        return True

    # -- signatures / control -------------------------------------------
    def register_signature(self, signature):
        if os.getpid() != self._owner_pid or not self._accepting or self._failure:
            return False
        # Caller misuse is rejected before the internal failure boundary.
        from flagprof.store.sig import compute_sid
        try:
            definition = dict(signature)
            sid = compute_sid(definition["name"], definition["overload"], definition["role"], definition["tree"])
            definition.update(t="sig", sid=definition.get("sid", sid))
            if definition["sid"] != sid or definition.get("trunc", False):
                return False
            self._validator.decode(definition)
            payload = marshal.dumps((1, definition))
        except (ValueError, TypeError, KeyError, AttributeError):
            return False
        except Exception as exc:
            self._fail("writer_internal_error:"+repr(exc))
            return False
        if (FRAME.size+len(payload)+7)//8*8 > self._size//2:
            self._ipc_drops["signature_too_large"] += 1
            return False
        try:
            # Registration must become visible before emit can reference it.
            with self._lock:
                if not self._accepting or self._failure:
                    return False
                known = sid in self._signatures
                if not super().register_signature(signature):
                    return False
                if not known:
                    payload = marshal.dumps((1, self._signatures[sid]))
                    if not self._reserve(payload, control=True):
                        self._fail('control_byte_capacity')
                        return False
                    self._batch.append(payload)
                    if len(self._batch) == 1:
                        self._batch_t0 = time.monotonic()
            return True
        except Exception as exc:
            self._fail("writer_internal_error:"+repr(exc))
            return False

    def _barrier(self, op, deadline):
        with _held(self._lock, deadline):
            self._request += 1
            request = self._request
            state = dict(covered=self.enqueued, next_seq=self._next_seq, dropped=dict(self._dropped))
            self._barriers[request] = state
            if op == "close":
                self._closing_request = request
            if not self._batch:
                self._batch_t0 = time.monotonic()
            payload = marshal.dumps((2 if op == 'flush' else 3, request, self._next_seq, state['dropped']))
            if not self._reserve(payload, control=True):
                self._fail('control_byte_capacity')
                return None
            self._batch.append(payload)
        try:
            while self._batch:
                if self._failure or time.monotonic() >= deadline:
                    if not self._failure:
                        self._fail(op+"_timeout")
                    return None
                self._publish(deadline)
            try:
                while True:
                    reply = self._recv(max(0., deadline-time.monotonic()))
                    if reply.get("op") == "failed":
                        return None
                    if reply.get("request") != request:
                        self._fail("invalid_durable_ack")
                        return None
                    break
            except (TimeoutError, EOFError, OSError):
                self._finish_failure_messages()
                if not self._failure:
                    self._fail(op+"_timeout")
                return None
            if (reply.get("instance") != self._instance or reply.get("consumed") != state["covered"] or
                    reply.get("op") != ("durable" if op == "flush" else "closed")):
                self._fail("invalid_durable_ack")
                return None
            return reply
        finally:
            with _held(self._lock, deadline):
                self._barriers.pop(request, None)

    def _manage(self, op, timeout):
        deadline = time.monotonic()+timeout
        if os.getpid() != self._owner_pid:
            return False
        if not self._management.acquire(timeout=max(0., deadline-time.monotonic())):
            self._fail(op+"_management_timeout")
            self._finish_failure_messages()
            return False
        result = False
        try:
            result = self._management_operation(op, deadline)
        except Exception as exc:
            self._finish_failure_messages()
            self._fail("writer_internal_error:"+repr(exc))
        finally:
            self._finish_failure_messages()
            try:
                if op == "close" and not self._disposed:
                    self._kill(deadline)
            except Exception as exc:
                self._fail("writer_internal_error:"+repr(exc))
            finally:
                self._management.release()
        return result and not self._failure

    def _management_operation(self, op, deadline):
        if self._disposed:
            return self._closed and not self._failure
        self._finish_failure_messages()
        if op == "close":
            with _held(self._lock, deadline):
                self._accepting = False
                self._closing = True
        if self._failure:
            return False
        if op in ("flush", "close"):
            reply = self._barrier(op, deadline)
            if reply is None:
                return False
            if op == "close":
                self.consumer_cpu_seconds = reply.get("consumer_cpu_seconds", 0.)
                self._consumer.wait(max(0., deadline-time.monotonic()))
                self._reaped_monotonic = time.monotonic()
                self._finish_failure_messages()
                if not self._closed_ack or self._consumer.returncode != 0 or self._failure:
                    if not self._failure:
                        self._fail("consumer_lost")
                    return False
                self._closed = True
            return not self._failure
        if op == "park" and self._barrier("flush", deadline) is None:
            return False
        self._send({"op": op})
        reply = self._recv(max(0., deadline-time.monotonic()))
        if reply.get("op") != {"park":"parked","resume":"resumed"}[op]:
            self._fail("invalid_management_ack")
        return not self._failure

    def flush(self, timeout=10):
        return self._manage("flush", timeout)

    def _park(self):
        return self._manage("park", 10)

    def _resume(self):
        return self._manage("resume", 10)

    def close(self, timeout=10):
        return self._manage("close", timeout)

    def pending(self):
        try:
            written = self._cons()[2]
        except Exception as exc:
            self._fail("writer_internal_error:"+repr(exc))
            written = self._cons_cache[2]
        return max(0, self.enqueued-written)

    def metrics(self):
        try:
            data = super().metrics()
            self._last_base_metrics = data.copy()
        except Exception as exc:
            self._fail("writer_internal_error:"+repr(exc))
            data = getattr(self, "_last_base_metrics", {}).copy()
        try:
            tail, consumed, written, durable, _ = self._cons()
        except Exception as exc:
            self._fail("writer_internal_error:"+repr(exc))
            tail, consumed, written, durable, _ = self._cons_cache
        data.update(persisted=written, written=written, durable=durable, consumed=consumed,
                    transport_backend='memfd', seal_hash_backend='incremental',
                    inflight_wire_bytes=max(0, self._wire_total-self._wire_consumed_cache),
                    inflight_reserved_peak=self._wire_peak, transient_reserve_bytes=3*self._max_frame,
                    encoded_total_bytes=self._wire_total, processed_wire_bytes=self._wire_consumed_cache,
                    control_pending_bytes=max(0, self._control_total-self._control_consumed_cache),
                    pending=max(0, self.enqueued-written), ring_bytes=self._head-tail,
                    batch_items=len(self._batch), ipc_drop_reasons=dict(self._ipc_drops),
                    writer_failure=self._failure, consumer_cpu_seconds=self.consumer_cpu_seconds,
                    consumer_pid=self._consumer.pid if self._consumer else None,
                    consumer_terminal_monotonic=self._terminal_monotonic,
                    consumer_reaped_monotonic=self._reaped_monotonic,
                    cleanup_state=self._cleanup_state, cleanup_errors=list(self._cleanup_errors),
                    flusher_alive=getattr(self, "_flusher", None) is not None and self._flusher.is_alive(),
                    writer_failure_monotonic=self._failure_monotonic, torn_reads=self._torn_reads+getattr(self, "_consumer_torn_reads", 0),
                    consumer_buffered_records_peak=self._buffered_peak)
        return data
