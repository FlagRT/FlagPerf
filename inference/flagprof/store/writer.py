"""One process owns its fragment, queue, and shard files.

emit() never waits for queue space or disk I/O. Closing/flush are explicit
management operations and may wait. Submitted nested payloads transfer ownership:
callers must not mutate them after emit; signatures are copied on registration.
"""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol
import copy
import gzip
import hashlib
import os
import threading
import time

from .manifest import allocate_writer, atomic_json, json_bytes, load_json, proc_key
from .schema import SCHEMA_VERSION, SchemaValidator, is_control


@dataclass(frozen=True)
class WriterConfig:
    queue_capacity: int = 65536
    signature_budget_bytes: int = 64 << 20
    truncated_signature_budget_bytes: int = 16 << 20
    shard_bytes: int = 64 << 20
    shard_records: int = 0  # Test/operational override; 0 disables record-count rotation.
    shard_period_seconds: float = 60.0
    drop_period_seconds: float = 10.0
    clock_period_seconds: float = 60.0
    fsync_period_seconds: float = 1.0
    heartbeat_period_ns: int = 1_000_000_000
    compress: bool = True
    gzip_level: int = 1

    def __post_init__(self) -> None:
        if self.queue_capacity < 1 or self.shard_bytes < 1 or self.shard_records < 0:
            raise ValueError("invalid queue/shard limit")
        if min(self.signature_budget_bytes, self.truncated_signature_budget_bytes) < 0:
            raise ValueError("signature budgets cannot be negative")
        if min(self.shard_period_seconds, self.drop_period_seconds, self.clock_period_seconds, self.fsync_period_seconds) <= 0:
            raise ValueError("periods must be positive")
        if self.heartbeat_period_ns <= 0 or not 0 <= self.gzip_level <= 9:
            raise ValueError("invalid heartbeat period/gzip level")


class EventWriter(Protocol):
    def emit(self, record: Mapping[str, Any]) -> bool: ...
    def register_signature(self, signature: Mapping[str, Any]) -> bool: ...
    def alloc_corr(self) -> int: ...


@dataclass
class EmitReceipt:
    """Importer receipt; persisted becomes authoritative after flush succeeds."""
    seq: int | None = None
    accepted: bool = False
    persisted: bool = False
    reason: str | None = "not_accepting"


@dataclass
class _Barrier:
    done: threading.Event


_PROCESS_ATTACH_LOCK = threading.Lock()
_PROCESS_ATTACH: set[tuple[str, int]] = set()


def _after_fork() -> None:
    global _PROCESS_ATTACH_LOCK, _PROCESS_ATTACH
    _PROCESS_ATTACH_LOCK = threading.Lock()
    _PROCESS_ATTACH = set()


os.register_at_fork(after_in_child=_after_fork)


class Writer:
    @property
    def batch_limit(self) -> int:
        """Valid events accepted from an empty, exclusively owned queue.

        Offline callers establish that precondition with a successful flush,
        including before the first batch (run_start also occupies the queue).
        """
        return self.config.queue_capacity

    def _process_identity(self):
        return proc_key()

    def _allocate_fragment(self, *, node_index, kind, rank):
        return allocate_writer(self.run_dir, node_index=node_index, kind=kind, rank=rank, key=self.proc)

    def __init__(self, run_dir: str | Path, *, kind: str = "probe", rank: int = 0,
                 node_index: int | None = None, probe_id: str = "user", config: WriterConfig | None = None) -> None:
        self.run_dir = Path(run_dir)
        self._owner_pid = os.getpid()
        self.config = config or WriterConfig()
        manifest = load_json(self.run_dir / "manifest.json")
        self.run_id = manifest["run_id"]
        self.proc = self._process_identity()
        # Only exception to the no-singleton rule: process attach ownership.
        with _PROCESS_ATTACH_LOCK:
            if any(run == self.run_id for run, _ in _PROCESS_ATTACH):
                raise RuntimeError("this process already attached to this run")
            self.fragment = self._allocate_fragment(node_index=node_index, kind=kind, rank=rank)
            self.writer_id = self.fragment["writer_id"]
            _PROCESS_ATTACH.add((self.run_id, self.writer_id))
        self.defaults = {"pid": os.getpid(), "rank": rank, "p": probe_id, "clk": self.fragment["clk"]}
        self._fragment_path = self.run_dir / "writers" / f"w-{self.writer_id}.json"
        self._fragment_lock = threading.Lock()
        self._lock = threading.Lock()
        self._available = threading.Condition(self._lock)
        self._queue: deque[Any] = deque()
        self._next_seq = 0
        self._corr_counter = 0
        self._dropped: Counter[str] = Counter()
        self._sig_lock = threading.Lock()
        self._signatures: dict[str, dict[str, Any]] = {}
        self._signature_bytes = 0
        self._truncated_bytes = 0
        self.sig_truncated = 0
        self._accepting = True
        self._closing = False
        self._closed = False
        self._close_lock = threading.Lock()
        self._ready = threading.Event()
        self._done = threading.Event()
        self._compressor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"flagprof-gzip-{self.writer_id}")
        self._compression_jobs = []
        self._validator = SchemaValidator()
        self.errors: list[str] = []
        self.enqueued = 0
        self.persisted = 0
        self.peak_backlog = 0
        self._shard: dict[str, Any] | None = None
        self._handle = None
        self._shard_sids: set[str] = set()
        self._shard_start = time.monotonic()
        self._last_fsync = self._shard_start
        self._last_drop = self._shard_start
        self._last_clock = self._shard_start
        self._reported_dropped: dict[str, int] = {}
        self._reported_sig_truncated = 0
        self.emit({"t": "meta", "kind": "run_start", "data": {"run_id": self.run_id, "proc_key": self.proc,
                  "heartbeat_period_ns": self.config.heartbeat_period_ns}})
        self._thread = threading.Thread(target=self._disk_main, name=f"flagprof-writer-{self.writer_id}", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self.errors:
            self.close()
            raise OSError(self.errors[-1])

    def alloc_corr(self) -> int:
        if os.getpid() != self._owner_pid:
            raise RuntimeError("inherited writer cannot allocate corr; attach a new writer after fork")
        with self._lock:
            if self._corr_counter >= (1 << 48) - 1:
                raise OverflowError("corr counter exhausted; ids cannot wrap")
            self._corr_counter += 1
            return (self.writer_id << 48) | self._corr_counter

    def register_signature(self, signature: Mapping[str, Any]) -> bool:
        """Register before emitting a reference; definitions never use the queue."""
        if os.getpid() != self._owner_pid:
            return False
        from .sig import compute_sid
        definition = copy.deepcopy(dict(signature))
        definition["t"] = "sig"
        try:
            sid = compute_sid(definition["name"], definition["overload"], definition["role"], definition["tree"])
            if definition.get("sid", sid) != sid or definition.get("trunc", False):
                raise ValueError("registration requires the complete, canonical signature")
            definition["sid"] = sid
            self._validator.decode(definition)
        except (KeyError, ValueError, TypeError) as exc:
            self.errors.append("signature registration: " + str(exc))
            return False
        encoded_size = len(json_bytes(definition))
        with self._sig_lock:
            previous = self._signatures.get(sid)
            if previous is not None:
                if not previous.get("trunc") and previous != definition:
                    self.errors.append("signature hash collision: " + sid)
                    return False
                return True
            if self._signature_bytes + encoded_size <= self.config.signature_budget_bytes:
                self._signature_bytes += encoded_size
                self._signatures[sid] = definition
                return True
            abbreviated = {"t": "sig", "sid": sid, "role": definition["role"], "name": definition["name"],
                           "overload": definition["overload"], "tree": [], "trunc": True}
            abbreviated_size = len(json_bytes(abbreviated))
            if self._truncated_bytes + abbreviated_size > self.config.truncated_signature_budget_bytes:
                return False
            self._truncated_bytes += abbreviated_size
            self.sig_truncated += 1
            self._signatures[sid] = abbreviated
            return True

    def emit(self, record: Mapping[str, Any], *, _receipt: EmitReceipt | None = None) -> bool:
        """Allocate seq and enqueue/drop at one linearization point."""
        if os.getpid() != self._owner_pid:
            return False
        event = dict(record)
        if is_control(event) or "seq" in event:
            # Control records have one owner: the disk thread.
            self.errors.append("emit rejected writer-owned control record/seq")
            if _receipt is not None:
                _receipt.reason = "writer_owned_record"
            return False
        event.setdefault("ts", time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
        event.setdefault("tid", threading.get_native_id())
        missing_signature = False
        if "sid" in event or "osid" in event:
            with self._sig_lock:
                missing_signature = any(event[k] not in self._signatures for k in ("sid", "osid") if k in event)
        with self._available:
            if not self._accepting:
                return False
            event["seq"] = self._next_seq
            self._next_seq += 1
            if _receipt is not None:
                _receipt.seq = event["seq"]
            if missing_signature or len(self._queue) >= self.config.queue_capacity:
                self._dropped[str(event.get("t", "unknown"))] += 1
                # Wake the writer for periodic drop visibility even without surviving events.
                self._available.notify()
                if _receipt is not None:
                    _receipt.reason = "missing_signature" if missing_signature else "queue_full"
                return False
            was_empty = not self._queue
            self._queue.append(event if _receipt is None else (event, _receipt))
            if _receipt is not None:
                _receipt.accepted = True
                _receipt.reason = "pending_barrier"
            self.enqueued += 1
            self.peak_backlog = max(self.peak_backlog, len(self._queue))
            if was_empty:
                self._available.notify()
            return True

    def emit_with_receipt(self, record: Mapping[str, Any]) -> EmitReceipt:
        receipt = EmitReceipt()
        self.emit(record, _receipt=receipt)
        return receipt

    def _publish_fragment(self) -> None:
        with self._fragment_lock:
            with self._lock:
                self.fragment["next_seq"] = self._next_seq
                self.fragment["dropped"] = dict(sorted(self._dropped.items()))
            atomic_json(self._fragment_path, self.fragment)

    def _write_line(self, record: Mapping[str, Any], shard: dict[str, Any] | None = None, handle: Any = None) -> None:
        shard = self._shard if shard is None else shard
        handle = self._handle if handle is None else handle
        payload = json_bytes(record) + b"\n"
        handle.write(payload)
        handle.flush()
        shard["records"] += 1
        if "seq" in record:
            if shard["first_seq"] is None:
                shard["first_seq"] = record["seq"]
            shard["last_seq"] = record["seq"]
        now = time.monotonic()
        if now - self._last_fsync >= self.config.fsync_period_seconds:
            os.fsync(handle.fileno())
            self._last_fsync = now

    def _meta(self, kind: str, data: Mapping[str, Any]) -> dict[str, Any]:
        return {"t": "meta", "ts": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), "clk": self.defaults["clk"],
                "kind": kind, "data": dict(data)}

    def _open_shard(self) -> None:
        index = len(self.fragment["shards"])
        relative = Path("events") / f"w-{self.writer_id}" / f"{index:06d}.jsonl"
        (self.run_dir / relative).parent.mkdir(exist_ok=True)
        self._handle = (self.run_dir / relative).open("xb")
        self._shard = {"index": index, "path": str(relative), "codec": "none", "records": 0,
                       "first_seq": None, "last_seq": None, "sealed": False}
        self._shard_sids = set()
        header = self._meta("shard_header", {"writer_id": self.writer_id, "shard_index": index, "defaults": self.defaults})
        header["v"] = SCHEMA_VERSION
        self._write_line(header)
        with self._fragment_lock:
            self.fragment["shards"].append(self._shard)
        self._shard_start = time.monotonic()
        self._publish_fragment()

    def _drop_report(self, *, shard: dict[str, Any] | None = None, handle: Any = None) -> None:
        with self._lock:
            dropped = dict(sorted(self._dropped.items()))
            watermark = self._next_seq
        self._write_line(self._meta("drop_report", {"dropped": dropped, "next_seq": watermark,
                         "sig_truncated": self.sig_truncated}), shard, handle)
        self._reported_dropped = dropped
        self._reported_sig_truncated = self.sig_truncated
        self._last_drop = time.monotonic()

    def _clock(self) -> None:
        self._write_line(self._meta("clock", {"mono_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), "wall_ns": time.time_ns()}))
        self._last_clock = time.monotonic()

    def _seal(self, shard: dict[str, Any], handle: Any) -> None:
        self._drop_report(shard=shard, handle=handle)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        # Step 3 commits record count while path/codec still name the plaintext.
        with self._fragment_lock:
            with (self.run_dir / shard["path"]).open("rb") as source:
                digest = hashlib.sha256()
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
            shard["sha256"] = digest.hexdigest()
            shard["sealed"] = True
        self._publish_fragment()
        if self.config.compress:
            self._compression_jobs.append(self._compressor.submit(self._compress, shard))

    def _rotate(self) -> None:
        old_shard, old_handle = self._shard, self._handle
        # Step 1: new header and active fragment are visible before old is sealed.
        self._open_shard()
        self._seal(old_shard, old_handle)

    def _compress(self, shard: dict[str, Any]) -> None:
        original = self.run_dir / shard["path"]
        compressed = original.with_name(original.name + ".gz")
        temporary = compressed.with_name(compressed.name + ".tmp")
        try:
            with original.open("rb") as source, temporary.open("xb") as target:
                with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0, compresslevel=self.config.gzip_level) as zipped:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        zipped.write(block)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, compressed)
            self._sync_directory(compressed.parent)
            with self._fragment_lock:
                shard["path"] = str(compressed.relative_to(self.run_dir))
                shard["codec"] = "gzip"
                shard["sha256"] = hashlib.sha256(compressed.read_bytes()).hexdigest()
            # Step 4: update path and codec together, then unlink plaintext.
            self._publish_fragment()
            original.unlink()
            self._sync_directory(original.parent)
        except Exception as exc:
            self.errors.append("compression: " + repr(exc))
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _sync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _persist_event(self, event: dict[str, Any]) -> bool:
        decoded = self._validator.decode(event, self.defaults)
        if decoded is None:
            with self._lock:
                self._dropped[str(event.get("t", "unknown"))] += 1
            return False
        for field in ("sid", "osid"):
            sid = event.get(field)
            if sid is not None and sid not in self._shard_sids:
                with self._sig_lock:
                    definition = self._signatures[sid]
                self._write_line(definition)
                self._shard_sids.add(sid)
        # Keep shard defaults on the header; explicit event overrides survive.
        encoded = {k: v for k, v in decoded.items() if k not in self.defaults or v != self.defaults[k]}
        self._write_line(encoded)
        self.persisted += 1
        return True

    def _durable_barrier(self, barrier: _Barrier) -> None:
        self._drop_report()
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._sync_directory(Path(self._handle.name).parent)
        self._sync_directory(self.run_dir / "events")
        self._publish_fragment()
        barrier.done.set()

    def _disk_main(self) -> None:
        try:
            self._open_shard()
            self._clock()
            self._ready.set()
            while True:
                with self._available:
                    if not self._queue and not self._closing:
                        self._available.wait(timeout=min(self.config.drop_period_seconds, self.config.clock_period_seconds, 0.1))
                    event = self._queue.popleft() if self._queue else None
                    done = self._closing and not self._queue and event is None
                if done:
                    break
                if event is not None:
                    if isinstance(event, _Barrier):
                        self._durable_barrier(event)
                        continue
                    receipt = None
                    if isinstance(event, tuple):
                        event, receipt = event
                    try:
                        written = self._persist_event(event)
                        if receipt is not None:
                            receipt.persisted = written
                            receipt.reason = None if written else "schema_rejected"
                    except (ValueError, TypeError, KeyError) as exc:
                        self.errors.append("record rejected: " + str(exc))
                        with self._lock:
                            self._dropped[str(event.get("t", "unknown"))] += 1
                        if receipt is not None:
                            receipt.reason = "schema_rejected"
                now = time.monotonic()
                with self._lock:
                    drops_changed = (dict(self._dropped) != self._reported_dropped or
                                     self.sig_truncated != self._reported_sig_truncated)
                if drops_changed and now - self._last_drop >= self.config.drop_period_seconds:
                    self._drop_report()
                    self._publish_fragment()
                if now - self._last_clock >= self.config.clock_period_seconds:
                    self._clock()
                if (self._handle.tell() >= self.config.shard_bytes or
                    (self.config.shard_records and self._shard["records"] >= self.config.shard_records) or
                    now - self._shard_start >= self.config.shard_period_seconds):
                    self._rotate()
            with self._lock:
                end_seq = self._next_seq
                self._next_seq += 1
            self._persist_event({"t": "meta", "kind": "run_end", "ts": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                                 "seq": end_seq, "data": {"run_id": self.run_id}})
            self._clock()
            self._seal(self._shard, self._handle)
            self._compressor.shutdown(wait=True)
            for job in self._compression_jobs:
                job.result()
            with self._fragment_lock:
                self.fragment["state"] = "closed"
            self._publish_fragment()
            self._closed = True
        except Exception as exc:
            self.errors.append("writer I/O: " + repr(exc))
            self._accepting = False
            self._compressor.shutdown(wait=False, cancel_futures=True)
            self._ready.set()
        finally:
            if self._handle is not None and not self._handle.closed:
                self._handle.close()
            if self.errors:
                log = self.run_dir / "logs" / f"writer-{self.writer_id}.log"
                try:
                    log.write_text("\n".join(self.errors) + "\n", encoding="utf-8")
                except OSError:
                    pass  # A full/unwritable filesystem must not kill the target.
            self._done.set()

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait for a disk-thread fsync barrier, including an in-flight record.

        Barriers bypass queue capacity and consume no event seq. Successful
        acknowledgement means data and its fragment/directory are durable.
        """
        if os.getpid() != self._owner_pid:
            return False
        barrier = _Barrier(threading.Event())
        with self._available:
            if self._done.is_set():
                return self._closed
            if not self._closing:
                self._queue.append(barrier)
                self._available.notify()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if barrier.done.wait(min(0.05, max(0, deadline - time.monotonic()))):
                return True
            if self._done.is_set():
                return self._closed
        return False

    def close(self) -> None:
        if os.getpid() != self._owner_pid:
            return
        with self._close_lock:
            if self._closed:
                return
            with self._available:
                self._accepting = False
                self._closing = True
                self._available.notify_all()
            self._thread.join()

    def metrics(self) -> dict[str, Any]:
        if os.getpid() != self._owner_pid:
            return {"errors": ["inherited writer is unavailable; attach a new writer after fork"]}
        with self._lock:
            return {"enqueued": self.enqueued, "persisted": self.persisted, "peak_backlog": self.peak_backlog,
                    "dropped": dict(sorted(self._dropped.items())), "next_seq": self._next_seq,
                    "sig_truncated": self.sig_truncated, "errors": list(self.errors)}

    def __enter__(self) -> Writer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
