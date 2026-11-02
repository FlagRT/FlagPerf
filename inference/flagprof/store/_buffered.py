"""Internal shared disk sink. Public Writer defaults remain unchanged.

BufferedFile and BufferedWriter serve in-process importers and the proxy sink.
Only the disk owner calls these methods; receipts require a successful barrier.
"""
import hashlib
import json
import os
from pathlib import Path
import time

from .writer import Writer, _Barrier
from ._validation import CachedValidator

BATCH_RECORDS = 512
BATCH_BYTES = 256 << 10
BATCH_SECONDS = .01


class BufferedFile:
    def __init__(self, raw, owner):
        self.raw, self.owner = raw, owner
        self.name = raw.name
        self.parts = []
        self.size = self.events = self.position = self.hashed_bytes = 0
        self.started = time.monotonic()
        self.digest = hashlib.sha256()
        self.closed = self.failed = False

    def append(self, payload, event=False):
        if self.closed or self.failed:
            raise OSError('unavailable buffered file')
        if len(payload) > BATCH_BYTES:
            self.flush(force=True)
            self.position += len(payload)
            self._write(payload, int(event))
            return
        if self.size+len(payload) > BATCH_BYTES:
            self.flush(force=True)
        if not self.parts:
            self.started = time.monotonic()
        self.parts.append(payload)
        self.size += len(payload)
        self.events += int(event)
        self.position += len(payload)
        self.owner.buffered_bytes_peak = max(self.owner.buffered_bytes_peak, self.size)
        self.owner.buffered_records_peak = max(self.owner.buffered_records_peak, self.events)
        self.flush()

    def _write(self, payload, events):
        try:
            written = self.raw.write(payload)
            if type(written) is int and 0 < written <= len(payload):
                self.digest.update(memoryview(payload)[:written])
                self.hashed_bytes += written
            if type(written) is not int or written != len(payload):
                raise OSError('short batch write')
            self.raw.flush()
        except BaseException:
            self.failed = True
            raise
        self.owner.written_events += events

    def flush(self, force=False):
        if self.failed:
            raise OSError('failed buffered file')
        if self.parts and (force or len(self.parts) >= BATCH_RECORDS or
                self.size >= BATCH_BYTES or time.monotonic()-self.started >= BATCH_SECONDS):
            self._write(b''.join(self.parts), self.events)
            self.parts.clear()
            self.size = self.events = 0
            self.started = time.monotonic()

    def fileno(self):
        self.flush(force=True)
        return self.raw.fileno()

    def tell(self):
        return self.position

    def close(self):
        if not self.closed:
            try:
                if not self.failed:
                    self.flush(force=True)
            finally:
                self.raw.close()
                self.closed = True


class BufferedWriter(Writer):
    """Same queue/thread/receipt/config contract, shared bounded disk mechanics."""
    def __init__(self, *args, **kwargs):
        self._encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True,
                                        separators=(',', ':'), allow_nan=False)
        self.buffered_bytes_peak = self.buffered_records_peak = self.written_events = 0
        self.consumer_cpu_seconds = 0.
        super().__init__(*args, **kwargs)

    def _write_line(self, record, shard=None, handle=None):
        shard = self._shard if shard is None else shard
        raw = self._handle if handle is None else handle
        if not isinstance(raw, BufferedFile):
            target = BufferedFile(raw, self)
            if raw is not self._handle:
                raise RuntimeError('unregistered raw shard handle')
            self._handle = target
        else:
            target = raw
        target.append(self._encoder.encode(record).encode('utf-8')+b'\n', 'seq' in record)
        if record.get('t') == 'meta' and record.get('kind') == 'shard_header':
            target.flush(force=True)
        shard['records'] += 1
        if 'seq' in record:
            if shard['first_seq'] is None:
                shard['first_seq'] = record['seq']
            shard['last_seq'] = record['seq']
        now = time.monotonic()
        if now-self._last_fsync >= self.config.fsync_period_seconds:
            os.fsync(target.fileno())
            self._last_fsync = now

    def _drop_report(self, *, shard=None, handle=None):
        super()._drop_report(shard=shard, handle=handle)
        (self._handle if handle is None else handle).flush(force=True)

    def _seal(self, shard, handle):
        self._drop_report(shard=shard, handle=handle)
        handle.flush(force=True)
        os.fsync(handle.fileno())
        handle.close()
        if handle.failed or handle.hashed_bytes != handle.tell():
            raise OSError('incremental SHA byte count mismatch')
        with self._fragment_lock:
            shard['sha256'] = handle.digest.hexdigest()
            shard['sealed'] = True
        self._publish_fragment()
        if self.config.compress:
            self._compression_jobs.append(self._compressor.submit(self._compress, shard))

    def _compress(self, shard):
        before = len(self.errors)
        super()._compress(shard)
        if any(e.startswith('compression:') for e in self.errors[before:]):
            raise OSError('compression failed; writer cannot complete')

    def _disk_main(self):
        started = time.thread_time()
        self._validator = CachedValidator()
        try:
            self._open_shard()
            self._clock()
            self._ready.set()
            while True:
                with self._available:
                    if not self._queue and not self._closing:
                        self._available.wait(timeout=min(BATCH_SECONDS,
                            self.config.drop_period_seconds, self.config.clock_period_seconds))
                    event = self._queue.popleft() if self._queue else None
                    done = self._closing and not self._queue and event is None
                if done:
                    break
                if isinstance(event, _Barrier):
                    self._durable_barrier(event)
                    continue
                if event is not None:
                    receipt = None
                    if isinstance(event, tuple):
                        event, receipt = event
                    try:
                        written = self._persist_event(event)
                        if receipt is not None:
                            receipt.persisted = written
                            receipt.reason = None if written else 'schema_rejected'
                    except (ValueError, TypeError, KeyError) as error:
                        self.errors.append('record rejected: '+str(error))
                        with self._lock:
                            self._dropped[str(event.get('t', 'unknown'))] += 1
                        if receipt is not None:
                            receipt.reason = 'schema_rejected'
                self._handle.flush()
                now = time.monotonic()
                with self._lock:
                    changed = (dict(self._dropped) != self._reported_dropped or
                               self.sig_truncated != self._reported_sig_truncated)
                if changed and now-self._last_drop >= self.config.drop_period_seconds:
                    self._drop_report()
                    self._publish_fragment()
                if now-self._last_clock >= self.config.clock_period_seconds:
                    self._clock()
                if (self._handle.tell() >= self.config.shard_bytes or
                        (self.config.shard_records and self._shard['records'] >= self.config.shard_records) or
                        now-self._shard_start >= self.config.shard_period_seconds):
                    self._rotate()
            with self._lock:
                seq = self._next_seq
                self._next_seq += 1
            self._persist_event(dict(t='meta', kind='run_end',
                ts=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), seq=seq, data=dict(run_id=self.run_id)))
            self._clock()
            self._seal(self._shard, self._handle)
            self._compressor.shutdown(wait=True)
            for job in self._compression_jobs:
                job.result()
            with self._fragment_lock:
                self.fragment['state'] = 'closed'
            self._publish_fragment()
            self._closed = True
        except Exception as error:
            self.errors.append('writer I/O: '+repr(error))
            self._accepting = False
            self._compressor.shutdown(wait=False, cancel_futures=True)
            self._ready.set()
        finally:
            self.consumer_cpu_seconds = time.thread_time()-started
            try:
                if self._handle is not None and not self._handle.closed:
                    self._handle.close()
            except OSError as error:
                self.errors.append('writer cleanup: '+repr(error))
            if self.errors:
                try:
                    (self.run_dir/'logs'/f'writer-{self.writer_id}.log').write_text('\n'.join(self.errors)+'\n')
                except OSError:
                    pass
            self._done.set()

    def metrics(self):
        data = super().metrics()
        data.update(written=self.written_events, buffered_bytes_peak=self.buffered_bytes_peak,
                    buffered_records_peak=self.buffered_records_peak,
                    validation_cache_bytes=self._validator.cache_bytes,
                    consumer_cpu_seconds=self.consumer_cpu_seconds)
        return data
