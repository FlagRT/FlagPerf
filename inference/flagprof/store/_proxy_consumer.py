"""Single registered physical disk owner; no SDK or experimental imports."""
from collections import Counter
import json
import marshal
import mmap
import os
from pathlib import Path
import resource
import select
import sys
import threading
import time
import zlib

from ._proxy_sink import ProxySink
from ._validation import CachedValidator
from .manifest import atomic_json, proc_key, process_alive
from .proxy import HDR, FRAME, PAD, frame_size, prefault_ring
from .writer import _Barrier


def main():
    identity = json.loads(Path(sys.argv[1]).read_text())
    ctrl_fd, ack_fd = int(sys.argv[2]), int(sys.argv[3])
    run = Path(identity['run_dir']); size = identity['size']
    lock = run/'writers'/('.proxy-'+str(identity['fragment']['writer_id']))
    # This permanent registration is never removed, including after death.
    with lock.open('x') as registered:
        json.dump(dict(process=proc_key(), owner=identity['proc'], instance=identity['instance']), registered)
        registered.flush(); os.fsync(registered.fileno())
    file = os.fdopen(identity['ring_fd'], 'r+b')
    memory = mmap.mmap(file.fileno(), HDR+size); words = memoryview(memory).cast('Q')
    prefault_ring(words)
    sink = ProxySink(identity); sink._validator = CachedValidator()
    head_cache = torn_reads = tail = expected = consumed = wire_done = control_done = 0
    rejects = Counter(); errors = []; parked = lost = False
    visibility_retries = 0; invisible_since = None; visibility_max = 0.
    instance = identity['instance']; ctrl_buf = b''

    def head():
        nonlocal head_cache, torn_reads
        value = words[0]
        if value < head_cache:
            torn_reads += 1; value = words[0]
        head_cache = max(value, head_cache)
        return head_cache

    def publish():
        for i, value in enumerate((tail, consumed, sink._written, sink._durable, time.monotonic_ns())):
            words[8+i] = value
        words[13] = sink.consumer_buffered_records_peak
        words[14] = torn_reads
        words[15] = wire_done
        words[16] = control_done

    def ack(value):
        payload = json.dumps(value).encode()+b'\n'
        if len(payload) > 65536: raise ValueError('ack size budget')
        if os.write(ack_fd, payload) != len(payload): raise OSError('short ack write')

    def commands(timeout=0):
        nonlocal ctrl_buf
        if select.select([ctrl_fd], [], [], timeout)[0]:
            chunk = os.read(ctrl_fd, 65536)
            if not chunk: return None
            ctrl_buf += chunk
        if len(ctrl_buf) > 65536: raise ValueError('control size budget')
        out = []
        while b'\n' in ctrl_buf:
            line, ctrl_buf = ctrl_buf.split(b'\n', 1); out.append(json.loads(line))
        return out

    try:
        ack(dict(op='ready', proc_key=proc_key(), instance=instance))
        deadline = time.monotonic()+10
        while True:
            cmds = commands(.05)
            if cmds is None or process_alive(identity['proc']) is False: return 2
            if any(c.get('op') == 'start' for c in cmds): break
            if time.monotonic() >= deadline: raise TimeoutError('owner did not commit start')
        if process_alive(identity['proc']) is False: return 2
        sink._open_shard(); sink._clock()
        ack(dict(op='started', instance=instance))
        last_check = 0.
        while True:
            now = time.monotonic()
            if now-last_check >= .02:
                last_check = now
                if process_alive(identity['proc']) is False:
                    lost, parked = True, False; sink._parent_lost = True
                cmds = commands()
                if cmds is None:
                    lost, parked = True, False; sink._parent_lost = True
                else:
                    for cmd in cmds:
                        if cmd['op'] == 'park': parked = True; ack(dict(op='parked'))
                        elif cmd['op'] == 'resume': parked = False; ack(dict(op='resumed'))
                        else: raise ValueError('unknown management command')
            item = None; payload = None; need = 0
            current = head()
            if not parked and tail < current:
                pos = tail % size; remaining = size-pos
                if remaining < FRAME.size:
                    tail += remaining; publish(); continue
                fid, value = words[(HDR+pos)//8], words[(HDR+pos)//8+1]
                length, crc = value >> 32, value & 0xFFFFFFFF
                visible = fid == expected+1
                if visible and length == PAD:
                    tail += remaining; expected += 1; publish(); continue
                if visible and (length < 1 or length > size//2-FRAME.size or FRAME.size+length > remaining or FRAME.size+length > current-tail):
                    visible = False
                if visible:
                    payload = memory[HDR+pos+FRAME.size:HDR+pos+FRAME.size+length]
                    visible = zlib.crc32(payload) == crc
                if not visible:
                    visibility_retries += 1; invisible_since = invisible_since or now
                    visibility_max = max(visibility_max, now-invisible_since)
                    if now-invisible_since > 1: raise ValueError('ring frame never became visible')
                else:
                    invisible_since = None; need = frame_size(payload)
                    item = marshal.loads(payload)
            if item is not None:
                kind = item[0]; closing = None
                if kind == 0:
                    event = item[1]; consumed += 1
                    sink._next_seq = max(sink._next_seq, event['seq']+1)
                    ok = False
                    if not any(event[k] not in sink._signatures for k in ('sid','osid') if k in event):
                        try: ok = sink._persist_event(event)
                        except (ValueError, TypeError, KeyError) as error:
                            errors.append('record rejected: '+str(error))
                    if not ok: rejects[str(event.get('t','unknown'))] += 1
                    if (sink._handle.tell() >= sink.config.shard_bytes or
                        (sink.config.shard_records and sink._shard['records'] >= sink.config.shard_records) or
                        now-sink._shard_start >= sink.config.shard_period_seconds): sink._rotate()
                elif kind == 1:
                    definition = item[1]
                    sink.sig_truncated += int(bool(definition.get('trunc')) and definition['sid'] not in sink._signatures)
                    sink._signatures[definition['sid']] = definition
                elif kind in (2,3):
                    _, request, next_seq, dropped = item
                    sink._next_seq = max(sink._next_seq, next_seq)
                    sink._dropped = Counter(dropped)+rejects
                    sink._durable_barrier(_Barrier(threading.Event()))
                    if kind == 3:
                        sink._persist_event(dict(t='meta',kind='run_end',seq=sink._next_seq,
                            ts=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),data=dict(run_id=sink.run_id)))
                        sink._next_seq += 1; sink._clock(); sink._seal(sink._shard,sink._handle)
                        sink._compressor.shutdown(wait=True)
                        for job in sink._compression_jobs: job.result()
                        sink._durable = sink._written; sink.fragment['state'] = 'closed'; sink._publish_fragment()
                    closing = dict(op='closed' if kind == 3 else 'durable',request=request,instance=instance,consumed=consumed)
                else: raise ValueError('unknown frame item')
                # All object references and the snapshot are released before
                # returned credits become visible. Disk batching stays bounded.
                payload = item = None
                if kind == 0: event = None
                if kind == 1: definition = None
                wire_done += need
                if kind != 0: control_done += need
                tail += need; expected += 1
                if kind != 0 or consumed % 512 == 0 or tail >= head(): publish()
                if closing:
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    closing['consumer_cpu_seconds'] = usage.ru_utime+usage.ru_stime
                    if kind == 3:
                        # No payload reader remains after the final barrier.
                        # Retire volatile payload pages in both mappings before
                        # either process exits. The shared status header remains
                        # valid for the final ack, metrics and A5 observer.
                        file.truncate(HDR)
                    ack(closing)
                    if kind == 3: return 0
            sink._handle.flush()
            now = time.monotonic()
            if now-sink._last_fsync >= sink.config.fsync_period_seconds:
                os.fsync(sink._handle.fileno()); sink._durable = sink._written; sink._last_fsync = now
            if item is None and now-sink._shard_start >= sink.config.shard_period_seconds: sink._rotate()
            if not need: publish()
            if lost and not parked and tail >= head():
                sink._dropped += rejects; sink._clock(); sink._seal(sink._shard,sink._handle)
                sink._compressor.shutdown(wait=True)
                for job in sink._compression_jobs: job.result()
                sink.fragment['state'] = 'active'; sink._publish_fragment(); return 3
            if not need: time.sleep(.0005)
    except Exception as error:
        errors.append(repr(error))
        try: ack(dict(op='failed',instance=instance,error=repr(error)))
        except OSError: pass
        return 1
    finally:
        try:
            atomic_json(Path(identity['attachment_dir'])/'consumer-result.json',dict(
                errors=errors+sink.errors, consumed=consumed,lost=lost,written=sink._written,durable=sink._durable,
                processed_wire_bytes=wire_done,control_wire_bytes=control_done,rejects=dict(rejects),
                visibility_retries=visibility_retries,torn_reads=torn_reads,visibility_max_wait_seconds=visibility_max,
                consumer_buffered_records_peak=sink.consumer_buffered_records_peak,affinity=sorted(os.sched_getaffinity(0))))
        finally:
            if sink._handle is not None and not sink._handle.closed: sink._handle.close()
            sink._compressor.shutdown(wait=True)
            words.release(); memory.close(); file.close()


if __name__ == '__main__':
    raise SystemExit(main())
