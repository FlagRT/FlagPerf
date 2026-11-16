"""Host PID/start-time checked process memory observations from vendor tables."""
from .contract import Observation
from .host_proc import process_stat


def observations(device, rows, targets, start, end, source, *, target_devices=None):
    by_pid={};duplicates=set()
    for pid,used in rows:
        if pid in by_pid:duplicates.add(pid)
        by_pid[pid]=used
    result=[Observation(device,{'process.count':len(rows)},start,end,source=source,
        detail=dict(process_table=[dict(host_pid=pid,bytes=used) for pid,used in rows],
                    attribution='only registered identities with matching start ticks'))]
    for pid,key in targets.items():
        if target_devices is not None and device not in target_devices.get(pid,()):continue
        try:
            if process_stat(pid)['start_ticks']!=key['proc_start_ticks']:continue
        except (OSError,ValueError):continue
        entity='proc:{pid_ns}:{pid}:{proc_start_ticks}'.format(**key)+'/'+device
        value=by_pid.get(pid)
        reason='read_error' if pid not in by_pid else 'invalid' if pid in duplicates else 'unsupported' if value==2**64-1 else None
        if reason:value=None
        result.append(Observation(entity,{'process.device_memory':value},start,end,
            {'process.device_memory':reason} if reason else {},source,
            detail=dict(device=device,worker_proc_key=key,host_pid=pid)))
    return result
