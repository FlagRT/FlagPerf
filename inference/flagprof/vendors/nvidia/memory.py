"""Kineto memory instant records. Totals are occupancy, Bytes is an event delta."""
from decimal import Decimal
from flagprof.offline.streaming import JSONStream


def artifacts(trace):
    return []  # The sealed trace itself carries allocator records.


def record(row,loc):
    if row.get('name')!='[memory]':return None
    args=row.get('args',{})
    if args.get('Device Type')!=1:
        return dict(loc=loc,status='excluded',reason='non_cuda_memory')
    values={key:int(args[field]) for field,key in (
        ('Total Allocated','allocated_bytes'),('Total Reserved','reserved_bytes'),('Bytes','delta_bytes')) if field in args}
    return dict(loc=loc,status='normalized',dev='cuda:'+str(args.get('Device Id','unknown')),
        component='PyTorch allocator',values=values,
        source_time_ns=int(Decimal(str(row['ts']))*1000) if row.get('ts') is not None else None,
        operation_status='allocator_record_has_no_device_stream',layer_status='explicit_project_call_missing')


def records(path):
    stream=JSONStream(path)
    try:
        for loc,(row,start,end) in enumerate(stream.records()):
            item=record(row,loc)
            if item is not None:yield item
    finally:stream.close()
