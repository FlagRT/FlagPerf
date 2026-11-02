"""Official torch_npu CSV memory fields, measured in binary MB/KB by the SDK."""
import csv
from decimal import Decimal


def artifacts(trace):
    return [p for name in ('memory_record.csv', 'operator_memory.csv')
            if (p := trace.parent/name).is_file()]


def number(value, scale=1):
    if value is None or str(value).strip() in ('', 'N/A', 'None'):
        return None
    return int(Decimal(str(value).strip())*scale)


def records(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'Component', 'Device Type'} if path.name == 'memory_record.csv' else {'Name', 'Size(KB)', 'Device Type'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError('unsupported official memory CSV columns')
        for line, row in enumerate(reader, 2):
            dev = row['Device Type'].strip()
            if not dev.startswith('NPU:'):
                yield dict(loc=line, status='excluded', reason='non_device_memory')
                continue
            if path.name == 'memory_record.csv':
                values = {key: value for field, key in (
                    ('Total Allocated(MB)', 'allocated_bytes'), ('Total Reserved(MB)', 'reserved_bytes'),
                    ('Total Active(MB)', 'active_bytes'))
                    if (value := number(row.get(field), 1024**2)) is not None}
                yield dict(loc=line, status='normalized', dev='npu:'+dev.split(':')[1],
                           component=row['Component'], source_time_ns=number(row.get('Timestamp(us)'), 1000), values=values)
            else:
                size = number(row.get('Size(KB)'), 1024)
                yield dict(loc=line, status='normalized', dev='npu:'+dev.split(':')[1], component='operator',
                    source_time_ns=number(row.get('Allocation Time(us)'), 1000),
                    values={k:v for k,v in dict(allocation_bytes=size,
                        release_source_time_ns=number(row.get('Release Time(us)'), 1000)).items() if v is not None},
                    native_owner=row['Name'], release_observed=number(row.get('Release Time(us)'),1000) is not None,
                    operation_status='stream_pointer_not_a_validated_device_stream',
                    layer_status='explicit_project_call_missing')
