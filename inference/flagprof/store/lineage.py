"""Explicit analysis lineage; content reuse never implies a new execution."""
from pathlib import Path
import os
from .manifest import json_bytes, load_json


def register_lineage(run, *, execution_id=None, source_run_id=None, source_attempt_id=None, attempt_id=None):
    """Declare supplied identifiers once, without guessing from paths or time."""
    fields=dict(execution_id=execution_id,source_run_id=source_run_id,
                source_attempt_id=source_attempt_id,attempt_id=attempt_id)
    if any(v is not None and (not isinstance(v,str) or not v.strip()) for v in fields.values()):
        raise ValueError('lineage identifiers must be nonempty strings or None')
    run=Path(run)
    data=dict(format_version=1,run_id=load_json(run/'manifest.json')['run_id'],
              role='analysis',status='declared',execution_count_semantics='not_a_new_execution',**fields)
    with (run/'analysis-lineage.json').open('xb') as stream:
        stream.write(json_bytes(data)+b'\n');stream.flush();os.fsync(stream.fileno())
    fd=os.open(run,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return data


def read_lineage(run):
    run=Path(run)
    try:data=load_json(run/'analysis-lineage.json')
    except FileNotFoundError:
        return dict(status='unknown',reason='lineage_not_declared',execution_id=None,
                    source_run_id=None,source_attempt_id=None,attempt_id=None)
    if data.get('format_version')!=1 or data.get('run_id')!=load_json(run/'manifest.json')['run_id']:
        raise ValueError('lineage version/run identity mismatch')
    if data.get('role')!='analysis' or data.get('execution_count_semantics')!='not_a_new_execution':
        raise ValueError('invalid analysis lineage semantics')
    return data
