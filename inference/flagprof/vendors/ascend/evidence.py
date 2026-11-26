# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Sealed profiler_info and SQLite self-description producer."""
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from .capture import analyse_trace
from ..evidence import FLAGS,_consensus


def profiler_databases(directory):
    return [p for p in directory.glob('ascend_pytorch_profiler*.db')
            if re.fullmatch(r'ascend_pytorch_profiler(?:_-?[0-9]+)?\.db', p.name)]


def stage_format_evidence(trace, destination, remaining):
    """Keep the database of this export only, within the existing seal budget."""
    from flagprof.offline.bundle import copy_checked
    files = profiler_databases(trace.parent)
    if len(files) == 1:
        db = files[0]
        if not db.is_symlink() and db.is_file() and 0 < db.stat().st_size <= remaining:
            copy_checked(db, destination/db.name, max_bytes=remaining)


def schema_version(path):
    try:
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1', uri=True)) as db:
            rows = db.execute("SELECT CASE WHEN typeof(value)='text' AND length(value) BETWEEN 1 AND 128 "
                              "THEN value END FROM META_DATA WHERE name='SCHEMA_VERSION' LIMIT 2").fetchall()
        return rows[0][0] if len(rows) == 1 else None
    except sqlite3.DatabaseError:
        return None

def derive(document, collector, trace_source, artifacts, configuration, exporter, evidence):
    configs=[];versions=[];schemas=[]
    for name,(file,source) in sorted(artifacts.items()):
        if file.name.startswith('profiler_info') and file.suffix=='.json':
            data=json.loads(file.read_bytes());common=data.get('config',{}).get('common_config',{})
            extra=data.get('config',{}).get('experimental_config',{})
            values={k:common.get(k) for k in FLAGS}
            values.update(profiler_level=extra.get('_profiler_level'),data_simplification=extra.get('_data_simplification'),
                activities=[str(x).split('.')[-1] for x in common.get('activities',[])],
                schedule={k:common.get('schedule',{}).get(k) for k in ('wait','warmup','active','repeat')})
            configs.append(dict(source=source,values=values,sdk_version=data.get('cann_version')))
            versions.append(dict(collector=data.get('torch_npu_version'),exporter=data.get('cann_version'),source=source))
            evidence['clock']['sources'].append(dict(source=source,start_info=data.get('start_info'),end_info=data.get('end_info')))
        if file.name=='ascend_pytorch_profiler.db':
            try:
                with sqlite3.connect(file.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
                    rows=db.execute("SELECT value FROM META_DATA WHERE name='SCHEMA_VERSION'").fetchall()
                    value=rows[0][0] if len(rows)==1 else None
                    schemas.append(dict(value=value,source=source,field='META_DATA[SCHEMA_VERSION]'))
                    try:
                        times=db.execute('SELECT * FROM SESSION_TIME_INFO').fetchall()
                        evidence['clock']['sources'].append(dict(source=source,SESSION_TIME_INFO=times))
                    except sqlite3.Error as error:
                        evidence['clock']['sources'].append(dict(source=source,status='unavailable',reason=str(error)))
            except sqlite3.Error as error:
                schemas.append(dict(value=None,source=source,error=str(error)))
    keys=FLAGS+('profiler_level','data_simplification','activities','schedule')
    actual={k:_consensus([r['values'].get(k) for r in configs]) for k in keys}
    configuration['actual']=dict(status='available' if configs else 'unknown',values=actual,sources=configs,
        reason='sealed profiler_info fields; conflicting or missing values remain null')
    exporter.update(source=[r['source'] for r in versions+schemas] or trace_source,
                    version=_consensus([r['exporter'] for r in versions]),schema_version=_consensus([r['value'] for r in schemas]))
    observed=_consensus([r['collector'] for r in versions])
    if versions and (observed is None or collector.get('version')!=observed):
        evidence['conflicts'].append(dict(field='collector.version',session=collector.get('version'),artifacts=versions))
        collector['version']=None
    if len(configs)>1 and any(v['values']!=configs[0]['values'] for v in configs):
        evidence['conflicts'].append(dict(field='actual',sources=configs))
    evidence['observations'].append(dict(versions=versions,schemas=schemas,
        end_markers=[source for name,(_,source) in artifacts.items() if name.endswith('.done')]))


def final_configuration(directory, measured, trace, trace_values, end_count, raw_manifest):
    from flagprof.offline.checkpoints import fileinfo
    from flagprof.offline.limits import read_json
    actual, sources, versions = {}, [], []
    exporter = dict(status='unknown')
    for name,info in raw_manifest['files'].items():
        if Path(name).name.startswith('profiler_info') and Path(name).suffix=='.json':
            if len(sources)>=16:raise ValueError('official configuration cardinality budget')
            config=read_json(directory/'sealed-raw'/name)
            common=config.get('config',{}).get('common_config',{})
            extra=config.get('config',{}).get('experimental_config',{})
            value={k:common.get(k) for k in FLAGS}
            value.update(activities=[str(x).split('.')[-1] for x in common.get('activities',[])],
                         profiler_level=extra.get('_profiler_level'),data_simplification=extra.get('_data_simplification'))
            sources.append(dict(path='sealed-raw/'+name,sha256=info['sha256'],values=value,
                                sdk_version=config.get('cann_version'), sdk_version_field='cann_version'))
            versions.append(config.get('cann_version'))
    for key in (*FLAGS,'activities','profiler_level','data_simplification'):
        values=[row['values'][key] for row in sources]
        actual[key]=values[0] if values and all(type(v) is type(values[0]) and v==values[0] for v in values) else None
    if sources:
        exporter=dict(source=[dict(path=r['path'],sha256=r['sha256']) for r in sources],
                      version=versions[0] if all(v==versions[0] for v in versions) else None,schema_version=None)
    databases = profiler_databases(trace.parent)
    if len(databases) == 1:
        db = databases[0]
        exporter.setdefault('source', []).append(dict(path=str(db.relative_to(directory)),
            sha256=fileinfo(db)['sha256'], field='META_DATA[SCHEMA_VERSION]'))
        exporter['schema_version'] = schema_version(db)
    return actual, sources, exporter
