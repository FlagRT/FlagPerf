"""Sealed profiler_info and SQLite self-description producer."""
import json
import sqlite3
from ..evidence import FLAGS,_consensus

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
