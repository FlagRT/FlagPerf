"""Version-bound Kineto self-description producer; no SDK import."""
from ..evidence import FLAGS
KINETO_SOURCE='https://github.com/pytorch/pytorch/blob/v2.8.0/torch/profiler/profiler.py#L192-L232'

def metadata_configuration(document, collector, source):
    events=document.get('traceEvents') if isinstance(document,dict) else None
    complete=(isinstance(events,list) and all(isinstance(e,dict) for e in events)
        and type(document.get('schemaVersion')) is int and document['schemaVersion']==1
        and isinstance(document.get('deviceProperties'),list) and bool(document['deviceProperties'])
        and all(type(document.get(k)) is int for k in ('cupti_version','cuda_runtime_version','cuda_driver_version'))
        and isinstance(document.get('traceName'),str)
        and any(e.get('name')=='Record Window End' for e in events))
    verified=(bool(source) and collector.get('name')=='torch.profiler'
              and collector.get('version')=='2.8.0+cu126' and complete)
    values={}
    for key in FLAGS:
        value=document.get(key) if isinstance(document,dict) else None
        if verified:
            values[key]=True if type(value) is int and value==1 else False if key not in document else None
        else:values[key]=None
    return dict(status='available' if verified else 'unknown',values=values,source=source,
        interpretation=KINETO_SOURCE,reason='version-bound sealed exporter metadata' if verified else 'unverified metadata context')


def derive(document, collector, trace_source, artifacts, configuration, exporter, evidence):
    configuration['actual']=metadata_configuration(document,collector,trace_source)
    if isinstance(document,dict):
        exporter['schema_version']=document.get('schemaVersion')
        evidence['observations'].append(dict(source=trace_source,fields={k:document.get(k) for k in
            ('schemaVersion','cupti_version','cuda_runtime_version','cuda_driver_version','deviceProperties')}))
        evidence['clock']['sources'].append(dict(source=trace_source,baseTimeNanoseconds=document.get('baseTimeNanoseconds'),
            interpretation='producer source timestamp domain; no host_mono transform certified'))
        events=document.get('traceEvents',[])
        markers=[i for i,e in enumerate(events) if isinstance(e,dict) and e.get('name')=='Record Window End']
        evidence['observations'].append(dict(source=trace_source,record_window_end_locs=markers,traceName=document.get('traceName')))
        if collector.get('version')=='2.8.0+cu126' and not markers:
            evidence['negative_evidence'].append(dict(kind='end_marker_missing',source=trace_source))
        from .format import Taxonomy
        from ..chrome import stamp_events
        if isinstance(events,list) and all(isinstance(e,dict) for e in events):
            stamped=stamp_events(events);fmt=Taxonomy();context=fmt.context(stamped)
            launches={e.get('args',{}).get('correlation') for e in stamped if fmt.is_launch(e)
                      and type(e.get('args',{}).get('correlation')) is int}
            orphans=[dict(loc=e['_linker_loc'],correlation=e['args']['correlation']) for e in stamped
                if e.get('ph')=='X' and fmt.is_device(e,context) and type(e.get('args',{}).get('correlation')) is int
                and e['args']['correlation'] not in launches]
            if orphans:evidence['negative_evidence'].append(dict(kind='exact_id_missing_launch',source=trace_source,items=orphans,
                meaning='broken observed ID closure; does not identify the cause'))
    # No independent exporter version: do not alias the torch version.
