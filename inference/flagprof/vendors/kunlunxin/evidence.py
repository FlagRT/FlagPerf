# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""P800 exporter observations; missing configuration fields stay unknown."""
from flagprof.vendors.evidence import FLAGS
def derive(document,collector,trace_source,artifacts,configuration,exporter,evidence):
    valid=(isinstance(document,dict) and collector.get('name')=='torch.profiler' and collector.get('version')=='2.9.0+cu129'
           and document.get('schemaVersion')==1 and isinstance(document.get('traceEvents'),list))
    values={k: True if valid and type(document.get(k)) is int and document[k]==1 else None for k in FLAGS}
    configuration['actual']=dict(status='available' if valid else 'unknown',values=values,source=trace_source,
        reason='explicit exporter flags only; absent flags and capture completeness are not inferred')
    exporter.update(schema_version=document.get('schemaVersion') if isinstance(document,dict) else None)
    evidence['observations'].append(dict(source=trace_source,trace_metadata={k:document.get(k) for k in ('schemaVersion','deviceProperties','baseTimeNanoseconds')}) if isinstance(document,dict) else dict(source=trace_source))

def final_configuration(directory, measured, trace, trace_values, end_count, raw_manifest):
    from flagprof.offline.checkpoints import fileinfo
    from flagprof.vendors.evidence import FLAGS
    source = dict(path='sealed-trace/data/trace.json', sha256=fileinfo(trace)['sha256'])
    version = measured.get('collector', {}).get('version')
    exporter = dict(source=source, version=None, schema_version=trace_values.get('schemaVersion'))
    supported = version == '2.9.0+cu129' and trace_values.get('schemaVersion') == 1
    missing = None
    if not supported:
        return {}, [], exporter
    values = {k: True if type(trace_values.get(k)) is int and trace_values[k] == 1
              else missing if k not in trace_values else None for k in FLAGS}
    return values, [dict(source, interpretation='P800 exporter '+version, values=values)], exporter
