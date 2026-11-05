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
