# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Bounded persisted native evidence, completion and source-name lookup."""
import json
from pathlib import Path
import sqlite3
import tempfile
from .analysis_read import ResourceLimit
from ..offline.bounded_verify import Document
from ..offline.streaming import JSONStream, sha_file, stat_identity


def rows(view, uid, *, limits=None, work=None):
    limits=limits or {};limit=limits.get('record_bytes',16 << 20)
    if work:
        Path(work).mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='native-',dir=work) as directory:
        doc=Document(Path(directory)/'native.sqlite')
        raw=None
        try:
            pages=max(1,limits.get('scratch_bytes',4 << 30)//doc.c.execute('pragma page_size').fetchone()[0])
            doc.c.execute('pragma max_page_count='+str(pages))
            descriptor_path=view.run/'traces'/(uid+'.json')
            bindings=[(descriptor_path,stat_identity(descriptor_path))]
            root,descriptor_sha,_=doc.parse(descriptor_path)
            descriptor=doc.get(root)
            if descriptor['trace_uid']!=uid or descriptor['sha256'][:16]!=uid:
                raise ValueError('native descriptor identity mismatch')
            derived=descriptor.get('derived',{})
            if 'native' not in derived:
                return
            source_state=dict(import_status='legacy',format_status='unavailable',coverage_scope='unavailable')
            completion=descriptor.get('completion_path')
            if completion:
                completion_path=view._path(completion,'imports');bindings.append((completion_path,stat_identity(completion_path)))
                comp_root,_,_=doc.parse(completion_path)
                comp=doc.get(comp_root)
                if comp.get('trace_uid')!=uid or comp.get('descriptor_sha256')!=descriptor_sha or comp.get('writer_id')!=descriptor['writer_id'] or comp.get('derived_sha256')!={k:v['sha256'] for k,v in derived.items()}:
                    raise ValueError('native completion identity/SHA mismatch')
                source_state['import_status']=comp['status']
            checked={}
            for kind in ('native','summary','index'):
                if kind not in derived:
                    continue
                info=derived[kind];path=view._path(info['path'],'derived')
                before=stat_identity(path)
                if path.stat().st_size!=info['bytes'] or sha_file(path)!=info['sha256']:
                    raise ValueError('derived SHA mismatch: '+kind)
                checked[kind]=(path,before)
            if 'summary' in checked:
                path=checked['summary'][0]
                if path.stat().st_size>limit:
                    raise ResourceLimit('resource_limit: native summary')
                summary=json.loads(path.read_bytes())
                source_state.update(format_status=summary.get('format_assessment',{}).get('status','unavailable'),
                    coverage_scope=summary.get('coverage_scope','unavailable'))
            if descriptor.get('raw_path') and 'index' in checked:
                raw_path=view._path(descriptor['raw_path'],'vendor_raw')
                raw_before=stat_identity(raw_path)
                if sha_file(raw_path)!=descriptor['sha256']:
                    raise ValueError('native raw SHA mismatch')
                raw=raw_path.open('rb')
                index_root,_,_=doc.parse(checked['index'][0]);index=doc.get(index_root)
                if index['trace_uid']!=uid or index['raw_sha256']!=descriptor['sha256'] or index['raw_bytes']!=raw_before[2]:
                    raise ValueError('native evidence index identity mismatch')
                doc.c.execute('create table offsets(loc text primary key,start integer,end integer,ordinal integer)')
                for child in doc.children(index_root,'entries'):
                    entry=doc.get(child)
                    doc.c.execute('insert into offsets values(?,?,?,?)',(str(entry['loc']),entry['start'],entry['end'],entry['ordinal']))
            def enrich(row):
                result=dict(row,source_state=source_state)
                owner=row.get('owner')
                if raw is not None and owner:
                    entry=doc.c.execute('select start,end,ordinal from offsets where loc=?',(str(owner['loc']),)).fetchone()
                    if entry is None:
                        raise ValueError('native owner evidence is missing')
                    start,end,ordinal=entry
                    if not 0<=start<end<=raw_before[2] or end-start>limit:
                        raise ResourceLimit('resource_limit: owner source record')
                    raw.seek(start);event=json.loads(raw.read(end-start))
                    if str(event.get('_linker_loc',ordinal))!=str(owner['loc']):
                        raise ValueError('native owner source loc mismatch')
                    result['owner_name']=event.get('name')
                return result
            path=checked['native'][0]
            if path.suffix=='.jsonl':
                with path.open('rb') as stream:
                    while line:=stream.readline(limit+1):
                        if len(line)>limit:
                            raise ResourceLimit('resource_limit: native record_bytes')
                        yield enrich(json.loads(line))
            else:
                parser=JSONStream(path,limit=limit)
                try:
                    for row,_,_ in parser.records():
                        yield enrich(row)
                finally:
                    parser.close()
            for path,before in checked.values():
                if stat_identity(path)!=before:
                    raise ValueError('native derived source changed')
            if raw is not None and stat_identity(raw_path)!=raw_before:
                raise ValueError('native raw changed')
            if 'summary' in checked and source_state['import_status']=='complete':
                bindings.append(checked['summary'])
                if any(stat_identity(path)!=before for path,before in bindings):raise ValueError('analysis metadata changed during read')
                view._analysis_verified[uid]=dict(trace_uid=uid,bindings=bindings,
                    context=summary.get('device_analysis',dict(schema='device-analysis/1',status='unavailable',reason='projection_not_recorded')),
                    intervals=derived.get('analysis_intervals'))
        except sqlite3.Error as error:
            raise ResourceLimit('resource_limit: native scratch: '+str(error)) from error
        finally:
            if raw is not None:
                raw.close()
            doc.c.close()
