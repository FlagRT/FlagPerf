# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Build a versioned device projection during the existing normalization pass."""
import importlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .disk import Sink, key, pack, unpack
from flagprof.vendors.chrome import nanoseconds
from .clock_bridge import collect, request_windows


def capture_context(directory, capture):
    from .limits import read_json
    from flagprof.store import StoreView
    directory=Path(directory);meta=capture['metadata'];out=dict(schema='device-analysis-input/2',calls=[],
        requests=[],request_steps=[],clock_samples=[],project_writers=[],
        capture_occurrence=capture.get('occurrence'),capture_process=meta.get('process'),
        capture_normal_close=bool(capture.get('normal_close') and meta.get('target_returncode')==0
                                  and not meta.get('closed_before_target_return') and not meta.get('errors')))
    collection=meta.get('collection_run');argv=[];identity={};config={};steps=[]
    if collection:
        run=directory/collection
        manifest=read_json(run/'manifest.json');argv=manifest.get('identity',{}).get('argv',[])
        out['source_run_id']=manifest['run_id']
        from flagprof.store.manifest import writer_fragments
        out['project_writers']=[{k:w[k] for k in ('writer_id','rank','proc_key','clk')} for w in writer_fragments(run)
                                if all(k in w for k in ('writer_id','rank','proc_key','clk'))]
        with StoreView(run) as view:
            attachments=view.probe_status()['attachments']
            configs=[a.get('effective_config') for a in attachments if a.get('effective_config')]
            identities=[a.get('oracle_identity') for a in attachments if a.get('oracle_identity')]
            config=configs[0] if len(configs)==1 else {}
            identity=identities[0] if len(identities)==1 else {}
            for stored in view.iter_events(types=('comm','span','meta')):
                e=stored.record
                collect(stored,out)
                if e['t']=='comm':
                    extra=e.get('ext',{}).get('communication',{})
                    if extra.get('marker'):
                        out['calls'].append(dict(marker=extra['marker'],group=e['group'],group_seq=e['group_seq'],
                            op=e['op'],rank=e.get('rank'),generation=extra.get('generation',{})))
                elif e.get('name')=='generation_step_complete':steps.append(e['attrs'])
    def option(name,default=None):
        try:return argv[argv.index('--'+name)+1]
        except (ValueError,IndexError):return default
    def number(name,default=None):
        value=option(name,default)
        try:return int(value) if value is not None else None
        except ValueError:return None
    out.update(window_identity=dict(warmup=number('warmup'),forwards=number('forwards'),
        repeats=number('repeats',1),profiler='on',dispatch=config.get('dispatch'),window_kind='official_capture',
        task=option('task'),max_new_tokens=number('max-new-tokens'),
        profile_memory=meta.get('requested_configuration',{}).get('profile_memory')),
        workload_identity=identity,steps=steps,metadata_sources=['collection manifest argv','attachment state','generation_step_complete'],
        clock=importlib.import_module('flagprof.vendors.'+capture['vendor']+'.analysis').clock(directory,capture))
    return out


class Builder:
    def __init__(self,db,vendor,context,uid,inputs):
        self.db=db;self.vendor=vendor;self.context=context;self.uid=uid;self.inputs=inputs or {}
        self.adapter=importlib.import_module('flagprof.vendors.'+vendor+'.analysis')
        self.calls={}
        for call in self.inputs.get('calls',[]):self.calls.setdefault(call['marker'],[]).append(call)
        self.primary=set();self.markers=[];self.explicit_windows=list(self.inputs.get('clock',{}).get('capture_windows',[]));self.counts=Counter()
        db.c.execute('create table device_analysis(id integer primary key, payload blob)')

    def observe(self,i,event,record):
        facts=self.adapter.activity(event,self.context,self.db.get('labels',event.get('pid'),''))
        if event.get('ph')=='X' and event.get('name','').startswith('PyTorch Profiler'):
            if record['ts'] is not None and record['dur'] is not None:
                self.explicit_windows.append(dict(kind='official_capture',start_ns=record['ts'],end_ns=record['ts']+record['dur'],
                    boundary_source=dict(trace_uid=self.uid,loc=record['loc'])))
        if facts is None:return
        row=dict(facts,source=dict(trace_uid=self.uid,loc=record['loc']),source_clock=self.uid,
                 name=record['name'],start_ns=record['ts'],duration_ns=record['dur'])
        if facts['representation']=='collective':self.primary.add(facts['collective'])
        self.counts[facts['category']]+=1
        if facts['step_marker'] and record['ts'] is not None and record['dur'] is not None:
            self.markers.append(dict(end_ns=record['ts']+record['dur'],dev=facts['dev'],source=row['source']))
        self.db.enqueue('insert into device_analysis values(?,?)',(i,pack(row)))

    def communication(self,record):
        """Flow ID joins first, then the launch's same-thread lexical scope."""
        from flagprof.vendors.native_common import launch_interval
        candidates={};proof=[];launch_count=0;unresolved=False
        for rule,launch,edges in self.db.edge_hits(record,self.vendor):
            launch_count+=1;found=False
            for owner in self.db.contains(launch_interval(launch),'scope'):
                for call in self.calls.get(owner['name'],[]):
                    found=True;k=(call['group'],call['group_seq'],call.get('rank'))
                    candidates[k]=call
                    if len(proof)<8:proof.append(dict(rule=rule,launch={'trace_uid':self.uid,'loc':launch['loc']},
                        marker={'trace_uid':self.uid,'loc':owner['loc']},flow=[{'trace_uid':self.uid,'loc':e['loc']} for e in edges]))
            unresolved |= not found
        status='linked' if len(candidates)==1 and not unresolved else 'ambiguous' if candidates else 'explicit_link_unavailable'
        return dict(status=status,reason=None if status=='linked' else 'conflicting_or_missing_marker' if candidates else
                    'flow_has_no_registered_marker' if launch_count else 'explicit_device_key_missing',
                    call=next(iter(candidates.values())) if status=='linked' else None,evidence=proof)

    def finish(self,path):
        self.db.commit();sink=Sink(path);counts=Counter();unknown=Counter();markers=defaultdict(list)
        for m in self.markers:markers[m['dev']].append(m)
        for i,payload in self.db.c.execute('select id,payload from device_analysis order by id'):
            row=unpack(payload);row['canonical']=not(row['representation']=='auxiliary_kernel' and row['collective'] in self.primary)
            if row['category']=='comm' and row['canonical']:
                record=unpack(self.db.c.execute('select payload from norm where id=?',(i,)).fetchone()[0])
                row['communication']=self.communication(record)
            counts['raw_records']+=1;counts['canonical_records']+=row['canonical']
            if row['category']=='unknown':unknown[str(row['task_type'])]+=1
            sink.row(row)
        windows=list(self.explicit_windows);steps=self.inputs.get('steps',[])
        requests=defaultdict(list)
        for step in steps:requests[step['request_id']].append(step)
        # Ordinal correspondence is admitted only for the frozen one-request
        # greedy capture, with a complete one-marker-per-generated-step sequence.
        for dev,items in markers.items():
            items.sort(key=lambda m:m['end_ns'])
            if len(requests)==1 and len(items)==len(steps) and len(items)>=2:
                request=next(iter(requests));ordered=requests[request]
                if [s['generation_step'] for s in ordered]!=list(range(len(items))):continue
                for index in range(1,len(items)):
                    windows.append(dict(kind='decode_adjacent_selection',request_id=request,step=index,dev=dev,
                        start_ns=items[index-1]['end_ns'],end_ns=items[index]['end_ns'],
                        boundary_source=[items[index-1]['source'],items[index]['source']],
                        boundary_rule='one_greedy_selection_per_step_in_single_request'))
        bridges,mappings,request_rows=request_windows(self.inputs,windows)
        windows.extend(request_rows)
        for window in windows:
            window.setdefault('window_identity',dict(self.inputs.get('window_identity',{}),window_kind=window['kind'],
                boundary_rule=window.get('boundary_rule',window['kind']+'/v1')))
        context={k:v for k,v in self.inputs.items() if k not in ('calls','steps','requests','request_steps','clock_samples','project_writers')}
        clock=dict(context.get('clock',{}),bridges=bridges,
                   project_host_alignment='available' if any(b['status']=='available' for b in bridges) else 'unavailable')
        clock['reason']='mapping evidence and synchronization error bounds are reported separately'
        clock['request_boundary_mapping']='available' if request_rows else 'unavailable'
        request_status=('available' if all(m['status']=='complete' for m in mappings) else 'partial') if request_rows else 'unavailable'
        context.update(schema='device-analysis/2',status='available',source_clock=self.uid,clock=clock,
            counts=dict(counts),raw_categories=dict(self.counts),unknown_types=dict(unknown),windows=windows,
            window_status='available' if windows else 'unavailable',
            window_reason=None if windows else 'explicit_boundaries_missing_or_marker_count_mismatch',
            representation_rule='collective representation preferred; matching auxiliary class retained but excluded from canonical totals',
            request_mappings=mappings,request_window_status=request_status,
            request_window_reason=None if request_status=='available' else 'see request_mappings' if mappings else 'project_request_spans_missing')
        return sink.close(),context
