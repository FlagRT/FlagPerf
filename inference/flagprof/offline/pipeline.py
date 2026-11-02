"""Isolated full importer prototype, using the unmodified production Writer."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
from .disk import DB,Rows,JSONFile,ArrayFile,JSONStream,Sink,key,pack,unpack,encoded,file_rows,tokens
from .metadata import ingest_context,new_scope_state,observe_scope,build_context,assess,ObjectRows
from . import relations
from . import fast_relations
fast_relations.install()
from flagprof.correlate.linker import ref_sort
from flagprof.vendors.chrome import _adapt, _call_map
from flagprof.vendors.format import load_adapter
from flagprof.vendors.prepare import source_identity
from flagprof.vendors.fingerprint import parser_identity
from flagprof.vendors.native_common import ScopeConfig
from flagprof.vendors.disposition import UnrepresentableError
from flagprof.store.schema import SchemaValidator
from flagprof.store.manifest import json_bytes
from flagprof.vendors.import_common import _sync_dir

CATS=('compute','comm','memcpy','unknown')


class Prepared:
    def __init__(self,directory):self.directory=Path(directory);self.fields={};self.files={};self.phases=[];self.subphases=[]
    def mark(self,name,start):self.phases.append(dict(name=name,wall_s=time.perf_counter()-start,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    def detail(self,name,start):self.subphases.append(dict(name=name,wall_s=time.perf_counter()-start))
    def store(self,name,value):
        s=Sink(self.directory/(name+'.json'));s.row(value);self.files[name]=s.close();self.fields[name]=JSONFile(s.path)
    def arrays(self,name):return Rows(lambda:file_rows(self.directory/(name+'.jsonl')))


def prepare(source,directory,*,vendor,registry=None,kind=None,calls=None,marker_prefix=None,scope_config=None,identities=None,taxonomy_version=None,profile_memory=False,analysis_input=None,scan_checkpoint=None,resume_scan=None,max_derived_record_bytes=4<<20,max_metadata_bytes=4<<20):
    from .checkpoints import identity as source_stat
    initial_source_stat=source_stat(source)
    directory=Path(directory);directory.mkdir();p=Prepared(directory);start=time.perf_counter()
    adapter=load_adapter(vendor);original_tax=adapter.device_taxonomy(taxonomy_version)
    calls=[] if calls is None else calls;encoded(calls);kind=kind or 'all'
    if kind not in ('compute','communication','all'):raise ValueError('invalid normalization kind')
    if marker_prefix is not None and (not isinstance(marker_prefix,str) or not marker_prefix):raise ValueError('invalid marker_prefix')
    registry=adapter.default_registry() if registry is None else registry
    config=(scope_config or ScopeConfig()).resolved(importlib.import_module('flagprof.vendors.'+vendor+'.native_rules').ANNOTATION_CATEGORIES)
    from .disk import scratch_path
    if resume_scan:
        shutil.copyfile(resume_scan,scratch_path(directory/'scratch.sqlite'))
    db=DB(directory/'scratch.sqlite',existing=bool(resume_scan));db.source=Path(source).open('rb')
    if not resume_scan:
        stream=JSONStream(source)
        try:
            batch=[];batchbytes=0
            for i,(raw,a,b) in enumerate(stream.records()):
                e=dict(raw);e.setdefault('_linker_loc',i);loc=str(e['_linker_loc']);stem=loc.split('+dup',1)[0];int(stem)
                ingest_context(db,original_tax,i,e)
                payload=pack(raw)
                # One source record may exceed the batch budget but remains
                # bounded by JSONStream's existing per-record cap.
                if batch and (len(batch)>=128 or batchbytes+len(payload)>1<<20):
                    db.c.executemany('INSERT INTO raw VALUES(?,?,?,?,?,?,?,?,?)',batch);batch.clear();batchbytes=0
                batch.append((i,loc,stem,loc[len(stem):],type(e['_linker_loc']).__name__,payload,None,a,b));batchbytes+=len(payload)
            if batch:db.c.executemany('INSERT INTO raw VALUES(?,?,?,?,?,?,?,?,?)',batch)
            del batch
            digest=stream.sha.hexdigest()
        finally:stream.close()
        db.commit()
        if scan_checkpoint:scan_checkpoint(db)
    else:
        from .disk import streaming
        digest=streaming.sha_file(source)
    uid=digest[:16];nraw=db.c.execute('SELECT count(*) FROM raw').fetchone()[0]
    p.mark('scan_reused' if resume_scan else 'scan',start)
    detail_start=time.perf_counter()
    tax=build_context(db,original_tax)
    call_map=_call_map(calls)
    unbound={c['marker']:c for c in calls if 'corr' not in c and {'marker','group','group_seq','op','rank'}<=c.keys()}
    p.files['unbound_calls']=None
    from .device_analysis import Builder
    analysis=Builder(db,vendor,tax.ctx,uid,analysis_input)
    rawcats=Counter();scope_state=new_scope_state()
    memory_sink=Sink(directory/'memory.jsonl') if profile_memory and hasattr(adapter,'memory_record') else None
    for i,e in enumerate(db.raws()):
        if memory_sink is not None:
            memory_row=adapter.memory_record(e,i)
            if memory_row is not None:memory_sink.row(memory_row)
        observe_scope(db,tax,vendor,e,scope_state)
        category=tax.category(e,tax.ctx) if tax.is_device(e,tax.ctx) else None
        if category is not None:
            rawcats[category]+=1;task=tax.task_type(e);taskkey=str(task) if task is not None else '<absent>'
            db.put('task_count',taskkey,db.get('task_count',taskkey,0)+1)
            db.put('inventory',i,dict(trace_uid=uid,loc=e['_linker_loc'],category=category,name=e.get('name',''),task_type=task,dur=__import__('flagprof.vendors.chrome',fromlist=['nanoseconds']).nanoseconds(e.get('dur'))))
        if vendor=='nvidia':reason='not_normalized' if category else 'scope_annotation' if e.get('cat') in tax.contract['roles']['allowed_non_device_categories'] else 'host_or_metadata'
        else:
            from flagprof.vendors.ascend.format import queue_key_valid
            reason='unrecognized_device_activity' if category else 'invalid_queue_key' if e.get('name','').startswith(('Enqueue','Dequeue')) and not queue_key_valid(e.get('args',{}).get('correlation_id')) else 'host_or_metadata'
        db.put('ledger',i,dict(trace_uid=uid,loc=e['_linker_loc'],category=category,disposition='unknown' if category is not None else 'source_only',reason=reason,outputs=[]))
        trace=_adapt([e],call_map,trace_uid=uid,kind=kind,marker_prefix=marker_prefix,taxonomy=tax,context=tax.ctx)
        r=trace.records[0]
        analysis.observe(i,e,r)
        if r['t']=='op' and r['name'] in unbound:
            r['t']='host';r.pop('corr',None);db.put('unbound',i,dict(trace_uid=uid,loc=r['loc'],reason='no_project_corr_in_source',call=unbound[r['name']]))
        ext=r['ext'];is_module=ext['chrome.ph']=='X' and (ext['chrome.cat'] in config.module_categories or r['name'].startswith(config.module_prefixes))
        ann=ext['chrome.ph']=='X' and (is_module or ext['chrome.cat'] in config.annotation_categories or r['name'].startswith(config.annotation_prefixes))
        call=ext['chrome.ph']=='X' and not ann and ext['chrome.cat']=='cpu_op'
        selected=bool(trace.devices)
        db.enqueue('INSERT INTO norm VALUES(?,?,?,?,?,?,?,?,?,?,?)',(i,pack(r),'device' if selected else r['t'],r.get('category'),key(r['host_pid']),key((r['host_pid'],r['tid'])),str(r['ts']) if r['ts'] is not None else None,str(r['dur']) if r['dur'] is not None else None,int(call),int(ann),int(is_module)))
        if r['ts'] is not None:
            end=r['ts']+(r['dur'] or 0)
            db.add_interval(i,r['ts'],end,key((r['host_pid'],r['tid'])))
        args=ext['chrome.args'];cat=ext['chrome.cat'];ph=ext['chrome.ph']
        if r['t']=='launch' or call:db.put('native_pid',r['host_pid'],r['host_pid'])
        flowcat='ac2g' if vendor=='nvidia' else 'HostToDevice'
        if cat==flowcat and ext['chrome.id'] is not None:
            if ph=='s' and r['t']=='launch':db.addkey('flow_start',(cat,str(ext['chrome.id'])),i)
            if ph=='f':
                db.addkey('flow_endpoint',(r['pid'],r['tid'],r['ts']),i);db.addkey('endpoint_flow',(cat,str(ext['chrome.id'])),i)
        if selected:
            db.addkey('device_endpoint',(r['pid'],r['tid'],r['ts']),i)
            field='correlation' if vendor=='nvidia' else 'connection_id'
            if args.get(field) is not None:db.addkey('device_launch',str(args[field]),i)
        if vendor=='nvidia':
            if cat in ('cuda_runtime','cuda_driver') and args.get('correlation') is not None:db.addkey('launch_key',str(args['correlation']),i)
            if args.get('External id') is not None:
                if cat in ('cpu_op','user_annotation'):db.addkey('project_external',(r['host_pid'],str(args['External id'])),i)
                if call:db.addkey('native_external',(r['host_pid'],str(args['External id'])),i)
        else:
            if r['name']=='Node@launch' and args.get('connection_id') is not None:db.addkey('launch_key',str(args['connection_id']),i)
            if r['name'].startswith('Enqueue') and queue_key_valid(args.get('correlation_id')):db.addkey('enqueue',(r['host_pid'],args['correlation_id']),i)
            if r['name'].startswith('Dequeue') and ph=='X':db.addkey('dequeue',0,i)
    db.commit();p.detail('context_normalize',detail_start)
    detail_start=time.perf_counter();db.build_intervals();p.detail('interval_index',detail_start)
    detail_start=time.perf_counter();db.build_device_edges();p.detail('device_edges',detail_start)
    p.mark('context_normalize_index',start);detail_start=time.perf_counter()
    pid_count=db.c.execute("SELECT count(*) FROM stage WHERE kind='native_pid'").fetchone()[0]
    single=(pid_count==1,next(db.stage('native_pid'),None))
    sinks={name:Sink(directory/(name+'.jsonl'),max_row_bytes=(16<<20) if name in ('events','devices') else max_derived_record_bytes) for name in ('events','devices','native','ledger','inventory','unbound_calls')}
    project_stats={cat:Counter() for cat in CATS};native_stats={cat:Counter() for cat in CATS};native_states=Counter();native_reasons=Counter()
    count=0;validator=SchemaValidator()
    for r in db.nodes("n.kind='device'"):
        i=db.rid(r);link=json.loads(encoded(relations.project(db,r,vendor,registry)))
        device=dict(r,link=link);event=None
        try:
            fields=adapter.device_fields(device,[dict(name='process_labels',pid=r['pid'],args={'labels':db.get('labels',r['pid'],'')})] if vendor=='ascend' else [])
        except UnrepresentableError as error:
            row=db.get('ledger',i);row.update(disposition='unrepresentable',reason='missing_identity',missing_fields=list(error.fields));db.put('ledger',i,row)
        else:
            dev,st=fields.pop('dev'),fields.pop('stream');db.put('emitted_device',dev,dev)
            ext=dict(device['ext'],**{'chrome.trace_uid':uid,'chrome.loc':r['loc'],'chrome.raw_record':db.raw(i,original=True),'chrome.clock_valid':False})
            event=dict(t='kernel',name=device['name'],dev=dev,stream=st,ktype=device['category'],ts=None,dur=device['dur'],clk='unmapped:'+uid,link=link,ext=ext)
            if device['category']=='memcpy' and {'kind','bytes'}<=fields.keys():event.update(t='memop',kind=fields['kind'],bytes=fields['bytes']);event.pop('name');event.pop('ktype')
            else:event.update({k:fields[k] for k in ('grid','block','regs','smem') if k in fields})
            encoded(event);validator.decode(dict(event,seq=0));sinks['events'].row(event);sinks['devices'].row(device)
            row=db.get('ledger',i);row.update(disposition='imported',reason=None,outputs=[count]);db.put('ledger',i,row);count+=1
            s=project_stats[r['category']];s['E']+=1
            if link['status']=='not_applicable':s['X']+=1
            else:
                s['A']+=1;s['dur']+=r['dur'] or 0;s['missing']+=r['dur'] is None;s['L']+=link['status']=='linked'
                if link['status']=='linked':s['linked_dur']+=r['dur'] or 0
                s['level_'+link['level']]+=1;s['leveldur_'+link['level']]+=r['dur'] or 0
        relation=relations.native(db,r,vendor,registry,single);sinks['native'].row(relation)
        native_states[relation['state']]+=1;native_reasons[str(relation['reason'])]+=1
        if event is not None and link['status']!='not_applicable':
            s=native_stats[r['category']];s['E']+=1;s['dur']+=r['dur'] or 0;s['missing']+=r['dur'] is None
            s['owned']+=relation['state']=='owned';s['owned_dur']+=(r['dur'] or 0) if relation['state']=='owned' else 0
            s['scoped']+=relation['module_scope'] is not None;s['scoped_dur']+=(r['dur'] or 0) if relation['module_scope'] is not None else 0
    p.files['analysis_intervals'],analysis_context=analysis.finish(directory/'analysis_intervals.jsonl')
    p.mark('all_relations_device_output',start)
    p.detail('relations_device_output',detail_start)
    categories={};dispositions=Counter();catdisp={cat:Counter() for cat in CATS}
    for row in db.ordered_stage('ledger'):
        sinks['ledger'].row(row);dispositions[row['disposition']]+=1
        if row['category'] in CATS:catdisp[row['category']][row['disposition']]+=1
    for row in db.ordered_stage('unbound'):sinks['unbound_calls'].row(row)
    for row in db.ordered_stage('inventory','r.loc'):sinks['inventory'].row(row)
    for name,s in sinks.items():p.files[name]=s.close();p.fields[name]=ArrayFile(s.path)
    for cat in CATS:
        d=catdisp[cat];assert sum(d.values())==rawcats[cat] and d['source_only']==0
        categories[cat]=dict(R=sum(d.values()),legacy_R=rawcats[cat],E=d['imported'],dispositions=dict(d),added=[])
    disposition=dict(records=nraw,dispositions=dict(dispositions),categories=categories,one_to_many=[])
    ratio=lambda a,b:a/b if b else None
    project_cov={};native_cov={}
    for cat in CATS:
        s=project_stats[cat];n=native_stats[cat]
        project_cov[cat]=dict(R=rawcats[cat],E=s['E'],X=s['X'],A=s['A'],L=s['L'],linked_count_ratio=ratio(s['L'],s['A']),linked_time_ratio=ratio(s['linked_dur'],s['dur']),import_completeness=ratio(s['E'],rawcats[cat]),dur_missing=s['missing'],level_distribution={v:dict(count=s['level_'+v],duration_ns=s['leveldur_'+v]) for v in ('A','B+','C')})
        native_cov[cat]=dict(native_owned_count_ratio=ratio(n['owned'],n['E']),native_owned_time_ratio=ratio(n['owned_dur'],n['dur']),scope_count_ratio=ratio(n['scoped'],n['E']),scope_time_ratio=ratio(n['scoped_dur'],n['dur']),eligible_count=n['E'],dur_missing=n['missing'])
    ids=source_identity(identities,vendor);ids['parser']['taxonomy']=original_tax.identity();ids['parser']['closure']=parser_identity(vendor,original_tax.version)
    if hasattr(adapter,'rules_identity'):ids['parser']['rules']=adapter.rules_identity()
    assessment=assess(db,tax,ids,p.arrays('ledger'),vendor,uid,scope_state)
    p.store('format_assessment',assessment)
    if p.files['format_assessment']['bytes']>max_metadata_bytes:raise ValueError('max_metadata_bytes; no complete published')
    p.fields.update(trace_uid=uid,sha256=digest,disposition=disposition,inventory=dict(categories={c:rawcats[c] for c in CATS},records=ArrayFile(directory/'inventory.jsonl')),
        identities=ids,coverage=dict(categories=project_cov,unrecognized_ktype=0),native_coverage=native_cov,
        normalization=dict(kind=kind,marker_prefix=marker_prefix,calls=calls),native_rules_version=importlib.import_module('flagprof.vendors.'+vendor+'.native_rules').VERSION,scope_config=asdict(config))
    p.store('_device_ids',Rows(lambda:(unpack(row[0]) for row in db.c.execute("SELECT payload FROM stage WHERE kind='emitted_device' ORDER BY pstr(payload) COLLATE BIGINT"))))
    p.store('_task_types',ObjectRows(lambda:((unpack(k),unpack(v)) for k,v in db.c.execute("SELECT k,payload FROM stage WHERE kind='task_count' ORDER BY pstr(k)"))))
    p.device_ids=p.fields['_device_ids'];p.native_states=dict(native_states);p.native_reasons=dict(native_reasons);p.tasks=p.fields['_task_types'];p.count=count
    # Legacy evidence-index byte order, streamed directly from source offsets.
    idx=dict(format_version=1,trace_uid=uid,raw_sha256=digest,raw_bytes=Path(source).stat().st_size,
        entries=Rows(lambda:(dict(loc=int(row[3]) if row[4]=='int' else row[3],ordinal=row[0],start=row[1],end=row[2]) for row in db.c.execute('SELECT id,start,end,loc,loctype FROM raw ORDER BY id'))))
    p.store('index',idx);db.commit();p.mark('metadata_summaries_index',start)
    p.spills=getattr(db,'spills',0);db.close()
    p.store('summary',dict(format_version=1,device_analysis=analysis_context,disposition=p.fields['disposition'],project=p.fields['coverage'],native=p.fields['native_coverage'],coverage_scope='prepared_events',
        unbound_calls=p.fields['unbound_calls'],format_assessment=p.fields['format_assessment'],native_states=p.native_states,native_reasons=p.native_reasons))
    if initial_source_stat!=source_stat(source):raise ValueError('source changed during prepare')
    if memory_sink is not None:
        p.files['memory']=memory_sink.close();p.fields['memory']=ArrayFile(memory_sink.path)
    p.source_stat=initial_source_stat
    return p


def import_prepared(source,writer,p,*,vendor,registry=None,calls_supplied=False,capture=None):
    from .checkpoints import identity as source_stat
    from .disk import streaming
    if source_stat(source)!=getattr(p,'source_stat',None) and streaming.sha_file(source)!=p.fields['sha256']:raise ValueError('source changed before Store')
    t=time.perf_counter();f=p.fields;uid=f['trace_uid'];adapter=load_adapter(vendor);registry=adapter.default_registry() if registry is None else registry
    d=Path('derived')/uid;rawpath=Path('vendor_raw')/(uid+'.json');compath=Path('imports')/uid/'completion.json'
    summary=dict(format_version=1,disposition=f['disposition'],project=f['coverage'],native=f['native_coverage'],coverage_scope='prepared_events',
        unbound_calls=f['unbound_calls'],format_assessment=f['format_assessment'],native_states=p.native_states,native_reasons=p.native_reasons)
    # summary is immutable after the prepare checkpoint
    files={'ledger':('ledger.jsonl',p.files['ledger']),'native':('native.jsonl',p.files['native']),
           'summary':('summary.json',p.files['summary']),'index':('evidence-index.json',p.files['index'])}
    if 'analysis_intervals' in p.files:files['analysis_intervals']=('analysis_intervals.jsonl',p.files['analysis_intervals'])
    normalization=dict(f['normalization'],calls_origin='argument' if calls_supplied else 'default',calls_sha256=hashlib.sha256(json_bytes(f['normalization']['calls'])).hexdigest(),calls_encoding='sorted compact UTF-8 JSON, no trailing newline')
    descriptor=dict(format_version=1,trace_uid=uid,vendor=vendor,source_path=str(source),sha256=f['sha256'],raw_path=str(rawpath),imported_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        device_ids=p.device_ids,rank=writer.fragment['rank'],inventory=dict(f['inventory'],raw_task_types=p.tasks),
        linker=dict(rules_version=f['identities']['parser'].get('rules',{}).get('version','t4-v2-20261011'),registry=registry,registry_sha256=hashlib.sha256(json_bytes(registry)).hexdigest(),registry_encoding='sorted compact UTF-8 JSON, no trailing newline'),
        clock_transform=dict(clock_valid=False,reason='fixture has no sourced transform to host_mono'),profiled_window=None,profiled_window_reason='fixture lacks measure/profiled window facts',
        writer_id=writer.writer_id,normalization=normalization,identities=f['identities'],capture=capture,disposition=f['disposition'],native=dict(rules_version=f['native_rules_version'],scope_config=f['scope_config']),
        derived={k:dict(path=str(d/name),sha256=meta['sha256'],bytes=meta['bytes'],format_version=1) for k,(name,meta) in files.items()},completion_path=str(compath))
    # Stream large inventory into the existing descriptor format. No Reader change.
    path=writer.run_dir/'traces'/(uid+'.json');s=Sink(path);s.row(descriptor);desc=s.close();_sync_dir(path.parent)
    for path in (writer.run_dir/d,(writer.run_dir/compath).parent):path.mkdir(parents=True,exist_ok=False);_sync_dir(path.parent)
    def copy(source,target):
        with Path(source).open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,65536);dst.flush();os.fsync(dst.fileno())
        _sync_dir(target.parent)
    before_copy=source_stat(source)
    copy(source,writer.run_dir/rawpath)
    if before_copy!=source_stat(source):raise ValueError('source changed while copying')
    for name,meta in files.values():copy(meta['path'],writer.run_dir/d/name)
    eventmap_path=writer.run_dir/'imports'/uid/'event_map.staging.jsonl'
    eventmap=Sink(eventmap_path);batch=[];batchbytes=0;attempted=enqueued=persisted=0;seqmin=None;seqmax=None;reasons=set();barrier=writer.flush(120)
    def flush_batch():
        nonlocal persisted,barrier,batchbytes
        barrier=writer.flush(120) if barrier else False
        for item,receipt in batch:
            item.update(asdict(receipt))
            if not barrier:item.update(persisted=False,reason='barrier_unconfirmed')
            persisted+=item['persisted']
            if item['reason']:reasons.add(item['reason'])
            eventmap.row(item)
        batch.clear();batchbytes=0
    for ordinal,event in enumerate(p.arrays('events')):
        payload=dict(event,ext={k:v for k,v in event['ext'].items() if k!='chrome.raw_record'})
        size=len(encoded(payload))
        if batch and (len(batch)>=min(128,writer.batch_limit) or batchbytes+size>1<<20):flush_batch()
        if not barrier:raise RuntimeError('barrier unconfirmed; incomplete attempt retained')
        receipt=writer.emit_with_receipt(payload);attempted+=1;enqueued+=receipt.accepted
        if receipt.seq is not None:seqmin=receipt.seq if seqmin is None else min(seqmin,receipt.seq);seqmax=receipt.seq if seqmax is None else max(seqmax,receipt.seq)
        batch.append((dict(ordinal=ordinal,trace_uid=uid,loc=event['ext']['chrome.loc']),receipt));batchbytes+=size
    flush_batch();eventmap.close();dropped=attempted-persisted;status='complete' if barrier and not dropped else 'partial'
    completion=dict(format_version=1,trace_uid=uid,descriptor_sha256=desc['sha256'],derived_sha256={k:v['sha256'] for k,v in descriptor['derived'].items()},writer_id=writer.writer_id,
        attempted=attempted,enqueued=enqueued,persisted=persisted,dropped=dropped,seq_range=[seqmin,seqmax+1] if seqmin is not None else None,
        event_map=Rows(lambda:file_rows(eventmap_path)),status=status,reasons=sorted(reasons))
    if barrier:
        s=Sink(writer.run_dir/compath);s.row(completion);s.close();_sync_dir((writer.run_dir/compath).parent)
    p.mark('publish_writer_completion',t)
    return dict(trace_uid=uid,source_sha256=f['sha256'],normalized_device_count=attempted,emitted=enqueued,persisted=persisted,dropped=dropped,status=status,barrier=barrier,summary=f['coverage'],trace_path=str(writer.run_dir/'traces'/(uid+'.json')))
