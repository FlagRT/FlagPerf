# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Paired accuracy analysis over public StoreView data; no torch dependency."""
import json
from . import Result
from .workspace import encode
from flagprof import replay
from flagprof.tensor_data import compare, FLOATS, ULP_BUCKETS

REQUIRED=('model_files','oracle_id','inputs_sha256','samples','batches','dtype','attention','pooling','boundaries','probe_alg')
DECLARABLE=('device','software','numerics','flaggems','include')
CTR_MODEL_FIELDS = ('kind','arch','config_sha256','weight_sha256','seed','init')
TASK_REQUIRED = {
    'ctr': ('task', *CTR_MODEL_FIELDS, 'score_semantics'),
    'rerank': ('task','model_files','attention','score_semantics','template_sha','yes_token_id','no_token_id','code_files'),
}


def required_fields(a, b):
    tasks = {v.get('task', 'embedding') for v in (a, b)}
    if tasks & TASK_REQUIRED.keys():
        common = tuple(k for k in REQUIRED if k not in ('model_files','attention','pooling'))
        return tuple(dict.fromkeys(common + tuple(k for task in sorted(tasks)
                           for k in TASK_REQUIRED.get(task, REQUIRED))))
    return tuple(k for k in REQUIRED if k!='pooling')+('task','generation') if 'generate' in tasks else REQUIRED


def call_config(event, cross_precision=False):
    value=event.get('ext',{}).get('call_config')
    if isinstance(value,list) and any(n.get('k')=='other' for arg in value for n in arg.get('value',[])):
        return None
    def without_layout(node):
        if isinstance(node,list):return [without_layout(n) for n in node]
        if not isinstance(node,dict):return node
        result = {k:without_layout(v) for k,v in node.items() if not (node.get('k')=='tensor' and k in ('contiguous','stride'))}
        if cross_precision and node.get('k') == 'tensor' and str(node.get('dtype')).removeprefix('torch.') in FLOATS:
            result['dtype'] = 'floating'
        return result
    return without_layout(value)


def _layout(event):
    def collect(node,path=()):
        if isinstance(node,list):
            return [entry for i,n in enumerate(node) for entry in collect(n,path+(i,))]
        if not isinstance(node,dict):return []
        own=[(path,{k:node[k] for k in ('contiguous','stride') if k in node})] if node.get('k')=='tensor' else []
        return own+[entry for k,v in node.items() for entry in collect(v,path+(k,))]
    return collect(event.get('ext',{}).get('call_config'))


def _numeric_difference(result):
    return bool((result.get('max_abs') or 0)>0 or result.get('nonfinite_pair_mismatches',0)>0)


def _numeric_equal(result):
    if result.get('status') != 'ok':
        return None
    if _numeric_difference(result):
        return False
    if result['level'] == 'full' or result.get('digest_equal'):
        return True
    return None


def _all_equal(previous, current):
    if previous is False or current is False:
        return False
    return True if previous is True and current is True else None


def identity(work):
    rows=[s['oracle_identity'] for s in work.statuses if s.get('oracle_identity')]
    if len(rows)!=1:return {}
    value=dict(rows[0])
    state=next(s for s in work.statuses if s.get('oracle_identity'))
    from flagprof.components.flaggems import gems_identity
    switch, source = gems_identity(value.get('flaggems'), state.get('flaggems_identity'))
    value['flaggems']=dict(enabled=switch,source_sha256=source)
    return value


def _operator(ref,cmp,left,right,row,result,find,mismatch,ordinal,cross_precision,replay_source):
    key=row['call_key']
    sources_a=left.get('src');sources_b=right.get('src')
    unknown=sources_a is None or sources_b is None or 'unknown' in (sources_a or [])+(sources_b or [])
    paired=not unknown and len(sources_a)==len(sources_b) and sources_a==sources_b
    config_a=call_config(left,cross_precision);config_b=call_config(right,cross_precision)
    config_equal=config_a is not None and config_b is not None and config_a==config_b
    if config_a is None or config_b is None:mismatch('call_config_unknown',key)
    elif not config_equal:mismatch('call_config_mismatch',key)
    if not unknown and not paired:mismatch('misaligned_producer',key)
    errors=[];all_equal=paired
    if paired:
        for source_a,source_b in zip(sources_a,sources_b):
            ea,eb=find(ref,source_a),find(cmp,source_b)
            if ea is None or eb is None:unknown=True;all_equal=False;continue
            cached=ref.db.execute("select payload from rows where panel='oracle.samples' and key=?",(source_a,)).fetchone()
            comparison=json.loads(cached[0]) if cached else compare(ref.view.run,ea,cmp.view.run,eb,work=ref.directory,limits=ref.limits,cross_precision=cross_precision)
            same=_numeric_equal(comparison) if cross_precision else comparison.get('digest_equal',False)
            if same is None:unknown=True
            all_equal = all_equal and same is True
            errors.append(0. if same else comparison.get('relative_l2'))
    if unknown:mismatch('source_unknown',key)
    error_in=max(errors) if paired and not unknown and errors and all(v is not None for v in errors) else None
    introduced=bool(config_equal and all_equal and not unknown and result['level']=='full' and _numeric_difference(result))
    rule={}
    if left.get('sid'):
        sig=ref.signature(left['sid'])
        if sig:
            op=sig['name'].removeprefix('aten::')+('.'+sig['overload'] if sig['overload'] else '')
            rule=next((s.get('route_table',{}).get(op,{}) for s in cmp.statuses if s.get('route_table')), {})
    item=dict(row,sid_ref=left.get('sid'),sid_cmp=right.get('sid'),input_sources_ref=sources_a,input_sources_cmp=sources_b,introduced=introduced,error_out=result.get('relative_l2'),error_in=error_in,
        amplification=result['relative_l2']/error_in if config_equal and error_in and result.get('relative_l2') is not None else None,
        input_status='unknown' if unknown else 'misaligned_producer' if not paired else 'equal' if all_equal else 'different',
        layout_differs=_layout(left)!=_layout(right),call_config_status='equal' if config_equal else 'unknown' if config_a is None or config_b is None else 'different',expected_route=rule or {'status':'not_registered'})
    status = replay_source
    if status == replay.READY:
        status, _ = replay.assess_call(dict(ref=ref, cmp=cmp), dict(ref=left, cmp=right), find)
    item['replay_status'] = status
    ref.row('oracle.op_sensitivity',key,item)
    ref.db.execute("update rows set metric=? where panel='oracle.op_sensitivity' and key=?",(-ordinal,key))
    if introduced:ref.row('oracle.introduced',key,item)
    if item['amplification'] is not None:
        ref.row('oracle.amplification',key,item)
        ref.db.execute("update rows set metric=? where panel='oracle.amplification' and key=?",(item['amplification'],key))


def _layer_sample(summaries,row,result,ordinal):
    scope,iteration,boundary,sample=(row[k] for k in ('scope','sample_pass','boundary','sample'))
    group=(scope,str(iteration),boundary)
    summary=summaries.setdefault(group,dict(scope=scope,boundary=boundary,sample_pass=int(iteration),samples=0,digest_equal=True,worst={},cosines=[],order=ordinal))
    if 'generation_step' in row:
        summary.update(generation_step=row['generation_step'],phase=row['phase'])
    summary['samples']+=1;summary['digest_equal'] = _all_equal(summary['digest_equal'],result['digest_equal'])
    if 'numeric_equal' in row:
        summary['numeric_equal'] = _all_equal(summary.get('numeric_equal',True),row['numeric_equal'])
    weights=summary.setdefault('weighting',dict(valid_samples=0,valid_elements=0,sample_mae_sum=0.,sample_mse_sum=0.,element_abs_sum=0.,element_square_sum=0.))
    n=result.get('valid_elements',0)
    if n and result.get('mae') is not None and result.get('mse') is not None:
        weights['valid_samples']+=1;weights['valid_elements']+=n
        weights['sample_mae_sum']+=result['mae'];weights['sample_mse_sum']+=result['mse']
        weights['element_abs_sum']+=result['mae']*n;weights['element_square_sum']+=result['mse']*n

    for metric in ('mse','mae','max_abs','relative_mean','relative_max','relative_l2','cosine_similarity'):
        value=result.get(metric)
        if value is None:continue
        old=summary['worst'].get(metric)
        if old is None or (value<old['value'] if metric=='cosine_similarity' else value>old['value']):
            chosen=dict(value=value,sample=sample)
            if metric=='max_abs':chosen.update(ulp=result.get('max_abs_ulp'),index=result.get('max_abs_index'))
            summary['worst'][metric]=chosen
    if result.get('cosine_similarity') is not None:summary['cosines'].append(result['cosine_similarity'])
    summary.setdefault('levels',set()).add(result['level'])
    if 'ulp_histogram' in result:
        histogram=summary.setdefault('ulp_histogram',dict.fromkeys(ULP_BUCKETS,0))
        for bucket,count in result['ulp_histogram'].items():histogram[bucket]+=count
        summary['ulp_samples']=summary.get('ulp_samples',0)+1
        if result.get('ulp_max') is not None:summary['ulp_max']=max(summary.get('ulp_max',0),result['ulp_max'])
    if 'ulp_histogram' in result or 'numeric_equal' in row:
        for field in ('valid_elements','differing_elements','paired_sentinel','finite_extreme_pair_mismatches','nonfinite_pair_mismatches'):
            summary[field]=summary.get(field,0)+result.get(field,0)


def _identity_result(ref,cmp,declare,cross_precision):
    declared=set((declare or {}).get('differences',[]))
    a,b=identity(ref),identity(cmp);blocked=None
    generation=a.get('task')=='generate' or b.get('task')=='generate'
    required=required_fields(a,b)
    if cross_precision and 'task' not in required and ('task' in a or 'task' in b):
        for side in (a,b):
            side.setdefault('task','embedding')
        required+=('task',)
    required += tuple(k for k in ('code_files','padding_side','adapter_names')
                      if k not in required and (k in a or k in b))
    declarable=DECLARABLE+('parallelism','comm_backend') if generation else DECLARABLE
    extra = tuple(k for k in ('compiler','communication','comm_backend')
                  if k not in declarable and (k in a or k in b))
    declarable += extra
    if not generation:
        required += tuple(k for k in ('parallelism','rank') if k in a or k in b)
    derived_id = False
    if cross_precision and all(side.get('dtype') in FLOATS for side in (a,b)):
        required=tuple(k for k in required if k!='dtype')
        declarable+=('dtype',)
        if (a['dtype']!=b['dtype'] and a.get('oracle_id')!=b.get('oracle_id')
                and all(side.get('task','embedding')=='embedding' for side in (a,b))):
            # Only relax an ID derived by the direct-input runner. Package and
            # custom IDs may encode additional contracts and must still match.
            from flagprof.runner.fixed import direct_oracle_id
            try:
                derived_id=(a['oracle_id']==direct_oracle_id(a)
                            and b['oracle_id']==direct_oracle_id(b)
                            and a['oracle_id']==direct_oracle_id(dict(b,dtype=a['dtype'])))
            except KeyError:
                pass
            if derived_id:
                required=tuple(k for k in required if k!='oracle_id')
                declarable+=('oracle_id',)
    def permitted(field, left, right):
        if field=='oracle_id':return derived_id
        if field in declared:return True
        if isinstance(left,dict) and isinstance(right,dict):
            return all(permitted(field+'.'+key,left.get(key),right.get(key))
                       for key in set(left)|set(right)
                       if left.get(key)!=right.get(key) or (key in left)!=(key in right))
        return False
    for field in required+declarable:
        equal=field in a and field in b and a[field]==b[field]
        if cross_precision and field=='oracle_id' and not a.get(field):
            equal=False
        allowed=field in declarable
        # Optional absent software components are values inside software, not missing identity fields.
        if not equal:
            if not allowed:blocked='identity_mismatch'
            elif not permitted(field,a.get(field),b.get(field)) and blocked!='identity_mismatch':blocked='undeclared_difference'
        ref.row('oracle.identity_diff',field,dict(field=field,ref=a.get(field),cmp=b.get(field),equal=equal,
            requirement='declarable' if allowed else 'must_equal',declared=permitted(field,a.get(field),b.get(field))))
    for field in ('compiler','communication'):
        if field not in a and field not in b:
            ref.row('oracle.identity_diff',field,dict(field=field,ref=None,cmp=None,equal=None,
                requirement='not_recorded',declared=False))
    panels=['oracle.layer','oracle.first_divergence','oracle.op_sensitivity']
    identity_result=Result('oracle.identity_diff','unavailable' if blocked else 'ok',ref.table('oracle.identity_diff'),{},[blocked] if blocked else [])
    return identity_result,blocked,panels


def _results(ref,cmp,mismatches,first,jumps,identity_result,panels):
    partial=any(mismatches.get(k) for k in ('only_ref','only_cmp','misaligned_shape','misaligned_dtype','misaligned_semantics','missing_numeric_evidence','limited_numeric_evidence','unscoped')) or ref.partial() or cmp.partial()
    op_partial=partial or any(mismatches[k] for k in ('misaligned_producer','call_config_mismatch','call_config_unknown','source_unknown'))
    status='partial' if partial else 'ok'
    if first:
        first['wording']=('已对齐边界中最早观测到的差异；' if partial else '')+('首个数值不等边界' if first['numeric_difference_observed'] else '首个字节不等边界，数值差异未确认')
    ref.row('oracle.first_divergence','first',dict(first_inequal=first,largest_jumps=sorted(
        (j for j in jumps if j['ratio'] is not None),key=lambda j:j['ratio'],reverse=True)[:5],
        zero_to_nonzero=[j for j in jumps if j['change']=='0_to_nonzero'][:ref.limits.evidence_samples]))
    routes={}
    for row in ref.table('oracle.introduced'):
        route=row.get('expected_route',{})
        if not route.get('registered'):continue
        impl=route.get('impl','unknown')
        group=routes.setdefault(impl,dict(impl=impl,occurrences=0,scopes=set()))
        group['occurrences']+=1;group['scopes'].add(row['scope'])
        if 'ulp_histogram' in row:
            histogram=group.setdefault('ulp_histogram',dict.fromkeys(ULP_BUCKETS,0))
            for bucket,count in row['ulp_histogram'].items():histogram[bucket]+=count
    for impl,group in sorted(routes.items()):
        group['scopes']=sorted(group['scopes']);ref.row('oracle.introduced_routes',impl,group)
    quality=dict(alignment=mismatches,alignment_samples=ref.table('oracle.alignment'),introduced=ref.table('oracle.introduced'),amplification=ref.table('oracle.amplification'),introduced_routes=ref.table('oracle.introduced_routes'))
    results=[identity_result]
    for panel in panels:
        available=len(ref.table(panel))>0
        panel_status=('partial' if op_partial else 'ok') if panel.endswith('op_sensitivity') else status
        results.append(Result(panel,panel_status if available else 'unavailable',ref.table(panel),quality,[] if available else ['no_tensor_ops' if panel.endswith('op_sensitivity') else 'no_paired_tensor']))
    return results

def analyze(ref,cmp,declare=None):
    differences=(declare or {}).get('differences',[])
    if not isinstance(differences,list):
        raise ValueError('Oracle differences must be a list of identity fields')
    cross_precision='dtype' in differences
    identity_result,blocked,panels=_identity_result(ref,cmp,declare,cross_precision)
    if blocked:return [identity_result]+[Result(panel,'unavailable',ref.table(panel),{},[blocked]) for panel in panels]
    generation_limits=None
    stores = dict(ref=ref, cmp=cmp)
    identities = {side: identity(ws) for side, ws in stores.items()}
    replay_source = replay.source_status(stores, identities)
    if identities['ref'].get('task')=='generate':
        from .generation import comparison
        generation_limits=comparison(ref,cmp)
    for work in (ref,cmp):
        work.db.execute("create index tensor_key on events(json_extract(payload,'$.call_key')) where type='tensor'")
    mismatches=dict(only_ref=0,only_cmp=0,misaligned_shape=0,misaligned_producer=0,unscoped=0,call_config_mismatch=0,call_config_unknown=0,source_unknown=0)
    summaries={};first=None;jumps=[];previous={};last_equal={};ordinal=0
    def find(work,key):
        rows=work.db.execute("select payload from events where type='tensor' and json_extract(payload,'$.call_key')=? limit 2",(key,)).fetchall()
        if len(rows)>1:raise ValueError('duplicate tensor call_key: '+key)
        return json.loads(rows[0][0]) if rows else None
    def mismatch(kind,key):
        mismatches[kind]=mismatches.get(kind,0)+1
        if mismatches[kind]<=ref.limits.evidence_samples:ref.row('oracle.alignment',kind+':'+str(mismatches[kind]),dict(kind=kind,call_key=key))
    for run,w,seq,left in ref.events(('tensor',)):
        key=left.get('call_key')
        if key is None:mismatch('unscoped',None);continue
        if generation_limits is not None:
            sample=key.split('#',3)[3]
            step=left.get('ext',{}).get('generation',{}).get('generation_step')
            if step is None:raise ValueError('generation tensor missing step binding')
            if step>=generation_limits[sample]:continue
        right=find(cmp,key)
        if right is None:mismatch('only_ref',key);continue
        if generation_limits is not None:
            binding=right.get('ext',{}).get('generation',{})
            if binding.get('generation_step')!=step or binding.get('phase')!=left['ext']['generation']['phase']:
                raise ValueError('paired generation tensor step binding mismatch: '+key)
        result=compare(ref.view.run,left,cmp.view.run,right,work=ref.directory,limits=ref.limits,cross_precision=cross_precision)
        if result['status']!='ok':mismatch(result['status'],key);continue
        scope,iteration,boundary,sample=key.split('#',3)
        row=dict(call_key=key,scope=scope,boundary=boundary,sample=sample,sample_pass=int(iteration),shape=left['shape'],dtype=left['dtype'],**result)
        if cross_precision:
            row.update(ref_dtype=left['dtype'],cmp_dtype=right['dtype'],numeric_equal=_numeric_equal(result))
            if result['level']!='full' and not result['digest_equal']:
                mismatch('limited_numeric_evidence',key)
        if generation_limits is not None:
            row.update(generation_step=step,phase=left['ext']['generation']['phase'])
        ref.row('oracle.samples',key,row)
        ordinal+=1
        ref.db.execute("update rows set metric=? where panel='oracle.samples' and key=?",(-ordinal,key))
        if boundary.startswith('op:'):
            _operator(ref,cmp,left,right,row,result,find,mismatch,ordinal,cross_precision,replay_source)
        elif left['role']=='out':
            _layer_sample(summaries,row,result,ordinal)
            sample_key=(sample,iteration)
            same=row.get('numeric_equal',result['digest_equal'])
            if same is True:last_equal[sample_key]=key
            elif same is False and first is None:
                first=dict(row,last_equal_boundary=last_equal.get(sample_key),numeric_difference_observed=_numeric_difference(result))
            current=result.get('relative_l2');prev=previous.get(sample_key)
            if prev and current is not None and prev[1] is not None and current != prev[1]:
                jumps.append(dict(sample=sample,before=prev[0],after=key,before_l2=prev[1],after_l2=current,
                    ratio=current/prev[1] if prev[1]>0 else None,
                    change='0_to_nonzero' if prev[1]==0 and current>0 else 'ratio'))
            previous[sample_key]=(key,current)
    for _,_,_,right in cmp.events(('tensor',)):
        if generation_limits is not None and right.get('call_key'):
            sample=right['call_key'].split('#',3)[3]
            step=right.get('ext',{}).get('generation',{}).get('generation_step')
            if step is None:raise ValueError('generation tensor missing step binding')
            if step>=generation_limits[sample]:continue
        if right.get('call_key') is None:mismatch('unscoped',None)
        elif find(ref,right['call_key']) is None:mismatch('only_cmp',right['call_key'])
    for group,summary in summaries.items():
        weights=summary['weighting'];ns=weights['valid_samples'];ne=weights['valid_elements']
        weights.update(sample_equal_mae=weights['sample_mae_sum']/ns if ns else None,
            sample_equal_mse=weights['sample_mse_sum']/ns if ns else None,
            element_equal_mae=weights['element_abs_sum']/ne if ne else None,
            element_equal_mse=weights['element_square_sum']/ne if ne else None,
            scope='paired finite elements; sampled and full levels remain explicitly labelled')
        cosines=summary.pop('cosines');summary['sample_cosine_mean']=sum(cosines)/len(cosines) if cosines else None
        summary['sample_cosine_min']=min(cosines) if cosines else None;summary['levels']=sorted(summary['levels'])
        if 'differing_elements' in summary:summary['differing_fraction']=summary['differing_elements']/summary['valid_elements'] if summary['valid_elements'] else None
        ref.row('oracle.layer',encode(group),summary)
        ref.db.execute("update rows set metric=? where panel='oracle.layer' and key=?",(-summary['order'],encode(group)))
    results=_results(ref,cmp,mismatches,first,jumps,identity_result,panels)
    if generation_limits is not None:
        results.append(Result('oracle.generation','ok',ref.table('oracle.generation'),{},
                              ['首个不同 token 的生成步包含在比较中，后续输入不同的步骤保留轨迹但不配对数值。']))
    return results


class PairedAnalyzer:
    def __init__(self,name):self.id='oracle.'+name

    def descriptor(self):
        return dict(id=self.id,kind='analyzer',version='1',module=__name__,factory='make_'+self.id.split('.')[1],requires=[],requires_min_schema='1.2')

    def run(self,store,params=None):
        from flagprof.store import StoreView
        from .workspace import Workspace,Limits
        import tempfile
        from pathlib import Path
        params=params or {}
        if 'cmp' not in params:raise ValueError('paired analyzer requires cmp Store path')
        temporary=tempfile.TemporaryDirectory(prefix='oracle-analyzer-')
        other=StoreView(params['cmp']);a=Workspace(store,Path(temporary.name)/'ref',Limits(**params.get('limits',{})))
        b=Workspace(other,Path(temporary.name)/'cmp',a.limits)
        class Owner:
            def close(self):a.close();b.close();other.close()
        owner=Owner()
        try:
            a.ingest();b.ingest();results=analyze(a,b,params.get('declare'))
            result=next(r for r in results if r.id==self.id);result._owner=(owner,temporary)
            return result
        except BaseException:owner.close();temporary.cleanup();raise


def make_identity_diff():return PairedAnalyzer('identity_diff')
def make_layer():return PairedAnalyzer('layer')
def make_first_divergence():return PairedAnalyzer('first_divergence')
def make_op_sensitivity():return PairedAnalyzer('op_sensitivity')
