# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Deterministic self-contained reporting from replayable result sources."""
from dataclasses import asdict
from html import escape
import itertools
import json
import os
from pathlib import Path
import resource
import tempfile
import time
from flagprof.store import StoreView
from flagprof.store.analysis_read import ResourceLimit
from flagprof.analyzers import Result
from flagprof.analyzers.workspace import Limits, Table, Workspace, encode
from flagprof.analyzers.core import ANALYZERS
from flagprof.analyzers.device import make_perf_device
from flagprof.analyzers.gap import make_perf_gap
from flagprof.analyzers.memory import make_memory


def write_json(stream, value):
    from .model import Rows
    if isinstance(value, (Table, Rows)):
        stream.write('[')
        for i, row in enumerate(value):
            if i:
                stream.write(',')
            stream.write(encode(row))
        stream.write(']')
    elif isinstance(value, dict):
        stream.write('{')
        for i, key in enumerate(sorted(value)):
            if i:
                stream.write(',')
            stream.write(encode(key)+':')
            write_json(stream, value[key])
        stream.write('}')
    elif isinstance(value, (tuple, list)):
        stream.write('[')
        for i, item in enumerate(value):
            if i:
                stream.write(',')
            write_json(stream, item)
        stream.write(']')
    else:
        stream.write(encode(value))


ORACLE_LAYER_COLUMNS=('scope','boundary','samples','digest_equal','relative_l2','cosine_similarity','max_abs',
    'differing_fraction','ulp_histogram','valid_elements','levels')
ORACLE_OP_COLUMNS=('boundary','input_status','replay_status','call_config_status','layout_differs','error_out',
    'amplification','differing_fraction','ulp_histogram','introduced','expected_route')


def _columns(panel,rows):
    if panel=='perf.device.comparison':return ('comparison_status',)+_columns('perf.device',rows)
    if panel=='communication.aligned':return ('group','group_seq','rank_device_union_ns','cross_rank_min_observed_ns','cross_rank_max_observed_ns','cross_rank_difference_ns','reason')
    if panel=='perf.gap.streams':return ('window_id','kind','step','dev','stream','window_ns','all_activity_union_ns','blank_ns','blank_segments_ns','categories')
    if panel=='perf.gap':return ('kind','request_id','coverage_status','step','dev','window_ns','compute_union_ns','memcpy_union_ns','comm_union_ns','wait_union_ns','all_activity_union_ns','blank_ns')
    if panel=='communication':return ('group','op','backend','role','calls','generation_calls','logical_input_bytes','logical_output_bytes','device_linked_calls','device_unlinked_calls','device_link_ratio','device_representation_records','device_sum_ns','device_link_status')
    if panel in ('communication.device_calls','perf.gap.device_calls'):return ('group','group_seq','rank','op','device_link_status','device_union_ns','device_sum_ns','device_records','evidence')
    if panel=='generation':return ('request_id','samples','mode','steps','complete','stop_reasons','tokens')
    if panel=='generation.steps':return ('request_id','generation_step','phase','host_duration_ns','kv_length_before','input_shape')
    if panel=='oracle.generation':return ('sample','first_token_divergence','compare_steps','ref_steps','cmp_steps','lengths_equal','ref','cmp')
    if panel=='perf.device':return ('dev','category','association_label','link_status','events','valid_durations','missing_durations','device_work_ns','window_identity','scope_note','evidence')
    if panel=='oracle.layer':
        return (('generation_step','phase') if any('generation_step' in row for row in rows) else ())+ORACLE_LAYER_COLUMNS
    if panel.startswith('oracle.op_sensitivity'):return ORACLE_OP_COLUMNS
    if panel=='oracle.first_divergence':
        return (('generation_step','phase') if any('generation_step' in (row.get('first_inequal') or {}) for row in rows) else ())+('wording','scope','boundary','sample','max_abs','max_abs_ulp','differing_fraction')
    if panel=='oracle.identity_diff':return ('field','equal','requirement','declared')
    return tuple(dict.fromkeys(key for row in rows for key in sorted(row)))


def _display_value(panel,row,key):
    if panel=='oracle.first_divergence':return (row.get('first_inequal') or {}).get(key)
    if panel=='oracle.layer' and key in ('relative_l2','cosine_similarity','max_abs'):
        value=row.get('worst',{}).get(key)
        if not value:return None
        if key=='max_abs':return dict(value=value['value'],ulp=value.get('ulp'),sample=value['sample'],index=value.get('index'))
        return value
    if key=='expected_route':return row.get(key,{}).get('impl',row.get(key,{}).get('status'))
    return row.get(key)


def html_rows(stream,panel,rows,limit):
    rows=list(itertools.islice(rows,limit))
    if not rows:return
    columns=_columns(panel,rows);oracle=panel.startswith('oracle.')
    # The full fields remain mapped inside details; only headline columns are visible.
    extras=[{k:v for k,v in row.items() if k not in columns} if oracle or panel=='perf.gap' else {} for row in rows]
    if panel=='oracle.layer':
        for extra in extras:
            worst=extra.get('worst',{})
            extra['worst']={k:v for k,v in worst.items() if k not in ('relative_mean','relative_max')}
    has_details=any(extras)
    stream.write('<table data-panel="'+escape(panel,quote=True)+'"><thead><tr>')
    for key in columns:stream.write('<th>'+escape(key)+'</th>')
    if has_details:stream.write('<th>详情</th>')
    stream.write('</tr></thead><tbody>')
    def cell(i,key,value):
        unit='ns' if key.endswith('_ns') else ''
        stream.write('<td data-panel="'+escape(panel,quote=True)+'" data-row="'+str(i)+'" data-field="'+escape(key,quote=True)+'" data-unit="'+unit+'">'+escape(encode(value))+'</td>')
    for i,row in enumerate(rows):
        stream.write('<tr data-row="'+str(i)+'">')
        for key in columns:cell(i,key,_display_value(panel,row,key))
        if has_details:
            stream.write('<td><details><summary>完整字段</summary><table>')
            for key,value in sorted(extras[i].items()):
                if key in ('relative_mean','relative_max'):continue
                stream.write('<tr><th>'+escape(key)+'</th>');cell(i,key,value);stream.write('</tr>')
            stream.write('</table></details></td>')
        stream.write('</tr>')
    stream.write('</tbody></table>')


def generate(run, out=None, *, limits=None, op_kinds=None, runner_results=None):
    from flagprof.offline.worker import output_operation
    target = out or Path(run).resolve().with_name(Path(run).name + '-report')
    return output_operation(target, [run], _generate, run, limits=limits, op_kinds=op_kinds, runner_results=runner_results)


def _generate(run, out, *, limits=None, op_kinds=None, runner_results=None):
    run = Path(run).resolve()
    out = Path(out).resolve() if out else run.parent/(run.name+'-report')
    if out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('report output must be outside input Store')
    limits = limits if isinstance(limits, Limits) else Limits(**(limits or {}))
    out.mkdir(parents=True, exist_ok=True)
    (out/'exports').mkdir(exist_ok=True)
    started = time.perf_counter()
    with StoreView(run) as view, tempfile.TemporaryDirectory(prefix='analysis-') as directory:
        workspace = Workspace(view, directory, limits)
        try:
            try:
                workspace.ingest()
            except (ValueError, OSError) as error:
                workspace.limit_error = str(error)
                workspace.notes.append(str(error))
            from flagprof.analyzers.forwards import prepare, finish
            prepare(workspace)
            from flagprof.analyzers.resource import make_resource
            results = []
            accuracy=any((s.get('effective_config') or {}).get('tensor_blob') is not None for s in workspace.statuses)
            for analyzer in (() if accuracy else (*ANALYZERS,make_perf_device(),make_memory(),make_perf_gap(),make_resource())):
                try:
                    result = analyzer.run(workspace, {})
                    workspace.check_budget()
                except (ValueError, OSError, KeyError) as error:
                    result = Result(analyzer.id, 'unavailable', workspace.table(analyzer.id),
                                    workspace.coverage(), [str(error)])
                results.append(result)
            from flagprof.analyzers.generation import analyze as generation, communication
            results.extend(generation(workspace));results.extend(communication(workspace))
            from flagprof.analyzers.training import analyze as training
            results.extend(training(workspace))
            forward_quality=finish(workspace, op_kinds)
            for result in results:
                if result.id in ('perf.op','perf.layer'):
                    result.coverage.update(forward_quality, headline=workspace.table(result.id+'.headline'))
                    result.notes.append(forward_quality['host_time_note'])
                elif result.id=='route.coverage':result.coverage['missing_ops']=workspace.table('missing_ops')
            from flagprof.analyzers.timeline import make_perf_timeline
            if (run/'monitor-ref.json').exists():
                results.append(make_perf_timeline().run(workspace,dict(out=out/'exports/timeline.json')))
            quality=workspace.coverage()
            quality.update(forward_quality)
            data = dict(schema='p1-result/1', limits=asdict(limits),quality=quality,
                        results={r.id: dict(status=r.status, data=r.data, coverage=r.coverage, notes=r.notes) for r in results})
            attachment = next((s for s in workspace.statuses if s.get('oracle_identity')), {})
            data['report_context']=dict(run=dict(run_id=run.name),identity=attachment.get('oracle_identity',{}))
            if attachment.get('model_name'):
                data['report_context']['model_name'] = attachment['model_name']
            if runner_results:
                from .performance import summarize
                data['runner_summary']=summarize(runner_results,data['report_context']['identity'],limits=limits)
            from flagprof.monitor.diagnostic_report import panel
            data['diagnosis'] = panel(run)
            with (out/'result.json').open('w') as stream:
                write_json(stream, data)
                stream.write('\n')
            for panel, filename in [('perf.op','perf-op'), ('perf.layer','perf-layer'),
                    ('route.coverage','route-coverage'), ('compile.health','compile-health'),
                    ('missing_ops','missing_ops'), ('compile_issues','compile_issues'), ('inventory_v1_view','inventory_v1_view'),
                    ('unscoped_by_tid','unscoped-by-tid'), ('forwards','forwards'),
                    ('perf.op.forwards','perf-op-forwards'), ('perf.layer.forwards','perf-layer-forwards'),
                    ('perf.gap','perf-gap'),('perf.gap.streams','perf-gap-streams'),('communication.device_calls','communication-device-calls'),('perf.device.headline','device-headline'),
                    ('perf.device','perf-device'),('perf.device.phases','device-phases'),('perf.device.coverage','device-coverage'),
                    ('perf.op.devices','perf-op-devices'),('perf.layer.devices','perf-layer-devices'),
                    ('resource','resource'),('resource.stages','resource-stages'),('resource.energy','resource-energy'),('resource.memory','resource-memory'),('resource.restrictions','resource-restrictions'),('resource.frequency','resource-frequency'),('memory','memory-peaks'),('memory.series','memory-series'),('memory.transfers','memory-transfers'),
                    ('memory.allocations','memory-allocations'),('generation','generation'),('generation.steps','generation-steps'),('generation.kv','generation-kv'),
                    ('communication','communication'),('communication.calls','communication-calls'),
                    ('training','training-steps'),('training.phases','training-phases')]:
                with (out/'exports'/(filename+'.jsonl')).open('w') as stream:
                    for row in workspace.table(panel):
                        stream.write(encode(row)+'\n')
            # Full folded membership is streamed independently of finite row samples.
            with (out/'exports'/'layer-members.jsonl').open('w') as stream:
                for panel, key, run_id, corr in workspace.db.execute('select * from members order by panel,key,run,corr'):
                    stream.write(encode(dict(panel=panel, key=key, run_id=run_id, corr=int(corr)))+'\n')
            from .product import write_report
            write_report(out,data,limits=limits)
            disk = sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())
            resources = dict(elapsed_s=time.perf_counter()-started,
                             peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             scratch_bytes=disk, limits=asdict(limits))
            (out/'resources.json').write_text(json.dumps(resources, sort_keys=True, indent=2)+'\n')
            return dict(output=str(out), panels={r.id:r.status for r in results})
        finally:
            workspace.close()


def generate_oracle(ref, cmp, out=None, *, declare=None, limits=None, perf_ref_run=None, perf_cmp_run=None,
                    replay_ref_run=None, replay_cmp_run=None, baseline=None, _delivery_root=None):
    from flagprof.offline.worker import output_operation
    target = out or Path(ref).resolve().with_name(Path(ref).name + ('-three-way' if baseline else '-oracle'))
    inputs = [ref, cmp, baseline, perf_ref_run, perf_cmp_run, replay_ref_run, replay_cmp_run]
    return output_operation(target, inputs, _generate_oracle, ref, cmp, declare=declare, limits=limits,
                            perf_ref_run=perf_ref_run, perf_cmp_run=perf_cmp_run, replay_ref_run=replay_ref_run,
                            replay_cmp_run=replay_cmp_run, baseline=baseline, delivery_root=_delivery_root or target)


def _generate_oracle(ref, cmp, out, *, declare, limits, perf_ref_run, perf_cmp_run, replay_ref_run, replay_cmp_run, baseline, delivery_root):
    if baseline:
        from .three_way import generate as three_way
        return three_way(ref,baseline,cmp,out,declare=declare,limits=limits,
            perf_ref_run=perf_ref_run,perf_cmp_run=perf_cmp_run,replay_ref_run=replay_ref_run,replay_cmp_run=replay_cmp_run,
            delivery_root=delivery_root)
    ref, cmp = Path(ref).resolve(), Path(cmp).resolve()
    out = Path(out).resolve() if out else ref.parent/(ref.name+'-oracle')
    if ref.is_file() or cmp.is_file():
        if not (ref.is_file() and cmp.is_file()):raise ValueError('both oracle inputs must be groups or both must be Stores')
        from .oracle_group import generate as generate_group_oracle
        return generate_group_oracle(ref,cmp,out,declare=declare,limits=limits,
            perf_ref_run=perf_ref_run,perf_cmp_run=perf_cmp_run,replay_ref_run=replay_ref_run,replay_cmp_run=replay_cmp_run,
            delivery_root=delivery_root)
    for source in (ref,cmp):
        if out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('oracle output must be outside input Stores')
    limits=limits if isinstance(limits,Limits) else Limits(**(limits or {}))
    out.mkdir(parents=True,exist_ok=True);(out/'exports').mkdir(exist_ok=True)
    started=time.perf_counter()
    from flagprof.analyzers.oracle import analyze
    with StoreView(ref) as av, StoreView(cmp) as bv, tempfile.TemporaryDirectory(prefix='oracle-') as directory:
        a=Workspace(av,Path(directory)/'ref',limits);b=Workspace(bv,Path(directory)/'cmp',limits)
        try:
            a.ingest();b.ingest();results=analyze(a,b,declare)
            identity_panel = next(r for r in results if r.id == 'oracle.identity_diff')
            blocking_fields = [row['field'] for row in identity_panel.data if row['equal'] is False
                               and (row['requirement'] == 'must_equal' or not row['declared'])]
            paired_tensors = len(a.table('oracle.samples'))
            if identity_panel.status != 'ok':
                comparison_status = 'not_comparable'
            elif not paired_tensors:
                comparison_status = 'no_pairs'
            elif any(r.status == 'partial' for r in results):
                comparison_status = 'partial'
            else:
                comparison_status = 'comparable'
            comparison = dict(comparison_status=comparison_status, blocking_fields=blocking_fields,
                              paired_tensors=paired_tensors, paired_boundaries=len(a.table('oracle.layer')))
            from flagprof.analyzers.perf_links import link
            links=link(a,b,dict(ref=perf_ref_run,cmp=perf_cmp_run),directory,limits)
            if links['requested']:
                results.append(Result('oracle.perf_links','partial' if links['unmatched'] else 'ok',
                    a.table('oracle.perf_links'),links,[]))
            if bool(replay_ref_run)!=bool(replay_cmp_run):raise ValueError('both replay Stores required')
            if replay_ref_run:
                from flagprof.replay import panel
                results.append(panel(a,ref,cmp,replay_ref_run,replay_cmp_run))
            quality=dict(ref=a.coverage(),cmp=b.coverage(),perf_links=links)
            data=dict(schema='oracle-result/1',declaration=declare or {},quality=quality,**comparison,
                      results={r.id:dict(status=r.status,data=r.data,coverage=r.coverage,notes=r.notes) for r in results})
            model_names = {}
            for side,workspace in (('ref',a),('cmp',b)):
                sources = [s for s in workspace.statuses if s.get('oracle_identity')]
                if len(sources) == 1 and sources[0].get('model_name'):
                    model_names[side] = sources[0]['model_name']
            context = {}
            if model_names:
                context['model_names'] = model_names
            first_replay = a.db.execute("""select key from rows where panel='oracle.op_sensitivity'
                and json_extract(payload,'$.replay_status')='replay_ready' order by metric desc,key limit 1""").fetchone()
            if first_replay:
                inputs = {name: os.path.relpath(Path(path).resolve(), out)
                          for name, path in [('ref', ref), ('cmp', cmp), ('perf_ref_run', perf_ref_run),
                                             ('perf_cmp_run', perf_cmp_run)] if path}
                context['replay'] = dict(inputs=inputs, call_key=first_replay[0])
            if context:
                data['report_context'] = context
            with (out/'result.json').open('w') as stream:write_json(stream,data);stream.write('\n')
            for panel in ('oracle.identity_diff','oracle.layer','oracle.first_divergence','oracle.op_sensitivity','oracle.samples','oracle.alignment','oracle.introduced','oracle.amplification','oracle.perf_links','oracle.introduced_routes','oracle.replay','oracle.generation'):
                with (out/'exports'/(panel+'.jsonl')).open('w') as stream:
                    for row in a.table(panel):stream.write(encode(row)+'\n')
            from .product import write_report
            write_report(out,data,limits=limits,replay_root=delivery_root)
            a.check_budget();b.check_budget()
            resources=dict(elapsed_s=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scratch_bytes=sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file()))
            (out/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
            return dict(output=str(out),panels={r.id:r.status for r in results},**comparison)
        finally:a.close();b.close()


def compare_device_reports(reports, out, *, declare=None, limits=None):
    from .comparison import generate
    return generate(reports,out,declare=declare,limits=limits)
