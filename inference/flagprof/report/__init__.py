# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Deterministic self-contained reporting from replayable result sources."""
from dataclasses import asdict
from html import escape
import itertools
import json
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
ORACLE_OP_COLUMNS=('boundary','input_status','call_config_status','layout_differs','error_out',
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


def memory_chart(stream, series, limit):
    """A bounded first-source-window curve; full points remain in the export."""
    groups={}
    for point in itertools.islice(series,limit):
        ts=point.get('source_time_ns')
        if ts is None:continue
        for metric in ('allocated_bytes','reserved_bytes','active_bytes','device_used_bytes'):
            value=point['values'].get(metric)
            if value is not None:groups.setdefault((point['dev'],point['component'],metric),[]).append((ts,value))
    for key,values in groups.items():
        values.sort();lo=values[0][0];span=max(1,values[-1][0]-lo);high=max(1,max(v for _,v in values))
        points=' '.join(f'{30+700*(t-lo)/span:.2f},{190-160*v/high:.2f}' for t,v in values)
        title=escape(' / '.join(key))
        stream.write('<figure><figcaption>'+title+'；首 '+str(limit)+' 条记录内的来源时间曲线</figcaption>'
            '<svg viewBox="0 0 760 220" width="760" role="img" aria-label="'+title+'">'
            '<path d="M30 20 V190 H740" fill="none" stroke="#aab4c3"/>'
            '<polyline points="'+points+'" fill="none" stroke="#1764bb" stroke-width="2"/>'
            '<text x="35" y="18">'+str(high)+' bytes</text><text x="35" y="212">0 ns</text>'
            '<text x="560" y="212">'+str(values[-1][0]-lo)+' ns</text></svg></figure>')


def gap_chart(stream, coverage):
    for kind, values in coverage.get('distributions',{}).items():
        window=values['window_ns']['mean'];active=values['all_activity_union_ns']['mean'];blank=values['blank_ns']['mean']
        if not window:continue
        width=600*active/window
        stream.write('<figure><figcaption>'+escape(kind)+'：均值贡献（活动并集＋空白）</figcaption>'
            '<svg viewBox="0 0 640 75" width="640" role="img" aria-label="窗口内活动并集与空白的均值贡献">'
            '<rect x="20" y="12" width="600" height="24" fill="#dce3ee"/>'
            '<rect x="20" y="12" width="'+str(width)+'" height="24" fill="#1764bb"/>'
            '<text x="20" y="62">活动 '+format(active/1e6,'.3f')+' ms；空白 '+format(blank/1e6,'.3f')+
            ' ms；窗口 '+format(window/1e6,'.3f')+' ms</text></svg></figure>')


def html(stream, results, limits, quality):
    oracle=any(r.id.startswith('oracle.') for r in results)
    title='训练负载分析' if any(r.id=='training' for r in results) else '推理负载分析'
    stream.write('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>FlagProf '+title+'</title><style>body{font:15px system-ui;margin:32px;color:#172338;background:#f6f8fc}'
        'section{background:white;padding:20px;margin:18px 0;border-radius:10px;overflow:auto}table{border-collapse:collapse}'
        'th,td{padding:8px;border:1px solid #d9dfeb;text-align:left;max-width:400px;overflow-wrap:anywhere}'
        'pre{white-space:pre-wrap}h1,h2{color:#17345c}</style><h1>'+title+'</h1>'
        '<p>先查看各面板的可用性与原因，再阅读指标及其覆盖范围。</p>')
    if oracle:
        stream.write('<h2>Oracle—Device 精度差分</h2><p>完整值给出逐元素误差；抽样值只描述抽样位置。digest 相同用于判等。先看首个差异，再查看逐层误差与算子输入来源。</p>')
        order=('oracle.identity_diff','oracle.generation','oracle.first_divergence','oracle.op_sensitivity','oracle.layer','oracle.replay')
        results=sorted(results,key=lambda r:order.index(r.id) if r.id in order else len(order))
    else:stream.write('<p>设备时长和表示记录的工作量，含重叠；区间并集表示指定窗口内的活动占用。跨来源时间轴能力见时钟依据。主机调用耗时与 runner 同步前向耗时分别解释。</p>')
    for result in results:
        panel = escape(result.id, quote=True)
        stream.write('<section id="'+panel+'"><h2>'+panel+' · '+escape(result.status)+'</h2>')
        if result.id in ('generation','oracle.generation'):
            stream.write('<p>生成步从 0 编号：step 0 为 prefill，之后为 decode。首次不同 token 所在步仍由共同输入产生，包含在数值比较中；后续完整轨迹保留。</p>')
        for note in result.notes:
            stream.write('<p class="reason">'+escape(note)+'</p>')
        displayed = result.coverage.get('headline', result.data)
        if result.id=='route.coverage' and 'missing_ops' in result.coverage:
            q=result.coverage;nonhit=sum(q.get(k,0) for k in ('fallback','excluded','error'))
            stream.write('<p>调用 '+str(q.get('calls',0))+' 次；非命中 '+str(nonhit)+' 次（含策略与设备排除）。完整 sid 明细 '+str(len(result.data))+' 行，见 route-coverage.jsonl。</p>')
            for section,title,note in [('main','回退主表','按前向内非命中次数、比例排序；主机时间为次要列。'),
                    ('composite','组合算子','覆盖情况取决于分解后的算子。'),('view','视图类','别名/视图类别不直接代表需要新增计算实现；处理方向仍看 reason。')]:
                stream.write('<h3>'+title+'</h3><p>'+note+'</p>')
                html_rows(stream,panel,(r for r in q['missing_ops'] if r['section']==section),limits.html_rows)
        else:
            stream.write('<p>显示前 '+str(min(len(displayed),limits.html_rows))+' 行；完整表 '+str(len(result.data))+' 行。完整数据见 result.json 与 exports。</p>')
            if result.id=='oracle.op_sensitivity':
                stream.write('<p>ULP 是两侧值之间的可表示值距离；分布描述差异量级，具体成因结合实现证据或定点实验判断。不同元素占比以有效有限配对元素为分母；成对极值哨兵单独计数并排除。布局差异单独显示，路由列为预期实现。</p>')
                for key,title in [('introduced','introduced：相同调用配置与完整相同输入上的数值差异'),('introduced_routes','按预期路由汇总的 introduced'),('amplification','可计算的误差放大 Top-10')]:
                    rows=result.coverage.get(key,[])
                    stream.write('<h3>'+title+' · '+str(len(rows))+' 行</h3>')
                    html_rows(stream,panel+'.'+key,rows,min(10,limits.html_rows))
                stream.write('<h3>按执行顺序的算子结果</h3>')
            html_rows(stream,panel,displayed,limits.html_rows)
        if result.id=='resource':
            for key,title in [('stages','按阶段汇总'),('energy','能耗与功率域'),('memory','扣除基线后的分配器外显存估算'),('restrictions','区间内限制记录'),('frequency','观测频率低于标称值的区间')]:
                stream.write('<h3>'+title+'</h3>')
                html_rows(stream,panel+'.'+key,result.coverage.get(key,()),limits.html_rows)
        if result.id=='perf.timeline':
            stream.write('<p><a href="exports/timeline.json">下载 Chrome / Perfetto 时间线</a></p><pre>'+escape(encode(result.coverage))+'</pre>')
        if result.id=='perf.gap':gap_chart(stream,result.coverage)
        if result.id=='memory':memory_chart(stream,result.coverage.get('series',()),limits.html_rows)
        if result.id in ('perf.device','perf.gap','communication','memory','generation','training'):
            for name,table in result.coverage.items():
                if isinstance(table,Table):
                    stream.write('<details><summary>'+escape(name)+'（完整记录见 exports）</summary>')
                    html_rows(stream,result.id+'.'+name,table,limits.html_rows)
                    stream.write('</details>')
        # Tables in coverage are exported, while scalar quality stays visible.
        panel_quality = {k: v for k, v in result.coverage.items() if not isinstance(v, Table)}
        stream.write('<details><summary>数据质量与覆盖范围</summary><pre>'+escape(encode(panel_quality))+'</pre></details></section>')
    stream.write('<section id="data-quality"><h2>数据质量</h2><pre>'+escape(encode(quality))+'</pre></section></html>\n')


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
            data['report_context']=dict(run=dict(run_id=run.name),
                identity=next((s['oracle_identity'] for s in workspace.statuses if s.get('oracle_identity')),{}))
            if runner_results:
                from .performance import summarize
                data['runner_summary']=summarize(runner_results,data['report_context']['identity'],limits=limits)
            from flagprof.monitor.diagnostic_report import panel, html as diagnostic_html
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
                    replay_ref_run=None, replay_cmp_run=None, baseline=None):
    from flagprof.offline.worker import output_operation
    target = out or Path(ref).resolve().with_name(Path(ref).name + ('-three-way' if baseline else '-oracle'))
    inputs = [ref, cmp, baseline, perf_ref_run, perf_cmp_run, replay_ref_run, replay_cmp_run]
    return output_operation(target, inputs, _generate_oracle, ref, cmp, declare=declare, limits=limits,
                            perf_ref_run=perf_ref_run, perf_cmp_run=perf_cmp_run, replay_ref_run=replay_ref_run,
                            replay_cmp_run=replay_cmp_run, baseline=baseline)


def _generate_oracle(ref, cmp, out, *, declare, limits, perf_ref_run, perf_cmp_run, replay_ref_run, replay_cmp_run, baseline):
    if baseline:
        from .three_way import generate as three_way
        return three_way(ref,baseline,cmp,out,declare=declare,limits=limits,
            perf_ref_run=perf_ref_run,perf_cmp_run=perf_cmp_run,replay_ref_run=replay_ref_run,replay_cmp_run=replay_cmp_run)
    ref, cmp = Path(ref).resolve(), Path(cmp).resolve()
    out = Path(out).resolve() if out else ref.parent/(ref.name+'-oracle')
    if ref.is_file() or cmp.is_file():
        if not (ref.is_file() and cmp.is_file()):raise ValueError('both oracle inputs must be groups or both must be Stores')
        from .oracle_group import generate as generate_group_oracle
        return generate_group_oracle(ref,cmp,out,declare=declare,limits=limits,
            perf_ref_run=perf_ref_run,perf_cmp_run=perf_cmp_run,replay_ref_run=replay_ref_run,replay_cmp_run=replay_cmp_run)
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
            data=dict(schema='oracle-result/1',declaration=declare or {},quality=quality,results={r.id:dict(status=r.status,data=r.data,coverage=r.coverage,notes=r.notes) for r in results})
            with (out/'result.json').open('w') as stream:write_json(stream,data);stream.write('\n')
            for panel in ('oracle.identity_diff','oracle.layer','oracle.first_divergence','oracle.op_sensitivity','oracle.samples','oracle.alignment','oracle.introduced','oracle.amplification','oracle.perf_links','oracle.introduced_routes','oracle.replay','oracle.generation'):
                with (out/'exports'/(panel+'.jsonl')).open('w') as stream:
                    for row in a.table(panel):stream.write(encode(row)+'\n')
            from .product import write_report
            write_report(out,data,limits=limits)
            a.check_budget();b.check_budget()
            resources=dict(elapsed_s=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scratch_bytes=sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file()))
            (out/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
            return dict(output=str(out),panels={r.id:r.status for r in results})
        finally:a.close();b.close()


def compare_device_reports(reports, out, *, declare=None, limits=None):
    from .comparison import generate
    return generate(reports,out,declare=declare,limits=limits)
