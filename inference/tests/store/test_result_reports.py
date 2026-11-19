"""Reader-facing regressions: actual omissions, scope, and portable redraw."""
import json
from pathlib import Path
import tempfile
import unittest

from flagprof.report.model import opened_model,read_document
from flagprof.report.product import write_report,redraw


class ResultReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def generate(self,document,name='report'):
        out=self.root/name;out.mkdir()
        (out/'result.json').write_text(json.dumps(document))
        write_report(out,document)
        return out

    def snapshot(self,out):
        return [json.loads(line) for line in (out/'report-data.jsonl').read_text().splitlines()]

    def test_empty_headline_never_hides_native_rows_or_changes_window(self):
        data=dict(schema='p1-result/1',results={
            'perf.op':dict(status='partial',data=[dict(name='native.add',source='native',calls=3,device_work_ns=1200)],
                           coverage=dict(headline=[],headline_note='测量前向')),
            'perf.device':dict(status='ok',data=[dict(name='named_kernel',events=3,category='compute',device_work_ns=1200)],coverage={})})
        out=self.generate(data);records=self.snapshot(out)
        self.assertTrue(any(r.get('section')=='perf.op' and r['value']['name']=='native.add' for r in records))
        self.assertIn('全采集窗口',next(r['value']['description'] for r in records if r['type']=='section' and r['value']['id']=='perf.op'))
        self.assertIn('named_kernel',(out/'report.md').read_text())
        self.assertIn('native.add',(out/'report.html').read_text())

    def test_zero_effective_progress_is_not_a_health_verdict(self):
        out=self.generate(dict(results={},diagnosis=dict(status='available',findings=[],evidence=[],
            progress_coverage=[dict(rank=0,registered_execution=0,source_gap=1)])))
        text=(out/'report.md').read_text()
        self.assertIn('没有有效执行进度观测',text)
        self.assertIn('未记录告警不能据此判断执行过程正常',text)

    def test_route_summary_columns_and_performance_link_are_visible(self):
        out=self.generate(dict(results={
            'oracle.op_sensitivity':dict(status='ok',data=[dict(boundary='pow',introduced=True)],coverage=dict(
                introduced_routes=[dict(impl='pow_tensor_scalar',occurrences=3,scopes=['layer.0'])])),
            'oracle.perf_links':dict(status='ok',data=[dict(side='cmp',sid='sig',call_key='call',matched_rows=1)],coverage={})}))
        records=self.snapshot(out)
        route=next(r['value'] for r in records if r['type']=='section' and r['value']['id'].endswith('introduced_routes'))
        self.assertEqual(route['columns'],['impl','occurrences','scopes'])
        self.assertIn('同签名的独立性能证据',(out/'report.md').read_text())

    def test_all_rows_survive_portable_redraw_and_script_names_are_escaped(self):
        out=self.generate(dict(results={'perf.device':dict(status='ok',coverage={},data=[
            dict(name='kernel-'+str(i),events=1,category='compute',device_work_ns=i+1) for i in range(205)])}))
        before=(out/'result.json').read_bytes()
        redrawn=self.root/'redrawn';redraw(out,redrawn)
        self.assertEqual(before,(out/'result.json').read_bytes())
        self.assertEqual(before,(redrawn/'result.json').read_bytes())
        records=self.snapshot(redrawn)
        self.assertEqual(sum(r.get('section')=='perf.device' for r in records),205)
        self.assertEqual((out/'report.md').read_text(),(redrawn/'report.md').read_text())
        document=dict(results={'perf.op':dict(status='ok',coverage={},data=[dict(name='</script><script>bad()</script>')])})
        page=(self.generate(document,'escaped')/'report.html').read_text()
        self.assertNotIn('<script>bad()</script>',page)

    def test_legacy_large_array_is_spooled_and_identity_samples_remain_values(self):
        path=self.root/'result.json'
        path.write_text(json.dumps(dict(results={'perf.device':dict(data=[dict(name='x')]*301,
            coverage=dict(workload_identity=dict(samples=['one','two'])))})))
        with opened_model() as model:
            doc=read_document(path,model)
            self.assertEqual(len(doc['results']['perf.device']['data']),301)
            self.assertEqual(doc['results']['perf.device']['coverage']['workload_identity']['samples'],['one','two'])

    def test_repeated_documents_do_not_mix_disk_backed_rows(self):
        a=self.root/'a.json';b=self.root/'b.json'
        a.write_text(json.dumps(dict(results={'perf.op':dict(data=[dict(name='a')])})))
        b.write_text(json.dumps(dict(results={'perf.op':dict(data=[dict(name='b')])})))
        with opened_model() as model:
            first=read_document(a,model);second=read_document(b,model)
            self.assertEqual(list(first['results']['perf.op']['data']),[dict(name='a')])
            self.assertEqual(list(second['results']['perf.op']['data']),[dict(name='b')])

    def test_detection_relative_time_uses_onset_not_last_revision(self):
        finding=dict(rule='hang.no_progress',rank=0,task_relative_ns=20,window=[120,None],
            first_detected=150,observed_at=190,evidence=[],detail={'name':'layer'},config_version='known')
        out=self.generate(dict(results={},diagnosis=dict(status='available',findings=[finding],evidence=[],
            rule_settings={'known':dict(progress_timeout_s=60)},progress_coverage=[])))
        row=next(r['value'] for r in self.snapshot(out) if r.get('section')=='diagnosis')
        self.assertEqual(row['first_detected_relative_ns'],50)
        self.assertEqual(row['observed_since_onset_ns'],70)
        self.assertEqual(row['configured_timeout_ns'],60000000000)

    def test_large_evidence_producer_links_reach_late_page(self):
        from flagprof.report.render import render
        out=self.root/'paged';out.mkdir()
        with opened_model() as model:
            model.section('oracle.samples','精度样本',('call_key',),kind='evidence')
            for index in range(2501):
                model.add('oracle.samples',dict(call_key='call-'+str(index),input_sources_ref=['call-2500'] if index==0 else []))
            render(model,out)
        self.assertIn('oracle-samples-6.html#row-2500',(out/'report.html').read_text())
        self.assertIn('oracle-samples-6.html#row-2500',(out/'evidence/oracle-samples-1.html').read_text())
        self.assertIn('id="row-2500"',(out/'evidence/oracle-samples-6.html').read_text())

    def test_shared_oracle_coverage_is_read_once(self):
        coverage=dict(amplification=[dict(boundary='op',ratio=2)])
        out=self.generate(dict(results={
            'oracle.first_divergence':dict(data=[dict(first_inequal=None)],coverage=coverage),
            'oracle.op_sensitivity':dict(data=[dict(boundary='op')],coverage=coverage),
            'oracle.layer':dict(data=[dict(scope='layer')],coverage=coverage)}))
        sections=[r['value'] for r in self.snapshot(out) if r['type']=='section']
        self.assertEqual([s['id'] for s in sections if s['title']=='误差放大观测'],['oracle.op_sensitivity.amplification'])

    def test_monitor_curve_uses_metric_sample_time(self):
        from unittest.mock import patch,MagicMock
        from flagprof.monitor.report import generate
        session=self.root/'session';session.mkdir()
        state=dict(id='time-source',started_raw_ns=1,ended_raw_ns=10000000000,status='stopped',gaps=[],
                   inventory=dict(descriptors=[dict(name='power.board',unit='W')]))
        view=MagicMock();view.__enter__.return_value=view
        view.iter_monitor_events.return_value=iter([dict(ts=9000000000,dev='device:0',values={'power.board':123},telemetry=dict(sample_times={'power.board':2000000000}))])
        view.iter_monitor_minutes.return_value=iter([])
        view.monitor_coverage.return_value=dict(read_report=dict(errors=[]))
        out=self.root/'timed-report'
        with patch('flagprof.monitor.report.load',return_value=state),patch('flagprof.monitor.report.StoreView',return_value=view),patch('flagprof.monitor.diagnostic_report.panel',return_value={}):
            from flagprof.monitor.report import _generate
            _generate(session,out,last='1h')
        row=json.loads((out/'result.json').read_text())['series'][0]
        self.assertEqual([p for p in row['samples_points'] if p[1] is not None],[[2000000000,123]])
        self.assertEqual(row['unit'],'W')


if __name__=='__main__':
    unittest.main()
