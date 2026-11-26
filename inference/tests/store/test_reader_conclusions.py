"""Regressions from reading real T19 result reports as an end user."""
import json
from pathlib import Path
import tempfile
import unittest

from flagprof.report.product import write_report, redraw


class ReaderConclusions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def report(self, doc, name='report'):
        out = self.root/name
        out.mkdir()
        (out/'result.json').write_text(json.dumps(doc))
        write_report(out)
        records = [json.loads(line) for line in (out/'report-data.jsonl').read_text().splitlines()]
        return out, records[0]['value'], records

    def test_per_rank_comparison_has_values_and_saved_identity(self):
        condition = dict(device='npu:0,1', dtype='bfloat16', comm_backend='hccl')
        sources = [dict(report=path, context=dict(workload_identity=dict(condition,comm_backend=backend)),
            runner=dict(groups=[dict(rank=0,scope='request',mean_ns=mean,observations=90,independent_repeats=0)]))
            for path,backend,mean in [('a','hccl',3100),('b','flagcx',3000)]]
        row = dict(source='b',rank=0,scope='request',metric='mean_ns',baseline=3100,candidate=3000,delta=-100,
                   relative_change=-100/3100,unit='ns',comparison_status='comparable')
        out, meta, records = self.report(dict(schema='device-comparison/1',sources=sources,metric_comparisons=[row]))
        text = (out/'report.md').read_text()
        self.assertIn('3.1 μs → 3 μs',text)
        self.assertIn('rank 0 request 平均时延变化',text)
        self.assertIn('独立进程重复依据不足',text)
        self.assertNotIn('未保存完整任务身份',text)
        section = next(r['value'] for r in records if r['type']=='section' and r['value']['id']=='comparison.metrics')
        self.assertIn('rank',section['columns'])
        self.assertIn('source_label',section['columns'])
        self.assertEqual(next(f for f in meta['findings'] if f['section']=='comparison.metrics')['row_index'],0)

    def test_three_way_prioritizes_output_and_keeps_mixed_directions(self):
        rows = [dict(scope=scope,boundary=boundary,metric=metric,reference_to_baseline=baseline,
                     reference_to_candidate=candidate,delta=candidate-baseline,relative_change=(candidate-baseline)/baseline,
                     baseline_to_candidate=.1,sample='sample',level='full')
            for scope,boundary,metric,baseline,candidate in [('<root>.pow','out','max_abs',392192,392216),
                ('<root>','output','max_abs',.004,.0042),('<root>','output','relative_l2',.039,.038)]]
        out,meta,_ = self.report(dict(schema='oracle-three-way/1',comparisons=rows))
        summaries=[f for f in meta['findings'] if f['level']=='result']
        self.assertEqual(len(summaries),2)
        self.assertIn('误差增大',summaries[0]['detail'])
        self.assertIn('误差减小',summaries[1]['detail'])
        self.assertNotIn('392216',(out/'report.md').read_text())
        self.assertEqual([f['row_index'] for f in summaries],[1,2])

    def test_oom_and_exit_join_direct_events_without_rewriting_detector_time(self):
        identity=['event','run',0,1]
        exit_identity=['event','monitor',0,9]
        doc=dict(diagnosis=dict(status='available',coverage=dict(reference=dict(
            diagnostic_identity=dict(run_id='run'),devices=[dict(logical_device='npu:0',phyid=1)])),
            findings=[dict(rule='oom.device',rank=0,run_id='run',task_relative_ns=20,window=[120,None],
                first_detected=125,evidence=[identity],detail=dict(message='allocate 61 GiB')),
                dict(rule='proc.exit',rank=0,run_id='run',window=[160,None],observed_at=160,evidence=[exit_identity])],
            evidence=[dict(identity=identity,record=dict(ts=120,clk='same',detail=dict(started_ns=100,identity=dict(run_id='run')))),
                dict(identity=exit_identity,record=dict(ts=150,clk='same',detail=dict(returncode=23)))]))
        out,meta,records=self.report(doc)
        rows=[r['value'] for r in records if r.get('section')=='diagnosis']
        self.assertEqual(rows[0]['first_detected_relative_ns'],25)
        self.assertEqual(rows[1]['occurrence_task_relative_ns'],50)
        self.assertEqual(rows[1]['observed_at'],160)
        self.assertEqual(rows[1]['trigger'],dict(returncode=23))
        text=(out/'report.md').read_text()
        self.assertIn('allocate 61 GiB',text)
        self.assertIn('进程返回码：23',text)
        self.assertIn('事件未记录活动层或算子',text)

    def test_no_cross_clock_task_time_is_invented(self):
        _,_,records=self.report(dict(diagnosis=dict(status='available',findings=[dict(rule='proc.exit',run_id='r',evidence=[['event','e',0,1]])],
            evidence=[dict(identity=['event','e',0,1],record=dict(ts=150,clk='b',detail=dict(returncode=23))),
                      dict(identity=['event','e',0,2],record=dict(ts=120,clk='a',detail=dict(started_ns=100,identity=dict(run_id='r'))))])))
        row=next(r['value'] for r in records if r.get('section')=='diagnosis')
        self.assertNotIn('occurrence_task_relative_ns',row)

    def test_two_rank_zero_runs_keep_alerts_and_exit_receipts_with_their_owner(self):
        normal = dict(run_id='first',rank=0,registered_execution=1,returncode=0,
                      start_clock='host',exit_clock='host',started_ns=100,exit_ns=130)
        failed = dict(normal,run_id='second',returncode=1,started_ns=150,exit_ns=200)
        alerts = [dict(id='exit',rule='proc.exit',rank=0,run_id='second',task_relative_ns=50),
                  dict(id='oom',rule='oom.device',rank=0,run_id='second',task_relative_ns=20)]
        doc = dict(diagnosis=dict(status='available',progress_coverage=[failed,normal],findings=alerts))
        _, meta, records = self.report(doc)
        self.assertEqual(meta['banners'][0]['title'],'设备 OOM、异常退出都属于任务 2')
        self.assertIn('没有重叠',meta['banners'][0]['detail'])
        roster = [r['value'] for r in records if r.get('section')=='diagnosis.coverage']
        self.assertEqual([(r['run_id'],r['alert_count']) for r in roster],[('first',0),('second',2)])
        events = [r['value'] for r in records if r.get('section')=='diagnosis']
        self.assertEqual([(r['task_label'],r['rule']) for r in events],[('任务 2','oom.device'),('任务 2','proc.exit')])
        normal['exit_ns'] = 180
        _, meta, _ = self.report(doc,'overlap')
        self.assertIn('部分任务的观测区间重叠',meta['banners'][0]['detail'])
        failed['start_clock'] = 'other'
        _, meta, _ = self.report(doc,'unmapped')
        self.assertNotIn('重叠',meta['banners'][0]['detail'])

    def test_revisions_count_as_one_alert_and_use_the_update_time(self):
        opened = dict(id='one',rule='hang.no_progress',rank=0,run_id='r',window=[110,None],
                      revision=1,state='open',task_relative_ns=10,first_detected=120,observed_at=120)
        resolved = dict(opened,revision=2,state='resolved',observed_at=140,window=[110,140])
        _, meta, records = self.report(dict(diagnosis=dict(status='available',findings=[resolved,opened])))
        self.assertIn('已解除',meta['banners'][0]['title'])
        events = [r['value'] for r in records if r.get('section')=='diagnosis']
        self.assertEqual([r['diagnostic_time_ns'] for r in events],[20,40])
        roster = next(r['value'] for r in records if r.get('section')=='diagnosis.coverage')
        self.assertEqual(roster['alert_count'],1)

    def test_node_axis_uses_session_origin_and_matching_source_clocks(self):
        # Work above JavaScript's safe integer range and request only the tail
        # of the resource window. Diagnostic times still use the session start.
        origin = 2**54
        first = dict(run_id='a',rank=0,started_ns=origin+10,start_clock='host',
                     exit_ns=origin+45,exit_clock='host',returncode=0)
        second = dict(first,run_id='b',started_ns=origin+35,exit_ns=origin+90,returncode=1)
        third = dict(first,run_id='c',start_clock='other',exit_clock='other')
        ref = ['event','fault',0,1]
        oom = dict(id='oom',rule='oom.device',rank=0,run_id='b',evidence=[ref],
                   observed_at=origin+70,window=[origin+58,None],task_relative_ns=23)
        source = dict(identity=ref,record=dict(ts=origin+58,clk='host'))
        doc = dict(schema='monitor-report/1',session_clock='host',window=[origin+80,origin+100],
                   coverage=dict(reference=dict(window=[origin,origin+100])),
                   diagnosis=dict(status='available',progress_coverage=[first,second,third],findings=[oom],evidence=[source]))
        _, meta, records = self.report(doc)
        self.assertEqual(meta['diagnosis_session_ns'],100)
        roster = {r['value']['run_id']:r['value'] for r in records if r.get('section')=='diagnosis.coverage'}
        self.assertEqual((roster['a']['node_start_ns'],roster['a']['node_exit_ns']),(10,45))
        self.assertEqual((roster['b']['node_start_ns'],roster['b']['node_exit_ns']),(35,90))
        self.assertNotIn('node_start_ns',roster['c'])
        self.assertNotIn('node_exit_ns',roster['c'])
        event = next(r['value'] for r in records if r.get('section')=='diagnosis')
        self.assertEqual((event['diagnostic_time_ns'],event['node_event_ns']),(23,58))
        source['record']['clk'] = 'other'
        _, _, records = self.report(doc,'foreign-event')
        event = next(r['value'] for r in records if r.get('section')=='diagnosis')
        self.assertNotIn('node_event_ns',event)
        doc['diagnosis']['evidence'] = []
        _, _, records = self.report(doc,'offset-only')
        event = next(r['value'] for r in records if r.get('section')=='diagnosis')
        self.assertEqual(event['node_event_ns'],58)
        del doc['session_clock']
        _, meta, _ = self.report(doc,'ambiguous-node-clock')
        self.assertNotIn('diagnosis_session_ns',meta)

    def test_refresh_rebuilds_stale_snapshot_and_preserves_machine_result(self):
        source,_,_=self.report(dict(results={'perf.device':dict(data=[dict(name='maximum',category='compute',device_work_ns=5)])}))
        original=(source/'result.json').read_bytes()
        snapshot=source/'report-data.jsonl'
        snapshot.write_text(snapshot.read_text().replace('全采集窗口计算热点','OLD PRESENTATION'))
        old=self.root/'old';redraw(source,old)
        new=self.root/'new';redraw(source,new,refresh_view=True)
        self.assertIn('OLD PRESENTATION',(old/'report.md').read_text())
        self.assertNotIn('OLD PRESENTATION',(new/'report.md').read_text())
        self.assertEqual(original,(new/'result.json').read_bytes())

    def test_missing_clock_does_not_establish_shared_clock(self):
        _,_,records=self.report(dict(diagnosis=dict(status='available',findings=[dict(rule='proc.exit',run_id='r',evidence=[['event','e',0,1]])],
            evidence=[dict(identity=['event','e',0,1],record=dict(ts=150,detail=dict(returncode=23))),
                      dict(identity=['event','e',0,2],record=dict(ts=120,detail=dict(started_ns=100,identity=dict(run_id='r'))))])))
        row=next(r['value'] for r in records if r.get('section')=='diagnosis')
        self.assertNotIn('occurrence_task_relative_ns',row)

    def test_findings_target_original_row_beyond_initial_page_and_empty_panels_removed(self):
        out,meta,records=self.report(dict(results={'perf.device':dict(data=[dict(name='k'+str(i),category='compute',device_work_ns=i) for i in range(205)])}))
        finding=meta['findings'][0]
        self.assertEqual(finding['row_index'],204)
        self.assertIn('section=perf.device&row=204#perf.device',(out/'report.md').read_text())
        self.assertIn('data-focus-row="204"',(out/'report.html').read_text())
        out,_,records=self.report(dict(schema='device-comparison/1',sources=[]),'empty')
        self.assertFalse(any(r['type']=='section' for r in records))


if __name__ == '__main__':
    unittest.main()
