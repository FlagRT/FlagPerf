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
