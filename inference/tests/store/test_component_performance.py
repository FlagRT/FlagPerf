# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Performance groups count real forwards even when precision is disabled."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flagprof.cli import main
from flagprof.report.group import generate_group
from flagprof.store import create_run, Writer, finalize


class ComponentPerformanceTests(unittest.TestCase):
    def fixture(self, root, *, missing=False, gap=False, roots=True, generation=False):
        group = root/'group';group.mkdir();ranks=[]
        for rank in range(2):
            run=create_run(group/('rank-'+str(rank)))
            identity=dict(model_files={'qwen':'fixture'},oracle_id='fixture',inputs_sha256='same',
                samples=['s'],batches=[['s']],dtype='bfloat16',attention='eager',pooling='last',
                device='npu:'+str(rank),software={},numerics={},flaggems='off',include=None,
                parallelism={'world_size':2,'plan':'auto'},rank=rank,comm_backend='hccl')
            if generation:identity.update(task='generate',generation={'mode':'performance'})
            state=run/'attachments/f/state.json';state.parent.mkdir(parents=True)
            state.write_text(json.dumps(dict(id='f',oracle_identity=identity)))
            with Writer(run,rank=rank) as writer:
                if roots:
                    for i,phase in enumerate(('warmup','measure','measure')):
                        writer.emit(dict(t='span',ph='x',cat='module',name='<root>',corr=i+1,
                            ts=1000*i,dur=900,attrs={'phase':phase,'cls':'Qwen3Model'}))
                        writer.emit(dict(t='span',ph='x',cat='module',name='<root>.layers.0',corr=i+10,
                            parent=i+1,ts=1000*i+1,dur=800,attrs={'phase':phase}))
                if generation:
                    self.assertTrue(writer.emit(dict(t='span',ph='x',cat='user',corr=999,name='generation_step_complete',ts=1,dur=1,
                        attrs=dict(request_id='r',samples=['s'],mode='performance',generation_step=0,
                            tokens=[1],done=True,stop_reasons=['max_new_tokens']))))
                count=73 if generation else 168
                for seq in range(count-(1 if rank==1 and missing else 0)):
                    writer.emit(dict(t='comm',op='allgather' if generation and seq==72 else 'allreduce',
                        group='g/model',group_seq=seq+(1 if gap and rank==1 and seq>=10 else 0),
                        bytes=4,dur=1,lib='hccl',phase='enqueue',dtype='torch.float32',count=1,
                        ext={'communication':{'role':'model','logical_input_bytes':4,'logical_output_bytes':4,
                                              'device_duration_ns':None}}))
            finalize(run)
            ranks.append(dict(rank=rank,group_id='g',status='complete',run=str(run.relative_to(group))))
        manifest=group/'group.json'
        manifest.write_text(json.dumps(dict(schema='flagprof-group/1',group_id='g',status='complete',
            world_size=2,backend='hccl',ranks=ranks)))
        return manifest

    def test_perf_without_tensors_counts_warmup_and_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);generate_group(self.fixture(root),root/'report')
            result=json.loads((root/'report/result.json').read_text())
            self.assertEqual(result['status'],'complete')
            for row in result['rank_counts']:
                self.assertEqual(row['forwards'],3)
                self.assertEqual(row['forward_phases'],dict(warmup=1,measure=2,unknown=0))
                self.assertEqual(row['expected_allreduce'],168)
                self.assertTrue(row['model_count_matches'])

    def test_missing_collective_and_sequence_gap_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);generate_group(self.fixture(root,missing=True),root/'report')
            result=json.loads((root/'report/result.json').read_text())
            self.assertEqual(result['status'],'partial')
            self.assertFalse(result['rank_counts'][1]['model_count_matches'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'sequence missing or out of order'):
                generate_group(self.fixture(root,gap=True),root/'report')

    def test_missing_forward_evidence_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);generate_group(self.fixture(root,roots=False),root/'report')
            result=json.loads((root/'report/result.json').read_text())
            self.assertEqual(result['status'],'partial')
            self.assertIsNone(result['rank_counts'][0]['expected_allreduce'])
            self.assertIsNone(result['rank_counts'][0]['model_count_matches'])

    def test_onerec_uses_steps_independently_of_root_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);generate_group(self.fixture(root,generation=True),root/'report')
            result=json.loads((root/'report/result.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['rank_counts'][0]['expected_allreduce'],72)
            self.assertEqual(result['rank_counts'][0]['expected_allgather'],1)

    def test_public_presets_forward_component_selection(self):
        for preset in ('perf','modules','perf-lite'):
            target='flagprof.record.record' if preset=='perf-lite' else 'flagprof.offline.capture.record_final'
            with self.subTest(preset=preset), patch(target,return_value=0) as record:
                self.assertEqual(main(['record','--store','unused','--vendor','ascend','--preset',preset,
                    '--flagtree','off','--vendor-compiler','/vendor','--','python','-c','pass']),0)
                self.assertEqual(record.call_args.kwargs['config']['flagtree'],'off')
                self.assertEqual(record.call_args.kwargs['config']['vendor_compiler'],'/vendor')
            with patch('flagprof.launch.launch',return_value=0) as launch:
                self.assertEqual(main(['launch','--store','unused','--preset',preset,'--flagcx','on',
                    '--flagtree','off','--vendor-compiler','/vendor','--','python','-c','pass']),0)
                self.assertEqual(launch.call_args.kwargs['flagcx'],'on')
                self.assertEqual(launch.call_args.kwargs['config']['flagtree'],'off')
                self.assertFalse(any(k.startswith('tensor_') for k in launch.call_args.kwargs['config']))

    def test_cross_rank_device_alignment_ignores_only_rank_identity(self):
        from flagprof.report import generate
        for changed in (False,True):
            with self.subTest(changed_dtype=changed), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);manifest=self.fixture(root)
                def rank_report(run, out, **kwargs):
                    result=generate(run,out,**kwargs);rank=int(out.name[-1])
                    value=json.loads((out/'result.json').read_text())
                    value['results']['perf.gap']['coverage']=dict(
                        workload_identity=dict(model_files={'m':'same'},inputs_sha256='same',samples=['s'],
                            dtype='fp32' if changed and rank==1 else 'bf16',attention='eager',
                            software={'torch':'same'},numerics={'mode':'default'},rank=rank,device='npu:'+str(rank)),
                        window_identity=dict(warmup=1,forwards=2,repeats=1,profiler='on',dispatch='off',window_kind='official_capture'))
                    (out/'result.json').write_text(json.dumps(value))
                    (out/'exports/communication-device-calls.jsonl').write_text(''.join(json.dumps(dict(
                        group='g/model',group_seq=i,rank=rank,device_union_ns=100+rank,device_sum_ns=100+rank,
                        device_records=1,evidence=[]))+'\n' for i in range(168)))
                    return result
                with patch('flagprof.report.group.generate',side_effect=rank_report):
                    generate_group(manifest,root/'report')
                result=json.loads((root/'report/result.json').read_text())
                self.assertEqual(result['comparison']['status'],'not_comparable' if changed else 'comparable')
                if changed:
                    self.assertEqual(result['comparison']['differences'][0]['field'],'workload_identity.dtype')
                    self.assertNotIn('cross_rank_difference_ns',result['device_calls'][0])
                else:
                    self.assertEqual(result['device_link_status'],'linked')
                    self.assertEqual(result['device_calls'][0]['cross_rank_difference_ns'],1)


if __name__=='__main__':unittest.main()
