# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""One small public-entry chain for capture, reopening and both evidence levels."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from flagprof.store import StoreView
from flagprof.tensor_data import verify_blob


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch CPU runtime')
class AccuracyEntryTests(unittest.TestCase):
    def test_collective_return_is_not_a_completed_tensor_producer(self):
        import torch
        from types import SimpleNamespace
        from flagprof import attach
        from flagprof.store import create_run,finalize
        calls=[]
        class Collective:
            _schema=SimpleNamespace(name='c10d::allreduce_')
            def __call__(self,value):calls.append(value);return [value],object()
        with tempfile.TemporaryDirectory() as tmp:
            run=create_run(tmp)
            with attach(run,'ascend',preset='accuracy') as handle:
                probe=handle.probe_manager.tensor;value=torch.ones(2)
                probe.remember(value,'previous-producer')
                result=probe.invoke(Collective(),(value,),{})
                self.assertIs(result[0][0],value);self.assertEqual(len(calls),1)
                self.assertEqual(probe.source(value),'unknown')
                self.assertEqual(probe.metrics['collective_boundaries_skipped'],1)
                self.assertEqual(probe.metrics['events'],0)
            finalize(run)
            with StoreView(run) as view:self.assertEqual(list(view.iter_events(types=('tensor',))),[])
    def test_boundary_presets_and_drill_capture_actual_scope(self):
        import torch
        from flagprof import attach
        from flagprof.store import create_run,finalize
        from tests.probes.cpu_target import Model
        torch.manual_seed(531);model=Model().eval();inputs=torch.arange(2).reshape(1,2)
        endpoints={('<root>','pooled'),('<root>','output')}
        layers={('<root>.embedding','out')}|{('<root>.layers.'+str(i),'out') for i in range(6)}
        cases=[('default',{},layers|endpoints),
               ('layer',dict(tensor_boundaries='layer'),layers|endpoints),
               ('model',dict(tensor_boundaries='model'),endpoints),
               ('empty',dict(tensor_boundaries=[]),endpoints),
               ('custom',dict(tensor_boundaries=['<root>.layers.2']),{('<root>.layers.2','out')}|endpoints),
               ('drill',dict(tensor_boundaries='layer',tensor_ops_in='<root>.layers.0'),layers|endpoints),
               ('model-drill',dict(tensor_boundaries='model',tensor_ops_in='<root>.layers.0'),endpoints)]
        with tempfile.TemporaryDirectory() as directory:
            outputs={}
            for label,config,expected in cases:
                with self.subTest(label=label):
                    run=create_run(Path(directory)/label)
                    # CPU fixture exercises tensor dispatch without any accelerator.
                    with patch('flagprof.devices.accelerator',return_value=True):
                        with attach(run,'ascend',preset='accuracy',config=config) as handle:
                            handle.register_model(model);handle.bind_tensor_inputs(['s'],{'input_ids':inputs})
                            with torch.inference_mode():
                                pooled=model(inputs).mean(1)
                                handle.record_tensor('pooled',pooled);handle.record_tensor('output',pooled)
                            self.assertEqual(handle.probe_manager.errors, [])
                    finalize(run)
                    with StoreView(run) as view:rows=[r.record for r in view.iter_events(types=('tensor',))]
                    observed={(r['call_key'].split('#')[0],r['boundary']) for r in rows if r['boundary'] in ('out','pooled','output')}
                    self.assertEqual(observed,expected)
                    ops=[r for r in rows if r['boundary'].startswith('op:')]
                    self.assertEqual(bool(ops),'tensor_ops_in' in config)
                    self.assertTrue(all(r['call_key'].split('#')[0] == '<root>.layers.0' or
                                        r['call_key'].split('#')[0].startswith('<root>.layers.0.') for r in ops))
                    outputs[label]=next(r['digest'] for r in rows if r['boundary']=='output')
            self.assertEqual(len(set(outputs.values())),1,'capture scope must preserve the model output')

    def test_fixed_input_identity_and_one_time_hashes(self):
        import torch
        from flagprof.runner import fixed
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prepared=root/'prepared';prepared.mkdir();model=root/'model';model.mkdir()
            (model/'config.json').write_text('{}')
            torch.save([{'inputs':{'input_ids':torch.tensor([[0,1,2],[3,4,5]]),
                'attention_mask':torch.tensor([[0,1,1],[1,1,1]])}}],prepared/'inputs.pt')
            (prepared/'forward-inputs.json').write_text(json.dumps({'batches':[{'sample_ids':['a','b']}]}))
            (prepared/'samples.json').write_text('[]')
            names=['prepared/inputs.pt','prepared/forward-inputs.json','prepared/samples.json']
            manifest=dict(oracle_id='fixture',files={n:dict(sha256=fixed.sha(root/n)) for n in names},
                contract={'model_files':{'config.json':fixed.sha(model/'config.json')},'inputs':{'batch_size':2}})
            (root/'manifest.json').write_text(json.dumps(manifest))
            with patch.object(fixed,'sha',wraps=fixed.sha) as hashes:
                rows,identity=fixed.load(root,samples=['a'],batch_size=1,model=model)
                self.assertEqual(hashes.call_count,4)
                # Reusing prepared inputs in forward loops performs no file validation.
                for _ in range(3):self.assertEqual(rows[0][1]['input_ids'].tolist(),[[1,2]])
                self.assertEqual(hashes.call_count,4)
            self.assertEqual(identity['batches'],[dict(samples=['a'],shape=[1,2],valid_tokens=[2])])
            (prepared/'inputs.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'fixed input SHA mismatch'):fixed.load(root)

    def test_multi_output_passes_and_weak_source_ownership(self):
        import torch
        from flagprof import attach
        from flagprof.store import create_run,finalize
        class Layer(torch.nn.Module):
            def forward(self,x):return torch.max(x,dim=1)
        with tempfile.TemporaryDirectory() as directory:
            run=create_run(directory);model=Layer()
            with patch('flagprof.devices.accelerator',return_value=True):
                with attach(run,'ascend',preset='accuracy',config=dict(tensor_ops_in='<root>',tensor_boundaries=['<root>'])) as handle:
                    handle.register_model(model)
                    with self.assertRaisesRegex(ValueError,'batch 1'):handle.bind_tensor_inputs(['s','other'],{})
                    handle.bind_tensor_inputs(['s'],{})
                    for _ in range(2):model(torch.tensor([[1.,2.]]))
                    probe=handle.probe_manager.tensor
                    old=torch.ones(2);other=torch.ones(2)
                    probe.remember(old,'stale');probe.sources[id(other)]=probe.sources[id(old)]
                    del old
                    self.assertEqual(probe.source(other),'unknown')
                    base=torch.ones(2);alias=base.view(2);probe.remember(alias,'before')
                    base.add_(1)
                    self.assertEqual(probe.source(alias),'unknown')
            finalize(run)
            with StoreView(run) as view:rows=[r.record for r in view.iter_events(types=('tensor',))]
            keys=[r['call_key'] for r in rows]
            self.assertEqual(len(keys),len(set(keys)))
            operations=[r['boundary'] for r in rows if r['boundary'].startswith('op:')]
            self.assertEqual(operations,['op:aten::max.dim/0/0','op:aten::max.dim/0/1'])
            self.assertTrue(any('#1#out#s' in k for k in keys))

    def test_left_padding_and_sample_keys_survive_batch_partition(self):
        import torch
        from flagprof import attach
        from flagprof.store import create_run,finalize
        from tests.probes.cpu_target import Model
        torch.manual_seed(531);model=Model().eval()
        with tempfile.TemporaryDirectory() as directory:
            observations=[]
            for label,ids,samples,mask in [('batch',[[0,1],[2,3]],['a','b'],[[0,1],[1,1]]),('single',[[0,1]],['a'],[[0,1]])]:
                run=create_run(Path(directory)/label)
                with attach(run,'ascend',preset='accuracy') as handle:
                    inputs=torch.tensor(ids);handle.register_model(model)
                    handle.bind_tensor_inputs(samples,dict(input_ids=inputs,attention_mask=torch.tensor(mask)))
                    with torch.inference_mode():model(inputs)
                finalize(run)
                with StoreView(run) as view:rows=[r.record for r in view.iter_events(types=('tensor',)) if r.record['call_key'].endswith('#a')]
                self.assertTrue(all(r['shape']==[1,8] and r['mask']['valid']==1 for r in rows))
                observations.append({r['call_key'] for r in rows})
            self.assertEqual(observations[0],observations[1])

    def test_statistics_sampling_and_blob_budget(self):
        import hashlib
        import torch
        from flagprof import attach
        from flagprof.store import create_run,finalize
        with tempfile.TemporaryDirectory() as directory:
            run=create_run(Path(directory)/'store')
            with attach(run,'ascend',preset='accuracy',config=dict(tensor_blob='all',tensor_blob_bytes=16)) as handle:
                handle.bind_tensor_inputs(['s'],{})
                handle.record_tensor('small',torch.tensor([[1.,2.]]))
                handle.record_tensor('sampled',torch.arange(513.).reshape(1,513))
                handle.record_tensor('nonfinite',torch.tensor([[float('nan'),float('inf')]]))
                self.assertEqual(handle.probe_manager.tensor.metrics['blob_skipped'],2)
            finalize(run)
            with StoreView(run) as view:rows={s.record['boundary']:s.record for s in view.iter_events(types=('tensor',))}
            self.assertEqual(rows['small']['stats']['mean'],1.5)
            self.assertEqual(rows['sampled']['chunks'][0]['count'],8)
            self.assertEqual(rows['sampled']['chunks'][-1]['count'],9)
            phase=int(hashlib.sha256(rows['sampled']['call_key'].encode()).hexdigest()[:16],16)%2
            self.assertEqual(rows['sampled']['probe'],list(range(phase,phase+512,2)))
            self.assertIsNone(rows['nonfinite']['stats']['mean'])
            self.assertEqual(rows['nonfinite']['stats']['nan'],1)
            self.assertNotIn('blob',rows['sampled'])
            selected=create_run(Path(directory)/'prefix')
            with attach(selected,'ascend',preset='accuracy',config=dict(tensor_blob=['<root>#0#selected#s'],tensor_boundaries=[])) as handle:
                handle.bind_tensor_inputs(['s'],{})
                handle.record_tensor('selected',torch.ones(1,2));handle.record_tensor('other',torch.ones(1,2))
            finalize(selected)
            with StoreView(selected) as view:rows={s.record['boundary']:s.record for s in view.iter_events(types=('tensor',))}
            self.assertIn('blob',rows['selected']);self.assertNotIn('blob',rows['other'])

    def test_selected_scope_tracks_inputs_signatures_and_repeated_calls(self):
        import torch
        from flagprof import attach
        from flagprof.store import create_run,finalize
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(2));self.register_buffer('scale',torch.tensor([2.,3.]))
            def forward(self,x,*,bias):
                y=x*self.weight
                y=y*self.scale
                y.add_(bias)
                return y.view(1,2)
        with tempfile.TemporaryDirectory() as directory:
            run=create_run(Path(directory)/'store');model=Layer().eval()
            with patch('flagprof.devices.accelerator',return_value=True):
                with attach(run,'ascend',preset='accuracy',config=dict(tensor_ops_in='<root>',tensor_blob='all',tensor_boundaries=['<root>'])) as handle:
                    handle.register_model(model);handle.bind_tensor_inputs(['s'],{'input_ids':torch.ones(1,2)})
                    expected=model(torch.ones(1,2),bias=torch.ones(1,2))
                    self.assertTrue(torch.equal(expected,torch.tensor([[3.,4.]])))
                    self.assertFalse(handle.probe_manager.tensor.failed,handle.state)
            finalize(run)
            with StoreView(run) as view:rows=[r.record for r in view.iter_events(types=('tensor',))]
            self.assertEqual(len({r['call_key'] for r in rows}),len(rows))
            operations=[r for r in rows if r['boundary'].startswith('op:')]
            self.assertEqual(operations[0]['src'][0],'<root>#0#in/0#s')
            self.assertEqual(len(operations),4,'probe snapshots must not appear as model operators')
            self.assertTrue(any('/1/0' in r['boundary'] for r in operations))
            self.assertTrue(all('sid' in r and 'call_config' in r['ext'] for r in operations))
            self.assertTrue(any(r['boundary'].startswith('in/kw:bias') for r in rows))
            self.assertTrue(any(r['boundary']=='buffer:scale' for r in rows))

    def test_public_record_and_oracle_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runs={}
            paths=root/'boundaries.json'
            paths.write_text(json.dumps(['<root>.embedding']+['<root>.layers.'+str(i) for i in range(6)]))
            cases=[('ref',0,'all',[]),('cmp',.25,'all',['--tensor-boundaries','layer']),
                   ('sampled',0,'none',['--tensor-boundaries',str(paths)]),
                   ('model-ref',0,'all',['--tensor-boundaries','model']),
                   ('model-cmp',.25,'all',['--tensor-boundaries','model'])]
            for label,delta,blob,boundaries in cases:
                subprocess.run([sys.executable,'-B','-m','flagprof','record','--store',str(root/label),'--vendor','ascend',
                    '--preset','accuracy','--tensor-blob',blob,*boundaries,'--',sys.executable,'-B','-m','tests.probes.accuracy_target',
                    '--outputs',str(root/(label+'-outputs')),'--delta',str(delta)],check=True,timeout=90)
                runs[label]=next(p.parent for p in (root/label).glob('*/manifest.json'))
                with StoreView(runs[label]) as view:
                    records=[s.record for s in view.iter_events() if s.record['t']=='tensor']
                    self.assertEqual(len(records),2 if label.startswith('model-') else 9)
                    if label.startswith('model-'):
                        self.assertEqual({r['boundary'] for r in records},{'pooled','output'})
                    self.assertEqual(len({r['call_key'] for r in records}),len(records))
                    for record in records:
                        if blob=='all':verify_blob(runs[label],record)
                        else:self.assertNotIn('blob',record)
            for label,ref,cmp in [('self','ref','ref'),('changed','ref','cmp'),('sampled','ref','sampled'),
                                  ('model-changed','model-ref','model-cmp')]:
                out=root/(label+'-report')
                subprocess.run([sys.executable,'-B','-m','flagprof','report','oracle','--ref',str(runs[ref]),'--cmp',str(runs[cmp]),'--out',str(out)],check=True)
                result=json.loads((out/'result.json').read_text())['results']
                first=result['oracle.first_divergence']['data'][0]['first_inequal']
                self.assertEqual(result['oracle.identity_diff']['status'],'ok')
                if label in ('changed','model-changed'):
                    self.assertEqual(first['boundary'],'output');self.assertAlmostEqual(first['max_abs'],.25,places=6)
                else:self.assertIsNone(first)
                self.assertTrue((out/'report.html').exists())


if __name__=='__main__':unittest.main()
