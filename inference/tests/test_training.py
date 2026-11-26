# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Training semantics: reference updates, accumulation and restart continuity."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None
from flagprof.training.engine import train
from flagprof.training.state import fingerprint


@unittest.skipIf(torch is None, 'training semantics require torch')
class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        torch.manual_seed(137); random.seed(137)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
        self.batches = [dict(x=torch.randn(2, 3), y=torch.tensor([0, 1])) for _ in range(4)]

    @staticmethod
    def loss(model, batch):
        return torch.nn.functional.cross_entropy(model(batch['x']), batch['y'])

    def run_model(self, name, model=None, **options):
        model = copy.deepcopy(self.model) if model is None else model
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        return model, train(model, optimizer, self.batches, self.loss,
                            outputs=self.root/name, steps=3, **options)

    def test_matches_uninstrumented_reference(self):
        model = copy.deepcopy(self.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        losses = []
        for index in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = self.loss(model, self.batches[index]); losses.append(float(loss.detach()))
            loss.backward(); optimizer.step()
        observed, result = self.run_model('observed')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(fingerprint(model.state_dict()), fingerprint(observed.state_dict()))
        self.assertEqual(losses, [row['loss'] for row in result['steps']])
        self.assertNotEqual(result['protocol']['initial_model_sha256'], result['final_model_sha256'])

    def test_accumulation_matches_full_batch(self):
        model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        full = {key: torch.cat([b[key] for b in self.batches[:2]]) for key in self.batches[0]}
        optimizer.zero_grad(set_to_none=True); self.loss(model, full).backward(); optimizer.step()
        other = copy.deepcopy(self.model)
        train(other, torch.optim.SGD(other.parameters(), lr=.01), self.batches, self.loss,
              outputs=self.root/'accumulated', steps=1, accumulation=2)
        for left, right in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)

    def test_restart_preserves_dropout_rng_and_optimizer(self):
        torch.manual_seed(222)
        initial = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.4), torch.nn.Linear(5, 2))
        torch.manual_seed(333)
        _, full = self.run_model('full', copy.deepcopy(initial))
        torch.manual_seed(333)
        checkpoint = self.root/'checkpoint.pt'
        _, paused = self.run_model('paused', copy.deepcopy(initial), checkpoint=checkpoint, stop_after=1)
        self.assertEqual(paused['status'], 'paused')
        torch.manual_seed(999)
        _, resumed = self.run_model('resumed', copy.deepcopy(initial), resume=checkpoint)
        self.assertEqual(resumed['start_step'], 1)
        self.assertEqual(full['final_model_sha256'], resumed['final_model_sha256'])
        self.assertEqual(full['final_optimizer_sha256'], resumed['final_optimizer_sha256'])
        self.assertEqual(full['steps'][1]['gradient_sha256'], resumed['steps'][0]['gradient_sha256'])

    def test_resume_rejects_protocol_difference(self):
        checkpoint = self.root/'checkpoint.pt'
        self.run_model('paused', checkpoint=checkpoint, stop_after=1)
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            self.run_model('invalid', resume=checkpoint, accumulation=2)
        result = json.loads((self.root/'invalid/training-result.json').read_text())
        self.assertEqual(result['status'], 'failed')

    def test_cli_resumes_v1_checkpoint_without_sharding_or_task_source(self):
        from flagprof.training.__main__ import main
        checkpoint = self.root/'v1.pt'
        options = ['--task', 'classification', '--steps', '2']
        main(options + ['--outputs', str(self.root/'paused'), '--stop-after', '1',
                        '--checkpoint', str(checkpoint)])
        saved = torch.load(checkpoint, weights_only=False)
        del saved['protocol']['task_source']
        del saved['protocol']['sharding']
        torch.save(saved, checkpoint)
        main(options + ['--outputs', str(self.root/'resumed'), '--resume', str(checkpoint)])
        result = json.loads((self.root/'resumed/training-result.json').read_text())
        self.assertEqual((result['status'], result['start_step']), ('complete', 1))

    @unittest.skipUnless(sys.platform == 'linux', 'Linux Store/attachment')
    def test_single_rank_observed_checkpoint_resumes_with_fresh_attachment(self):
        from flagprof.training.__main__ import main
        checkpoint = self.root/'observed.pt'
        settings = dict(FLAGPROF_TRAIN_OBSERVE=str(self.root/'observations'),
                        FLAGPROF_TRAIN_OBSERVE_VENDOR='ascend', FLAGPROF_TRAIN_PRESET='coverage',
                        FLAGPROF_TRAIN_MONITOR='off', FLAGPROF_TRAIN_REPORT='off')
        with patch.dict(os.environ, settings):
            main(['--steps', '2', '--outputs', str(self.root/'paused'), '--stop-after', '1',
                  '--checkpoint', str(checkpoint)])
            main(['--steps', '2', '--outputs', str(self.root/'resumed'), '--resume', str(checkpoint)])
        paused, resumed = [json.loads((self.root/name/'training-result.json').read_text())
                           for name in ('paused', 'resumed')]
        self.assertEqual((resumed['status'], resumed['start_step']), ('complete', 1))
        self.assertEqual(paused['protocol']['observation'], resumed['protocol']['observation'])
        self.assertEqual(resumed['protocol']['observation']['preset'], 'coverage')
        self.assertNotEqual(paused['attachment_id'], resumed['attachment_id'])

    def test_zero_drift_comparison_explicitly_declares_observation_only(self):
        from flagprof.training.__main__ import main
        from flagprof.training.compare import compare
        base, observed = self.root/'base', self.root/'observed'
        options = ['--device', 'cpu', '--task', 'classification', '--steps', '1']
        main(options + ['--outputs', str(base)])
        with patch.dict(os.environ, FLAGPROF_TRAIN_OBSERVE=str(self.root/'store'),
                        FLAGPROF_TRAIN_OBSERVE_VENDOR='ascend', FLAGPROF_TRAIN_PRESET='coverage',
                        FLAGPROF_TRAIN_MONITOR='off', FLAGPROF_TRAIN_REPORT='off'):
            main(options + ['--outputs', str(observed)])
        with self.assertRaisesRegex(ValueError, 'protocol differs: observation'):
            compare(base, observed)
        result = compare(base, observed, declare=['observation'])
        self.assertEqual(result['declared_protocol_differences'], ['observation'])
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['environments'][1]['observation']['preset'], 'coverage')
        metadata = observed/'training-result.json'
        run = json.loads(metadata.read_text())
        run['protocol']['lr'] *= 2
        metadata.write_text(json.dumps(run))
        with self.assertRaisesRegex(ValueError, 'protocol differs: lr'):
            compare(base, observed, declare=['observation'])

    def test_failure_keeps_completed_steps_and_does_not_reuse_output(self):
        calls = [0]
        def fail_second(model, batch):
            calls[0] += 1
            if calls[0] == 2: raise RuntimeError('injected failure')
            return self.loss(model, batch)
        model = copy.deepcopy(self.model)
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            train(model, torch.optim.SGD(model.parameters(), lr=.01), self.batches, fail_second,
                  outputs=self.root/'failure', steps=3)
        result = json.loads((self.root/'failure/training-result.json').read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['next_step'], 1)
        self.assertTrue((self.root/'failure/step-0.pt').is_file())
        with self.assertRaises(FileExistsError): self.run_model('failure')

    def test_performance_has_same_updates_without_snapshots(self):
        _, accuracy = self.run_model('accuracy')
        _, performance = self.run_model('performance', mode='performance')
        self.assertEqual(accuracy['final_model_sha256'], performance['final_model_sha256'])
        self.assertFalse(list((self.root/'performance').glob('step-*.pt')))


    def test_compare_detects_gradient_mutation(self):
        from flagprof.training.compare import compare
        self.run_model('left'); self.run_model('right')
        result = compare(self.root/'left', self.root/'right')
        self.assertEqual(result['status'], 'pass')
        path = self.root/'right/step-1.pt'
        snapshot = torch.load(path, weights_only=True)
        next(iter(snapshot['gradients'].values())).view(-1)[0] += 1
        torch.save(snapshot, path)
        result = compare(self.root/'left', self.root/'right')
        self.assertEqual(result['status'], 'different')
        self.assertTrue(any(r['status']=='different' and '/gradients/' in r['path'] for r in result['rows']))

    def test_integer_buffers_compare_exactly_above_float64_range(self):
        from flagprof.training.compare import compare
        self.run_model('left'); self.run_model('right')
        for name, offset in [('left', 0), ('right', 1)]:
            path=self.root/name/'step-1.pt'
            snapshot=torch.load(path,weights_only=True)
            snapshot['model']['counter']=torch.tensor([2**60+offset],dtype=torch.int64)
            torch.save(snapshot,path)
        result=compare(self.root/'left',self.root/'right',atol=10.,rtol=1.)
        self.assertEqual(result['status'],'different')
        row=next(r for r in result['rows'] if r['path']=='snapshot/model/counter')
        self.assertEqual(row['mismatches'],1)

    def test_cpu_bfloat16_amp_keeps_fp32_parameters(self):
        _, result = self.run_model('amp-bfloat16', amp='bfloat16')
        self.assertEqual(result['status'], 'complete')
        snapshot=torch.load(self.root/'amp-bfloat16/step-0.pt',weights_only=True)
        self.assertTrue(all(v.dtype==torch.float32 for v in snapshot['model'].values()))
        self.assertIn('scaler',snapshot)
        self.assertIsNone(snapshot['scaler'])

    def test_phase_restores_context_on_failure(self):
        import types
        from contextlib import contextmanager
        from unittest.mock import patch
        from flagprof.training import phase
        state = {'training': {'previous': True}}
        handle = types.SimpleNamespace(context=types.SimpleNamespace(probe_state=lambda:state))
        seen = []
        @contextmanager
        def scope(name, **attrs):
            try: yield
            finally: seen.append(attrs)
        with patch('flagprof.training.attachment',return_value=handle), patch('flagprof.scope',scope):
            with self.assertRaisesRegex(RuntimeError,'phase failure'):
                with phase('backward',step=1,microstep=0): raise RuntimeError('phase failure')
        self.assertEqual(state, {'training': {'previous': True}})
        self.assertEqual(seen[0]['outcome']['status'], 'failed')
        self.assertEqual(seen[0]['outcome']['error'], 'RuntimeError')

    def test_summary_does_not_certify_missing_rank(self):
        from flagprof.training.summary import summarize
        self.run_model('single')
        self.assertEqual(summarize(self.root/'single')['status'], 'complete')
        path=self.root/'single/training-result.json'
        result=json.loads(path.read_text());result['protocol']['world_size']=2
        path.write_text(json.dumps(result))
        summary=summarize(self.root/'single')
        self.assertEqual(summary['status'], 'partial')
        self.assertFalse(summary['rank_coverage_complete'])

    def test_nonfinite_loss_preserves_failure(self):
        model=copy.deepcopy(self.model)
        def invalid(model,batch):return self.loss(model,batch)*float('nan')
        with self.assertRaises(FloatingPointError):
            train(model,torch.optim.SGD(model.parameters(),lr=.01),self.batches,invalid,
                  outputs=self.root/'nonfinite',steps=1)
        self.assertEqual(json.loads((self.root/'nonfinite/training-result.json').read_text())['status'],'failed')

    @unittest.skipUnless(sys.platform == 'linux', 'Linux Store/attachment')
    def test_coverage_records_backward_and_optimizer_with_explicit_steps(self):
        from flagprof import attach
        from flagprof.store import create_run, finalize, StoreView
        run = create_run(self.root/'store')
        with attach(run, 'ascend', preset='coverage'):
            self.run_model('observed')
        finalize(run)
        with StoreView(run) as view:
            events = [dict(item.record) for item in view.iter_events()]
        phases = {e['corr']:e['attrs']['phase'] for e in events
                  if e['t']=='span' and e.get('attrs',{}).get('training_schema')=='training/1'}
        calls = [e for e in events if e['t']=='op']
        self.assertTrue(any(phases.get(e.get('scope'))=='backward' for e in calls))
        self.assertTrue(any(phases.get(e.get('scope'))=='optimizer' for e in calls))


class SummaryTests(unittest.TestCase):
    def test_rank_local_attachment_ids_preserve_summary_but_different_presets_do_not(self):
        from flagprof.training.summary import summarize
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in (0, 1):
                path = root/('rank-' + str(rank))/'training-result.json'
                path.parent.mkdir()
                protocol = dict(task='classification', world_size=2, rank=rank, sharding='ddp-replica',
                                observation=dict(preset='coverage', attachment=str(rank)))
                result = dict(protocol=protocol, status='complete', next_step=1,
                              steps=[dict(step=0)], final_model_sha256='equal-final')
                path.write_text(json.dumps(result))
            self.assertEqual(summarize(root)['status'], 'complete')
            result['protocol']['observation']['preset'] = 'perf'
            path.write_text(json.dumps(result))
            self.assertEqual(summarize(root)['status'], 'partial')


class TestCommunicationProbeCoverage(unittest.TestCase):
    def test_fsdp_collectives_are_wrapped(self):
        """FSDP2 bypasses torch.distributed: its functional entries must be wrapped."""
        from flagprof.probes.communication import DIST_NAMES, FUNCTIONAL_NAMES, OP_BY_NAME
        self.assertIn('reduce_scatter_tensor', DIST_NAMES)
        self.assertIn('reduce_scatter_tensor', FUNCTIONAL_NAMES)
        self.assertEqual(OP_BY_NAME['reduce_scatter_tensor'], 'reducescatter')
        for name in ('all_reduce', 'all_gather_tensor'):
            self.assertIn(name, FUNCTIONAL_NAMES)
        for name in DIST_NAMES:
            self.assertIn(name, OP_BY_NAME)
        for name in FUNCTIONAL_NAMES:
            self.assertIn(name, OP_BY_NAME)

    def test_pipeline_and_p2p_entries_are_wrapped(self):
        """Megatron-style pipeline parallelism moves activations over p2p entries;
        rooted collectives beyond the FSDP pair must be visible too."""
        from flagprof.probes.communication import DIST_NAMES, OP_BY_NAME, P2P_NAMES
        for name in ('broadcast', 'reduce', 'scatter', 'gather', 'gather_object',
                     'scatter_object_list', 'all_to_all', 'all_to_all_single',
                     'send', 'recv', 'isend', 'irecv', 'batch_isend_irecv'):
            self.assertIn(name, DIST_NAMES, name)
        self.assertEqual(set(P2P_NAMES), {'send', 'recv', 'isend', 'irecv', 'batch_isend_irecv'})
        self.assertTrue(set(P2P_NAMES).issubset(set(DIST_NAMES)))
        self.assertEqual(OP_BY_NAME['batch_isend_irecv'], 'batch_p2p')
        self.assertEqual(OP_BY_NAME['all_to_all_single'], 'alltoall')

    def test_phase_accepts_data_and_rejects_unknown(self):
        """The data-preparation phase is a first-class training/1 phase;
        unknown names stay rejected so typos cannot silently drop spans."""
        from flagprof.training import phase
        with phase('data', step=0):
            pass
        with self.assertRaisesRegex(ValueError, 'unknown training phase'):
            with phase('tokenize', step=0):
                pass

    @unittest.skipUnless(sys.platform == 'linux', 'Linux Store/attachment')
    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'transformers')),
                         'requires torch and Transformers (real capture attachment)')
    def test_driver_records_data_preparation_phase(self):
        """End to end through the driver: batch materialisation must surface as
        a training/1 span with phase=data, so layer-1 segment reports see it."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from flagprof import attach
            from flagprof.store import create_run, finalize, StoreView
            from flagprof.training.__main__ import main
            run = create_run(root/'driver-store')
            with attach(run, 'ascend', preset='coverage'):
                rc = main(['--steps', '1', '--outputs', str(root/'driver-out')])
            self.assertEqual(rc, 0)
            finalize(run)
            with StoreView(run) as view:
                events = [dict(item.record) for item in view.iter_events()]
            phases = [e['attrs']['phase'] for e in events
                      if e['t'] == 'span' and e.get('attrs', {}).get('training_schema') == 'training/1']
            self.assertIn('data', phases)
            self.assertIn('step', phases)
            self.assertIn('forward', phases)


if __name__ == '__main__':
    unittest.main()
