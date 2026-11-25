# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Training oracle: boundary capture into a real store and the cross-run join.

POSIX only (store locking). The capture emits the inference accuracy probe's
`t='tensor'` payload, so the events must survive the schema and the join must
read them back through the same store view the analyzers use.
"""
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
except ImportError:  # -S CPU smoke runs without site-packages
    torch = None
    nn = None

from flagprof.store import StoreView, Writer, create_run, finalize
from flagprof.training.oracle import OracleCapture, join_runs

HAVE_TORCH = torch is not None

if nn is not None:
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(8, 8)

        def forward(self, x):
            return torch.tanh(self.layer(x))


    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(8, 8)
            self.blocks = nn.ModuleList([Block(), Block()])
            self.head = nn.Linear(8, 4)

        def forward(self, x):
            h = self.embed(x)
            for block in self.blocks:
                h = block(h)
            return self.head(h)


def make_handle(writer, step_box):
    return SimpleNamespace(writer=writer, monitor_enabled=False, state={},
                           context=SimpleNamespace(probe_state=lambda: {'training': {'step': step_box[0]}}))


def capture_run(root, seed, *, perturb=False):
    torch.manual_seed(seed)
    run = create_run(root)
    step = [0]
    handle = make_handle(Writer(run), step)
    capture = OracleCapture(handle, boundaries=['<root>.blocks.0', '<root>.blocks.1'])
    model = Model()
    capture.register_model(model)
    generator = torch.Generator().manual_seed(seed+7)
    with handle.writer:
        for step_value in range(2):
            step[0] = step_value
            x = torch.randn(4, 8, generator=generator)
            if perturb and step_value == 1:
                x = x + 0.05
            model(x)
    capture.close()
    finalize(run)
    return run


class OracleEventJoinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_run(self, side, events):
        run = create_run(self.root/side)
        with Writer(run) as writer:
            for key, values, shape in events:
                raw = struct.pack('<'+'f'*len(values), *(float(v) for v in values))
                writer.emit(dict(t='tensor', role='out', boundary='out', call_key=key,
                                 shape=shape, dtype='torch.float32', stats={}, chunks=[],
                                 probe=values, probe_alg='p4v1',
                                 digest='sha256:'+hashlib.sha256(raw).hexdigest()))
        finalize(run)
        return run

    def test_first_divergence_follows_capture_order(self):
        keys = [f'<root>.layers.{i}#0#out#0' for i in (2, 10)]
        left = self.write_run('left', [(key, [0.0], [1]) for key in keys])
        right = self.write_run('right', [(key, [1.0], [1]) for key in keys])
        self.assertEqual(join_runs(left, right)['first_divergence']['key'], keys[0])

    def test_shape_and_nonfinite_differences_and_unsampled_values(self):
        cases = [('shape', [1], [1], [1, 1], [2]),
                 ('nonfinite', ['nan', 0.0], [2], [0.0, 0.0], [2]),
                 ('no-samples', [], [2], [1], [2]),
                 ('combined-tolerance', [10.0, 20.0], [2], [11.0, 21.5], [2])]
        left = self.write_run('left', [(name+'#0#out#0', a, sa) for name, a, sa, b, sb in cases])
        right = self.write_run('right', [(name+'#0#out#0', b, sb) for name, a, sa, b, sb in cases])
        result = join_runs(left, right, atol=0.5, rtol=0.06)
        self.assertEqual([r['verdict'] for r in result['rows']],
                         ['differs', 'differs', 'incomparable', 'within_tolerance'])
        self.assertEqual(result['counts']['incomparable'], 1)

    def test_right_only_boundary_precedes_later_numeric_drift(self):
        a, b = ['<root>.'+i+'#0#out#0' for i in ('first', 'next')]
        left = self.write_run('left', [(b, [0], [1])])
        right = self.write_run('right', [(a, [0], [1]), (b, [1], [1])])
        first = join_runs(left, right)['first_divergence']
        self.assertEqual((first['key'], first['verdict']), (a, 'missing_side'))

    def test_rejects_ambiguous_historical_keys(self):
        key = '<root>.layer#0#out#0'
        left = self.write_run('left', [(key, [1], [1]), (key, [2], [1])])
        with self.assertRaisesRegex(ValueError, 'ambiguous repeated tensor key'):
            join_runs(left, left)


@unittest.skipUnless(HAVE_TORCH and __import__('os').name == 'posix',
                     'torch and POSIX store locking required')
class TrainingOracleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_capture_events_survive_schema_and_join_self_exact(self):
        run = capture_run(self.root/'a', seed=1)
        with StoreView(run) as view:
            events = [stored.record for stored in view.iter_events(types=('tensor',))]
        self.assertEqual(len(events), 4, 'two boundaries x two steps expected')
        keys = {e['call_key'] for e in events}
        self.assertIn('<root>.blocks.0#0#out#0', keys)
        self.assertTrue(all(e['role'] == 'out' and e['blob'] for e in events))
        result = join_runs(run, run, out=self.root/'join')
        self.assertEqual(result['counts']['differs'], 0)
        self.assertEqual(result['counts']['missing'], 0)
        self.assertIsNone(result['first_divergence'])
        self.assertTrue((self.root/'join'/'oracle.md').exists())

    def test_join_locates_first_divergence_across_runs(self):
        run_a = capture_run(self.root/'a', seed=3)
        run_b = capture_run(self.root/'b', seed=3, perturb=True)
        result = join_runs(run_a, run_b, out=self.root/'join')
        self.assertEqual(result['counts']['rows'], 4)
        self.assertEqual(result['counts']['exact'], 2, 'step 0 identical; step 1 perturbed')
        first = result['first_divergence']
        self.assertIsNotNone(first)
        self.assertEqual(first['step'], 1)
        self.assertIn(first['key'], ('<root>.blocks.0#1#out#0', '<root>.blocks.1#1#out#0'))

    def test_budget_and_repeated_calls_across_microsteps(self):
        def run_with_first(value, side):
            run = create_run(self.root/side)
            step = [0]
            handle = make_handle(Writer(run), step)
            state = {'step': 0, 'microstep': 0}
            handle.context.probe_state = lambda: {'training': state}
            capture = OracleCapture(handle, boundaries=['<root>'], budget_bytes=1)
            model = nn.Identity()
            capture.register_model(model)
            with handle.writer:
                model(torch.tensor([value]))
                model(torch.tensor([2.0]))
                state['microstep'] = 1
                model(torch.tensor([2.0]))
            capture.close()
            finalize(run)
            self.assertEqual(capture.metrics['blob_skipped'], 3)
            return run

        result = join_runs(run_with_first(1.0, 'a'), run_with_first(9.0, 'b'))
        self.assertEqual(result['counts']['rows'], 3)
        self.assertEqual(result['counts']['exact'], 2)
        self.assertEqual(result['first_divergence']['key'], '<root>#0#out#0')
        keys = [r['key'] for r in result['rows']]
        self.assertIn('<root>#0#micro:0#call:1#out#0', keys)
        self.assertIn('<root>#0#micro:1#call:0#out#0', keys)

    def test_dropped_tensor_and_capture_error_qualify_surviving_exact_rows(self):
        from flagprof.probes.tensor import tensor_event
        run = create_run(self.root/'loss')
        handle = make_handle(Writer(run), [0])
        capture = OracleCapture(handle, budget_bytes=0)
        event, raw, fmt = tensor_event('layer#0#out#0', 'layer', 0, 'out', '0', torch.tensor([1.]))
        with handle.writer:
            capture._emit(event, raw, fmt)
            # An unregistered signature exercises the real writer rejection path.
            capture._emit(dict(event, sid='unknown'), raw, fmt)
        capture.safe(int, 'invalid')
        capture.close()
        self.assertEqual((capture.metrics['events'], capture.metrics['events_rejected']), (1, 1))
        self.assertEqual(capture.metrics['blob_skipped'], 2)
        finalize(run)
        state = run/'attachments'/'one'/'state.json'
        state.parent.mkdir(parents=True)
        state.write_text(json.dumps(handle.state))
        out = self.root/'join'
        result = join_runs(run, run, out=out)
        self.assertEqual(result['counts']['exact'], 1)
        self.assertTrue(result['coverage']['partial'])
        self.assertEqual(result['coverage']['left'],
                         dict(events_rejected=1, capture_errors=1, writer_tensor_drops=1))
        self.assertIn('采集记录不完整', (out/'oracle.md').read_text())

    def test_rejects_unsupported_snapshot_objects(self):
        import pickle
        run = capture_run(self.root/'a', seed=1)
        torch.save({'gradients': {}, 'unexpected': SimpleNamespace()}, run/'step-0.pt')
        with self.assertRaises(pickle.UnpicklingError):
            join_runs(run, run)

    def test_lossy_samples_cannot_certify_integer_or_complex_values(self):
        from flagprof.probes.tensor import tensor_event
        from flagprof.training.oracle import _probe_comparison, _tensor_comparison
        for a, b, verdict in ((torch.tensor([2**60]), torch.tensor([2**60+1]), 'differs'),
                              (torch.tensor([1+2j]), torch.tensor([1+3j]), 'incomparable')):
            def event(value):
                return tensor_event('layer#0#out#0', 'layer', 0, 'out', '0', value)[0]
            left, right = event(a), event(b)
            self.assertEqual(left['probe'], right['probe'])
            self.assertNotEqual(left['digest'], right['digest'])
            self.assertEqual(_probe_comparison(left, right, 0., 0.)['verdict'], verdict)
        limits = torch.iinfo(torch.int64)
        for a, b in ((2**60, 2**60+1), (limits.min, limits.max)):
            result = _tensor_comparison(torch.tensor([a]), torch.tensor([b]), 10., 1.)
            self.assertEqual(result['verdict'], 'differs')
            self.assertEqual(result['max_abs'], b-a)

    def test_gradient_rows_join_from_snapshots(self):
        run_a = capture_run(self.root/'a', seed=5)
        run_b = capture_run(self.root/'b', seed=5)
        snap = {'gradients': {'blocks.0.layer.weight': torch.tensor([[0.5, 1.0]])}}
        torch.save(snap, Path(run_a)/'step-0.pt')
        torch.save({'gradients': {'blocks.0.layer.weight': torch.tensor([[0.5, 1.0]])}}, Path(run_b)/'step-0.pt')
        result = join_runs(run_a, run_b)
        grads = [r for r in result['rows'] if r['kind'] == 'gradient']
        self.assertEqual(len(grads), 1)
        self.assertEqual(grads[0]['verdict'], 'exact')

    def test_gradient_metadata_nonfinite_and_tolerance(self):
        left = capture_run(self.root/'a', seed=5)
        right = capture_run(self.root/'b', seed=5)
        torch.save({'gradients': {'a': torch.tensor([0], dtype=torch.int32),
                                  'b': torch.tensor([float('nan')]), 'c': torch.tensor([10.0])}}, left/'step-0.pt')
        torch.save({'gradients': {'a': torch.tensor([0], dtype=torch.float32),
                                  'b': torch.tensor([0.0]), 'c': torch.tensor([11.0])}}, right/'step-0.pt')
        rows = join_runs(left, right, atol=0.5, rtol=0.06)['rows']
        self.assertEqual([r['verdict'] for r in rows if r['kind'] == 'gradient'],
                         ['differs', 'differs', 'within_tolerance'])

    def test_ops_drill_capture_and_replay(self):
        from flagprof.training.oracle import replay
        run = create_run(self.root/'ops')
        step = [0]
        handle = make_handle(Writer(run), step)
        handle.state['status'] = 'running'
        capture = OracleCapture(handle, boundaries=['<root>.blocks.0'], ops_module='<root>.blocks.0')
        model = Model()
        torch.manual_seed(9)
        capture.register_model(model)
        x = torch.randn(4, 8)
        with handle.writer:
            model(x)
            model(x+1.0)
        capture.close()
        with StoreView(run) as view:
            ops = [stored.record for stored in view.iter_events(types=('tensor',))
                   if '#op:' in stored.record['call_key'] and '#out:' in stored.record['call_key']]
        self.assertTrue(ops, 'no drilled op outputs captured')
        replay_key = ops[0]['call_key']
        result = replay(run, replay_key)
        self.assertTrue(result['match'], 'same-machine replay must match the captured output')
        self.assertGreaterEqual(result['inputs'], 1)

        second = replay_key.replace('#0#op:', '#0#micro:0#call:1#op:')
        self.assertNotEqual(replay_key, second)
        self.assertTrue(replay(run, second)['match'])

    def test_replay_preserves_alpha_and_rejects_unsupported_overloads(self):
        from flagprof.training.oracle import replay
        class AddAndMultiply(nn.Module):
            def forward(self, x):
                product = torch.mm(x, x.transpose(0, 1))
                total = torch.add(product, product, alpha=2)
                return torch.sum(total, dim=0)

        run = create_run(self.root/'calls')
        handle = make_handle(Writer(run), [0])
        model = AddAndMultiply()
        capture = OracleCapture(handle, boundaries=['<root>'], ops_module='<root>')
        capture.register_model(model)
        with handle.writer:
            model(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        capture.close()
        finalize(run)
        with StoreView(run) as view:
            outputs = {e.record['ext']['op_schema']: e.record for e in view.iter_events(types=('tensor',))
                       if e.record['ext'].get('capture_scope') == 'oracle-replay' and e.record['role'] == 'out'}
        mm = replay(run, outputs['aten::mm']['call_key'])
        self.assertTrue(mm['match'])
        self.assertEqual(mm['input_layouts'], ['contiguous', 'transposed-last-two'])
        add = replay(run, outputs['aten::add.Tensor']['call_key'])
        self.assertTrue(add['match'])
        self.assertEqual(add['kwargs'], {'alpha': 2})
        with self.assertRaisesRegex(ValueError, 'unsupported replay'):
            replay(run, outputs['aten::sum.dim_IntList']['call_key'])

        from unittest.mock import patch
        from flagprof.training.oracle import _read_tensor_events
        legacy = _read_tensor_events(run, ops_only=True)
        for event in legacy.values():
            event['ext'].pop('replay_layout', None)
            event['ext'].pop('replay_kwargs', None)
        with patch('flagprof.training.oracle._read_tensor_events', return_value=legacy):
            self.assertTrue(replay(run, outputs['aten::mm']['call_key'])['match'])
            with self.assertRaisesRegex(ValueError, 'unrecorded replay arguments'):
                replay(run, outputs['aten::add.Tensor']['call_key'])


if __name__ == '__main__':
    unittest.main()
