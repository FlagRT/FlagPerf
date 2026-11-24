# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Training oracle: boundary capture into a real store and the cross-run join.

POSIX only (store locking). The capture emits the inference accuracy probe's
`t='tensor'` payload, so the events must survive the schema and the join must
read them back through the same store view the analyzers use.
"""
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
        capture.close()
        with StoreView(run) as view:
            ops = [stored.record for stored in view.iter_events(types=('tensor',))
                   if '#op:' in stored.record['call_key'] and '#out:' in stored.record['call_key']]
        self.assertTrue(ops, 'no drilled op outputs captured')
        replay_key = ops[0]['call_key']
        result = replay(run, replay_key)
        self.assertTrue(result['match'], 'same-machine replay must match the captured output')
        self.assertEqual(result['inputs'] >= 1, True)
        run_a = capture_run(self.root/'a', seed=5)
        run_b = capture_run(self.root/'b', seed=5)
        snap = {'gradients': {'blocks.0.layer.weight': torch.tensor([[0.5, 1.0]])}}
        torch.save(snap, Path(run_a)/'step-0.pt')
        torch.save({'gradients': {'blocks.0.layer.weight': torch.tensor([[0.5, 1.0]])}}, Path(run_b)/'step-0.pt')
        result = join_runs(run_a, run_b)
        grads = [r for r in result['rows'] if r['kind'] == 'gradient']
        self.assertEqual(len(grads), 1)
        self.assertEqual(grads[0]['verdict'], 'exact')


if __name__ == '__main__':
    unittest.main()
