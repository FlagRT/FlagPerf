# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import diagnostics, cli
from runtime.evidence import write, seal, report, merge_runs
from vendors.ascend.adapter import Adapter


class FailureDiagnosticsTests(unittest.TestCase):
    def test_dependency_classification_requires_stack(self):
        adapter = Adapter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            error = 'Keyword argument SPLIT_K was specified but unrecognised'
            self.assertIsNone(adapter.diagnose(error, root))
            (root / 'probe.log').write_text('flag_gems/ops/mm.py triton/runtime/jit.py')
            self.assertIn('SPLIT_K', adapter.diagnose(error, root))
            self.assertIsNone(adapter.diagnose('unrelated exception', root))
            (root / 'probe.log').write_text('flag_gems/ops/mul.py')
            self.assertIn('scalar mul', adapter.diagnose("aten::mul Expected a value of type 'Tensor'", root))
            (root / 'probe.log').write_text('flag_gems/utils/random_utils.py')
            self.assertIn('RNG', adapter.diagnose('too many values to unpack', root))

    def test_report_includes_numeric_and_partial_without_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'summary.json', {'status': 'failed', 'counts': {}, 'tasks': [
                {'case': 'mm', 'dtype': 'FP32', 'oplib': 'nativetorch', 'directory': 'mm',
                 'status': 'failed', 'correctness': {'status': 'failed'}},
                {'case': 'linear', 'dtype': 'FP16', 'oplib': 'nativetorch', 'directory': 'linear',
                 'status': 'partial'}]})
            seal(root)
            text = report(root).read_text()
            section = text.split('## 异常与下一步')[1]
            self.assertIn('正确性门禁失败', section)
            self.assertIn('目标执行路径待确认', section)
            self.assertIn('未记录', section)

    def test_diagnostic_error_preserves_original_gate_and_closes_pool(self):
        class Pool:
            closed = False
            def phase(self, *args): raise RuntimeError('watchdog')
            def close(self): self.closed = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('inputs.pt', 'reference.pt'): (root / name).touch()
            pool = Pool()
            item = {'status': 'failed', 'correctness': {'status': 'failed'}}
            diagnostics.collect(pool, root, {'diagnostics_mode': 'failures'}, item)
            self.assertEqual(item['status'], 'failed')
            self.assertTrue(pool.closed)
            self.assertEqual(item['diagnostics']['results']['reference-check']['status'], 'failed')
            self.assertEqual(item['diagnostics']['results']['diagnose']['execution_status'], 'skipped')

    def test_cleanup_failure_not_swallowed(self):
        class Pool:
            def phase(self, *args): raise RuntimeError('watchdog')
            def close(self): raise cli.CleanupError('still running')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('inputs.pt', 'reference.pt'): (root / name).touch()
            with self.assertRaises(cli.CleanupError):
                diagnostics.collect(Pool(), root, {'diagnostics_mode': 'failures'},
                                    {'correctness': {'status': 'failed'}})
            data = json.loads((root / 'diagnostic-checks.json').read_text())
            self.assertEqual(data['status'], 'failed')

    def test_error_statistics_near_zero_and_nonfinite(self):
        import torch
        result = diagnostics.errors(torch.tensor([1e-3, float('nan'), float('inf')]),
                                    torch.tensor([0., float('nan'), float('inf')]), 1e-5, 1e-4)
        self.assertEqual(result['mismatch_count'], 1)
        self.assertEqual(result['nonfinite_elements'], 2)
        self.assertAlmostEqual(result['max_tolerance_ratio'], 100, places=3)
        self.assertEqual(result['worst_flat_index'], 0)
        json.dumps(result, allow_nan=False)

    def test_diagnostic_modes_do_not_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = []
            for mode in ('off', 'failures'):
                source = root / mode;source.mkdir();sources.append(source)
                write(source / 'summary.json', {'status': 'passed', 'actual_image_id': 'same',
                      'vendor': 'ascend', 'execution': 'docker', 'preflight': {'host': 'host'},
                      'tasks': [{'case': 'mm', 'dtype': 'FP32', 'oplib': 'nativetorch',
                                 'diagnostics_mode': mode, 'status': 'passed'}]})
                seal(source)
            merge_runs(sources, root / 'combined')
            summary = json.loads((root / 'combined/summary.json').read_text())
            self.assertEqual(len(summary['tasks']), 2)

    def test_numeric_diagnostic_uses_saved_inputs_and_separates_gradients(self):
        import torch
        from unittest.mock import patch
        from runtime.worker import invocation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = torch.tensor([0.25, 0.75], requires_grad=True)
            expected = invocation(torch.square, (x.double(),), True, nonzero=True)
            actual = (expected[0].float() + 0.1, (expected[1][0].float(),))
            torch.save({'inputs': (x,), 'state': None}, root / 'inputs.pt')
            torch.save(expected, root / 'reference.pt')
            torch.save(actual, root / 'output.pt')
            torch.save(actual, root / 'measurement-output.pt')
            write(root / 'correctness.json', {'atol': 1e-5, 'rtol': 1e-4, 'status': 'failed'})
            with patch('runtime.worker.build', return_value=(torch.square, None, True, None)):
                diagnostics.numeric(root, {})
            result = json.loads((root / 'numeric-diagnostic.json').read_text())
            checks = result['comparisons']['measure']
            self.assertEqual([c['role'] for c in checks], ['output', 'input-gradient'])
            self.assertEqual(checks[0]['device_vs_fp64']['mismatch_count'], 2)
            self.assertEqual(checks[0]['cpu_fp32_vs_fp64']['mismatch_count'], 0)
            self.assertEqual(checks[1]['device_vs_fp64']['mismatch_count'], 0)
            self.assertFalse(result['gate_changed'])

    def test_diagnostics_off_never_starts_worker(self):
        class Pool:
            def phase(self, *args): raise AssertionError('must not launch')
        item = {'status': 'failed', 'correctness': {'status': 'failed'}}
        diagnostics.collect(Pool(), None, {'diagnostics_mode': 'off'}, item)
        self.assertNotIn('diagnostics', item)

    def test_cpu_api_event_is_not_device_route_proof(self):
        probe = {'device': 'flagos:0', 'output_devices': ['flagos:0'], 'calls': [],
                 'events': [{'name': 'aten::matmul'}]}
        log = 'OPERATION_TARGET_BEGIN\n[flagos dispatch] t -> ascend\nOPERATION_TARGET_END'
        self.assertEqual(Adapter().route(probe, log)['status'], 'partial')

    def test_dropout_does_not_invent_cross_device_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostics.numeric(root, {'case': 'dropout'})
            result = json.loads((root / 'numeric-diagnostic.json').read_text())
            self.assertEqual(result['status'], 'unsupported')
            self.assertFalse(result['gate_changed'])

    def test_cached_launcher_requires_actual_execution_state(self):
        import copy
        probe = {'device': 'flagos:0', 'output_devices': ['flagos:0'], 'calls': [
            {'path': '/site/flag_gems/ops/groupnorm.py', 'function': 'group_norm'},
            {'path': '/site/triton/compiler/compiler.py', 'function': 'runner'},
            {'path': '/site/triton/backends/ascend/driver.py', 'function': '__call__',
             'launcher_states': [{'class': 'NPULauncher', 'compile_only': False, 'register_tensor_only': False}]}]}
        log = 'OPERATION_TARGET_BEGIN\nOPERATION_TARGET_END'
        route = Adapter().route(probe, log)
        self.assertEqual(route['status'], 'passed')
        self.assertTrue(route['cached_launch_proven'])
        for field in ('compile_only', 'register_tensor_only'):
            other = copy.deepcopy(probe)
            other['calls'][-1]['launcher_states'][0][field] = True
            self.assertEqual(Adapter().route(other, log)['status'], 'partial')
        probe['calls'][-1].pop('launcher_states')
        self.assertEqual(Adapter().route(probe, log)['status'], 'partial')

    def test_compile_only_log_cannot_prove_execution(self):
        probe = {'device': 'flagos:0', 'output_devices': ['flagos:0'], 'calls': [
            {'path': '/site/flag_gems/ops/mm.py', 'function': 'mm'},
            {'path': '/site/triton/runtime/jit.py', 'function': 'run'}]}
        log = 'OPERATION_TARGET_BEGIN\n[INFO]: skip running kernel\nOPERATION_TARGET_END'
        self.assertEqual(Adapter().route(probe, log)['status'], 'partial')

    def test_vendor_owns_launcher_observation(self):
        from types import SimpleNamespace
        launcher = type('NPULauncher', (), {'compile_only': False, 'enable_msprof_register_tensor': False})()
        frame = SimpleNamespace(f_code=SimpleNamespace(co_filename='/site/triton/backends/ascend/driver.py', co_name='__call__'), f_locals={'self': launcher})
        self.assertEqual(Adapter().profile_call(frame), {'class': 'NPULauncher', 'compile_only': False, 'register_tensor_only': False})
        frame.f_code.co_filename = '/site/unrelated.py'
        self.assertIsNone(Adapter().profile_call(frame))
        self.assertNotIn('/triton/backends/ascend', (ROOT / 'runtime/worker.py').read_text())


if __name__ == '__main__':
    unittest.main()
