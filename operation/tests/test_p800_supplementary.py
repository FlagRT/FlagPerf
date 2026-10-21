# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Diagnostic captures must never grant route or performance qualification."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import cli
from runtime import diagnosis
from runtime.evidence import read, write
from vendors import get_vendor
from vendors.kunlunxin.p800 import Adapter


def eligible():
    return {'status': 'partial', 'correctness': {'status': 'passed'},
            'measurement': {'correctness': {'status': 'passed'}},
            'phase_consistency': {'status': 'passed'}, 'routing': {'status': 'partial'},
            'measurement_fallback_observed': False}


class SupplementaryGateTests(unittest.TestCase):
    def test_replay_records_lease_release_even_when_probe_fails(self):
        for release_fails in (False, True):
            with self.subTest(release_fails=release_fails), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'source-task').mkdir()
                args=cli.parser().parse_args(['diagnose','--source-task',str(root/'source-task'),
                                             '--replay','--device-ids','7','--allow-shared-device'])
                adapter=MagicMock();adapter.preflight.return_value={'host':'fixture'}
                adapter.runtime_identity.return_value={'fixture':True}
                lease=MagicMock();lease.record.return_value={'resource_keys':['kunlunxin/test-uuid']}
                if release_fails:lease.release.side_effect=RuntimeError('release failed')
                data={'checks':[]}
                task={'case':'abs','dtype':'FP32','oplib':'nativetorch','vendor':'kunlunxin'}
                with patch.object(diagnosis,'get_vendor',return_value=adapter), \
                     patch.object(diagnosis.subprocess,'check_output',return_value='[{"Id":"image"}]'), \
                     patch.object(cli,'make_lease',return_value=lease), \
                     patch.object(cli,'install_termination_handler',return_value=None), \
                     patch.object(cli,'Workers'), \
                     patch.object(diagnosis.diagnostics,'run_check',side_effect=RuntimeError('probe failed')):
                    with self.assertRaisesRegex(RuntimeError,'probe failed'):
                        diagnosis.replay(root,args,task,{'actual_image_id':'image'},data)
                lease.release.assert_called_once()
                self.assertEqual(data['lease']['released'],not release_fails)
                self.assertEqual('cleanup_error' in data,release_fails)

    def test_partial_correct_task_gets_diagnostic_capture_without_mutation(self):
        for phase in ('memory', 'profiling'):
            task = eligible(); before = copy.deepcopy(task)
            gate = cli.supplementary_gate(Adapter(), task, phase)
            self.assertTrue(gate['allowed'])
            self.assertEqual(gate['scope'], 'diagnostic-only')
            self.assertEqual(task, before)

    def test_failure_missing_evidence_and_fallback_never_opt_in(self):
        mutations = [('status', 'failed'), ('status', 'blocked'), ('status', 'not-run'),
                     ('correctness', {'status': 'failed'}), ('measurement', {}),
                     ('phase_consistency', {}), ('routing', {'status': 'failed'}),
                     ('routing', {}), ('measurement_fallback_observed', True),
                     ('measurement_fallback_observed', None),
                     ('routing', {'status': 'partial', 'target_cpu_fallback': True}),
                     ('error', 'failed'), ('cleanup_error', 'failed'),
                     ('postflight_error', 'failed'), ('diagnostic_execution_blocked', True)]
        for key, value in mutations:
            task = eligible();task[key] = value
            for phase in ('memory', 'profiling'):
                with self.subTest(key=key, value=value, phase=phase):
                    self.assertFalse(cli.supplementary_gate(Adapter(), task, phase)['allowed'])

    def test_existing_vendor_policy_is_unchanged(self):
        for vendor in ('ascend', 'nvidia', 'kunlunxin'):
            adapter = get_vendor(vendor)
            self.assertFalse(cli.supplementary_gate(adapter, eligible(), 'memory')['allowed'])
            self.assertTrue(cli.supplementary_gate(adapter, {'status': 'passed'}, 'memory')['allowed'])

    def test_orchestrator_runs_memory_but_keeps_partial_and_separate_worker(self):
        identity = {k: {'fixture': 1} for k in ('vendor_runtime', 'packages', 'case_sha256',
                                              'input_sha256', 'registration', 'runtime_config', 'device', 'input_devices')}
        with tempfile.TemporaryDirectory() as tmp:
            args = cli.parser().parse_args(['run', '--vendor', 'kunlunxin', '--runtime-profile',
                'p800-xpytorch29', '--execution', 'local', '--device-ids', '0', '--case', 'abs',
                '--allow-shared-device', '--diagnostics', 'off', '--result-root', tmp])
            adapter, plan = cli.resolve(args)
            phases = []
            def phase(root, task, mode):
                phases.append(mode)
                if mode == 'probe':
                    write(root/'probe.json', identity)
                    write(root/'correctness.json', {'status': 'passed'})
                    (root/'probe.log').write_text('')
                elif mode == 'measure':
                    write(root/'measurement.json', {**identity, 'correctness': {'status': 'passed'}, 'median_us': 2.0})
                    (root/'measure.log').write_text('')
                elif mode == 'memory':
                    write(root/'memory.json', {**identity, 'status': 'measured', 'incremental_allocated_bytes': 4096})
            with patch.object(cli.Workers, 'phase', side_effect=phase), patch.object(cli.Workers, 'close') as close, \
                 patch.object(cli.Lease, 'acquire'), patch.object(cli.Lease, 'release'), \
                 patch.object(cli, 'snapshot'), patch.object(adapter, 'route', return_value={'status': 'partial'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.execute(args, adapter, plan), 2)
            self.assertEqual(phases, ['reference', 'probe', 'measure', 'memory'])
            close.assert_any_call(mode='memory')
            summary_path = next(Path(tmp).glob('*/summary.json'))
            item = read(summary_path)['tasks'][0]
            self.assertEqual(item['status'], 'partial')
            self.assertEqual(item['performance_qualification']['status'], 'partial')
            self.assertEqual(item['routing']['status'], 'partial')
            self.assertEqual(item['memory']['status'], 'measured')
            self.assertEqual(item['memory']['qualification']['scope'], 'diagnostic-only')
            self.assertEqual(read(summary_path.parent/item['directory']/'memory.json'), item['memory'])

    def test_memory_identity_drift_keeps_raw_but_suppresses_target_metrics(self):
        fields = ('vendor_runtime', 'packages', 'case_sha256', 'input_sha256',
                  'registration', 'runtime_config', 'device', 'input_devices')
        identity = {key: {'value': 1} for key in fields}
        for key in fields:
            item = {'measurement': identity, 'memory': {**copy.deepcopy(identity), 'status': 'measured',
                    'raw_peak': {'allocated_bytes': 4096}, 'peak_allocated_bytes': 4096}}
            item['memory'][key] = {'value': 2}
            Adapter().qualify_supplement(item, 'memory')
            self.assertEqual(item['memory']['status'], 'partial')
            self.assertEqual(item['memory']['phase_consistency']['mismatches'], [key])
            self.assertNotIn('peak_allocated_bytes', item['memory'])
            self.assertEqual(item['memory']['raw_peak']['allocated_bytes'], 4096)


if __name__ == '__main__':
    unittest.main()
