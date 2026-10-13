# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""P800 planning/host contracts: no Docker daemon or accelerator required."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import cli
from vendors import get_vendor
from vendors.kunlunxin.p800 import Adapter, PROFILE


def arguments(*extra):
    return cli.parser().parse_args(['run', '--vendor', 'kunlunxin', '--runtime-profile',
                                   'p800-xpytorch29', '--device-ids', '4', *extra])


class P800PlanningTests(unittest.TestCase):
    def test_full_matrix_and_multi_device(self):
        for devices, expected in [('4', (728, 308, 420)), ('4,7', (1456, 616, 840))]:
            _, plan = cli.resolve(arguments('--device-ids', devices, '--oplib', 'both', '--dtype',
                                           'FP32', 'FP16', 'BF16', 'INT32', 'INT16', 'BOOL', 'INT64'))
            tasks = plan['tasks']
            applicable = sum(t['applicable'] for t in tasks)
            self.assertEqual((len(tasks), applicable, len(tasks)-applicable), expected)
            self.assertTrue(all(t['worker_device'] is None for t in tasks))
            self.assertTrue(all(t['runtime_profile'] == 'p800-xpytorch29' for t in tasks))
            self.assertEqual(len({t['runtime_config']['profile_sha256'] for t in tasks}), 1)

    def test_default_types_literal_all_sizes_and_modes(self):
        self.assertEqual(len(cli.resolve(arguments('--oplib', 'both'))[1]['tasks']), 104)
        self.assertEqual([t['case'] for t in cli.resolve(arguments('--case', 'all'))[1]['tasks']], ['all'])
        for diagnostics in ('off', 'failures'):
            _, plan = cli.resolve(arguments('--execution', 'local', '--device-ids', '9', '--case', 'mm',
                                            '--size', 'M=7', '--size', 'M=13', '--oplib', 'both',
                                            '--diagnostics', diagnostics, '--workload', 'smoke'))
            self.assertEqual(len(plan['tasks']), 4)
            self.assertEqual({t['case_config']['M'] for t in plan['tasks']}, {7, 13})
            self.assertTrue(all(t['worker_device'] == 9 for t in plan['tasks']))
            self.assertTrue(all(t['device_binding']['status'] == 'externally-managed' for t in plan['tasks']))
            self.assertTrue(all(t['diagnostics_mode'] == diagnostics for t in plan['tasks']))

    def test_reject_bad_profile_and_device_selections(self):
        for extra in [('--runtime-profile', 'unknown'), ('--vendor', 'ascend'),
                      ('--device-ids', '1,1'), ('--device-ids', '8'), ('--device-ids', '3-1'),
                      ('--device-ids', '-1'), ('--allow-privileged-root',),
                      ('--container-device', '/dev/xpu3'), ('--mount', '/tmp:/tmp'),
                      ('--profiling', 'timeline')]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                cli.resolve(arguments(*extra))
        self.assertEqual(cli.resolve(arguments('--device-ids', '1'))[1]['device_ids'], [1])
        self.assertEqual(json.loads(PROFILE.read_text())['excluded_physical_ids'], [])

    def test_workload_alias_and_runtime_profile_are_independent(self):
        for flag in ('--workload', '--profile'):
            self.assertEqual(cli.resolve(arguments(flag, 'smoke'))[1]['workload'], 'smoke')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments('--workload', 'smoke', '--profile', 'smoke')

    def test_old_vendor_routes_still_selected(self):
        from vendors.compat import Adapter as Compat
        from vendors.ascend.adapter import Adapter as Ascend
        self.assertIs(type(get_vendor('kunlunxin')), Compat)
        self.assertIs(type(get_vendor('ascend')), Ascend)
        args = cli.parser().parse_args(['run', '--vendor', 'kunlunxin', '--device-ids', '9', '--image', 'legacy'])
        _, plan = cli.resolve(args)
        self.assertNotIn('runtime_profile', plan)
        self.assertEqual(plan['image'], 'legacy')
        self.assertEqual(len(plan['tasks']), 52)
        self.assertEqual(plan['tasks'][0]['worker_device'], 0)

    def test_locked_image_checked_without_claiming_packages_verified(self):
        adapter = Adapter()
        for info in ({}, {'Id': 'sha256:wrong'}):
            with self.assertRaisesRegex(ValueError, 'image ID'):
                adapter.runtime_identity(adapter.default_image(), info)
        identity = adapter.runtime_identity('equivalent-alias', {'Id': adapter.lock['image_id']})
        self.assertFalse(identity['validated'])
        self.assertFalse(identity['packages_observed'])

    def test_host_and_each_worker_entry_select_same_profile(self):
        from runtime import worker
        adapter, plan = cli.resolve(arguments('--case', 'abs'))
        task = plan['tasks'][0]
        self.assertIs(type(get_vendor(task['vendor'], task['runtime_profile'])), type(adapter))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli.write(root / 'task.json', task)
            for mode in ('probe', 'measure', 'trace', 'memory', 'profiling', 'profile-target'):
                with patch('vendors.get_vendor', side_effect=RuntimeError('selection-stop')) as select, \
                     contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(worker.run_phase(root, mode), 1)
                select.assert_called_once_with('kunlunxin', 'p800-xpytorch29')

    def test_planning_is_stdlib_only_and_never_touches_host(self):
        script = '''
import contextlib, io, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
from runtime import cli
with patch.object(cli.subprocess, 'run', side_effect=AssertionError('process')), patch.object(cli.subprocess, 'check_output', side_effect=AssertionError('process')), patch.object(cli.Lease, 'acquire', side_effect=AssertionError('lease')), contextlib.redirect_stdout(io.StringIO()):
    assert cli.main(['list']) == 0
    assert cli.main(['run', '--vendor', 'kunlunxin', '--runtime-profile', 'p800-xpytorch29', '--device-ids', '4', '--dry-run']) == 0
assert not any(n == 'torch' or n.startswith(('torch.', 'torch_xmlir', 'flag_gems')) for n in sys.modules)
'''
        result = subprocess.run([sys.executable, '-B', '-S', '-c', script, str(ROOT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_profile_included_in_source_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(cli.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='fixture')):
                cli.snapshot(root)
            self.assertEqual((root / 'source/vendors/kunlunxin/profiles/p800-xpytorch29.json').read_bytes(), PROFILE.read_bytes())

    def test_unimplemented_execution_fails_before_docker_or_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments('--case', 'abs', '--result-root', tmp)
            adapter, plan = cli.resolve(args)
            with patch.object(cli, 'snapshot'), patch.object(cli.subprocess, 'check_output', side_effect=AssertionError('docker')), \
                 patch.object(cli.Workers, 'start', side_effect=AssertionError('worker')), \
                 patch.object(cli.Lease, 'acquire', side_effect=AssertionError('lease')), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.execute(args, adapter, plan), 1)
            summary = cli.read(next(Path(tmp).glob('*/summary.json')))
            self.assertIn('device workers are not integrated', summary['error'])
            self.assertEqual(summary['tasks'][0]['status'], 'not-run')


if __name__ == '__main__':
    unittest.main()
