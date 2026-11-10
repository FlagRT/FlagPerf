# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Published CLI contracts across accuracy, components and vendor adapters."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flagprof.cli import main
from flagprof.launch import launch
from flagprof.probes.config import resolve


class ComponentCLITests(unittest.TestCase):
    def test_accuracy_boundaries_and_drill_forward_to_both_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp)/'boundaries.json'
            custom.write_text('["<root>.layers.2"]')
            for entry in ('record', 'launch'):
                for boundary, expected in [('model', []), ('layer', None),
                                           (str(custom), ['<root>.layers.2'])]:
                    with self.subTest(entry=entry, boundary=boundary):
                        target = 'flagprof.record.record' if entry == 'record' else 'flagprof.launch.launch'
                        with patch(target, return_value=0) as capture:
                            self.assertEqual(main([entry, '--store', 'unused', '--vendor', 'ascend',
                                '--preset', 'accuracy', '--tensor-boundaries', boundary,
                                '--tensor-ops-in', '<root>.layers.0', '--', 'python', 'target.py']), 0)
                        _, config = resolve('accuracy', capture.call_args.kwargs['config'])
                        self.assertEqual(config['tensor_boundaries'], expected)
                        self.assertEqual(config['tensor_ops_in'], '<root>.layers.0')
                        self.assertEqual(config['tensor_blob'], 'none' if entry == 'record' else 'all')

    def test_launch_coverage_passes_components_and_vendor(self):
        with patch('flagprof.launch.launch', return_value=0) as capture:
            self.assertEqual(main(['launch', '--store', 'unused', '--vendor', 'kunlunxin',
                '--comm-backend', 'flagcx', '--preset', 'coverage', '--flaggems', 'on',
                '--flaggems-include', '/inputs/include.json', '--flagtree', 'on',
                '--', 'python', 'target.py']), 0)
        options = capture.call_args.kwargs
        self.assertEqual(options['vendor'], 'kunlunxin')
        self.assertEqual(options['comm_backend'], 'flagcx')
        self.assertEqual(options['preset'], 'coverage')
        self.assertEqual(options['config'], dict(flaggems='on', flaggems_include='/inputs/include.json', flagtree='on'))

    def test_shortcut_conflicts_fail_before_launching_workers(self):
        for options, reason in [
            (dict(flagcx='on', comm_backend='hccl'), 'conflicts'),
            (dict(flagcx='on', vendor='kunlunxin'), 'shortcut requires Ascend'),
            (dict(flagcx='on', vendor='thead'), 'shortcut requires Ascend'),
            (dict(flagcx='off', vendor='nvidia'), 'shortcut requires Ascend'),
        ]:
            with self.subTest(options=options), patch('flagprof.launch.subprocess.Popen') as process:
                with self.assertRaisesRegex(ValueError, reason):
                    launch(['python', 'target.py'], store='/unused', **options)
                process.assert_not_called()

    def test_launch_default_and_shortcuts_select_backend(self):
        class StopBeforeOutput(Exception):
            pass
        for options, backend in [({}, 'hccl'), (dict(flagcx='off'), 'hccl'),
                                 (dict(flagcx='on'), 'flagcx')]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                with patch('flagprof.offline.workspace.check_disk'), \
                     patch('flagprof.launch.socket.socket') as socket_factory, \
                     patch('flagprof.launch.atomic_json', side_effect=StopBeforeOutput) as save:
                    socket_factory.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 12345)
                    with self.assertRaises(StopBeforeOutput):
                        launch(['python', 'target.py'], store=Path(tmp)/backend, **options)
                    self.assertEqual(save.call_args.args[1]['backend'], backend)

    def test_normalized_include_never_becomes_sdk_configuration(self):
        for preset in ('coverage', 'accuracy', 'perf', 'modules', 'perf-lite'):
            _, first = resolve(preset, dict(flaggems='on', flaggems_include=['add']))
            sdk, second = resolve(preset, first)
            self.assertEqual(sdk, {})
            self.assertEqual(first, second)

    def test_launch_preserves_official_profiler_options(self):
        class StopBeforeWorkers(Exception):
            pass
        for preset in ('perf', 'modules'):
            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as tmp:
                with patch('flagprof.offline.workspace.check_disk'), \
                     patch('flagprof.launch.socket.socket') as socket_factory, \
                     patch('flagprof.launch.atomic_json', side_effect=StopBeforeWorkers) as save:
                    socket_factory.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 12345)
                    with self.assertRaises(StopBeforeWorkers):
                        main(['launch', '--store', str(Path(tmp)/'launch'), '--preset', preset,
                              '--profile-memory', '--flagtree', 'on', '--', 'python', 'target.py'])
                config = save.call_args.args[1]['config']
                sdk, probes = resolve(preset, config)
                self.assertEqual(sdk, {'profile_memory': True})
                self.assertEqual(probes['flagtree'], 'on')

    def test_monitor_is_not_part_of_this_release(self):
        from flagprof.registry import Registry
        from flagprof.store.schema import SCHEMA_VERSION
        self.assertEqual(SCHEMA_VERSION, '1.2')
        self.assertFalse({'resource', 'perf.timeline'} & {d['id'] for d in Registry().descriptors()})
        for argv in [['watch'], ['record', '--store', 'unused', '--vendor', 'ascend', '--monitor', 'on'],
                     ['launch', '--store', 'unused', '--preset', 'monitor']]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(argv)
                self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
