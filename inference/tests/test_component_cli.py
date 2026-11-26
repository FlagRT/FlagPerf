# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Published CLI contracts across accuracy, components and vendor adapters."""
import contextlib
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

from flagprof.cli import main
from flagprof.launch import launch
from flagprof.probes.config import resolve


class ComponentCLITests(unittest.TestCase):
    def test_window_option_correction_preserves_target_arguments(self):
        target = ['python', 'my script.py', '--windows', "a b'$", '--capture-mode=final']
        cases = [
            (['--preset', 'accuracy', '--capture-mod=final', '--act=1'], True),
            (['--preset', 'accuracy', '--capture-only', '--active', '1'], True),
            (['--capture-mode', 'windowed', '--capture-mode', 'final', '--act', '1'], True),
            (['--capture-only', '--act', '1', '--win=2', '--invalid-pol=continue',
              '--log-file', 'a b', '--log-file=--looks-like-a-path', '--plugins=unused'], False),
        ]
        for options, windowed in cases:
            error = io.StringIO()
            args = ['record', '--vendor', 'nvidia', '--store=--windows', *options, '--', *target]
            with self.subTest(options=options), contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as failure:
                main(args)
            self.assertEqual(failure.exception.code, 2)
            self.assertIn('--active', error.getvalue())
            corrected = shlex.split(error.getvalue().splitlines()[-1])
            controls = corrected[corrected.index('record'):corrected.index('--')]
            self.assertEqual(corrected[corrected.index('--')+1:], target)
            self.assertIn('--store=--windows', controls)
            backend = 'flagprof.record.record' if windowed else 'flagprof.offline.capture.record_final'
            with patch(backend, return_value=0) as capture:
                self.assertEqual(main(corrected[corrected.index('record'):]), 0)
                self.assertEqual(capture.call_args.args, (target,))
                if windowed:
                    self.assertEqual(capture.call_args.kwargs['capture_mode'], 'windowed')
                    self.assertEqual(capture.call_args.kwargs['config']['active'], 1)
                    self.assertNotIn('final', controls)
                else:
                    self.assertTrue(capture.call_args.kwargs['capture_only'])

    def test_final_record_source_budget(self):
        for options, expected in [
            ([], 4 << 30),
            (['--source-max-bytes', str(1 << 30)], 1 << 30),
            (['--source-max-bytes', str(8 << 30)], 8 << 30),
        ]:
            with self.subTest(options=options), patch('flagprof.offline.capture.record_final', return_value=0) as capture:
                self.assertEqual(main(['record', '--store', 'unused', '--vendor', 'ascend',
                                       *options, '--', 'python', 'target.py']), 0)
                self.assertEqual(capture.call_args.kwargs['source_max_bytes'], expected)

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

    def test_incompatible_backend_fails_before_launching_workers(self):
        for vendor, backend in [('ascend','nccl'), ('nvidia','hccl'), ('thead','hccl'), ('kunlunxin','hccl')]:
            with self.subTest(vendor=vendor), patch('flagprof.launch.subprocess.Popen') as process:
                with self.assertRaisesRegex(ValueError, 'supports --comm-backend'):
                    launch(['python', 'target.py'], store='/unused', vendor=vendor, comm_backend=backend)
                process.assert_not_called()

    def test_selected_vendor_chooses_its_backend(self):
        class StopBeforeOutput(Exception):
            pass
        for options, backend in [(dict(vendor='ascend'), 'hccl'), (dict(vendor='nvidia'), 'nccl'),
                                 (dict(vendor='kunlunxin', comm_backend='flagcx'), 'flagcx')]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                with patch('flagprof.launch.socket.socket') as socket_factory, \
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
                with patch('flagprof.launch.socket.socket') as socket_factory, \
                     patch('flagprof.launch.atomic_json', side_effect=StopBeforeWorkers) as save:
                    socket_factory.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 12345)
                    with self.assertRaises(StopBeforeWorkers):
                        main(['launch', '--vendor', 'ascend', '--store', str(Path(tmp)/'launch'), '--preset', preset,
                              '--profile-memory', '--flagtree', 'on', '--', 'python', 'target.py'])
                config = save.call_args.args[1]['config']
                sdk, probes = resolve(preset, config)
                self.assertEqual(sdk, {'profile_memory': True})
                self.assertEqual(probes['flagtree'], 'on')

    def test_monitor_is_available_and_opt_in(self):
        from flagprof.registry import Registry
        from flagprof.store.schema import SCHEMA_VERSION
        from flagprof.probes.config import resolve
        self.assertEqual(SCHEMA_VERSION, '1.3')
        self.assertTrue({'resource', 'perf.timeline', 'telemetry.host', 'telemetry.ascend',
                         'telemetry.nvidia'} <= {d['id'] for d in Registry().descriptors()})
        self.assertEqual(resolve('perf')[1]['monitor'], 'off')
        self.assertEqual(resolve('monitor')[1]['monitor'], 'on')
        for verb in ('record', 'launch', 'watch', 'export'):
            with self.subTest(verb=verb), contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as error:
                    main([verb, '--help'])
                self.assertEqual(error.exception.code, 0)
                if verb in ('record', 'launch'):
                    self.assertIn('--diagnose', output.getvalue())
                    self.assertIn('--monitor', output.getvalue())


if __name__ == '__main__':
    unittest.main()
