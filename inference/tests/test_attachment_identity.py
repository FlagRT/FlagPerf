# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Vendor identity validation at the non-profiler attachment entrypoint."""
from contextlib import contextmanager, ExitStack
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from flagprof.runtime import Attachment


class AttachmentIdentityTests(unittest.TestCase):
    @contextmanager
    def attachment(self, root, vendor, preset, checker):
        writer = Mock(errors=[])
        writer.metrics.return_value = {}

        def probes(context, options, state, **kwargs):
            state['probes'] = {}
            return Mock(errors=[])

        with ExitStack() as stack:
            stack.enter_context(patch('flagprof.runtime.online_writer', return_value=writer))
            stack.enter_context(patch('flagprof.runtime.Context'))
            stack.enter_context(patch('flagprof.probes.manager.ProbeManager', side_effect=probes))
            stack.enter_context(patch('flagprof.vendors.' + vendor + '.session.hardware_identity', checker))
            handle = Attachment(root, vendor, preset=preset, capture_mode='windowed')
            try:
                yield handle, writer
            finally:
                handle.close()

    def test_non_profiler_presets_persist_validated_identity(self):
        for vendor in ('thead', 'kunlunxin'):
            for preset in ('perf-lite', 'accuracy', 'coverage'):
                with self.subTest(vendor=vendor, preset=preset), tempfile.TemporaryDirectory() as root:
                    identity = dict(uuid='validated-device', logical_device=0, vendor=vendor)
                    checker = Mock(return_value=identity)
                    with self.attachment(root, vendor, preset, checker) as (handle, writer):
                        checker.assert_called_once_with()
                        self.assertEqual(handle.state['status'], 'running')
                        self.assertEqual(handle.state['hardware_identity'], identity)
                        saved = json.loads((handle.path / 'state.json').read_bytes())
                        self.assertEqual(saved['hardware_identity'], identity)
                        self.assertEqual(saved['capture_status'], 'not_requested')

    def test_non_profiler_presets_reject_invalid_identity(self):
        for vendor in ('thead', 'kunlunxin'):
            for preset in ('perf-lite', 'accuracy', 'coverage'):
                with self.subTest(vendor=vendor, preset=preset), tempfile.TemporaryDirectory() as root:
                    checker = Mock(side_effect=ValueError('physical/framework UUID mismatch'))
                    with self.attachment(root, vendor, preset, checker) as (handle, writer):
                        checker.assert_called_once_with()
                        self.assertTrue(handle.closed)
                        self.assertEqual(handle.state['status'], 'failed')
                        self.assertNotIn('hardware_identity', handle.state)
                        self.assertIsNone(handle.probe_manager)
                        writer.close.assert_called_once_with()
                        saved = json.loads((handle.path / 'state.json').read_bytes())
                        self.assertEqual(saved['status'], 'failed')
                        self.assertIn('ValueError: physical/framework UUID mismatch', saved['errors'])
