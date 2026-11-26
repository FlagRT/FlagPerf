# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""NVML-compat identity matching: target and telemetry must agree on the
exact entity string, so matching is on the UUID body across vendor prefixes.

Pure stdlib (ctypes only); runs under the -S smoke discovery.
"""
import os
import unittest
from unittest.mock import patch

from flagprof.telemetry import nvml_compat
from flagprof.telemetry.nvml_compat import candidate_libraries, match_uuid, strip_prefix


class NvmlCompatTests(unittest.TestCase):
    def test_strip_prefix_variants(self):
        self.assertEqual(strip_prefix('GPU-abc'), 'abc')
        self.assertEqual(strip_prefix('PPU-abc'), 'abc')
        self.assertEqual(strip_prefix('P800-abc'), 'abc')
        self.assertEqual(strip_prefix('abc'), 'abc')
        self.assertEqual(strip_prefix(None), '')

    def test_match_uuid_is_body_based(self):
        candidates = ['GPU-011ef111-aaa', 'GPU-011ef111-bbb', None]
        self.assertEqual(match_uuid(candidates, 'PPU-011ef111-bbb'),
                         (1, 'GPU-011ef111-bbb'))
        self.assertEqual(match_uuid(candidates, '011ef111-aaa'),
                         (0, 'GPU-011ef111-aaa'))
        self.assertEqual(match_uuid(candidates, 'P800-011ef111-aaa'),
                         (0, 'GPU-011ef111-aaa'))

    def test_env_override_is_tried_first(self):
        with patch.dict(os.environ, dict(FLAGPROF_TELEMETRY_LIBRARY='/opt/vendor/libnvidia-ml.so')):
            candidates = candidate_libraries()
        self.assertEqual(candidates[0], '/opt/vendor/libnvidia-ml.so')

    def test_load_error_enumerates_every_attempt(self):
        with patch.dict(os.environ, dict(FLAGPROF_TELEMETRY_LIBRARY='/nope/lib.so')), \
             patch('ctypes.CDLL', side_effect=OSError('cannot open')):
            candidates = candidate_libraries()
            with self.assertRaises(OSError) as caught:
                nvml_compat.load()
        message = str(caught.exception)
        self.assertIn('tried ' + str(len(candidates)) + ' path(s)', message)
        for path in candidates:
            self.assertIn(path, message)

    def test_check_command_is_success_or_diagnosable(self):
        """`flagprof telemetry check` must either resolve the provider or
        report every attempted path plus an actionable hint."""
        import contextlib
        import io
        from flagprof.telemetry import check
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = check.run('thead', 'all', True)
        output = stream.getvalue()
        self.assertIn(code, (0, 2), 'exit code must signal success or failure')
        if code == 0:
            self.assertIn('source', output)
            self.assertIn('library', output)
        else:
            self.assertIn('tried', output)
            self.assertIn('hint', output)

    def test_match_uuid_rejects_unknown_and_empty(self):
        candidates = ['GPU-011ef111-aaa']
        self.assertEqual(match_uuid(candidates, 'deadbeef'), (None, None))
        self.assertEqual(match_uuid(candidates, ''), (None, None))
        self.assertEqual(match_uuid(candidates, None), (None, None))
        self.assertEqual(match_uuid([None, None], 'anything'), (None, None))


if __name__ == '__main__':
    unittest.main()
