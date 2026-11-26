# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""NVML-compat identity matching: target and telemetry must agree on the
exact entity string, so matching is on the UUID body across vendor prefixes.

Pure stdlib (ctypes only); runs under the -S smoke discovery.
"""
import unittest

from flagprof.telemetry.nvml_compat import match_uuid, strip_prefix


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

    def test_match_uuid_rejects_unknown_and_empty(self):
        candidates = ['GPU-011ef111-aaa']
        self.assertEqual(match_uuid(candidates, 'deadbeef'), (None, None))
        self.assertEqual(match_uuid(candidates, ''), (None, None))
        self.assertEqual(match_uuid(candidates, None), (None, None))
        self.assertEqual(match_uuid([None, None], 'anything'), (None, None))


if __name__ == '__main__':
    unittest.main()
