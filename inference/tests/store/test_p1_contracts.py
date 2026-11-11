# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Canonical accuracy boundary modes shared by CLI and Python API."""
import unittest
from flagprof.probes.config import resolve


class P1Contracts(unittest.TestCase):
    def test_accuracy_boundary_presets_are_canonical(self):
        default = resolve('accuracy')[1]
        self.assertEqual(resolve('accuracy', {'tensor_boundaries':'layer'})[1], default)
        self.assertEqual(resolve('accuracy', {'tensor_boundaries':'model'})[1],
                         resolve('accuracy', {'tensor_boundaries':[]})[1])
        scope = '<root>.layers.0'
        for boundary in ('model', 'layer', [scope]):
            with self.subTest(boundary=boundary):
                config = resolve('accuracy', {'tensor_boundaries':boundary, 'tensor_ops_in':scope})[1]
                self.assertEqual(config['tensor_ops_in'], scope)
                self.assertEqual(config['tensor_blob'], 'none')
        for invalid in ('operator', {}, 1, [1]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'tensor_boundaries'):
                resolve('accuracy', {'tensor_boundaries':invalid})
        with self.assertRaisesRegex(ValueError, 'tensor options require accuracy'):
            resolve('coverage', {'tensor_boundaries':'model'})
