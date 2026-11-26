# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class KernelProfileTests(unittest.TestCase):
    def test_chrome_microseconds_are_reported_in_nanoseconds(self):
        script = Path(__file__).resolve().parents[1] / 'tools' / 'kernel_profile.py'
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder) / 'trace.json'
            report = Path(folder) / 'report.json'
            events = [dict(cat='kernel', name='ncclKernel', dur=1000),
                      dict(cat='kernel', name='compute', dur=0.375)]
            trace.write_text(json.dumps(dict(traceEvents=events)))
            result = subprocess.run([sys.executable, str(script), str(trace), str(report)],
                                    capture_output=True, text=True, check=True)
            profile = json.loads(report.read_text())
        self.assertEqual(profile['kernel_time_ns'], 1_000_375)
        self.assertEqual(profile['comm_kernel_time_ns'], 1_000_000)
        self.assertEqual(profile['top'][1]['median_ns'], 375)
        self.assertIn('kernel_time_ms=1.0', result.stdout)
