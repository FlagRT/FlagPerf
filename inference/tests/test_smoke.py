# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CPU smoke checks for the published tree: no device, no torch.

Run from inference/: python3 -B -S -m unittest discover -s tests -v
Device acceptance is the full record -> inspect -> report chain in docs/usage.md.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PLUGINS = [dict(id='fake', kind='vendor', version='1', module='smoke_backend',
                factory='FakeBackend', requires=[], format_vendor='nvidia')]
TARGET = ('import flagprof; h = flagprof.current_attachment(); '
          'assert h is not None; assert h.step(); assert h.step()')
PANELS = {'perf.op', 'perf.layer', 'route.coverage', 'compile.health', 'perf.device', 'memory', 'perf.gap'}


def short_tmp():
    # Writer proxies use AF_UNIX sockets below the run directory (107-byte limit).
    base = tempfile.gettempdir()
    return base if len(os.fsencode(base)) <= 40 else '/tmp'


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.env = dict(os.environ, PYTHONPATH=str(HERE), PYTHONDONTWRITEBYTECODE='1')

    def flagprof(self, *argv):
        return subprocess.run([sys.executable, '-B', '-S', '-m', 'flagprof', *argv], cwd=ROOT,
                              env=self.env, capture_output=True, text=True, timeout=300)

    def test_offline_readers_import_without_torch(self):
        code = ('import importlib.util, sys\n'
                'import flagprof.store.view, flagprof.analyzers, flagprof.report\n'
                'assert not any(n.startswith(("flagprof.vendors", "flagprof.correlate")) for n in sys.modules)\n'
                'assert importlib.util.find_spec("torch") is None\n')
        env = {k: v for k, v in self.env.items() if k != 'PYTHONPATH'}
        result = subprocess.run([sys.executable, '-B', '-S', '-c', code], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fake_vendor_record_import_report(self):
        with tempfile.TemporaryDirectory(prefix='fps-', dir=short_tmp()) as temp:
            temp = Path(temp)
            plugins = temp/'plugins.json'
            plugins.write_text(json.dumps(PLUGINS))
            result = self.flagprof('record', '--capture-mode', 'windowed', '--vendor', 'fake',
                                   '--plugins', str(plugins), '--active', '1', '--windows', '2',
                                   '--store', str(temp/'runs'), '--', sys.executable, '-c', TARGET)
            self.assertEqual(result.returncode, 0, (result.stdout, result.stderr))
            summary = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(summary['status'], 'complete', summary)
            run = Path(summary['run'])
            status = json.loads((run/'record-status.json').read_text())
            self.assertEqual(status['import_completion'], 'complete', status)
            self.assertEqual(len(status['handshakes']), 1)
            self.assertTrue(status['root_handshake'])

            result = self.flagprof('inspect', str(run), '--json')
            self.assertEqual(result.returncode, 0, (result.stdout, result.stderr))
            inspection = json.loads(result.stdout)
            self.assertEqual(inspection['schema'], 'inspect/2')
            self.assertEqual(inspection['run']['state'], 'finalized')
            self.assertEqual(len(inspection['traces']), 2)  # one per window
            for trace in inspection['traces']:
                self.assertEqual(trace['import_status'], 'complete')
                self.assertIn('integrity', trace)

            out = temp/'report'
            result = self.flagprof('report', str(run), '--out', str(out), '--work', str(temp/'work'))
            self.assertEqual(result.returncode, 0, (result.stdout, result.stderr))
            data = json.loads((out/'result.json').read_text())
            self.assertEqual(data['schema'], 'p1-result/1')
            self.assertLessEqual(PANELS, set(data['results']))
            for name, panel in data['results'].items():
                # The fake trace has no project or native linkage, so panels may be unavailable.
                self.assertIn(panel['status'], {'ok', 'partial', 'unavailable'}, name)
            self.assertTrue((out/'report.html').stat().st_size > 0)


if __name__ == '__main__':
    unittest.main()
