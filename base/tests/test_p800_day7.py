"""Regression for bounded CLI early skip without privileged side effects."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from executors.benchmark import BenchmarkExecutor, BenchmarkRunRequest, add_cli_arguments
from base.vendors.protocol import ConfigurationError

class BoundedEarlySkipTests(unittest.TestCase):
    def request(self, root, case='computation-FP64', extra=()):
        parser=argparse.ArgumentParser()
        add_cli_arguments(parser)
        args=parser.parse_args(['--config',str(BASE/'configs/kunlunxin_p800_xpytorch29.yaml'),
            '--case',case+':P800','--physical-device-ids','5','--nproc-per-node','1',
            '--timeout','300','--allow-candidate-runtime','--result-dir',str(root),
            '--privilege-command','sudo -n','--reservation-reference','fixture',
            '--reservation-end','2000-01-01T00:00:00+00:00',*extra])
        return BenchmarkRunRequest.from_namespace(args)

    def test_every_unsupported_case_skips_without_commands_and_preserves_reason(self):
        for case in ('computation-FP64','computation-FP8','computation-TF32',
                     'interconnect-MPI_interserver','interconnect-P2P_interserver'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'run'
                with patch('subprocess.run',side_effect=AssertionError('unexpected external command')), \
                     patch('executors.bounded_benchmark.execute',side_effect=AssertionError('unexpected device lifecycle')), \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(BenchmarkExecutor().execute(self.request(root,case)),0)
                summary=json.loads((root/'summary.json').read_text())
                self.assertEqual(summary['status'],'skipped')
                self.assertTrue(summary['skip_reason'])
                self.assertEqual(json.loads((root/'benchmark-result.json').read_text())['metrics'],[])
                self.assertFalse((root/'lease.json').exists())
                self.assertFalse((root/'container-create.json').exists())

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'; root.mkdir()
            marker=root/'summary.json'; marker.write_text('historical')
            with self.assertRaises(FileExistsError):
                BenchmarkExecutor().execute(self.request(root))
            self.assertEqual(marker.read_text(),'historical')

    def test_conflicting_output_paths_rejected_before_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            with self.assertRaises(ConfigurationError):
                BenchmarkExecutor().execute(self.request(root,extra=['--result-root',tmp]))
            self.assertFalse(root.exists())

if __name__=='__main__':
    unittest.main()
