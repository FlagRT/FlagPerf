"""Configuration provenance, process consumption and negative path tests."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from benchmarks.case_assets import resolve_case_assets, portable_assets, verify_worker_assets, load_case_config
from base.vendors.protocol import ConfigurationError


class CaseAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.case = self.base / "benchmarks/example"
        self.chip = self.case / "fixture/Chip"
        self.chip.mkdir(parents=True)
        for directory, text in [(self.case, "A: 1\nB: 1\nC: 1\nD: 1\n"), (self.chip.parent, "B: 2\nC: 2\nD: 2\n"), (self.chip, "C: 3\nD: 3\n")]:
            (directory / "case_config.yaml").write_text(text)
            (directory / "env.sh").write_text("true\n")
            (directory / "main.py").write_text("pass\n")
        (self.chip.parent / "runtime_requirements.json").write_text(json.dumps({"schema_version": 1, "supported": True}))

    def resolve(self, override=None):
        return resolve_case_assets(self.base, "example:Chip", "fixture", override)

    def test_all_layers_and_actual_case_consumer_agree(self):
        result = self.base / "results"
        (result / "case-config").mkdir(parents=True)
        override = result / "case-config/override.yaml"
        override.write_text("D: 4\n")
        assets = self.resolve(override)
        self.assertEqual(assets["merged_config"], {"A": 1, "B": 2, "C": 3, "D": 4})
        self.assertEqual(assets["entrypoint"]["path"], str(self.chip / "main.py"))
        self.assertEqual(len(assets["environments"]), 3)
        contract = result / "case-assets.json"
        contract.write_text(json.dumps(portable_assets(assets)))
        verified = verify_worker_assets(self.base, contract, "example:Chip", "fixture")
        self.assertEqual(assets, verified)
        with patch.dict(os.environ, {"FLAGPERF_CASE_ASSETS": str(contract)}):
            self.assertEqual(load_case_config(self.case, "fixture/Chip"), assets["merged_config"])
            with self.assertRaises(ConfigurationError):
                load_case_config(self.case, "fixture/OtherChip")
        override.write_text("D: 9\n")
        with self.assertRaisesRegex(ConfigurationError, "hash changed"):
            verify_worker_assets(self.base, contract, "example:Chip", "fixture")

    def test_entrypoint_falls_back_through_vendor_before_generic(self):
        (self.chip / "main.py").unlink()
        self.assertEqual(self.resolve()["entrypoint"]["path"], str(self.chip.parent / "main.py"))
        (self.chip.parent / "main.py").unlink()
        self.assertEqual(self.resolve()["entrypoint"]["path"], str(self.case / "main.py"))

    def test_unsupported_is_a_contract_and_cannot_be_relaxed(self):
        path = self.chip.parent / "runtime_requirements.json"
        path.write_text(json.dumps({"schema_version": 1, "supported": False, "unsupported_reason": "fixture capability"}))
        self.assertFalse(self.resolve()["requirements"]["supported"])
        (self.chip / "runtime_requirements.json").write_text(json.dumps({"schema_version": 1, "supported": True}))
        with self.assertRaisesRegex(ConfigurationError, "conflicting"):
            self.resolve()

    def test_missing_and_malformed_contracts_fail_closed(self):
        path = self.chip.parent / "runtime_requirements.json"
        path.unlink()
        with self.assertRaisesRegex(ConfigurationError, "requirements missing"):
            self.resolve()
        for value in ([], {"schema_version": 1, "supported": "false"}, {"schema_version": 1, "process_scope": {"nnodes": True}}):
            path.write_text(json.dumps(value))
            with self.assertRaises(ConfigurationError):
                self.resolve()

    def test_traversal_and_symlink_escape_are_rejected(self):
        for spec,vendor in [("..", "fixture"), ("example:..", "fixture"), ("example:Chip:Other", "fixture"), ("example", "../outside")]:
            with self.subTest(spec=spec,vendor=vendor), self.assertRaises(ConfigurationError):
                resolve_case_assets(self.base, spec, vendor)
        (self.chip / "main.py").unlink()
        outside = self.base / "outside.py"
        outside.write_text("pass\n")
        (self.chip / "main.py").symlink_to(outside)
        # A source inside the repo but outside its case is also not a case asset.
        with self.assertRaises(ConfigurationError):
            self.resolve()

    def test_no_implicit_a100_for_new_vendor(self):
        for p in self.chip.parent.glob('case_config.yaml'):
            p.unlink()
        assets = resolve_case_assets(self.base, "example", "fixture")
        self.assertEqual(assets["selector"], "fixture")
        self.assertIsNone(assets["chip"])

    def test_legacy_ascend_cases_now_have_explicit_contracts(self):
        for case in (BASE / "benchmarks").iterdir():
            if (case / "main.py").is_file():
                with self.subTest(case=case.name):
                    assets = resolve_case_assets(BASE, case.name, "ascend")
                    self.assertTrue(assets["requirement_files"])

    def test_actual_worker_consumes_contract_and_sourced_environment(self):
        import subprocess
        result = self.base / 'result'
        (result/'case-config').mkdir(parents=True)
        override = result/'case-config/override.yaml'
        override.write_text('D: 4\n')
        (self.chip/'env.sh').write_text('export FIXTURE_ENV=from-chip\n')
        (self.chip/'main.py').write_text(
            'import json,os\nfrom pathlib import Path\n'
            'from benchmarks.case_assets import load_case_config\n'
            'print(json.dumps({"config":load_case_config(Path.cwd(),"fixture/Chip"),"env":os.environ["FIXTURE_ENV"]},sort_keys=True))\n')
        contract=result/'case-assets.json'
        contract.write_text(json.dumps(portable_assets(self.resolve(override))))
        binary=self.base/'bin';binary.mkdir()
        launcher=binary/'torchrun'
        launcher.write_text('#!'+sys.executable+'\nimport os,sys\n'
                            'args=sys.argv[1:]\nindex=next(i for i,a in enumerate(args) if not a.startswith("--"))\n'
                            'os.execv(sys.executable,[sys.executable,*args[index:]])\n')
        launcher.chmod(0o755)
        env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ.get('PATH',''),'PYTHONPATH':str(BASE),
             'PYTHONDONTWRITEBYTECODE':'1'}
        command=[sys.executable,str(BASE/'benchmark_worker.py'),'--case_name','example:Chip','--vendor','fixture',
                 '--case-assets',str(contract),'--perf_path',str(self.base),'--log_dir',str(result),
                 '--nnodes','1','--nproc_per_node','1','--node_rank','0','--master_addr','127.0.0.1',
                 '--master_port','29721','--host_addr','127.0.0.1','--log_level','INFO','--monitor-events']
        proc=subprocess.run(command,env=env,capture_output=True,text=True,timeout=15)
        self.assertEqual(proc.returncode,0,proc.stderr+proc.stdout)
        output=json.loads((result/'example/127.0.0.1_noderank0/benchmark.log.txt').read_text())
        self.assertEqual(output,{'config':{'A':1,'B':2,'C':3,'D':4},'env':'from-chip'})
        self.assertEqual(json.loads((result/'benchmark-events/torchrun-window.json').read_text())['returncode'],0)


if __name__ == "__main__":
    unittest.main()
