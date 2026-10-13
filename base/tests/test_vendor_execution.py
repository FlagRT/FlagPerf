"""Exercise real orchestration with a test-only provider and no hardware access."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import run
from executors import benchmark, common
from base.vendors.protocol import DeviceBinding, ConfigurationError, runtime_root
from base.vendors.registry import PROVIDERS
from generate_benchmark_report import generate_and_record


class FixtureProvider:
    name = "fixture"
    display_name = "Fixture Accelerator"
    default_runtime_profile = "locked"
    fallback_markers = ("fixture CPU fallback",)

    def runtime_root(self, base, profile):
        return runtime_root(base, self.name, profile or "locked")

    def validate_config(self, config):
        if config.get("runtime_environment"):
            raise ConfigurationError("unknown fixture environment")

    def validate_selection(self, context):
        context.validate(require_selection=True)
        if context.npu_ids is not None:
            raise ConfigurationError("--npu-ids only supports Ascend")

    def preflight(self, result, config, context, *, label="host-preflight"):
        p = result / label / "summary.json"
        common.write_json(p, {"selection": {"selected_device_ids": [8]}, "status": "passed"})
        return p

    def bindings(self, preflight, selected):
        return [DeviceBinding("fixture", 8, "0000:01:00.0", "fixture-uuid", self.node, "/dev/fixture0",
                              0, 3, "fixture:3", 0, "fixture/fixture-uuid")]

    def container_policy(self, config):
        return {"network":"none","ipc_namespace":"private","pid_namespace":"private"}

    def docker_args(self, config, bindings):
        return ["--network=none", "-e", "FIXTURE_VISIBLE=3", f"--device={self.node}:/dev/fixture0:rwm"]

    def bootstrap(self, base, config):
        return ""

    def worker_python(self, config):
        return "/fixture/python"

    def container_preflight(self, config, bindings):
        return ["/fixture/python", "probe.py"]

    def monitor_policy(self, enabled):
        return {"enabled":enabled,"vendor":"fixture","collector":"fixture usage",
                "required_samples_per_target":10,"target_resource":"fixture-core",
                "metric_fields":[{"key":"busy","label":"Device busy","unit":"%"}],
                "automatic_workload_extension":False}

    def monitor_targets(self, preflight, selected):
        return [{"device_id":"fixture/fixture-uuid"}]

    def create_monitor(self, targets):
        raise RuntimeError("fixture collector unavailable")


class VendorExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        case = self.base / "benchmarks/example/fixture/Chip"
        case.mkdir(parents=True)
        (case / "case_config.yaml").write_text("VALUE: 2\n")
        (case.parent.parent / "case_config.yaml").write_text("VALUE: 1\n")
        (case / "main.py").write_text("pass\n")
        self.requirements = case / "runtime_requirements.json"
        self.requirements.write_text(json.dumps({"schema_version":1,"supported":True}))
        runtime = self.base / "vendors/fixture/locked"
        runtime.mkdir(parents=True)
        (runtime / "stack.lock.yaml").write_text("{}\n")
        (runtime / "image-manifest.json").write_text(json.dumps({"schema_version":1,"image":"fixture:image","image_id":"sha256:fixture","validated":True}))
        self.config = self.base / "host.json"
        self.config.write_text(json.dumps({"schema_version":1,"image":"fixture:image","vendor":"fixture","shm_size":"1g",
            "expected_device_ids":[8],"result_root":str(self.base/'results'),"required_devices":[],"host_mounts":[]}))
        self.provider = FixtureProvider()
        self.provider.node = str(self.base / "fake-device")
        Path(self.provider.node).touch()
        for patcher in (patch.dict(PROVIDERS, fixture=self.provider), patch.object(common,"BASE_DIR",self.base), patch.object(benchmark,"BASE_DIR",self.base)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def args(self, *extra):
        return ["benchmark","run","--config",str(self.config),"--case","example:Chip","--physical-device-ids","8",*extra]

    def test_dry_run_has_no_runtime_or_file_side_effect(self):
        def forbidden(*args,**kwargs):
            raise AssertionError("runtime operation during dry-run")
        before = sorted(p.relative_to(self.base).as_posix() for p in self.base.rglob('*'))
        stdout = io.StringIO()
        with patch.object(benchmark,"docker_inspect",side_effect=forbidden), patch.object(benchmark,"run_host_preflight",side_effect=forbidden), \
             patch.object(benchmark,"DeviceLease",side_effect=forbidden), patch.object(benchmark,"create_usage_monitor",side_effect=forbidden), \
             patch.object(subprocess,"run",side_effect=forbidden), redirect_stdout(stdout):
            self.assertEqual(run.main(self.args("--dry-run")),0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["vendor"],"fixture")
        self.assertEqual(plan["permissions"]["network"],"none")
        self.assertEqual(plan["case_assets"]["merged_config"],{"VALUE":2})
        self.assertNotIn("ascend",stdout.getvalue().lower())
        self.assertEqual(before,sorted(p.relative_to(self.base).as_posix() for p in self.base.rglob('*')))

    def test_fixture_execution_uses_provider_and_records_bindings(self):
        commands=[]
        def fake_run(argv, **kwargs):
            commands.append(argv)
            result = next((self.base/'results').iterdir())
            log=result/'example/127.0.0.1_noderank0/benchmark.log.txt'
            log.write_text("[FlagPerf Result]Rank 0's example=1.25GB/s\n")
            return subprocess.CompletedProcess(argv,0,"fixture worker complete\n")
        actual_lease=common.DeviceLease
        def lease(*args,**kwargs):
            return actual_lease(*args,root=self.base/'leases',**kwargs)
        with patch.object(benchmark,"docker_inspect",return_value={"Id":"sha256:fixture"}), \
             patch.object(benchmark,"DeviceLease",side_effect=lease), patch.object(benchmark.subprocess,"run",side_effect=fake_run), redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(self.args("--monitor","off")),0)
        result = next((self.base/'results').iterdir())
        summary=json.loads((result/'summary.json').read_text())
        self.assertEqual(summary["status"],"passed")
        self.assertEqual(summary["device_bindings"][0]["framework_logical_id"],3)
        self.assertEqual(summary["device_bindings"][0]["framework_local_rank"],0)
        self.assertEqual(summary["device_lease"]["resource_keys"],["fixture/fixture-uuid"])
        command=' '.join(commands[0])
        for token in ["FIXTURE_VISIBLE=3","--vendor fixture","/fixture/python probe.py"]:
            self.assertIn(token,command)
        for token in ["ascend","davinci","HCCL"]:
            self.assertNotIn(token,command)
        report=(result/'report.md').read_bytes()
        generate_and_record(result)
        self.assertEqual(report,(result/'report.md').read_bytes())
        self.assertNotIn(b'npu-smi',(result/'report_monitor.md').read_bytes())
        self.assertEqual(json.loads((result/'summary.json').read_text())["status"],"passed")

    def test_unsupported_contract_skips_before_preflight_and_lease(self):
        self.requirements.write_text(json.dumps({"schema_version":1,"supported":False,"unsupported_reason":"fixture unsupported operation"}))
        with patch.object(benchmark,"docker_inspect") as docker, patch.object(benchmark,"run_host_preflight") as preflight, \
             patch.object(benchmark,"DeviceLease") as lease, redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(self.args()),0)
        docker.assert_not_called();preflight.assert_not_called();lease.assert_not_called()

    def test_missing_contract_is_not_a_successful_skip(self):
        self.requirements.unlink()
        with self.assertRaises(ConfigurationError):
            benchmark.BenchmarkExecutor().plan(benchmark.BenchmarkRunRequest.from_namespace(run.build_parser().parse_args(self.args())))

    def test_missing_lock_and_candidate_identity_remain_closed(self):
        manifest = self.base / "vendors/fixture/locked/image-manifest.json"
        value = json.loads(manifest.read_text())
        value["validated"] = False
        manifest.write_text(json.dumps(value))
        config = json.loads(self.config.read_text())
        with self.assertRaisesRegex(ConfigurationError,"candidate"):
            common.validate_runtime_identity(config,{"Id":"sha256:fixture"})
        record = common.validate_runtime_identity(config,{"Id":"sha256:fixture"},allow_candidate=True)
        self.assertFalse(record["image_manifest"]["validated"])
        with self.assertRaisesRegex(ConfigurationError,"does not match"):
            common.validate_runtime_identity(config,{"Id":"sha256:drift"},allow_candidate=True)
        manifest.unlink()
        with self.assertRaisesRegex(ConfigurationError,"missing"):
            common.runtime_lock_record(vendor="fixture")

    def test_unknown_vendor_cannot_read_another_vendors_runtime(self):
        config = json.loads(self.config.read_text())
        config["vendor"]="unregistered"
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ConfigurationError,"unregistered vendor"):
            common.load_host_config(self.config)

    def test_runtime_imports_are_forbidden_during_real_cli_dry_run(self):
        script = """
import builtins,sys
sys.path.insert(0, BASE)
real_import=builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','torch_fl','torch_xmlir','torch_xray','torch_npu','flagcx','flag_gems'}:
        raise AssertionError('runtime import during planning: '+name)
    return real_import(name,*args,**kwargs)
builtins.__import__=checked
import run
raise SystemExit(run.main(['benchmark','run','--case','computation-FP32','--device-ids','14','--dry-run']))
""".replace("BASE", repr(str(BASE)))
        proc=subprocess.run([sys.executable,"-B","-c",script],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["vendor"],"ascend")


if __name__ == "__main__":
    unittest.main()
