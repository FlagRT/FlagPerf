# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from unittest import mock

BASE_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = BASE_DIR / "benchmarks"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

import run as base_run  # noqa: E402
from executors.common import (  # noqa: E402
    ConfigurationError,
    runtime_lock_record,
    validate_runtime_identity,
)


QUALIFIER_PATH = (
    BASE_DIR / "vendors" / "ascend" / "torch_fl_2.10_flagcx" /
    "qualify_candidate.py"
)


def load_qualifier_module():
    spec = importlib.util.spec_from_file_location("test_flagcx_qualifier", QUALIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_p2p_module():
    path = BENCHMARKS_DIR / "interconnect-P2P_intraserver" / "main.py"
    spec = importlib.util.spec_from_file_location("test_ascend_p2p_case", path)
    module = importlib.util.module_from_spec(spec)
    fake_torch = types.ModuleType("torch")
    fake_dist = types.ModuleType("torch.distributed")
    fake_torch.distributed = fake_dist
    fake_torch.float32 = "float32"
    fake_utils = types.ModuleType("drivers.utils")
    for name in (
        "accelerator_device", "benchmark_measurement_finish",
        "benchmark_measurement_start", "bootstrap_vendor",
        "host_device_sync", "multi_device_sync", "set_ieee_float32",
    ):
        setattr(fake_utils, name, mock.Mock(name=name))
    fake_drivers = types.ModuleType("drivers")
    fake_drivers.__path__ = []
    fake_yaml = types.ModuleType("yaml")
    with mock.patch.dict(sys.modules, {
        "torch": fake_torch,
        "torch.distributed": fake_dist,
        "drivers": fake_drivers,
        "drivers.utils": fake_utils,
        "yaml": fake_yaml,
    }):
        spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self):
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


class AscendP2PCaseTest(unittest.TestCase):
    def test_case_preserves_blocking_send_and_legacy_formula(self):
        module = load_p2p_module()
        tensor = FakeTensor()
        config = Namespace(vendor="ascend")
        case_config = Namespace(Melements=1, WARMUP=1, ITERS=2)

        with mock.patch.object(
                    module.torch, "rand", return_value=tensor, create=True
                ), \
                mock.patch.object(
                    module, "accelerator_device", return_value="flagos:0"
                ) as device, \
                mock.patch.object(module, "set_ieee_float32"), \
                mock.patch.object(module, "host_device_sync"), \
                mock.patch.object(module, "multi_device_sync"), \
                mock.patch.object(module.dist, "send", create=True) as send, \
                mock.patch.object(module.dist, "recv", create=True) as recv, \
                mock.patch.object(
                    module, "benchmark_measurement_start", return_value="event"
                ), \
                mock.patch.object(module, "benchmark_measurement_finish") as finish, \
                mock.patch.object(
                    module.time, "perf_counter", side_effect=(10.0, 12.0)
                ):
            gb, gib = module.main(
                config, case_config, rank=0, world_size=2,
                local_rank=0, select_gpus=[0, 1],
            )

        device.assert_called_once_with("ascend", 0)
        self.assertEqual(tensor.to_calls, ["flagos:0"])
        self.assertEqual(send.call_count, 3)
        recv.assert_not_called()
        finish.assert_called_once_with("event")
        self.assertEqual(gb, 0.01)
        self.assertEqual(gib, 0.01)

    def test_ascend_contract_routes_flagos_through_communication_profile(self):
        case_root = BENCHMARKS_DIR / "interconnect-P2P_intraserver" / "ascend"
        case_config = (case_root / "case_config.yaml").read_text(encoding="utf-8")
        requirements = json.loads(
            (case_root / "runtime_requirements.json").read_text(encoding="utf-8")
        )
        self.assertIn('DIST_BACKEND: "flagos"', case_config)
        self.assertEqual(requirements["runtime_profile"], "torch_fl_2.10_flagcx")
        self.assertEqual(requirements["process_scope"], {
            "nnodes": 1,
            "nproc_per_node": 2,
        })


class FlagCXRuntimeControlTest(unittest.TestCase):
    def setUp(self):
        self.candidate_config = (
            BASE_DIR / "configs" / "ascend910_cann9_p2p_candidate.yaml"
        )
        self.smoke_config = (
            BASE_DIR / "benchmarks" / "interconnect-P2P_intraserver" /
            "ascend" / "case_config.smoke.yaml"
        )

    def test_default_runtime_is_rejected_for_p2p(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = base_run.main([
                "benchmark", "run",
                "--case", "interconnect-P2P_intraserver",
                "--device-ids", "14,15",
                "--dry-run",
            ])
        self.assertEqual(code, 2)
        self.assertIn("requires runtime profile", stderr.getvalue())

    def test_candidate_plan_records_lock_contract_and_explicit_permission(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = base_run.main([
                "benchmark", "run",
                "--config", str(self.candidate_config),
                "--case", "interconnect-P2P_intraserver",
                "--device-ids", "14,15",
                "--nproc-per-node", "2",
                "--case-config", str(self.smoke_config),
                "--allow-candidate-runtime",
                "--dry-run",
            ])
        self.assertEqual(code, 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(
            plan["runtime_lock"]["runtime_profile"],
            "torch_fl_2.10_flagcx",
        )
        self.assertEqual(
            plan["runtime_requirements"]["requirements"]["distributed_backend"],
            "flagos",
        )
        self.assertTrue(plan["permissions"]["candidate_runtime"])

    def test_candidate_plan_rejects_any_rank_count_other_than_two(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = base_run.main([
                "benchmark", "run",
                "--config", str(self.candidate_config),
                "--case", "interconnect-P2P_intraserver",
                "--device-ids", "14,15",
                "--nproc-per-node", "1",
                "--case-config", str(self.smoke_config),
                "--dry-run",
            ])
        self.assertEqual(code, 2)
        self.assertIn("requires exactly 2 local ranks", stderr.getvalue())

    def test_candidate_plan_rejects_unbounded_default_case_config(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = base_run.main([
                "benchmark", "run",
                "--config", str(self.candidate_config),
                "--case", "interconnect-P2P_intraserver",
                "--device-ids", "14,15",
                "--nproc-per-node", "2",
                "--dry-run",
            ])
        self.assertEqual(code, 2)
        self.assertIn("explicit bounded --case-config", stderr.getvalue())

    def test_candidate_plan_rejects_modified_smoke_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            modified = Path(temporary) / "case_config.yaml"
            modified.write_text(
                self.smoke_config.read_text(encoding="utf-8").replace(
                    "Melements: 1", "Melements: 1024"
                ),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = base_run.main([
                    "benchmark", "run",
                    "--config", str(self.candidate_config),
                    "--case", "interconnect-P2P_intraserver",
                    "--device-ids", "14,15",
                    "--nproc-per-node", "2",
                    "--case-config", str(modified),
                    "--allow-candidate-runtime",
                    "--dry-run",
                ])
        self.assertEqual(code, 2)
        self.assertIn("bounded configuration allowlist", stderr.getvalue())

    def test_candidate_requires_explicit_permission_until_promoted(self):
        lock = runtime_lock_record("torch_fl_2.10_flagcx")
        image_id = lock["image_manifest"]["image_id"]
        self.assertTrue(image_id.startswith("sha256:"))
        config = {
            "image": lock["image_manifest"]["image"],
            "runtime_profile": "torch_fl_2.10_flagcx",
        }
        with self.assertRaisesRegex(ConfigurationError, "unvalidated candidate"):
            validate_runtime_identity(
                config, {"Id": image_id},
            )
        validated = validate_runtime_identity(
            config, {"Id": image_id}, allow_candidate=True,
        )
        self.assertFalse(validated["image_manifest"]["validated"])

    def test_build_contract_is_pinned_and_excludes_torch_npu(self):
        profile = BASE_DIR / "vendors" / "ascend" / "torch_fl_2.10_flagcx"
        dockerfile = (profile / "Dockerfile").read_text(encoding="utf-8")
        build_script = (profile / "build_image.sh").read_text(encoding="utf-8")
        stack_lock = (profile / "stack.lock.yaml").read_text(encoding="utf-8")
        self.assertIn("FLAGCX_TORCH_BACKEND=flagos", dockerfile)
        self.assertIn("FLAGCX_ADAPTOR=ascend", dockerfile)
        self.assertIn("^torch[-_]npu==", dockerfile)
        self.assertIn(
            "55eb2ffff6988ae1db5e6ecb325472aecc93d238",
            build_script,
        )
        self.assertIn(
            "bc84ea26eee3783085ac140e8792455433299fb0bb2330ec9d2680b6d2b7736d",
            build_script,
        )
        self.assertIn(
            "524654689fdfe896c2f935f89e884bca4c57e50cd263cc145659e3b8b8ac7f0a",
            build_script,
        )
        self.assertIn("--network=none", build_script)
        self.assertIn("_get_backend_name", (
            profile / "verify_flagcx_p2p.py"
        ).read_text(encoding="utf-8"))
        for artifact in (
            "source-artifacts.sha256",
            "validation-summary.json",
            "image-manifest.json",
        ):
            self.assertTrue((profile / artifact).is_file(), artifact)
            self.assertIn(artifact, stack_lock)

    def test_candidate_environment_includes_hccl_initialization_control(self):
        config = json.loads(self.candidate_config.read_text(encoding="utf-8"))
        self.assertEqual(config["runtime_environment"], {
            "FLAGCX_TORCH_BACKEND": "flagos",
            "HCCL_WHITELIST_DISABLE": "1",
        })


class FlagCXQualificationRunnerTest(unittest.TestCase):
    def test_rank_record_parser_ignores_diagnostics_and_validates_both_ranks(self):
        qualifier = load_qualifier_module()
        records = qualifier.parse_rank_record_lines("\n".join([
            "HCCL diagnostic",
            json.dumps({
                "schema_version": 1, "status": "passed", "rank": 1,
                "world_size": 2, "public_backend": "flagos",
                "inner_backend": "ProcessGroupFlagCX",
                "stream_mode": "default",
                "payloads": [
                    {"elements": value} for value in (1, 4, 257, 65536)
                ],
            }),
            json.dumps({
                "schema_version": 1, "status": "passed", "rank": 0,
                "world_size": 2, "public_backend": "flagos",
                "inner_backend": "ProcessGroupFlagCX",
                "stream_mode": "default",
                "payloads": [
                    {"elements": value} for value in (1, 4, 257, 65536)
                ],
            }),
        ]))
        qualifier.validate_sentinel_records(records)
        self.assertEqual(sorted(item["rank"] for item in records), [0, 1])

        concatenated = "diagnostic\n" + "".join(
            json.dumps(record) for record in records
        )
        reparsed = qualifier.parse_rank_record_lines(concatenated)
        qualifier.validate_sentinel_records(reparsed)
        self.assertEqual(sorted(item["rank"] for item in reparsed), [0, 1])

    def test_rank_record_validation_rejects_non_flagcx_inner_backend(self):
        qualifier = load_qualifier_module()
        records = [
            {
                "schema_version": 1, "status": "passed", "rank": rank,
                "world_size": 2, "public_backend": "flagos",
                "inner_backend": "ProcessGroupHCCL",
                "stream_mode": "default",
                "payloads": [
                    {"elements": value} for value in (1, 4, 257, 65536)
                ],
            }
            for rank in (0, 1)
        ]
        with self.assertRaisesRegex(RuntimeError, "did not select FlagCX"):
            qualifier.validate_sentinel_records(records)

    def test_timeout_cleanup_gate_is_deterministic_and_fail_closed(self):
        qualifier = load_qualifier_module()
        config = {
            "image": "candidate:locked",
            "shm_size": "1g",
            "runtime_environment": {},
            "required_devices": [],
            "host_mounts": [],
        }
        command = qualifier.build_container_command(
            config, (14, 15), container_name="flagcx-timeout-test",
            master_port=29821, gate="c3-timeout-cleanup",
        )
        self.assertEqual(command[-1], "exec sleep 600")

        qualifier.validate_expected_timeout_record({
            "returncode": 124,
            "timed_out": True,
            "timeout_cleanup": {
                "remove_returncode": 0,
                "inspect_returncode": 1,
            },
        })
        with self.assertRaisesRegex(RuntimeError, "still exists"):
            qualifier.validate_expected_timeout_record({
                "returncode": 124,
                "timed_out": True,
                "timeout_cleanup": {
                    "remove_returncode": 0,
                    "inspect_returncode": 0,
                },
            })

    def test_acl_event_gate_requires_timeline_failure_and_sync_success(self):
        qualifier = load_qualifier_module()
        probe = {
            "schema_version": 1,
            "kind": "acl-event-flag-ab",
            "device": 0,
            "create_stream_rc": 0,
            "destroy_stream_rc": 0,
            "results": [{
                "name": "timeline", "flag": 0x8,
                "create_rc": 0, "record_rc": 0, "wait_rc": 207000,
                "synchronize_rc": None, "destroy_rc": 0,
            }, {
                "name": "sync", "flag": 0x1,
                "create_rc": 0, "record_rc": 0, "wait_rc": 0,
                "synchronize_rc": 0, "destroy_rc": 0,
            }],
        }
        qualifier.validate_acl_event_records([probe])
        probe["results"][1]["wait_rc"] = 207000
        with self.assertRaisesRegex(RuntimeError, "sync event wait_rc failed"):
            qualifier.validate_acl_event_records([probe])


if __name__ == "__main__":
    unittest.main()
