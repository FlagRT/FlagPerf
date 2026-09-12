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
from executors.common import runtime_lock_record, validate_runtime_identity  # noqa: E402


QUALIFIER_PATH = (
    BASE_DIR / "vendors" / "ascend" / "torch_fl_2.10_flagcx" /
    "qualify_runtime.py"
)
CALIBRATOR_PATH = (
    BASE_DIR / "vendors" / "ascend" / "torch_fl_2.10_flagcx" /
    "run_p2p_calibration.py"
)
FORMAL_RUNNER_PATH = (
    BASE_DIR / "vendors" / "ascend" / "torch_fl_2.10_flagcx" /
    "run_p2p_qualification.py"
)
QUALIFICATION_PLAN_PATH = (
    BASE_DIR / "benchmarks" / "interconnect-P2P_intraserver" / "ascend" /
    "p2p-qualification-plan.json"
)


def load_qualifier_module():
    spec = importlib.util.spec_from_file_location("test_flagcx_qualifier", QUALIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_calibrator_module():
    spec = importlib.util.spec_from_file_location(
        "test_flagcx_p2p_calibrator", CALIBRATOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_formal_runner_module():
    spec = importlib.util.spec_from_file_location(
        "test_flagcx_p2p_qualification", FORMAL_RUNNER_PATH,
    )
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
        self.formal_config = (
            BASE_DIR / "configs" / "ascend910_cann9_p2p.yaml"
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

    def test_formal_plan_records_validated_lock_without_candidate_permission(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = base_run.main([
                "benchmark", "run",
                "--config", str(self.formal_config),
                "--case", "interconnect-P2P_intraserver",
                "--device-ids", "14,15",
                "--nproc-per-node", "2",
                "--case-config", str(self.smoke_config),
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
        self.assertTrue(plan["runtime_lock"]["image_manifest"]["validated"])
        self.assertFalse(plan["permissions"]["candidate_runtime"])

    def test_formal_plan_rejects_any_rank_count_other_than_two(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = base_run.main([
                "benchmark", "run",
                "--config", str(self.formal_config),
                "--case", "interconnect-P2P_intraserver",
                "--device-ids", "14,15",
                "--nproc-per-node", "1",
                "--case-config", str(self.smoke_config),
                "--dry-run",
            ])
        self.assertEqual(code, 2)
        self.assertIn("requires exactly 2 local ranks", stderr.getvalue())

    def test_formal_plan_rejects_unbounded_default_case_config(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = base_run.main([
                "benchmark", "run",
                "--config", str(self.formal_config),
                "--case", "interconnect-P2P_intraserver",
                "--device-ids", "14,15",
                "--nproc-per-node", "2",
                "--dry-run",
            ])
        self.assertEqual(code, 2)
        self.assertIn("explicit bounded --case-config", stderr.getvalue())

    def test_formal_plan_rejects_modified_smoke_config(self):
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
                    "--config", str(self.formal_config),
                    "--case", "interconnect-P2P_intraserver",
                    "--device-ids", "14,15",
                    "--nproc-per-node", "2",
                    "--case-config", str(modified),
                    "--dry-run",
                ])
        self.assertEqual(code, 2)
        self.assertIn("bounded configuration allowlist", stderr.getvalue())

    def test_formal_runtime_is_validated_without_candidate_permission(self):
        lock = runtime_lock_record("torch_fl_2.10_flagcx")
        image_id = lock["image_manifest"]["image_id"]
        self.assertTrue(image_id.startswith("sha256:"))
        config = {
            "image": lock["image_manifest"]["image"],
            "runtime_profile": "torch_fl_2.10_flagcx",
        }
        validated = validate_runtime_identity(
            config, {"Id": image_id},
        )
        self.assertTrue(validated["image_manifest"]["validated"])
        self.assertEqual(
            validated["image_manifest"]["release_stage"], "validated",
        )

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

    def test_formal_environment_includes_hccl_initialization_control(self):
        config = json.loads(self.formal_config.read_text(encoding="utf-8"))
        self.assertEqual(config["runtime_environment"], {
            "FLAGCX_TORCH_BACKEND": "flagos",
            "HCCL_WHITELIST_DISABLE": "1",
        })

    def test_p2p_calibration_configs_are_pinned_but_not_formal_results(self):
        calibrator = load_calibrator_module()
        plan = calibrator.load_and_validate_plan(calibrator.DEFAULT_PLAN)
        matrix = calibrator.build_matrix(
            plan, None, master_port=29921, timeout=300,
        )
        self.assertFalse(plan["formal_baseline_eligible"])
        self.assertEqual(
            {item["expected_path"] for item in matrix}, {"SIO", "HCCS_SW"},
        )
        self.assertEqual(len(matrix), 8)
        self.assertEqual(
            sorted({item["message_bytes"] for item in matrix}),
            [4194304, 16777216, 67108864, 268435456],
        )
        requirements = json.loads((
            BENCHMARKS_DIR / "interconnect-P2P_intraserver" / "ascend" /
            "runtime_requirements.json"
        ).read_text(encoding="utf-8"))
        allowed = {
            item["sha256"]: item.get("scope")
            for item in requirements["allowed_case_configs"]
        }
        for point in plan["size_points"]:
            self.assertEqual(allowed[point["sha256"]], "p2p-calibration-only")

    def test_p2p_calibration_defaults_to_nonexecuting_plan(self):
        calibrator = load_calibrator_module()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = calibrator.main([])
        output = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(output["mode"], "plan-only")
        self.assertFalse(output["formal_baseline_eligible"])
        self.assertEqual(output["run_count"], 8)
        self.assertTrue(all(
            "--allow-candidate-runtime" not in item["command"]
            and "--allow-privileged-root" in item["command"]
            and item["command"][item["command"].index("--monitor") + 1] == "off"
            for item in output["matrix"]
        ))

    def test_p2p_calibration_rejects_config_hash_drift(self):
        calibrator = load_calibrator_module()
        plan = json.loads(calibrator.DEFAULT_PLAN.read_text(encoding="utf-8"))
        plan["size_points"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(
                calibrator.CalibrationError, "config hash drifted",
            ):
                calibrator.load_and_validate_plan(path)

    def test_p2p_qualification_plan_is_pinned_and_nonexecuting_by_default(self):
        formal = load_formal_runner_module()
        plan = formal.load_and_validate_plan(formal.DEFAULT_PLAN)
        self.assertEqual(plan["_protocol_id"], "p2p-single-node-v1")
        self.assertEqual(plan["status"], "qualified")
        self.assertEqual(set(formal.PROTOCOLS), {"p2p-single-node-v1"})
        tasks = formal.build_tasks(
            plan, master_port=30021, result_root=BASE_DIR / "result",
        )
        self.assertEqual(len(tasks), 30)
        self.assertTrue(all(
            task["phase"] == "size-curve" and task["monitor"] == "off"
            for task in tasks[:24]
        ))
        self.assertEqual(
            [task["monitor"] for task in tasks[24:]],
            ["off", "on", "on", "off", "off", "on"],
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = formal.main([])
        output = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(output["mode"], "plan-only")
        self.assertEqual(output["run_count"], 30)
        self.assertEqual(output["estimated_total_measurement_minutes"], 30)

    def test_p2p_qualification_preserves_matrix_and_pinned_window(self):
        formal = load_formal_runner_module()
        plan = formal.load_and_validate_plan(QUALIFICATION_PLAN_PATH)
        tasks = formal.build_tasks(
            plan, master_port=30121, result_root=BASE_DIR / "result",
        )
        self.assertEqual(plan["_protocol_id"], "p2p-single-node-v1")
        self.assertEqual(
            plan["measurement_contract"]["minimum_measurement_seconds_per_run"],
            45,
        )
        self.assertEqual(
            plan["measurement_contract"]["target_measurement_seconds_per_run"],
            60,
        )
        self.assertEqual(len(tasks), 30)
        self.assertEqual(len(plan["matrix"]), 8)
        self.assertEqual(
            [task["monitor"] for task in tasks[24:]],
            ["off", "on", "on", "off", "off", "on"],
        )

        requirements = json.loads((
            BENCHMARKS_DIR / "interconnect-P2P_intraserver" / "ascend" /
            "runtime_requirements.json"
        ).read_text(encoding="utf-8"))
        allowed = {
            item["sha256"]: item.get("scope")
            for item in requirements["allowed_case_configs"]
        }
        for item in plan["matrix"]:
            self.assertEqual(allowed[item["sha256"]], "p2p-qualification")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = formal.main(["--plan", str(QUALIFICATION_PLAN_PATH)])
        output = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(output["protocol_id"], "p2p-single-node-v1")
        self.assertEqual(output["mode"], "plan-only")
        self.assertEqual(output["run_count"], 30)
        self.assertEqual(output["estimated_total_measurement_minutes"], 30)
        self.assertTrue(all(
            "--allow-candidate-runtime" not in item["command"]
            and "--allow-privileged-root" in item["command"]
            for item in output["tasks"]
        ))

    def test_p2p_qualification_rejects_window_below_protocol_minimum(self):
        formal = load_formal_runner_module()
        plan = json.loads(QUALIFICATION_PLAN_PATH.read_text(encoding="utf-8"))
        plan["measurement_contract"]["minimum_measurement_seconds_per_run"] = 44
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "compact-plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(
                formal.common.CalibrationError, "measurement contract drifted",
            ):
                formal.load_and_validate_plan(path)

    def test_p2p_qualification_rejects_scope_and_threshold_changes(self):
        formal = load_formal_runner_module()
        for field in ("matrix", "acceptance", "monitor_ab"):
            plan = json.loads(QUALIFICATION_PLAN_PATH.read_text(encoding="utf-8"))
            if field == "matrix":
                plan[field][0]["device_ids"] = [0, 1]
            elif field == "acceptance":
                plan[field]["maximum_cv_pct_per_topology_size_cell"] = 100.0
            else:
                plan[field]["message_bytes"] = 4194304
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "plan.json"
                path.write_text(json.dumps(plan), encoding="utf-8")
                with self.assertRaisesRegex(formal.common.CalibrationError, "drifted"):
                    formal.load_and_validate_plan(path)

    def test_p2p_qualification_rejects_missing_distributable_evidence(self):
        formal = load_formal_runner_module()
        plan = json.loads(QUALIFICATION_PLAN_PATH.read_text(encoding="utf-8"))
        plan["calibration_evidence"]["path"] = "base/vendors/ascend/missing-evidence.json"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(formal.common.CalibrationError, "calibration evidence"):
                formal.load_and_validate_plan(path)

    def test_p2p_resume_preserves_failure_and_reaudits_completed_runs(self):
        formal = load_formal_runner_module()
        plan = formal.load_and_validate_plan(QUALIFICATION_PLAN_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary)
            tasks = formal.build_tasks(
                plan, master_port=30121, result_root=result_root,
            )
            session = result_root / "flagcx-p2p-qualification-test"
            session.mkdir()
            snapshot = session / QUALIFICATION_PLAN_PATH.name
            snapshot.write_bytes(QUALIFICATION_PLAN_PATH.read_bytes())

            def record_for(task, returncode, audit=None):
                record = {
                    "phase": task["phase"],
                    "monitor": task["monitor"],
                    "pair_id": task["cell"]["pair_id"],
                    "message_bytes": task["cell"]["message_bytes"],
                    "repeat": task.get("repeat"),
                    "sequence": task.get("sequence"),
                    "command": task["command"],
                    "returncode": returncode,
                    "child_result": str(session / "child"),
                }
                if audit is not None:
                    record["audit"] = audit
                return record

            audit = {"runtime_image_id": "sha256:candidate"}
            summary = {
                "schema_version": 1,
                "kind": "flagcx-p2p-qualification",
                "protocol_id": "p2p-single-node-v1",
                "status": "failed",
                "error": "formal child execution failed",
                "formal_execution_eligible": True,
                "production_eligible": True,
                "plan": {
                    "path": snapshot.name,
                    "sha256": plan["_sha256"],
                },
                "runs": [
                    record_for(tasks[0], 0, audit),
                    record_for(tasks[1], 137),
                ],
            }
            formal.common.write_json(session / "summary.json", summary)
            with mock.patch.object(formal, "audit_child", return_value=audit):
                _, resumed, start_index, runtime_id = formal.prepare_resumed_session(
                    plan, tasks, result_root, session,
                )
            self.assertEqual(start_index, 1)
            self.assertEqual(runtime_id, "sha256:candidate")
            self.assertEqual(resumed["status"], "running")
            self.assertNotIn("error", resumed)
            self.assertEqual(len(resumed["runs"]), 1)
            self.assertEqual(resumed["failed_attempts"][0]["returncode"], 137)
            self.assertEqual(resumed["failed_attempts"][0]["task_index"], 2)

    def test_p2p_qualification_aggregate_enforces_variance_and_monitor_delta(self):
        formal = load_formal_runner_module()
        plan = formal.load_and_validate_plan(formal.DEFAULT_PLAN)
        runs = []
        for item in plan["matrix"]:
            for repeat in range(1, 4):
                runs.append({
                    "phase": "size-curve", "monitor": "off",
                    "pair_id": item["pair_id"],
                    "message_bytes": item["message_bytes"],
                    "repeat": repeat, "audit": {"rank_mean_gb_s": 100.0},
                })
        for sequence, monitor in enumerate(plan["monitor_ab"]["order"], start=1):
            runs.append({
                "phase": "monitor-ab", "monitor": monitor,
                "pair_id": plan["monitor_ab"]["pair_id"],
                "message_bytes": plan["monitor_ab"]["message_bytes"],
                "sequence": sequence,
                "audit": {"rank_mean_gb_s": 102.0 if monitor == "on" else 100.0},
            })
        self.assertEqual(formal.aggregate(plan, runs)["status"], "passed")
        runs[2]["audit"]["rank_mean_gb_s"] = 120.0
        self.assertEqual(formal.aggregate(plan, runs)["status"], "failed")

    def test_p2p_topology_parser_uses_explicit_matrix_direction(self):
        calibrator = load_calibrator_module()
        topology = """\
        Phy-ID0    X          SIO        HCCS_SW
        Phy-ID1    SIO        X          HCCS_SW
        Phy-ID2    HCCS_SW    HCCS_SW    X
        """
        self.assertEqual(calibrator.topology_path(topology, 0, 1), "SIO")
        self.assertEqual(calibrator.topology_path(topology, 0, 2), "HCCS_SW")
        with self.assertRaises(calibrator.CalibrationError):
            calibrator.topology_path(topology, 3, 0)


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
            master_port=29821, gate="timeout-cleanup",
        )
        self.assertEqual(command[-1], "exec sleep 600")

        qualifier.validate_expected_timeout_record({
            "returncode": 124,
            "timed_out": True,
            "timeout_cleanup": {
                "remove_returncode": 0,
                "inspect_returncode": 1,
                "absence_confirmed": True,
            },
        })
        with self.assertRaisesRegex(RuntimeError, "absence is unproven"):
            qualifier.validate_expected_timeout_record({
                "returncode": 124,
                "timed_out": True,
                "timeout_cleanup": {
                    "remove_returncode": 0,
                    "inspect_returncode": 0,
                    "absence_confirmed": False,
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

    def test_peer_failure_gate_requires_both_ranks_and_fail_fast(self):
        qualifier = load_qualifier_module()
        records = [{
            "schema_version": 1,
            "kind": "flagcx-fault-injection",
            "fault": "peer-exit",
            "rank": 0,
            "world_size": 2,
            "public_backend": "flagos",
            "inner_backend": "ProcessGroupFlagCX",
            "role": "peer-waiter",
            "peer": 1,
        }, {
            "schema_version": 1,
            "kind": "flagcx-fault-injection",
            "fault": "peer-exit",
            "rank": 1,
            "world_size": 2,
            "public_backend": "flagos",
            "inner_backend": "ProcessGroupFlagCX",
            "role": "injected-exit",
            "injected_exit_code": 42,
        }]
        qualifier.validate_peer_failure_record({
            "returncode": 1, "timed_out": False,
            "container_postcondition": {
                "inspect_returncode": 1, "absence_confirmed": True,
            },
        }, records)
        with self.assertRaisesRegex(RuntimeError, "before the host deadline"):
            qualifier.validate_peer_failure_record({
                "returncode": 124, "timed_out": True,
                "container_postcondition": {
                    "inspect_returncode": 1, "absence_confirmed": True,
                },
            }, records)
        with self.assertRaisesRegex(RuntimeError, "absence is unproven"):
            qualifier.validate_peer_failure_record({
                "returncode": 1, "timed_out": False,
                "container_postcondition": {
                    "inspect_returncode": 0, "absence_confirmed": False,
                },
            }, records)

    def test_hung_p2p_gate_requires_real_waiters_and_cleanup(self):
        qualifier = load_qualifier_module()
        records = [{
            "schema_version": 1,
            "kind": "flagcx-fault-injection",
            "fault": "hung-p2p",
            "rank": rank,
            "world_size": 2,
            "public_backend": "flagos",
            "inner_backend": "ProcessGroupFlagCX",
            "role": "recv-waiter",
            "peer": 1 - rank,
            "observed_at": "2026-09-04T06:32:10Z",
        } for rank in (0, 1)]
        qualifier.validate_hung_p2p_record({
            "returncode": 124,
            "timed_out": True,
            "timeout_observed_at": "2026-09-04T06:32:20Z",
            "timeout_cleanup": {
                "remove_returncode": 0,
                "inspect_returncode": 1,
                "absence_confirmed": True,
            },
        }, records)
        records[1]["peer"] = 1
        with self.assertRaisesRegex(RuntimeError, "wait evidence drifted"):
            qualifier.validate_hung_p2p_record({
                "returncode": 124,
                "timed_out": True,
                "timeout_observed_at": "2026-09-04T06:32:20Z",
                "timeout_cleanup": {
                    "remove_returncode": 0,
                    "inspect_returncode": 1,
                    "absence_confirmed": True,
                },
            }, records)

    def test_hung_p2p_gate_requires_a_real_observation_interval(self):
        qualifier = load_qualifier_module()
        records = [{
            "schema_version": 1,
            "kind": "flagcx-fault-injection",
            "fault": "hung-p2p",
            "rank": rank,
            "world_size": 2,
            "public_backend": "flagos",
            "inner_backend": "ProcessGroupFlagCX",
            "role": "recv-waiter",
            "peer": 1 - rank,
            "observed_at": "2026-09-04T06:32:18Z",
        } for rank in (0, 1)]
        with self.assertRaisesRegex(RuntimeError, "minimum interval"):
            qualifier.validate_hung_p2p_record({
                "returncode": 124,
                "timed_out": True,
                "timeout_observed_at": "2026-09-04T06:32:20Z",
                "timeout_cleanup": {
                    "remove_returncode": 0,
                    "inspect_returncode": 1,
                    "absence_confirmed": True,
                },
            }, records)

    def test_container_absence_rejects_daemon_failures(self):
        qualifier = load_qualifier_module()
        self.assertTrue(qualifier.docker_absence_confirmed(
            1, "", "Error: No such object: exact-name",
        ))
        self.assertFalse(qualifier.docker_absence_confirmed(
            1, "", "Cannot connect to the Docker daemon",
        ))

    def test_fault_gate_commands_mount_the_snapshot(self):
        qualifier = load_qualifier_module()
        config = {
            "image": "candidate:locked",
            "shm_size": "1g",
            "runtime_environment": {},
            "required_devices": [],
            "host_mounts": [],
        }
        probe = Path("/evidence/verify_flagcx_faults.py")
        for gate, fault in (
            ("peer-failure", "peer-exit"),
            ("hung-p2p", "hung-p2p"),
        ):
            command = qualifier.build_container_command(
                config, (14, 15), container_name="flagcx-fault-test",
                master_port=29821, gate=gate, fault_probe_path=probe,
            )
            self.assertIn(
                f"{probe}:/opt/flagrt/verify_flagcx_faults.py:ro", command,
            )
            self.assertTrue(command[-1].endswith(f"--fault {fault}"))


if __name__ == "__main__":
    unittest.main()
