"""Regression tests for prelaunch skips, OOM recovery and AllReduce semantics."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from argparse import Namespace

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tests"))
import run
from executors.benchmark import validate_resolved_process_scope
from executors.common import ConfigurationError
from test_ascend_transfer_benchmarks import load_case_module


class FailureFixTests(unittest.TestCase):
    def test_unsupported_cases_skip_before_docker_or_devices(self):
        for case in ("computation-FP64", "computation-FP8", "computation-TF32:A3",
                     "interconnect-MPI_interserver", "interconnect-P2P_interserver"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root, \
                    mock.patch("executors.benchmark.docker_inspect") as docker, \
                    mock.patch("executors.benchmark.run_host_preflight") as preflight, \
                    mock.patch("executors.benchmark.DeviceLease") as lease, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                rc = run.main(["benchmark", "run", "--case", case, "--npu-ids", "2",
                               "--result-root", root])
                self.assertEqual(rc, 0)
                self.assertIn("SKIPPED", output.getvalue())
                docker.assert_not_called()
                preflight.assert_not_called()
                lease.assert_not_called()
                summary_path = next(Path(root).glob("*/summary.json"))
                summary = json.loads(summary_path.read_text())
                self.assertEqual(summary["status"], "skipped")
                self.assertTrue(summary["skip_reason"])
                self.assertNotIn("runtime", summary)
                report = (summary_path.parent / "report.md").read_text()
                self.assertIn("启动前跳过", report)

    def test_allreduce_rejects_operator_runtime_before_launch(self):
        with mock.patch("executors.benchmark.docker_inspect") as docker, \
                contextlib.redirect_stderr(io.StringIO()):
            rc = run.main(["benchmark", "run", "--case", "interconnect-MPI_intraserver",
                           "--npu-ids", "2,3"])
        self.assertEqual(rc, 2)
        docker.assert_not_called()

    def test_allreduce_requires_multiple_ranks_but_allows_four(self):
        req = {"requirements": {"process_scope": {"nnodes": 1, "min_nproc_per_node": 2}}}
        with self.assertRaises(ConfigurationError):
            validate_resolved_process_scope("allreduce", req, nnodes=1, nproc=1)
        validate_resolved_process_scope("allreduce", req, nnodes=1, nproc=4)

    def test_capacity_recovers_from_locked_allocator_error(self):
        module = load_case_module("main_memory-capacity")
        failure = RuntimeError("CachingDeviceAllocator: failed to allocate 2097152 bytes on device 0")
        with mock.patch.object(module.torch, "empty", side_effect=[failure, object(), failure]) as allocate, \
                mock.patch.object(module, "accelerator_device", return_value="cpu"):
            result = module.main(Namespace(vendor="ascend"),
                                 Namespace(INITSIZE=2, POST_TEST_WAIT_SECONDS=0), 0, 1, 0)
        self.assertEqual(result, 1)
        self.assertEqual([call.args[0] for call in allocate.call_args_list], [524288, 262144, 262144])

    def test_capacity_does_not_swallow_device_errors_or_report_zero_as_success(self):
        module = load_case_module("main_memory-capacity")
        for message in ("device context lost", "CachingDeviceAllocator: failed to allocate 4 bytes on device 0"):
            with self.subTest(message=message), \
                    mock.patch.object(module.torch, "empty", side_effect=RuntimeError(message)), \
                    mock.patch.object(module, "accelerator_device", return_value="cpu"):
                with self.assertRaisesRegex(RuntimeError, message):
                    module.main(Namespace(vendor="ascend"),
                                Namespace(INITSIZE=1, POST_TEST_WAIT_SECONDS=0), 0, 1, 0)

    def test_capacity_hint_bounds_attempts_without_reporting_queried_capacity(self):
        module = load_case_module("main_memory-capacity")
        failure = RuntimeError("CachingDeviceAllocator: failed to allocate 1048576 bytes on device 0")
        with mock.patch.object(module.ascend_driver, "capacity_request_mib", side_effect=[3, 1]), \
                mock.patch.object(module.torch, "empty", side_effect=[object(), failure]) as allocate, \
                mock.patch.object(module, "accelerator_device", return_value="cpu"):
            result = module.main(Namespace(vendor="ascend"),
                                 Namespace(INITSIZE=65536, POST_TEST_WAIT_SECONDS=0,
                                           BOUND_REQUEST_BY_FREE_MEMORY=True), 0, 1, 0)
        self.assertEqual(result, 3)
        self.assertEqual([call.args[0] for call in allocate.call_args_list], [786432, 262144])

    def test_hbm_query_failures_propagate_and_zero_free_still_attempts_one_mib(self):
        module = load_case_module("main_memory-capacity")
        driver = module.ascend_driver
        for rc, free, total, expected in ((0, 0, 8192, 1), (0, 4 * 1048576, 8 * 1048576, 2),
                                          (1, 0, 0, None), (0, 8192, 4096, None)):
            def query(attr, free_ptr, total_ptr):
                self.assertEqual(attr, 1)
                free_ptr._obj.value = free
                total_ptr._obj.value = total
                return rc
            runtime = mock.Mock()
            runtime.aclrtGetMemInfo.side_effect = query
            with self.subTest(rc=rc, free=free), mock.patch.object(driver.ctypes, "CDLL", return_value=runtime):
                if expected is None:
                    with self.assertRaises(RuntimeError):
                        driver.capacity_request_mib(65536)
                else:
                    self.assertEqual(driver.capacity_request_mib(65536), expected)

    def test_allreduce_checks_nonzero_sum_and_keeps_timed_values_finite(self):
        module = load_case_module("interconnect-MPI_intraserver")
        seen = []
        def reduce(tensor, op):
            seen.append(tensor.flatten()[0].item())
            tensor.mul_(2)
            if len(seen) == 1:
                tensor.fill_(3)  # ranks 0 and 1 hold 1 and 2
        with mock.patch.object(module, "accelerator_device", return_value="cpu"), \
                mock.patch.object(module, "host_device_sync"), \
                mock.patch.object(module, "multi_device_sync"), \
                mock.patch.object(module.dist, "all_reduce", side_effect=reduce):
            module.main(Namespace(vendor="ascend", node_size=2),
                        Namespace(Melements=1, WARMUP=2, ITERS=140), 0, 2, 0)
        self.assertEqual(seen, [1] + [0] * 142)

    def test_allreduce_rejects_wrong_nonzero_sum(self):
        module = load_case_module("interconnect-MPI_intraserver")
        with mock.patch.object(module, "accelerator_device", return_value="cpu"), \
                mock.patch.object(module, "host_device_sync"), \
                mock.patch.object(module.dist, "all_reduce"):
            with self.assertRaisesRegex(RuntimeError, "correctness failed"):
                module.main(Namespace(vendor="ascend", node_size=2),
                            Namespace(Melements=1, WARMUP=2, ITERS=2), 0, 2, 0)


if __name__ == "__main__":
    unittest.main()
