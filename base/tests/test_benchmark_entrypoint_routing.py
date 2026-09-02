# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark_worker.py"
SPEC = importlib.util.spec_from_file_location("base_benchmark_worker", MODULE_PATH)
BENCHMARK_WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK_WORKER)


class BenchmarkEntrypointRoutingTest(unittest.TestCase):
    def test_direct_vendor_entrypoint_is_preferred(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            vendor_dir = base / "benchmarks" / "computation-INT8" / "ascend"
            vendor_dir.mkdir(parents=True)
            (vendor_dir / "case_config.yaml").touch()
            (vendor_dir / "main.py").touch()

            _, _, entrypoint, selector = (
                BENCHMARK_WORKER.resolve_benchmark_entrypoint(
                    temp_dir, "computation-INT8", "ascend"
                )
            )

        self.assertEqual(entrypoint, str(vendor_dir / "main.py"))
        self.assertEqual(selector, "ascend")

    def test_generic_entrypoint_keeps_default_a100_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            case_dir = base / "benchmarks" / "computation-FP32"
            (case_dir / "nvidia" / "A100").mkdir(parents=True)
            (case_dir / "main.py").touch()

            _, _, entrypoint, selector = (
                BENCHMARK_WORKER.resolve_benchmark_entrypoint(
                    temp_dir, "computation-FP32", "nvidia"
                )
            )

        self.assertEqual(entrypoint, str(case_dir / "main.py"))
        self.assertEqual(selector, "nvidia/A100")

    def test_direct_vendor_config_can_use_generic_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            case_dir = base / "benchmarks" / "computation-FP16"
            vendor_dir = case_dir / "ascend"
            vendor_dir.mkdir(parents=True)
            (vendor_dir / "case_config.yaml").touch()
            (case_dir / "main.py").touch()

            _, _, entrypoint, selector = (
                BENCHMARK_WORKER.resolve_benchmark_entrypoint(
                    temp_dir, "computation-FP16", "ascend"
                )
            )

        self.assertEqual(entrypoint, str(case_dir / "main.py"))
        self.assertEqual(selector, "ascend")

    def test_current_ascend_transfer_cases_use_generic_entrypoint(self):
        base_dir = Path(__file__).resolve().parents[1]

        for case_name in (
            "main_memory-bandwidth",
            "interconnect-h2d",
            "interconnect-d2h",
        ):
            with self.subTest(case_name=case_name):
                _, case_dir, entrypoint, selector = (
                    BENCHMARK_WORKER.resolve_benchmark_entrypoint(
                        str(base_dir), case_name, "ascend"
                    )
                )

                self.assertEqual(
                    entrypoint, str(Path(case_dir) / "main.py")
                )
                self.assertEqual(selector, "ascend")


if __name__ == "__main__":
    unittest.main()
