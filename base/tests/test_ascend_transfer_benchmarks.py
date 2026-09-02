# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = BASE_DIR / "benchmarks"
sys.path.insert(0, str(BENCHMARKS_DIR))


def load_case_module(case_name):
    module_path = BENCHMARKS_DIR / case_name / "main.py"
    spec = importlib.util.spec_from_file_location(
        "test_" + case_name.replace("-", "_"), module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self):
        self.copy_calls = []
        self.clone_count = 0
        self.pin_count = 0
        self.to_calls = []

    def to(self, device, non_blocking=False):
        self.to_calls.append((device, non_blocking))
        return self

    def pin_memory(self):
        self.pin_count += 1
        return self

    def copy_(self, source, non_blocking=False):
        self.copy_calls.append((source, non_blocking))
        return self

    def clone(self):
        self.clone_count += 1
        return self


class AscendTransferBenchmarkTest(unittest.TestCase):
    def test_d2d_preserves_clone_workload_and_uses_adapter_device(self):
        module = load_case_module("main_memory-bandwidth")
        tensor = FakeTensor()
        config = Namespace(vendor="ascend")
        case_config = Namespace(Melements=1, WARMUP=2, ITERS=3)

        with mock.patch.object(module.torch, "rand", return_value=tensor), \
                mock.patch.object(
                    module, "accelerator_device", return_value="flagos:4"
                ) as device, \
                mock.patch.object(module, "host_device_sync") as device_sync, \
                mock.patch.object(module, "multi_device_sync"), \
                mock.patch.object(
                    module.time, "perf_counter", side_effect=(10.0, 11.0)
                ):
            module.main(config, case_config, 0, 1, 4)

        device.assert_called_once_with("ascend", 4)
        self.assertEqual(tensor.to_calls, [("flagos:4", False)])
        self.assertEqual(tensor.clone_count, 5)
        self.assertEqual(device_sync.call_count, 3)

    def test_h2d_reuses_destination_and_pinned_source(self):
        module = load_case_module("interconnect-h2d")
        source = FakeTensor()
        destination = FakeTensor()
        config = Namespace(vendor="ascend")
        case_config = Namespace(
            Melements=1,
            WARMUP=2,
            ITERS=3,
            PIN_MEMORY=True,
            NON_BLOCKING=False,
            REUSE_DESTINATION=True,
        )

        with mock.patch.object(module.torch, "rand", return_value=source), \
                mock.patch.object(
                    module.torch, "empty", return_value=destination
                ) as empty, \
                mock.patch.object(
                    module, "accelerator_device", return_value="flagos:5"
                ), \
                mock.patch.object(module, "host_device_sync"), \
                mock.patch.object(module, "multi_device_sync"), \
                mock.patch.object(
                    module.time, "perf_counter", side_effect=(10.0, 11.0)
                ):
            module.main(config, case_config, 0, 1, 5)

        self.assertEqual(source.pin_count, 1)
        empty.assert_called_once_with(
            (1, 1024, 1024), dtype=module.torch.float32,
            device="flagos:5"
        )
        self.assertEqual(
            destination.copy_calls, [(source, False)] * 5
        )

    def test_d2h_reuses_pinned_destination(self):
        module = load_case_module("interconnect-d2h")
        source = FakeTensor()
        destination = FakeTensor()
        config = Namespace(vendor="ascend")
        case_config = Namespace(
            Melements=1,
            WARMUP=2,
            ITERS=3,
            PIN_MEMORY=True,
            NON_BLOCKING=False,
        )

        with mock.patch.object(module.torch, "rand", return_value=source), \
                mock.patch.object(
                    module.torch, "empty", return_value=destination
                ), \
                mock.patch.object(
                    module, "accelerator_device", return_value="flagos:6"
                ), \
                mock.patch.object(module, "host_device_sync"), \
                mock.patch.object(module, "multi_device_sync"), \
                mock.patch.object(
                    module.time, "perf_counter", side_effect=(10.0, 11.0)
                ):
            module.main(config, case_config, 0, 1, 6)

        self.assertEqual(source.to_calls, [("flagos:6", False)])
        self.assertEqual(destination.pin_count, 1)
        self.assertEqual(
            destination.copy_calls, [(source, False)] * 5
        )

    def test_ascend_configs_declare_blocking_pinned_copy_semantics(self):
        for case_name in ("interconnect-h2d", "interconnect-d2h"):
            with self.subTest(case_name=case_name):
                config_path = (
                    BENCHMARKS_DIR / case_name / "ascend" /
                    "case_config.yaml"
                )
                config = yaml.safe_load(config_path.read_text())
                self.assertEqual(config["DIST_BACKEND"], "gloo")
                self.assertIs(config["PIN_MEMORY"], True)
                self.assertIs(config["NON_BLOCKING"], False)


if __name__ == "__main__":
    unittest.main()
