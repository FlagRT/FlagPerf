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

"""Ascend INT8 Base benchmark using scaled MatMul with INT32 accumulation."""

import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch_fl  # noqa: F401 - registers torch.flagos before vendor imports
import torch
import torch.distributed as dist
import yaml
from flag_gems.ops.scaled_mm import scaled_mm


CASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = CASE_DIR.parent
sys.path.insert(0, str(BENCHMARKS_DIR))

from drivers.utils import (  # noqa: E402
    accelerator_device,
    benchmark_measurement_finish,
    benchmark_measurement_start,
    bootstrap_vendor,
    host_device_sync,
    multi_device_sync,
)


OUTPUT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args():
    parser = ArgumentParser(description="Ascend scaled INT8 matrix multiply")
    parser.add_argument("--vendor", type=str, required=True)
    parser.add_argument("--node_size", type=int, required=True)
    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


def load_case_config():
    from case_assets import load_case_config as resolve_config
    return Namespace(**resolve_config(CASE_DIR, "ascend"))


def resolve_output_dtype(name):
    try:
        return OUTPUT_DTYPES[name.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "Ascend INT8 OUTPUT_DTYPE must be one of: "
            + ", ".join(sorted(OUTPUT_DTYPES))
        ) from exc


def main(config, case_config, rank, world_size, local_rank):
    if rank == 0:
        print("finish initialization")

    m = case_config.M
    n = case_config.N
    k = case_config.K
    device = accelerator_device(config.vendor, local_rank)
    matrix_a = torch.ones((m, k), dtype=torch.int8, device=device)
    matrix_b = torch.ones((k, n), dtype=torch.int8, device=device)
    scale_a = torch.full(
        (), case_config.SCALE_A, dtype=torch.float32, device=device
    )
    scale_b = torch.full(
        (), case_config.SCALE_B, dtype=torch.float32, device=device
    )
    output_dtype = resolve_output_dtype(case_config.OUTPUT_DTYPE)

    if rank == 0:
        print(
            "[FlagPerf Config] Ascend INT8 kernel=flag_gems.scaled_mm "
            "accumulation=int32 output_dtype={} scale_a={} scale_b={}".format(
                case_config.OUTPUT_DTYPE,
                case_config.SCALE_A,
                case_config.SCALE_B,
            )
        )

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start warmup")

    for _ in range(case_config.WARMUP):
        _result = scaled_mm(
            matrix_a,
            matrix_b,
            scale_a,
            scale_b,
            out_dtype=output_dtype,
        )

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start test")

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    measurement_event = benchmark_measurement_start()
    start_time = time.perf_counter()

    for _ in range(case_config.ITERS):
        _result = scaled_mm(
            matrix_a,
            matrix_b,
            scale_a,
            scale_b,
            out_dtype=output_dtype,
        )

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    end_time = time.perf_counter()
    benchmark_measurement_finish(measurement_event)

    operations = case_config.ITERS * 2 * m * n * k
    return round(operations / (end_time - start_time) / 1e12, 2)


if __name__ == "__main__":
    config = parse_args()
    bootstrap_vendor(config.vendor)
    case_config = load_case_config()

    dist.init_process_group(backend=case_config.DIST_BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % config.node_size

    result = main(config, case_config, rank, world_size, local_rank)

    multi_device_sync(config.vendor)
    for output_rank in range(config.node_size):
        if local_rank == output_rank:
            print(
                r"[FlagPerf Result]Rank {}'s computation-INT8=".format(
                    dist.get_rank()
                )
                + str(result)
                + "TOPS"
            )
        multi_device_sync(config.vendor)

    dist.destroy_process_group()
