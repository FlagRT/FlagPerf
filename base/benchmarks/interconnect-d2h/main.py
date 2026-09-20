# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import sys
import time
from argparse import ArgumentParser, Namespace

import torch
import torch.distributed as dist
import yaml

sys.path.append("..")
from drivers.utils import *


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument(
        "--vendor", type=str, required=True, help="vendor name like ascend"
    )
    parser.add_argument(
        "--node_size", type=int, required=True, help="for pytorch"
    )

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


def main(config, case_config, rank, world_size, local_rank):
    if rank == 0:
        print("finish initialization")

    device = accelerator_device(config.vendor, local_rank)
    Melements = case_config.Melements
    torchsize = (Melements, 1024, 1024)
    non_blocking = getattr(case_config, "NON_BLOCKING", False)
    pin_memory = getattr(case_config, "PIN_MEMORY", False)

    tensor = torch.rand(torchsize, dtype=torch.float32).to(device)
    destination = torch.empty(torchsize, dtype=torch.float32)
    if pin_memory:
        destination = destination.pin_memory()

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start warmup")
        print(
            "D2H transfer semantics: host_memory={}, api=copy_, "
            "non_blocking={}".format(
                "pinned" if pin_memory else "pageable", non_blocking
            )
        )

    for _ in range(case_config.WARMUP):
        destination.copy_(tensor, non_blocking=non_blocking)

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    measurement_event = benchmark_measurement_start()
    start_time = time.perf_counter()

    for _ in range(case_config.ITERS):
        destination.copy_(tensor, non_blocking=non_blocking)

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    end_time = time.perf_counter()
    benchmark_measurement_finish(measurement_event)

    elapsed_time = end_time - start_time
    datasize = case_config.ITERS * (
        Melements * 1024 * 1024 * 4 / 1E9
    )
    bandwidth = datasize / elapsed_time
    bandwidth_gib = bandwidth * 1E9 / (1024**3)

    return round(bandwidth, 2), round(bandwidth_gib, 2)


if __name__ == "__main__":
    config = parse_args()
    bootstrap_vendor(config.vendor)
    from pathlib import Path
    from case_assets import load_case_config
    case_config = load_case_config(Path.cwd(), config.vendor)
    case_config = Namespace(**case_config)

    dist.init_process_group(backend=case_config.DIST_BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % config.node_size

    gb, gib = main(config, case_config, rank, world_size, local_rank)

    multi_device_sync(config.vendor)
    for output_rank in range(config.node_size):
        if local_rank == output_rank:
            print(
                r"[FlagPerf Result]Rank {}'s d2h-bandwidth=".format(
                    dist.get_rank()
                )
                + str(gb)
                + "GB/s"
            )
            print(
                r"[FlagPerf Result]Rank {}'s d2h-bandwidth=".format(
                    dist.get_rank()
                )
                + str(gib)
                + "GiB/s"
            )
        multi_device_sync(config.vendor)

    dist.destroy_process_group()
