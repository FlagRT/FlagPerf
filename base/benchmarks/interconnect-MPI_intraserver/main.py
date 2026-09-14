# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# cambricon mlu import
try:
    from torch_mlu.utils.model_transfer import transfer
except ImportError:
    pass

import torch
import torch.distributed as dist
import os
import time
from argparse import ArgumentParser, Namespace
import yaml
import sys
sys.path.append("..")
from drivers.utils import *


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")

    parser.add_argument("--node_size",
                        type=int,
                        required=True,
                        help="for pytorch")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


def main(config, case_config, rank, world_size, local_rank):    
    set_ieee_float32(config.vendor)
    if rank == 0:
        print("finish initialization")

    if "iluvatar" in config.vendor:
        torch.cuda.set_device(local_rank)
        
    Melements = case_config.Melements
    torchsize = (Melements, 1024, 1024)
    device = accelerator_device(config.vendor, local_rank)
    if config.vendor.split("/", 1)[0].lower() == "ascend":
        # Validate nonzero SUM once, outside warmup and measurement. All ranks
        # inspect the entire result; zeros keep repeated in-place SUM finite.
        tensor = torch.full(torchsize, float(rank + 1), dtype=torch.float32).to(device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        host_device_sync(config.vendor)
        expected = world_size * (world_size + 1) // 2
        if not torch.all(tensor.cpu() == expected).item():
            raise RuntimeError(f"AllReduce SUM correctness failed on rank {rank}")
        print(f"Rank {rank}: AllReduce SUM correctness passed; expected={expected}")
        tensor = torch.zeros(torchsize, dtype=torch.float32).to(device)
    else:
        tensor = torch.ones(torchsize, dtype=torch.float32).to(device)

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start warmup")
    
    for _ in range(case_config.WARMUP):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    
    measurement_event = benchmark_measurement_start()
    start_time = time.perf_counter()

    for _ in range(case_config.ITERS):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    end_time = time.perf_counter()
    benchmark_measurement_finish(measurement_event)
    if config.vendor.split("/", 1)[0].lower() == "ascend":
        if not torch.all(tensor.cpu() == 0).item():
            raise RuntimeError(f"AllReduce timed output corrupted on rank {rank}")

    elapsed_time = end_time - start_time

    '''
        algbw = S/t
    Considering that each rank has a bandwidth to the outside world of B, the time to perform an allReduce operation of S elements is at best :
        t = (S*2*(n-1)) / (n*B)
    Indeed, we have S elements, 2*(n-1) operations per element, and n links of bandwidth B to perform them. Reordering the equation, we find that
        t = (S/B) * (2*(n-1)/n)
    Therefore, to get an AllReduce bandwidth measurement which we can compare to the hardware peak bandwidth, we compute :
        B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)
    More details can be found in https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
    
    NVIDIA specifies the 600GBps for intra-server connect as a bidirectional bandwidth, 
    meaning each node can simultaneously upload and download at 300GBps. 
    To better reflect the ratio of the tested value to the specified value and 
    to align with common understanding of NVIDIA's product capabilities, 
    we have multiplied the bandwidth result here by two.
    '''
    datasize = case_config.ITERS * (Melements * 1024 * 1024 * 4 / 1E9)
    algbw = datasize / elapsed_time
    bandwidth = algbw * (2 * (config.node_size - 1) / config.node_size)
    bandwidth *= 2
    bandwidth_gib = bandwidth * 1E9 / (1024**3)

    return round(bandwidth, 2), round(bandwidth_gib, 2)


if __name__ == "__main__":    
    config = parse_args()
    bootstrap_vendor(config.vendor)
    with open("case_config.yaml", "r") as file:
        case_config = yaml.safe_load(file)
    with open(os.path.join(config.vendor, "case_config.yaml"), "r") as file:
        case_config_vendor = yaml.safe_load(file)
    case_config.update(case_config_vendor)
    case_config = Namespace(**case_config)

    if config.vendor.split("/", 1)[0].lower() == "ascend":
        if case_config.DIST_BACKEND != "flagos":
            raise ValueError("Ascend AllReduce requires the FlagCX flagos backend")
        if int(os.environ.get("WORLD_SIZE", "1")) < 2:
            raise ValueError("AllReduce requires at least two ranks")
        accelerator_device(config.vendor, int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend=case_config.DIST_BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % config.node_size

    gb, gib = main(config, case_config, rank, world_size, local_rank)

    multi_device_sync(config.vendor)
    for output_rank in range(config.node_size):
        if local_rank == output_rank:
            print(r"[FlagPerf Result]Rank {}'s interconnect-MPI_intraserver-bandwidth=".format(dist.get_rank()) + str(gb) + "GB/s")
            print(r"[FlagPerf Result]Rank {}'s interconnect-MPI_intraserver-bandwidth=".format(dist.get_rank()) + str(gib) + "GiB/s")
        multi_device_sync(config.vendor)

    dist.destroy_process_group()
