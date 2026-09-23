# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""P800 two-rank in-server AllReduce over the resolved FlagCX backend."""
from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
import sys
sys.path.append("..")

# The bounded container runs with --network none, so loopback is the only
# interface.  The BKCL/FlagCX socket bootstrap reads this at native-runtime
# load, which happens during the first torch import below, so it must be set
# before any torch import in this module.
os.environ.setdefault("BKCL_SOCKET_IFNAME", "lo")

import torch.distributed as dist


def parse_args():
    parser = ArgumentParser(description="P800 interconnect-MPI_intraserver")
    parser.add_argument("--vendor", type=str, required=True)
    parser.add_argument("--node_size", type=int, required=True)
    args, _unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    from drivers.utils import bootstrap_vendor
    bootstrap_vendor(config.vendor)
    from case_assets import load_case_config
    from drivers import kunlunxin
    from drivers.day6 import run_allreduce
    case_config = Namespace(**load_case_config(Path.cwd(), config.vendor))
    dist.init_process_group(backend=case_config.DIST_BACKEND)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = rank % config.node_size
    run_allreduce(kunlunxin, vars(case_config), rank, world_size, local_rank)
    dist.destroy_process_group()
