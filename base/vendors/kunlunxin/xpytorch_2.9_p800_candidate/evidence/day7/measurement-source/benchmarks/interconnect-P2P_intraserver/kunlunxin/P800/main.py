# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""P800 two-rank in-server P2P send/recv over the resolved FlagCX backend."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
sys.path.append("..")

import torch.distributed as dist


def parse_args():
    parser = ArgumentParser(description="P800 interconnect-P2P_intraserver")
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
    from drivers.day6 import run_p2p
    case_config = Namespace(**load_case_config(Path.cwd(), config.vendor))
    dist.init_process_group(backend=case_config.DIST_BACKEND)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = rank % config.node_size
    run_p2p(kunlunxin, vars(case_config), rank, world_size, local_rank)
    dist.destroy_process_group()
