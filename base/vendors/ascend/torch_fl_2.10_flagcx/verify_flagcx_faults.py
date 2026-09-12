#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Two-rank FlagCX fault probe for peer-exit and hung-P2P lifecycle gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import sys

import torch_fl  # noqa: F401 - Torch-FL must own PrivateUse1 before consumers
import flagcx  # noqa: F401 - registers the FlagCX backend for flagos
import torch
import torch.distributed as dist

from verify_flagcx_p2p import resolve_flagcx_backend


INJECTED_EXIT_CODE = 42


def emit(record: dict[str, object]) -> None:
    print(json.dumps(
        record | {
            "observed_at": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
        },
        sort_keys=True,
    ), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fault", choices=("peer-exit", "hung-p2p"), required=True,
    )
    args = parser.parse_args()

    if os.environ.get("FLAGCX_TORCH_BACKEND") != "flagos":
        raise RuntimeError("FLAGCX_TORCH_BACKEND must be exactly 'flagos'")
    if os.environ.get("HCCL_WHITELIST_DISABLE") != "1":
        raise RuntimeError("HCCL_WHITELIST_DISABLE must be exactly '1'")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.flagos.set_device(local_rank)
    device = torch.device(f"flagos:{local_rank}")
    dist.init_process_group("flagos", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(
            f"FlagCX fault probe requires exactly 2 ranks, got {world_size}"
        )
    public_name, inner_name = resolve_flagcx_backend(device)
    common = {
        "schema_version": 1,
        "kind": "flagcx-fault-injection",
        "fault": args.fault,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "public_backend": "flagos",
        "public_process_group": public_name,
        "inner_backend": inner_name,
    }

    # Ensure both ranks have initialized the real communication backend before
    # either process enters the intentionally terminal fault path.
    dist.barrier()
    if args.fault == "peer-exit":
        if rank == 1:
            emit(common | {
                "status": "injected",
                "phase": "before-peer-exit",
                "role": "injected-exit",
                "injected_exit_code": INJECTED_EXIT_CODE,
            })
            os._exit(INJECTED_EXIT_CODE)
        emit(common | {
            "status": "waiting",
            "phase": "before-blocking-recv",
            "role": "peer-waiter",
            "peer": 1,
        })
        inbound = torch.empty(1, dtype=torch.float32, device=device)
        dist.recv(inbound, src=1)
        raise RuntimeError("peer-exit receive unexpectedly completed")

    peer = 1 - rank
    emit(common | {
        "status": "waiting",
        "phase": "before-blocking-recv",
        "role": "recv-waiter",
        "peer": peer,
    })
    # Both ranks issue recv first, deliberately creating a real unmatched P2P
    # wait. The host qualification runner owns the deadline and cleanup proof.
    inbound = torch.empty(1, dtype=torch.float32, device=device)
    dist.recv(inbound, src=peer)
    raise RuntimeError("hung-P2P receive unexpectedly completed")


if __name__ == "__main__":
    sys.exit(main())
