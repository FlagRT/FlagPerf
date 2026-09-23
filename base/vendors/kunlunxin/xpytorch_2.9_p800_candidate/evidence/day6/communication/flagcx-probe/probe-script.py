import os
print("rank-env BKCL_SOCKET_IFNAME=", os.environ.get("BKCL_SOCKET_IFNAME"), flush=True)
import torch
import torch.distributed as dist
try:
    dist.init_process_group(backend="cpu:gloo,cuda:flagcx")
    print("INIT-OK rank", dist.get_rank(), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    print("DONE rank", dist.get_rank(), flush=True)
except Exception as exc:
    print("INIT-FAIL rank", dist.get_rank(), type(exc).__name__, str(exc)[:300], flush=True)
    raise SystemExit(1)
