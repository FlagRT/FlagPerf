# FlagCX capability probe notes (2026-09-23)

Unified-entry smoke `comm-smoke-allreduce-auto042727-56` (cards 5,6) failed inside
the bounded container with:

    NET/Socket : no ipv4/ipv6 net card found, expected 1
    net_socket_init failed / get_unique_id: init_root failed
    Undefined: flagcxComm is not fully initialized.

while Gloo completed the two-rank rendezvous over loopback and the XCCL runtime
itself loaded (xccl version 0792b03 [rdma], build Feb 9 2026). Root cause: the
container policy is `--network none`, and the BKCL socket bootstrap enumerates
network interfaces but does not accept loopback by default.

Device-less diagnostic (same image, same `--network none`, no XPU attached, not a
Base measurement): with `BKCL_SOCKET_IFNAME=lo` the socket warnings disappear and
init proceeds until `flagcxUnhandledDeviceError`, which is expected in a
device-less container. Fix adopted: both P800 intraserver entrypoints set
`BKCL_SOCKET_IFNAME=lo` before `init_process_group`; official verdict runs are the
unified-entry smokes in this directory.

Diagnostic command (host):

    sudo -n docker run --rm --network none \
      -v /tmp/flagcx-probe.py:/tmp/flagcx-probe.py:ro -e BKCL_SOCKET_IFNAME=lo \
      flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608 \
      /root/miniconda/envs/python310_torch29_cuda/bin/python \
      -m torch.distributed.run --nproc-per-node=2 --master-port=29531 /tmp/flagcx-probe.py
