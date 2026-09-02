#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Compare CANN ACL timeline and sync events for cross-stream waiting."""

from __future__ import annotations

import ctypes
import json

import torch_fl  # noqa: F401
import torch


def main() -> None:
    torch.flagos.set_device(0)
    acl = ctypes.CDLL("libascendcl.so")
    void_p = ctypes.c_void_p
    acl.aclrtCreateStream.argtypes = [ctypes.POINTER(void_p)]
    acl.aclrtCreateStream.restype = ctypes.c_int
    acl.aclrtDestroyStream.argtypes = [void_p]
    acl.aclrtDestroyStream.restype = ctypes.c_int
    acl.aclrtSynchronizeStream.argtypes = [void_p]
    acl.aclrtSynchronizeStream.restype = ctypes.c_int
    acl.aclrtCreateEventWithFlag.argtypes = [ctypes.POINTER(void_p), ctypes.c_uint]
    acl.aclrtCreateEventWithFlag.restype = ctypes.c_int
    acl.aclrtDestroyEvent.argtypes = [void_p]
    acl.aclrtDestroyEvent.restype = ctypes.c_int
    acl.aclrtRecordEvent.argtypes = [void_p, void_p]
    acl.aclrtRecordEvent.restype = ctypes.c_int
    acl.aclrtStreamWaitEvent.argtypes = [void_p, void_p]
    acl.aclrtStreamWaitEvent.restype = ctypes.c_int

    flagos = ctypes.CDLL(torch_fl.__path__[0] + "/lib/libflagos.so")
    flagos.GetCurrentStream.argtypes = [ctypes.c_int]
    flagos.GetCurrentStream.restype = void_p
    current_stream = flagos.GetCurrentStream(0)
    wait_stream = void_p()
    create_stream_rc = acl.aclrtCreateStream(ctypes.byref(wait_stream))
    results = []
    try:
        for name, flag in (("timeline", 0x8), ("sync", 0x1)):
            event = void_p()
            create_rc = acl.aclrtCreateEventWithFlag(ctypes.byref(event), flag)
            record_rc = wait_rc = synchronize_rc = destroy_rc = None
            if create_rc == 0:
                record_rc = acl.aclrtRecordEvent(event, current_stream)
                if record_rc == 0:
                    wait_rc = acl.aclrtStreamWaitEvent(wait_stream, event)
                    if wait_rc == 0:
                        synchronize_rc = acl.aclrtSynchronizeStream(wait_stream)
                destroy_rc = acl.aclrtDestroyEvent(event)
            results.append({
                "name": name,
                "flag": flag,
                "create_rc": create_rc,
                "record_rc": record_rc,
                "wait_rc": wait_rc,
                "synchronize_rc": synchronize_rc,
                "destroy_rc": destroy_rc,
            })
    finally:
        destroy_stream_rc = acl.aclrtDestroyStream(wait_stream)

    print(json.dumps({
        "schema_version": 1,
        "kind": "acl-event-flag-ab",
        "device": 0,
        "current_stream": int(current_stream or 0),
        "create_stream_rc": create_stream_rc,
        "destroy_stream_rc": destroy_stream_rc,
        "results": results,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
