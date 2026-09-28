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

"""Minimal Torch-FL adapter for the existing FlagPerf Base cases."""

import importlib
import re
import ctypes

import torch


_INITIALIZED = False


def is_allocation_failure(error):
    """Recognize OOM plus the locked Torch-FL allocator's untyped failure.

    The compatibility message does not retain the ACL error code. Log the raw
    error at the call site; never classify unrelated RuntimeErrors as OOM.
    """
    return (
        isinstance(error, torch.OutOfMemoryError)
        or "out of memory" in str(error).lower()
        or re.fullmatch(
            r"CachingDeviceAllocator: failed to allocate \d+ bytes on device \d+",
            str(error).strip(),
        ) is not None
    )


def capacity_request_mib(request_mib):
    """Bound a real allocation attempt using current-device free HBM.

    This is only a search hint, never the reported capacity. The caller keeps
    successful tensors alive and still attempts 1 MiB when the hint is zero.
    """
    runtime = ctypes.CDLL("libascendcl.so")
    query = runtime.aclrtGetMemInfo
    query.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
                      ctypes.POINTER(ctypes.c_size_t)]
    query.restype = ctypes.c_int
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    # ACL_HBM_MEM = 1 in the locked CANN 9 acl_rt.h.
    rc = query(1, ctypes.byref(free), ctypes.byref(total))
    if rc != 0:
        raise RuntimeError(f"aclrtGetMemInfo failed: {rc}")
    if total.value == 0 or free.value > total.value:
        raise RuntimeError("aclrtGetMemInfo returned invalid HBM sizes")
    # Free HBM is not necessarily one allocatable block. Taking at most half
    # avoids a slow near-total request while converging to the same 1 MiB tail.
    return max(1, min(request_mib, free.value // (2 * 1024 * 1024)))


def initialize():
    """Register Torch-FL's PrivateUse1 backend and fail early if unavailable."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    try:
        importlib.import_module("torch_fl")
    except ImportError as exc:
        raise RuntimeError(
            "Ascend Base benchmarks require Torch-FL in the runtime image"
        ) from exc

    if not hasattr(torch, "flagos"):
        raise RuntimeError(
            "Torch-FL was imported but torch.flagos is not registered"
        )
    _INITIALIZED = True


def device(local_rank):
    """Return the Torch-FL device addressed by torchrun's local rank."""
    initialize()
    torch.flagos.set_device(local_rank)
    return torch.device("flagos:{}".format(local_rank))


def synchronize():
    """Wait for queued work on the current Torch-FL device."""
    initialize()
    torch.flagos.synchronize()
