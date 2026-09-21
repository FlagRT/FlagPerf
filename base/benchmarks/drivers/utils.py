# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch

from . import ascend as ascend_driver
from .events import benchmark_measurement_finish, benchmark_measurement_start

# mthreads torch_musa import
try:
    import torch_musa
except ImportError:
    pass


def _vendor_name(vendor):
    name = vendor.split("/", 1)[0].lower()
    if name not in {"nvidia", "ascend", "kunlunxin", "mthreads", "cambricon", "iluvatar", "metax", "dcu", "tsingmicro"}:
        raise ValueError(f"unsupported runtime vendor: {name!r}")
    return name


def bootstrap_vendor(vendor):
    """Initialize only the runtime extension required by the selected vendor."""
    if _vendor_name(vendor) == "ascend":
        ascend_driver.initialize()
    elif _vendor_name(vendor) == "kunlunxin":
        from . import kunlunxin
        kunlunxin.initialize()


def accelerator_device(vendor, local_rank):
    """Resolve the explicit device for vendors that require an adapter."""
    if _vendor_name(vendor) == "ascend":
        return ascend_driver.device(local_rank)
    if _vendor_name(vendor) == "kunlunxin":
        from . import kunlunxin
        return kunlunxin.device(local_rank)
    return local_rank


def set_ieee_float32(vendor):
    _vendor_name(vendor)
    if _vendor_name(vendor) == "kunlunxin":
        from . import kunlunxin
        return kunlunxin.set_float32()
    if vendor == "nvidia":
        torch.backends.cuda.matmul.allow_tf32 = False
    elif _vendor_name(vendor) == "ascend":
        print("Ascend FP32 uses Torch-FL backend defaults; strict IEEE FP32 "
              "mode is not enforced")
    elif "cambricon" in vendor:
        torch.backends.mlu.matmul.allow_tf32 = False
        torch.backends.cnnl.allow_tf32 = False
    elif "mthreads" in vendor:
        torch.backends.mudnn.allow_tf32 = False
    else:
        print("unspecified vendor {}, do nothing".format(vendor))


def unset_ieee_float32(vendor):
    _vendor_name(vendor)
    if vendor == "nvidia":
        torch.backends.cuda.matmul.allow_tf32 = True
    elif _vendor_name(vendor) == "ascend":
        pass
    elif "cambricon" in vendor:
        torch.backends.mlu.matmul.allow_tf32 = True
        torch.backends.cnnl.allow_tf32 = True
    elif "mthreads" in vendor:
        torch.backends.mudnn.allow_tf32 = True
    else:
        print("unspecified vendor {}, do nothing".format(vendor))


def host_device_sync(vendor):
    if _vendor_name(vendor) == "kunlunxin":
        from . import kunlunxin
        return kunlunxin.synchronize()
    if _vendor_name(vendor) == "nvidia":
        torch.cuda.synchronize()
    elif _vendor_name(vendor) == "ascend":
        ascend_driver.synchronize()
    elif "mthreads" in vendor:
        torch.musa.synchronize()
    elif _vendor_name(vendor) == "cambricon":
        torch.mlu.synchronize()
    else:
        # Explicit legacy CUDA-compatible adapters, never an unknown vendor.
        torch.cuda.synchronize()


def multi_device_sync(vendor):
    if _vendor_name(vendor) == "kunlunxin":
        torch.distributed.barrier()
        return
    if _vendor_name(vendor) == "nvidia":
        torch.distributed.barrier()
    elif _vendor_name(vendor) == "ascend":
        torch.distributed.barrier()
    elif "mthreads" in vendor:
        torch.distributed.barrier()
    else:
        print("unspecified vendor {}, using default pytorch \"torch.distributed.barrier\"".format(vendor))
        torch.distributed.barrier()
        
