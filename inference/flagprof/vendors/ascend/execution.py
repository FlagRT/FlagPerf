# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""NPU worker setup, imported only inside an executing worker."""


def prepare_components():
    # Component registration uses torch.npu even before a runner selects a device.
    import torch_npu


def initialize(rank):
    import torch
    import torch_npu
    device = 'npu:' + str(rank)
    torch.npu.set_device(device)
    return device


def allocator_memory(device):
    import torch
    return {'allocated_bytes': torch.npu.memory_allocated(device),
            'reserved_bytes': torch.npu.memory_reserved(device)}


def physical_identity(device):
    import ctypes as C
    import torch
    library = C.CDLL('libascendcl.so')
    function = library.aclrtGetPhyDevIdByLogicDevId
    function.argtypes = [C.c_int, C.POINTER(C.c_int)]
    function.restype = C.c_int
    logical = torch.device(device).index
    logical = torch.npu.current_device() if logical is None else logical
    physical = C.c_int()
    result = function(logical, C.byref(physical))
    if result:
        raise RuntimeError('aclrtGetPhyDevIdByLogicDevId: '+str(result))
    return dict(entity='chip:'+str(physical.value), phyid=physical.value)
