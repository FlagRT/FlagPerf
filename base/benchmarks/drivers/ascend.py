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

import torch


_INITIALIZED = False


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
