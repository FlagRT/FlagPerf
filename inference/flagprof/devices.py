# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Device predicates shared by producers and offline consumers; no torch import."""


def device_types(dev):
    values = dev.values() if isinstance(dev, dict) else [dev] if dev is not None else []
    return {value.split(':', 1)[0] for value in values}


def accelerator(dev):
    return bool(device_types(dev) - {'cpu', 'meta'})
