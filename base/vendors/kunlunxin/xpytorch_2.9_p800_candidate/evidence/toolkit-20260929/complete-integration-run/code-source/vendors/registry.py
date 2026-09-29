# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Explicit host capability registration, not device autodetection."""
from base.vendors.protocol import ConfigurationError, VendorProvider
from base.vendors.ascend.provider import AscendProvider
from base.vendors.kunlunxin.provider import KunlunxinProvider

PROVIDERS = {"ascend": AscendProvider(), "kunlunxin": KunlunxinProvider()}


def get_provider(vendor: str) -> VendorProvider:
    if not isinstance(vendor, str) or vendor not in PROVIDERS:
        raise ConfigurationError(f"unregistered vendor {vendor!r}: host provider is not implemented")
    return PROVIDERS[vendor]
