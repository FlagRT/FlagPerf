# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Pure host contracts. Runtime discovery is an explicit execution operation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Protocol, Sequence


class ConfigurationError(RuntimeError):
    """A request cannot form a safe execution plan."""


def checked_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value) or value in (".", ".."):
        raise ConfigurationError(f"invalid {label}: {value!r}")
    return value


@dataclass(frozen=True)
class DeviceBinding:
    vendor: str
    host_physical_id: int
    pci_bdf: str | None
    serial_or_uuid: str | None
    host_device_node: str
    container_device_node: str
    framework_local_rank: int
    framework_logical_id: int
    framework_device_name: str
    request_index: int
    resource_key: str
    legacy_logical_id: int | None = None

    def record(self) -> dict[str, Any]:
        return asdict(self)


class VendorProvider(Protocol):
    name: str
    display_name: str
    default_runtime_profile: str
    fallback_markers: tuple[str, ...]

    def runtime_root(self, base_dir: Path, profile: str | None) -> Path: ...
    def validate_config(self, config: dict[str, Any]) -> None: ...
    def validate_selection(self, context: Any) -> None: ...
    def preflight(self, result_dir: Path, config: dict[str, Any], context: Any, *, label: str = "host-preflight") -> Path: ...
    def bindings(self, preflight: dict[str, Any], selected: Sequence[int]) -> list[DeviceBinding]: ...
    def container_policy(self, config: dict[str, Any]) -> dict[str, Any]: ...
    def docker_args(self, config: dict[str, Any], bindings: Sequence[DeviceBinding]) -> list[str]: ...
    def bootstrap(self, base_dir: Path, config: dict[str, Any]) -> str: ...
    def worker_python(self, config: dict[str, Any]) -> str: ...
    def container_preflight(self, config: dict[str, Any], bindings: Sequence[DeviceBinding]) -> list[str]: ...
    def monitor_policy(self, enabled: bool) -> dict[str, Any]: ...
    def monitor_targets(self, preflight: dict[str, Any], selected: Sequence[int]) -> list[dict[str, Any]]: ...
    def create_monitor(self, targets: Sequence[dict[str, Any]]) -> Any: ...
    def rank_target(self, targets: Sequence[dict[str, Any]], selected: Sequence[int], local_rank: int, binding: DeviceBinding | None) -> dict[str, Any] | None: ...
    def measurement_identity(self, target: dict[str, Any], binding: DeviceBinding | None) -> dict[str, Any]: ...


def runtime_root(base_dir: Path, vendor: str, profile: str) -> Path:
    vendors = (base_dir / "vendors").resolve()
    parent = (vendors / checked_component(vendor, "vendor")).resolve()
    if not parent.is_relative_to(vendors) or parent == vendors:
        raise ConfigurationError("vendor root escapes runtime directory")
    root = (parent / checked_component(profile, "runtime profile")).resolve()
    if not root.is_relative_to(parent) or root == parent:
        raise ConfigurationError("runtime profile escapes vendor root")
    return root
