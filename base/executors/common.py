# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Shared, side-effect-free planning and host resource helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from base.vendors.protocol import ConfigurationError
from base.vendors.registry import get_provider


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HOST_CONFIG = BASE_DIR / "configs" / "ascend910_cann9_local.yaml"
DEFAULT_LEASE_ROOT = Path("/tmp/flagperf-base-device-leases")
DEFAULT_RUNTIME_PROFILE = "torch_fl_2.10"
RUNTIME_PROFILE_RE = re.compile(r"[A-Za-z0-9_.-]+")


class DeviceLeaseError(RuntimeError):
    """One or more requested logical devices are already leased."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_artifact_paths(runtime_profile: str | None = None, *, vendor: str = "ascend") -> tuple[Path, Path]:
    root = get_provider(vendor).runtime_root(BASE_DIR, runtime_profile)
    paths = root / "stack.lock.yaml", root / "image-manifest.json"
    if any(not path.resolve().is_relative_to(root) for path in paths):
        raise ConfigurationError("runtime asset escapes locked profile")
    return paths


def runtime_lock_record(
    runtime_profile: str | None = None, *, vendor: str = "ascend",
) -> dict[str, Any]:
    stack_lock, image_manifest = runtime_artifact_paths(runtime_profile, vendor=vendor)
    if not stack_lock.is_file() or not image_manifest.is_file():
        raise ConfigurationError(
            "runtime lock or image manifest is missing under "
            f"{stack_lock.parent}"
        )
    manifest = json.loads(image_manifest.read_text(encoding="utf-8"))
    return {
        "vendor": vendor,
        "runtime_profile": runtime_profile or get_provider(vendor).default_runtime_profile,
        "stack_lock": {
            "path": str(stack_lock.resolve()),
            "sha256": sha256_file(stack_lock),
        },
        "image_manifest": {
            "path": str(image_manifest.resolve()),
            "sha256": sha256_file(image_manifest),
            "schema_version": manifest.get("schema_version"),
            "image": manifest.get("image"),
            "image_id": manifest.get("image_id"),
            "validated": manifest.get("validated"),
            "release_stage": manifest.get("release_stage"),
            "validation_status": manifest.get("validation_status"),
            "validation_scope": manifest.get("validation_scope"),
        },
    }


def validate_runtime_identity(
    config: dict[str, Any], image_info: dict[str, Any], *,
    allow_candidate: bool = False,
) -> dict[str, Any]:
    record = runtime_lock_record(config.get("runtime_profile"), vendor=config.get("vendor", "ascend"))
    expected = record["image_manifest"]
    if config.get("image") != expected.get("image"):
        raise ConfigurationError(
            "host profile image does not match the locked image manifest: "
            f"config={config.get('image')}, locked={expected.get('image')}"
        )
    if not isinstance(expected.get("image_id"), str) or not expected["image_id"]:
        raise ConfigurationError(
            "runtime candidate has no built image ID in its image manifest"
        )
    if image_info.get("Id") != expected.get("image_id"):
        raise ConfigurationError(
            "local Docker image ID does not match the validated image manifest: "
            f"actual={image_info.get('Id')}, locked={expected.get('image_id')}"
        )
    if expected.get("validated") is not True and not allow_candidate:
        raise ConfigurationError(
            "runtime image is still an unvalidated candidate; run the bounded "
            "runtime qualification gates and explicitly pass --allow-candidate-runtime "
            "only for authorized candidate testing"
        )
    return record


def load_host_config(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"host config does not exist: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"host config is not valid JSON: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError("host config must be a JSON object")
    if value.get("schema_version") != 1:
        raise ConfigurationError(
            f"unsupported host config schema: {value.get('schema_version')}"
        )
    required = {
        "image", "vendor", "shm_size", "expected_device_ids", "result_root",
        "required_devices", "host_mounts",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ConfigurationError(f"host config is missing required fields: {missing}")
    expected = value["expected_device_ids"]
    if (
        not isinstance(expected, list)
        or not expected
        or any(type(item) is not int or item < 0 for item in expected)
        or len(expected) != len(set(expected))
    ):
        raise ConfigurationError(
            "expected_device_ids must be a non-empty unique integer list"
        )
    provider = get_provider(value["vendor"])
    runtime_artifact_paths(value.get("runtime_profile"), vendor=provider.name)
    runtime_environment = value.get("runtime_environment", {})
    if not isinstance(runtime_environment, dict):
        raise ConfigurationError("runtime_environment must be a JSON object")
    for key in ("required_devices", "host_mounts"):
        if not isinstance(value[key], list) or any(not isinstance(item, str) or not item.startswith("/") for item in value[key]):
            raise ConfigurationError(f"{key} must be a list of absolute paths")
    provider.validate_config(value)
    return resolved, value


def parse_id_spec(value: str, label: str, *, preserve_order: bool = False) -> tuple[int, ...]:
    """Parse comma-separated IDs and inclusive ranges without hiding mistakes."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a non-empty ID/range expression")
    resolved: list[int] = []
    seen: set[int] = set()
    for raw_token in value.split(","):
        token = raw_token.strip()
        if re.fullmatch(r"\d+", token):
            values = [int(token)]
        elif match := re.fullmatch(r"(\d+)-(\d+)", token):
            start, end = map(int, match.groups())
            if start > end:
                raise ConfigurationError(f"{label} range is descending: {token}")
            values = list(range(start, end + 1))
        else:
            raise ConfigurationError(
                f"{label} contains an invalid token: {token or '<empty>'}"
            )
        duplicate = next((item for item in values if item in seen), None)
        if duplicate is not None:
            raise ConfigurationError(f"{label} contains duplicate ID {duplicate}")
        resolved.extend(values)
        seen.update(values)
    return tuple(resolved if preserve_order else sorted(resolved))


@dataclass(frozen=True)
class BaseRunContext:
    config: Path = DEFAULT_HOST_CONFIG
    npu_ids: str | None = None
    device_ids: str | None = None
    physical_device_ids: str | None = None
    result_root: Path | None = None
    timeout: int = 3600
    dry_run: bool = False

    def validate(self, *, require_selection: bool) -> None:
        if sum(item is not None for item in (self.npu_ids, self.device_ids, self.physical_device_ids)) > 1:
            raise ConfigurationError(
                "--physical-device-ids, --npu-ids and --device-ids are mutually exclusive"
            )
        if require_selection and not (self.npu_ids or self.device_ids or self.physical_device_ids):
            raise ConfigurationError(
                "an explicit --npu-ids or --device-ids selection is required (or --physical-device-ids)"
            )
        if self.physical_device_ids is not None:
            parse_id_spec(self.physical_device_ids, "physical device IDs", preserve_order=True)
        if self.npu_ids is not None:
            parse_id_spec(self.npu_ids, "physical NPU IDs")
        if self.device_ids:
            parse_id_spec(self.device_ids, "logical Device IDs")
        if self.timeout <= 0:
            raise ConfigurationError("--timeout must be positive")

    def selection_request(self) -> dict[str, Any]:
        if self.physical_device_ids is not None:
            return {"source": "physical-device-ids", "expression": self.physical_device_ids,
                    "requested_ids": list(parse_id_spec(self.physical_device_ids, "physical device IDs", preserve_order=True)),
                    "resolution_status": "deferred-until-host-preflight"}
        if self.npu_ids:
            return {
                "source": "npu-ids",
                "expression": self.npu_ids,
                "requested_ids": list(parse_id_spec(self.npu_ids, "physical NPU IDs")),
                "resolution_status": "deferred-until-host-preflight",
            }
        if self.device_ids:
            values = list(parse_id_spec(self.device_ids, "logical Device IDs"))
            return {
                "source": "device-ids",
                "expression": self.device_ids,
                "requested_ids": values,
                "selected_device_ids": values,
                "resolution_status": "static",
            }
        return {
            "source": "default-inventory",
            "resolution_status": "deferred-until-host-preflight",
        }

    def resolved_result_root(self, config: dict[str, Any]) -> Path:
        if self.result_root is not None:
            return self.result_root.expanduser().resolve()
        configured = Path(config["result_root"])
        if configured.is_absolute():
            return configured.resolve()
        return (BASE_DIR / configured).resolve()


def context_record(context: BaseRunContext) -> dict[str, Any]:
    value = asdict(context)
    value["config"] = str(context.config)
    value["result_root"] = (
        str(context.result_root) if context.result_root is not None else None
    )
    return value


class DeviceLease:
    """Stable resource locks plus optional legacy logical locks, backed by flock."""

    def __init__(
        self,
        device_ids: Iterable[int],
        *,
        run_id: str,
        kind: str,
        root: Path = DEFAULT_LEASE_ROOT,
        resource_keys: Iterable[str] = (),
        compatibility_paths: Iterable[Path] = (),
    ) -> None:
        self.device_ids = tuple(sorted(set(device_ids)))
        self.resource_keys = tuple(sorted(set(resource_keys)))
        self.compatibility_paths = tuple(sorted(set(Path(p) for p in compatibility_paths)))
        if any(not p.is_absolute() for p in self.compatibility_paths):
            raise DeviceLeaseError("compatibility lock paths must be absolute")
        if any(not isinstance(key, str) or not key or "/" not in key for key in self.resource_keys):
            raise DeviceLeaseError("resource keys require vendor/physical-resource identity")
        if not self.device_ids and not self.resource_keys:
            raise DeviceLeaseError("cannot acquire an empty device lease")
        self.run_id = run_id
        self.kind = kind
        self.root = root
        self._streams: list[Any] = []
        self.acquired_at: str | None = None

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            names = [f"logical-device-{item}.lock" for item in self.device_ids]
            names += ["resource-" + hashlib.sha256(key.encode()).hexdigest() + ".lock" for key in self.resource_keys]
            paths = sorted(set(self.root / name for name in names) | set(self.compatibility_paths))
            for path in paths:
                name = str(path)
                fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666)
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    os.close(fd)
                    raise DeviceLeaseError("lease must be a regular file with one link")
                stream = os.fdopen(fd, "r+", encoding="utf-8")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    stream.seek(0)
                    owner = stream.read().strip() or "unknown owner"
                    stream.close()
                    raise DeviceLeaseError(
                        f"resource {name} is already leased: {owner}"
                    ) from exc
                self._streams.append(stream)
            self.acquired_at = utc_now()
            metadata = json.dumps(
                {
                    "run_id": self.run_id,
                    "kind": self.kind,
                    "pid": os.getpid(),
                    "device_ids": list(self.device_ids),
                    "resource_keys": list(self.resource_keys),
                    "acquired_at": self.acquired_at,
                },
                sort_keys=True,
            )
            for stream in self._streams:
                stream.seek(0)
                stream.truncate()
                stream.write(metadata + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self.release()
            raise

    def record(self) -> dict[str, Any]:
        return {
            "backend": "flock",
            "root": str(self.root),
            "compatibility_paths": [str(p) for p in self.compatibility_paths],
            "device_ids": list(self.device_ids),
                    "resource_keys": list(self.resource_keys),
            "run_id": self.run_id,
            "pid": os.getpid(),
            "acquired_at": self.acquired_at,
        }

    def release(self) -> None:
        for stream in reversed(self._streams):
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()
        self._streams.clear()

    def __enter__(self) -> "DeviceLease":
        self.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()
