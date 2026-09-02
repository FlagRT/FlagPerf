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
from typing import Any, Iterable


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HOST_CONFIG = BASE_DIR / "configs" / "ascend910_cann9_local.yaml"
DEFAULT_LEASE_ROOT = Path("/tmp/flagperf-base-device-leases")
DEFAULT_RUNTIME_PROFILE = "torch_fl_2.10"
RUNTIME_PROFILE_RE = re.compile(r"[A-Za-z0-9_.-]+")


class ConfigurationError(RuntimeError):
    """A request cannot form a safe execution plan."""


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


def runtime_artifact_paths(runtime_profile: str | None = None) -> tuple[Path, Path]:
    """Resolve one repository-owned runtime profile without path traversal."""
    profile = runtime_profile or DEFAULT_RUNTIME_PROFILE
    if not isinstance(profile, str) or not RUNTIME_PROFILE_RE.fullmatch(profile):
        raise ConfigurationError(f"invalid Ascend runtime profile: {profile!r}")
    root = BASE_DIR / "vendors" / "ascend" / profile
    return root / "stack.lock.yaml", root / "image-manifest.json"


def runtime_lock_record(
    runtime_profile: str | None = None,
) -> dict[str, Any]:
    stack_lock, image_manifest = runtime_artifact_paths(runtime_profile)
    if not stack_lock.is_file() or not image_manifest.is_file():
        raise ConfigurationError(
            "Ascend runtime lock or image manifest is missing under "
            f"{stack_lock.parent}"
        )
    manifest = json.loads(image_manifest.read_text(encoding="utf-8"))
    return {
        "runtime_profile": runtime_profile or DEFAULT_RUNTIME_PROFILE,
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
    record = runtime_lock_record(config.get("runtime_profile"))
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
            "communication gates and explicitly pass --allow-candidate-runtime "
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
        or any(not isinstance(item, int) or item < 0 for item in expected)
        or len(expected) != len(set(expected))
    ):
        raise ConfigurationError(
            "expected_device_ids must be a non-empty unique integer list"
        )
    runtime_profile = value.get("runtime_profile", DEFAULT_RUNTIME_PROFILE)
    runtime_artifact_paths(runtime_profile)
    runtime_environment = value.get("runtime_environment", {})
    if not isinstance(runtime_environment, dict):
        raise ConfigurationError("runtime_environment must be a JSON object")
    allowed_runtime_environment = {
        "FLAGCX_TORCH_BACKEND",
        "HCCL_WHITELIST_DISABLE",
    }
    unknown_environment = sorted(
        set(runtime_environment) - allowed_runtime_environment
    )
    if unknown_environment:
        raise ConfigurationError(
            "runtime_environment contains unsupported keys: "
            f"{unknown_environment}"
        )
    if runtime_environment.get("FLAGCX_TORCH_BACKEND") not in (None, "flagos"):
        raise ConfigurationError(
            "FLAGCX_TORCH_BACKEND must be exactly 'flagos' for this runtime"
        )
    if runtime_environment.get("HCCL_WHITELIST_DISABLE") not in (None, "1"):
        raise ConfigurationError(
            "HCCL_WHITELIST_DISABLE must be exactly '1' when configured"
        )
    return resolved, value


def parse_id_spec(value: str, label: str) -> tuple[int, ...]:
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
    return tuple(sorted(resolved))


@dataclass(frozen=True)
class BaseRunContext:
    config: Path = DEFAULT_HOST_CONFIG
    npu_ids: str | None = None
    device_ids: str | None = None
    result_root: Path | None = None
    timeout: int = 3600
    dry_run: bool = False

    def validate(self, *, require_selection: bool) -> None:
        if self.npu_ids and self.device_ids:
            raise ConfigurationError(
                "--npu-ids and --device-ids are mutually exclusive"
            )
        if require_selection and not (self.npu_ids or self.device_ids):
            raise ConfigurationError(
                "an explicit --npu-ids or --device-ids selection is required"
            )
        if self.npu_ids:
            parse_id_spec(self.npu_ids, "physical NPU IDs")
        if self.device_ids:
            parse_id_spec(self.device_ids, "logical Device IDs")
        if self.timeout <= 0:
            raise ConfigurationError("--timeout must be positive")

    def selection_request(self) -> dict[str, Any]:
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
    """Non-blocking per-logical-device host lease backed by ``flock``."""

    def __init__(
        self,
        device_ids: Iterable[int],
        *,
        run_id: str,
        kind: str,
        root: Path = DEFAULT_LEASE_ROOT,
    ) -> None:
        self.device_ids = tuple(sorted(set(device_ids)))
        if not self.device_ids:
            raise DeviceLeaseError("cannot acquire an empty device lease")
        self.run_id = run_id
        self.kind = kind
        self.root = root
        self._streams: list[Any] = []
        self.acquired_at: str | None = None

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            for device_id in self.device_ids:
                path = self.root / f"logical-device-{device_id}.lock"
                stream = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    stream.seek(0)
                    owner = stream.read().strip() or "unknown owner"
                    stream.close()
                    raise DeviceLeaseError(
                        f"logical Device {device_id} is already leased: {owner}"
                    ) from exc
                self._streams.append(stream)
            self.acquired_at = utc_now()
            metadata = json.dumps(
                {
                    "run_id": self.run_id,
                    "kind": self.kind,
                    "pid": os.getpid(),
                    "device_ids": list(self.device_ids),
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
        except Exception:
            self.release()
            raise

    def record(self) -> dict[str, Any]:
        return {
            "backend": "flock",
            "root": str(self.root),
            "device_ids": list(self.device_ids),
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
