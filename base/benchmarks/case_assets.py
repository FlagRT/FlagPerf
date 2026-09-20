# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""One side-effect-free asset resolver for host, worker and case configuration."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

# Direct case entrypoints historically put only benchmarks/ on sys.path.
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))
from base.vendors.protocol import ConfigurationError, checked_component


def file_record(path: Path, base: Path, *, override: bool = False) -> dict[str, Any]:
    path = path.resolve()
    if not override and not path.is_relative_to(base):
        raise ConfigurationError(f"case asset escapes repository: {path}")
    content = path.read_bytes()
    return {"path": str(path), "relative_path": "@override" if override else path.relative_to(base).as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def read_config(path: Path) -> dict[str, Any]:
    import yaml
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"invalid case configuration {path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigurationError(f"case configuration must be a string-keyed mapping: {path}")
    return value


def merge_requirements(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    result = dict(parent)
    for key, value in child.items():
        if key in parent and parent[key] != value:
            if isinstance(parent[key], dict) and isinstance(value, dict):
                value = merge_requirements(parent[key], value)
            elif key == "supported" and parent[key] is True and value is False:
                pass
            elif key == "min_nproc_per_node" and type(value) is int and type(parent[key]) is int and value >= parent[key]:
                pass
            else:
                raise ConfigurationError(f"conflicting runtime requirement: {key}")
        result[key] = value
    return result


def resolve_case_assets(base: Path, case_spec: str, vendor: str, override: Path | None = None,
                        *, require_contract: bool = True) -> dict[str, Any]:
    base = base.resolve()
    parts = case_spec.split(":")
    if len(parts) > 2:
        raise ConfigurationError("ambiguous case/chip selector")
    case = checked_component(parts[0], "case")
    vendor = checked_component(vendor, "vendor")
    chip = checked_component(parts[1], "chip") if len(parts) == 2 else None
    root = base / "benchmarks" / case
    if not root.resolve().is_relative_to(base / "benchmarks") or not root.is_dir():
        raise ConfigurationError(f"invalid case directory: {root}")
    vendor_dir = root / vendor
    if chip is None and not (vendor_dir / "case_config.yaml").is_file() and vendor == "nvidia":
        # Explicit compatibility for legacy standalone NVIDIA workers only.
        chip = "A100"
    layers = [("generic", root), (vendor, vendor_dir)]
    if chip is not None:
        layers.append(("chip", vendor_dir / chip))
    for _, directory in layers:
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ConfigurationError("case layer escapes case directory")
    configs, environments, contracts = {}, [], []
    merged, requirements = {}, {}
    entrypoint = None
    for label, directory in layers:
        for name in ("case_config.yaml", "main.py", "env.sh", "runtime_requirements.json"):
            candidate = directory / name
            if candidate.is_file() and not candidate.resolve().is_relative_to(root.resolve()):
                raise ConfigurationError(f"case asset escapes case directory: {candidate}")
        config = directory / "case_config.yaml"
        if config.is_file():
            configs[label] = file_record(config, base)
            merged.update(read_config(config))
        if (directory / "main.py").is_file():
            entrypoint = file_record(directory / "main.py", base)
        if (directory / "env.sh").is_file():
            environments.append(file_record(directory / "env.sh", base))
        path = directory / "runtime_requirements.json"
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ConfigurationError(f"invalid runtime requirements: {path}") from exc
            if not isinstance(value, dict) or value.get("schema_version") != 1:
                raise ConfigurationError(f"unsupported runtime requirements schema: {path}")
            if "supported" in value and type(value["supported"]) is not bool:
                raise ConfigurationError("supported must be a boolean")
            scope = value.get("process_scope", {})
            if not isinstance(scope, dict) or any(type(v) is not int or v <= 0 for v in scope.values()):
                raise ConfigurationError("process_scope must contain positive integers")
            requirements = merge_requirements(requirements, value)
            contracts.append(file_record(path, base))
    if require_contract and not contracts:
        raise ConfigurationError(f"runtime requirements missing for {vendor}/{case_spec}")
    if override is not None:
        override = override.expanduser().resolve()
        if not override.is_file():
            raise ConfigurationError(f"case override does not exist: {override}")
        configs["override"] = file_record(override, base, override=True)
        merged.update(read_config(override))
    if requirements.get("supported") is False:
        if not isinstance(requirements.get("unsupported_reason"), str) or not requirements["unsupported_reason"].strip():
            raise ConfigurationError("unsupported case requires a reason")
    elif entrypoint is None:
        raise ConfigurationError(f"Benchmark entrypoint does not exist: {case_spec}")
    if require_contract and requirements.get("supported") is not False and not configs:
        raise ConfigurationError("supported case has no configuration")
    return {"schema_version": 1, "case": case_spec, "case_name": case, "vendor": vendor, "chip": chip,
            "selector": vendor + ("/" + chip if chip else ""), "entrypoint": entrypoint,
            "configurations": configs, "merged_config": merged, "environments": environments,
            "requirement_files": contracts, "requirements": requirements}


def portable_assets(assets: dict[str, Any]) -> dict[str, Any]:
    """Only repository-relative paths cross the host/container boundary."""
    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key != "path"}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value
    return strip(assets)


def verify_worker_assets(base: Path, contract_path: Path, case_spec: str, vendor: str) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("case") != case_spec or contract.get("vendor") != vendor:
        raise ConfigurationError("host/worker case identity mismatch")
    override = contract_path.parent / "case-config" / "override.yaml" if "override" in contract["configurations"] else None
    actual = resolve_case_assets(base, case_spec, vendor, override)
    if portable_assets(actual) != contract:
        raise ConfigurationError("host/worker case assets or configuration hash changed")
    return actual


def load_case_config(case_dir: Path, selector: str) -> dict[str, Any]:
    """Used by actual case processes, including legacy direct invocations."""
    case_dir = case_dir.resolve()
    parts = selector.split("/")
    if len(parts) > 2:
        raise ConfigurationError("ambiguous vendor/chip selector")
    spec = case_dir.name + (":" + parts[1] if len(parts) == 2 else "")
    contract = os.environ.get("FLAGPERF_CASE_ASSETS")
    if contract:
        record = json.loads(Path(contract).read_text(encoding="utf-8"))
        if record.get("selector") != selector or record.get("case_name") != case_dir.name:
            raise ConfigurationError("case process selector differs from worker contract")
        assets = verify_worker_assets(case_dir.parent.parent, Path(contract), record["case"], parts[0])
    else:
        assets = resolve_case_assets(case_dir.parent.parent, spec, parts[0], require_contract=False)
    if assets["requirements"].get("supported") is False:
        raise ConfigurationError("unsupported case cannot execute")
    return assets["merged_config"]
