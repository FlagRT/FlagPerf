# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Ascend host policy; importing this module is device-free."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from base.vendors.protocol import ConfigurationError, DeviceBinding, runtime_root


class AscendProvider:
    name = "ascend"
    display_name = "Ascend"
    default_runtime_profile = "torch_fl_2.10"
    fallback_markers = ("[flagos cpu_fallback]",)

    def runtime_root(self, base_dir: Path, profile: str | None) -> Path:
        return runtime_root(base_dir, self.name, profile or self.default_runtime_profile)

    def validate_config(self, config: dict[str, Any]) -> None:
        environment = config.get("runtime_environment", {})
        allowed = {"FLAGCX_TORCH_BACKEND": "flagos", "HCCL_WHITELIST_DISABLE": "1"}
        unknown = sorted(set(environment) - allowed.keys())
        if unknown:
            raise ConfigurationError(f"runtime_environment contains unsupported keys: {unknown}")
        for key, value in environment.items():
            if value != allowed[key]:
                raise ConfigurationError(f"{key} must be exactly {allowed[key]!r} for this runtime")

    def validate_selection(self, context: Any) -> None:
        context.validate(require_selection=True)

    def preflight(self, result_dir: Path, config: dict[str, Any], context: Any, *, label: str = "host-preflight") -> Path:
        raw_path = run_host_preflight(result_dir, config["expected_device_ids"],
                                     npu_ids=context.physical_device_ids or context.npu_ids,
                                     device_ids=context.device_ids, label=label)
        if context.physical_device_ids is None:
            return raw_path
        # Preserve the legacy preflight evidence and normalize only the new
        # selector's order. One physical NPU can expand into multiple ranks.
        record = json.loads(raw_path.read_text(encoding="utf-8"))
        requested = context.selection_request()["requested_ids"]
        selected = [item["logic_id"] for physical in requested
                    for item in sorted(record["actual_device_map"], key=lambda item: item["logic_id"])
                    if item["npu_id"] == physical]
        if len(selected) != len(set(selected)) or set(selected) != set(record["selection"]["selected_device_ids"]):
            raise RuntimeError("physical selection differs from verified Ascend preflight")
        record["selection"].update(source="physical-device-ids", requested_ids=requested,
                                   selected_npu_ids=requested, selected_device_ids=selected)
        record["raw_preflight"] = str(raw_path.relative_to(result_dir))
        path = result_dir / f"{label}-physical-selection.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return path

    def bindings(self, preflight: dict[str, Any], selected: Sequence[int]) -> list[DeviceBinding]:
        targets = self.monitor_targets(preflight, selected)
        by_id = {item["logic_id"]: item for item in targets}
        inventory = {item["logic_id"]: item for item in preflight["actual_device_map"]}
        selection = preflight.get("selection", {})
        physical_request = selection.get("source") in ("physical-device-ids", "npu-ids")
        requested = selection.get("requested_ids", [])
        result = []
        for rank, logical in enumerate(selected):
            item = by_id[logical]
            raw = inventory[logical]
            result.append(DeviceBinding(
                vendor=self.name, host_physical_id=item["npu_id"],
                pci_bdf=raw.get("pci_bdf") or raw.get("pci_bus_id"),
                serial_or_uuid=raw.get("uuid") or raw.get("serial"),
                host_device_node=f"/dev/davinci{logical}",
                container_device_node=f"/dev/davinci{logical}",
                framework_local_rank=rank, framework_logical_id=rank,
                framework_device_name=f"flagos:{rank}",
                request_index=requested.index(item["npu_id"]) if physical_request else rank,
                resource_key=f"ascend/npu-{item['npu_id']}/chip-{item['chip_id']}",
                legacy_logical_id=logical,
            ))
        return result

    def container_policy(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"network": "host", "ipc_namespace": "host", "pid_namespace": "private"}

    def docker_args(self, config: dict[str, Any], bindings: Sequence[DeviceBinding]) -> list[str]:
        selected = [item.legacy_logical_id for item in bindings]
        args = ["--network=host", "--ipc=host",
                "-e", "ASCEND_RT_VISIBLE_DEVICES=" + ",".join(map(str, selected)),
                "-e", "GEMS_VENDOR=ascend", "-e", "TRITON_ENABLE_TASKQUEUE=false",
                "-e", "FLAGOS_LOG_FALLBACK=1"]
        for item in bindings:
            args.append(f"--device={item.host_device_node}:{item.container_device_node}:rwm")
        for device in config["required_devices"]:
            args.append(f"--device={device}:{device}:rwm")
        for mount in config["host_mounts"]:
            if Path(mount).exists():
                args.extend(["-v", f"{mount}:{mount}:ro"])
        for key, value in sorted(config.get("runtime_environment", {}).items()):
            args.extend(["-e", f"{key}={value}"])
        return args

    def bootstrap(self, base_dir: Path, config: dict[str, Any]) -> str:
        return ""

    def worker_python(self, config: dict[str, Any]) -> str:
        return "python3"

    def container_preflight(self, config: dict[str, Any], bindings: Sequence[DeviceBinding]) -> list[str]:
        # Preserve existing Ascend execution; P800 will provide its UUID gate.
        return []

    def monitor_policy(self, enabled: bool) -> dict[str, Any]:
        from monitoring.ascend_usage import BENCHMARK_USAGE_FIELDS
        return {
            "enabled": enabled, "vendor": self.name, "vendor_display_name": self.display_name,
            "collector": "npu-smi info -t usages", "required_fields": list(BENCHMARK_USAGE_FIELDS),
            "metric_fields": [{"key": key, "label": label, "unit": "%"} for label, key in BENCHMARK_USAGE_FIELDS.items()],
            "target_interval_s": 1.0, "command_timeout_s": 5, "required_samples_per_target": 10,
            "coverage_window": "rank-local exact measurement window", "automatic_workload_extension": False,
            "target_resource": "benchmark-npu-core",
        }

    def monitor_targets(self, preflight: dict[str, Any], selected: Sequence[int]) -> list[dict[str, int]]:
        device_map = preflight.get("actual_device_map")
        if not isinstance(device_map, list):
            raise RuntimeError("host preflight did not preserve the npu-smi device map")
        targets = [{key: int(item[key]) for key in ("npu_id", "chip_id", "logic_id")}
                   for item in device_map if isinstance(item, dict) and item.get("logic_id") in selected]
        targets.sort(key=lambda item: (item["npu_id"], item["chip_id"], item["logic_id"]))
        if {item["logic_id"] for item in targets} != set(selected) or len(targets) != len(selected):
            raise RuntimeError("host preflight did not map every selected logical Device to a unique NPU/Chip target")
        return targets

    def create_monitor(self, targets: Sequence[dict[str, Any]]) -> Any:
        from monitoring.ascend_usage import BENCHMARK_USAGE_FIELDS, UsageMonitor
        return UsageMonitor(targets, field_specs=BENCHMARK_USAGE_FIELDS, interval_s=1.0, command_timeout_s=5)

    def rank_target(self, targets, selected, local_rank, binding):
        logical = binding.legacy_logical_id if binding is not None else selected[local_rank]
        matches = [item for item in targets if item["logic_id"] == logical]
        return matches[0] if len(matches) == 1 else None

    def measurement_identity(self, target: dict[str, Any], binding: DeviceBinding | None) -> dict[str, Any]:
        value = {"logical_device_id": target["logic_id"], "npu_id": target["npu_id"], "chip_id": target["chip_id"]}
        if binding is not None:
            value.update(vendor=self.name, device_id=binding.resource_key, binding=binding.record())
        return value


def run_host_preflight(result_dir: Path, expected_device_ids: list[int], *, npu_ids: str | None = None,
                       device_ids: str | None = None, label: str = "host-preflight") -> Path:
    output = result_dir / label
    command = [sys.executable, str(Path(__file__).with_name("preflight.py")), "--output", str(output),
               "--expected-device-ids", ",".join(map(str, expected_device_ids))]
    if npu_ids is not None:
        command.extend(["--npu-ids", npu_ids])
    if device_ids is not None:
        command.extend(["--device-ids", device_ids])
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    summaries = list(output.glob("*/summary.json"))
    if proc.returncode != 0 or len(summaries) != 1:
        raise RuntimeError("Ascend host preflight failed; inspect " + str(output))
    return summaries[0]
