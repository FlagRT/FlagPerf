"""Strict PPU host inventory and occupancy checks; imports never query devices.

ppu-smi has no machine-readable table mode, so the inventory is one full
`ppu-smi -q` dump (day-1 probe D: 0.887s for 16 cards) parsed by the same
section grammar the monitor uses, plus one `ppu-smi -i N -q` per selected
card for fresh per-device evidence.
"""
import json
import os
from pathlib import Path
import stat

from base.vendors.thead import runtime
from base.vendors.thead.provider import (
    KNOWN_READONLY_MONITORS, authorized_monitors, normalize_bdf, validate_device_set,
)
from monitoring.thead_usage import one, parse_card, parse_q_sections

# Host-side ppu-smi binary (day-3 `readlink -f $(which ppu-smi)` evidence).
PPU_SMI = "/usr/local/bin/ppu-smi"

# Idle cards report a small driver reserve that already drifted between
# observations (10 MiB in the day-2 16-card snapshot, 18 MiB on day 3), so
# used memory is not an occupancy signal near that scale. A live foreign
# context always appears in the Processes table, and any real allocation
# (torch context alone exceeds 500 MiB) exceeds this threshold.
IDLE_RESERVE_MIB = 2048


def machine_inventory(raw):
    rows = []
    for bdf, section in parse_q_sections(raw).items():
        row = parse_card(section, bdf)
        row["host_physical_id"] = row["device_minor"] - 1
        if not (0 <= row["host_physical_id"] <= 15):
            raise ValueError("physical index outside the PPU slot range")
        rows.append(row)
    if not rows:
        raise ValueError("empty ppu-smi inventory")
    for field in ("host_physical_id", "pci_bdf", "uuid", "serial"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError("duplicate machine identity: " + field)
    return rows


def query_record(raw, bdf):
    canonical = normalize_bdf(bdf)
    sections = parse_q_sections(raw)
    if canonical not in sections:
        raise ValueError("single-card query misses the selected device section")
    identity = parse_card(sections[canonical], canonical)
    # The driver line sits in the shared header above all card sections.
    identity["driver"] = one(raw, r"^Driver Version[ \t]*:[ \t]*(\S+)[ \t]*$", "driver")
    return identity


def check_nodes(devices, control="/dev/alixpu"):
    for device in devices:
        info = Path(device["host_device_node"]).lstat()
        if not stat.S_ISCHR(info.st_mode) or os.minor(info.st_rdev) != device["device_minor"]:
            raise RuntimeError("device node type/minor differs from queried identity")
        device["device_major"] = os.major(info.st_rdev)
    if not stat.S_ISCHR(Path(control).lstat().st_mode):
        raise RuntimeError("missing character control device")


def readonly_smi_arguments(arguments):
    return (arguments in (["-q"], []) or
            len(arguments) == 3 and arguments[0] == "-i" and arguments[1].isdigit() and arguments[2] == "-q")


def inspect_handles(device, commands, directory, allow_observers=False):
    node = device["host_device_node"]
    for attempt in range(3):
        handles = commands.run(["fuser", node], privileged=True,
                               evidence=directory / ("handles-%d-%d.json" % (device["host_physical_id"], attempt)))
        if handles["returncode"] == 1 and not handles["stdout"].strip() and not handles["stderr"].strip():
            return []
        pids = handles["stdout"].split()
        if (not allow_observers or handles["returncode"] != 0 or not pids or
                any(not pid.isdigit() for pid in pids) or handles["stderr"].strip() != node + ":"):
            raise RuntimeError("device has open handles or handle check failed")
        observers = []
        vanished = False
        for pid in pids:
            prefix = directory / ("observer-%s-%d" % (pid, attempt))
            executable = commands.run(["readlink", "-e", "/proc/%s/exe" % pid], privileged=True,
                                      evidence=prefix.with_suffix(".exe.json"))
            command = commands.run(["cat", "/proc/%s/cmdline" % pid], privileged=True,
                                   evidence=prefix.with_suffix(".argv.json"))
            if executable["returncode"] or command["returncode"]:
                existence = commands.run(["test", "-d", "/proc/%s" % pid], privileged=True,
                                         evidence=prefix.with_suffix(".exists.json"))
                if existence["returncode"] == 1 and not existence["stdout"] and not existence["stderr"]:
                    observers.append({"pid": int(pid), "classification": "exited-before-inspection"})
                    continue
                vanished = True
                break
            arguments = command["stdout"].rstrip("\0").split("\0")
            exe = executable["stdout"].strip()
            name = next((n for n, path in KNOWN_READONLY_MONITORS.items()
                         if exe == path and arguments == [n]), None)
            if exe == PPU_SMI and arguments and readonly_smi_arguments(arguments[1:]):
                observers.append({"pid": int(pid), "executable": exe,
                                  "arguments": arguments[1:], "classification": "read-only-observer"})
            elif name and name in authorized_monitors():
                observers.append({"pid": int(pid), "executable": exe, "arguments": arguments,
                                  "agent": name, "classification": "read-only-monitor-agent",
                                  "authorized_by": "THEAD_KNOWN_READONLY_MONITORS"})
            else:
                raise RuntimeError("device handle is not a verified read-only ppu-smi observer")
        if not vanished:
            return observers
    raise RuntimeError("device handle owners changed before they could be verified")


def inspect_host(config, requested, commands, directory):
    directory.mkdir(parents=True, exist_ok=False)
    machine = commands.checked(["ppu-smi", "-q"], timeout=15, evidence=directory / "machine.json")
    inventory = machine_inventory(machine["stdout"])
    if sorted(row["host_physical_id"] for row in inventory) != sorted(config["expected_device_ids"]):
        raise RuntimeError("host physical inventory differs from configured inventory")
    by_id = {row["host_physical_id"]: row for row in inventory}
    devices = []
    for index in requested:
        if index not in by_id:
            raise RuntimeError("selected physical device absent")
        row = by_id[index]
        query = commands.checked(["ppu-smi", "-i", str(index), "-q"], timeout=15,
                                 evidence=directory / ("query-%d.json" % index))
        info = query_record(query["stdout"], row["pci_bdf"])
        locked = runtime.locked_host_observation()
        if info["driver"] != locked["driver"]:
            raise RuntimeError("host driver differs from locked observation")
        occupied = bool(row["utilization_percent"] or info["utilization_percent"] or
                        not row["processes_empty"] or not info["processes_empty"] or
                        row["used_memory_mib"] > IDLE_RESERVE_MIB or info["used_memory_mib"] > IDLE_RESERVE_MIB)
        foreign_authorized = os.environ.get("THEAD_ALLOW_FOREIGN_HANDLES") == "1"
        if occupied and not foreign_authorized:
            raise RuntimeError("selected physical device %d is occupied" % index)
        if occupied and foreign_authorized:
            info["foreign_occupancy_observed"] = True
            info["foreign_handles_authorized"] = True
        if row["total_memory_mib"] != info["total_memory_mib"]:
            raise RuntimeError("inventory/query memory capacity mismatch")
        for field in ("uuid", "device_minor", "pci_bdf"):
            if row[field] != info[field]:
                raise RuntimeError("inventory/query identity drift: " + field)
        devices.append({**info, "host_physical_id": index, "container_node": info["host_device_node"]})
    validate_device_set(devices)
    check_nodes(devices)
    observers = {}
    for device in devices:
        if device.get("foreign_occupancy_observed") and os.environ.get("THEAD_ALLOW_FOREIGN_HANDLES") == "1":
            observers[str(device["host_physical_id"])] = [{"classification": "foreign-handles-authorized",
                                                           "inspection": "skipped-by-explicit-user-authorization"}]
        else:
            observers[str(device["host_physical_id"])] = inspect_handles(
                device, commands, directory, config.get("allow_readonly_smi_handles", False))
    result = {"schema_version": 1, "status": "passed", "devices": devices,
              "foreign_handles_authorized": os.environ.get("THEAD_ALLOW_FOREIGN_HANDLES") == "1",
              "foreign_occupancy_observed": any(d.get("foreign_occupancy_observed") for d in devices),
              "readonly_observers": observers,
              "authorized_monitors": sorted(authorized_monitors()),
              "selection": {"source": "physical-device-ids", "requested_ids": requested},
              "machine_memory_unit": "query-section MiB cross-checked against the same -q dump"}
    (directory / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def same_identity(first, second):
    fields = ("host_physical_id", "uuid", "pci_bdf", "host_device_node", "container_node", "device_minor", "device_major")
    if [[d[k] for k in fields] for d in first["devices"]] != [[d[k] for k in fields] for d in second["devices"]]:
        raise RuntimeError("device identity changed after acquiring leases")
