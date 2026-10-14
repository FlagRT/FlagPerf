"""Locked PPU runtime manifest contract; importable on host and in-container.

Mirrors the P800 verify_runtime state machine: a candidate manifest is usable
only with explicit per-run authorization, and a validated manifest must carry
its own scoped validation record. Nothing here promotes a runtime.
"""
import hashlib
import json
import re
from pathlib import Path

PROFILE = Path(__file__).parent / "torch210_ppu_candidate"


def load_lock():
    return json.loads((PROFILE / "stack.lock.yaml").read_text(encoding="utf-8"))


def locked_host_observation():
    return load_lock()["host_observed"]


def validate_manifest(manifest, observed=None):
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    if manifest.get("vendor") != "thead" or manifest.get("chip") != "PPU":
        raise ValueError("wrong vendor/chip")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest.get("image_id", "")):
        raise ValueError("immutable image ID required")
    state = (manifest.get("release_stage"), manifest.get("validated"))
    if state == ("candidate", False):
        pass
    elif state == ("validated", True):
        record_ref = manifest.get("validation_record") or {}
        if record_ref.get("path") != "validation-summary.json":
            raise ValueError("validated runtime requires its scoped validation record")
        raw = (PROFILE / record_ref["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record_ref.get("sha256"):
            raise ValueError("validation record hash mismatch")
        record = json.loads(raw)
        if (manifest.get("validation_status") != "partial"
                or not isinstance(manifest.get("validation_scope"), list)
                or not manifest["validation_scope"]
                or not all(isinstance(item, str) and item.strip() for item in manifest["validation_scope"])):
            raise ValueError("validated runtime requires an explicit partial scope")
        for field in ("schema_version", "vendor", "chip", "runtime_profile", "image_id",
                      "release_stage", "validated", "validation_status", "validation_scope"):
            if record.get(field) != manifest.get(field):
                raise ValueError("validation record mismatch: " + field)
    else:
        raise ValueError("inconsistent runtime release state")
    if manifest.get("platform") != "linux/amd64":
        raise ValueError("expected linux/amd64")
    if manifest.get("hardware_model") != "PPU-ZW810E":
        raise ValueError("unexpected PPU hardware model declaration")
    if observed is not None:
        if isinstance(observed, list):
            if len(observed) != 1:
                raise ValueError("inspect must describe exactly one image")
            observed = observed[0]
        for field, expected in (("Id", manifest["image_id"]), ("Architecture", "amd64")):
            if observed.get(field) != expected:
                raise ValueError("image identity mismatch: " + field)
        if set(observed.get("RepoDigests") or []) != set(manifest.get("repo_digests") or []):
            raise ValueError("RepoDigests changed; review provenance before use")
