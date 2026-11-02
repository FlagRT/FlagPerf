"""Run creation, Linux process identity, allocation, and read-only finalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import datetime as dt
import fcntl
import fnmatch
import hashlib
import json
import os
import socket
import uuid

from .schema import SCHEMA_VERSION

# L-field: eight launcher node bits and eight process-local allocation bits.
NODE_BITS = 8
LOCAL_BITS = 16 - NODE_BITS
MAX_WRITER_ID = (1 << 16) - 1
REDACTION_PATTERNS = ("*TOKEN*", "*SECRET*", "*KEY*", "*PASSWORD*", "*_AK", "*_SK")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def writer_fragments(run: Path) -> list[dict[str, Any]]:
    """Initial O_EXCL reservations are visible only after their owner unlocks."""
    with (run / "writers" / ".lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        result = []
        for path in sorted((run / "writers").glob("w-*.json"), key=lambda p: int(p.stem[2:])):
            try:
                result.append(load_json(path))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # A creator killed during its initial reservation still occupies
                # the id, but has no safely recoverable process identity.
                result.append({"writer_id": int(path.stem[2:]), "state": "active", "proc_key": None,
                               "shards": [], "dropped": {}, "next_seq": 0, "initialization_incomplete": True})
        return result


def proc_key(pid: int | None = None) -> dict[str, Any]:
    pid = os.getpid() if pid is None else pid
    stat = Path(f"/proc/{pid}/stat").read_text()
    # comm can contain spaces and parentheses; fields after the final ')' start at 3.
    ticks = int(stat[stat.rfind(")") + 2:].split()[19])
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_ns": str(os.stat(f"/proc/{pid}/ns/pid").st_ino),
        "pid": pid,
        "proc_start_ticks": ticks,
    }


def process_alive(key: Mapping[str, Any] | None) -> bool | None:
    """None means inaccessible or from another host/namespace, never 'lost'."""
    current = proc_key()
    if key is None:
        return None
    if key["boot_id"] != current["boot_id"] or key["pid_ns"] != current["pid_ns"]:
        return None
    try:
        actual = proc_key(int(key["pid"]))
    except (FileNotFoundError, ProcessLookupError):
        return False
    except (PermissionError, OSError):
        return None
    return actual == dict(key)


def environment_identity(environ: Mapping[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    result, redacted = {}, []
    for key, value in sorted((os.environ if environ is None else environ).items()):
        if any(fnmatch.fnmatchcase(key.upper(), pattern) for pattern in REDACTION_PATTERNS):
            result[key] = "<redacted:" + hashlib.sha256(value.encode()).hexdigest()[:8] + ">"
            redacted.append(key)
        else:
            result[key] = value
    return result, redacted


def create_run(store: str | Path, *, run_id: str | None = None, identity: Mapping[str, Any] | None = None,
               mode: str = "observe", experiment: Mapping[str, Any] | None = None) -> Path:
    now = dt.datetime.now(dt.timezone.utc)
    run_id = run_id or now.strftime("%Y%m%dT%H%M%SZ") + "-" + socket.gethostname() + "-" + uuid.uuid4().hex[:8]
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one directory name")
    if mode not in {"observe", "seal"}:
        raise ValueError("invalid run mode")
    env, redacted = environment_identity()
    identity = dict(identity or {})
    # This is a measured identity, not caller metadata. A supplied env cannot
    # bypass full capture or the same redaction applied to the real process.
    identity["env"] = env
    if mode == "seal" and "inputs" not in identity:
        raise ValueError("seal mode requires input identity")
    root = Path(store) / run_id
    root.mkdir(parents=True, exist_ok=False)
    for name in ("writers", "traces", "events", "blobs", "vendor_raw", "logs"):
        (root / name).mkdir()
    (root / "writers" / ".lock").touch()
    atomic_json(root / "manifest.json", {
        "run_id": run_id, "created_at": now.isoformat(), "schema_version": SCHEMA_VERSION,
        "status": "running", "probes": [], "identity": identity, "mode": mode,
        "experiment": dict(experiment or {}), "collection": {"redacted_env": redacted},
        "retention": [], "writers": {},
    })
    return root


def allocate_writer(run: Path, *, node_index: int | None, kind: str, rank: int, key: Mapping[str, Any]) -> dict[str, Any]:
    if node_index is not None and (type(node_index) is not int or not 0 <= node_index < (1 << NODE_BITS)):
        raise ValueError("node_index exceeds writer_id node bit capacity")
    if kind not in {"probe", "counter", "importer"}:
        raise ValueError("invalid writer kind")
    directory = run / "writers"
    with (directory / ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        occupied = [int(p.stem[2:]) for p in directory.glob("w-*.json")]
        if node_index is None:
            writer_id = max(occupied, default=-1) + 1
        else:
            local = [w & ((1 << LOCAL_BITS) - 1) for w in occupied if w >> LOCAL_BITS == node_index]
            next_local = max(local, default=-1) + 1
            if next_local >= (1 << LOCAL_BITS):
                raise OverflowError("node-local writer_id exhausted; ids cannot be reused")
            writer_id = (node_index << LOCAL_BITS) | next_local
        if writer_id > MAX_WRITER_ID:
            raise OverflowError("16-bit writer_id exhausted; ids cannot be reused")
        fragment = {"writer_id": writer_id, "kind": kind, "rank": rank, "proc_key": dict(key),
                    "clk": "host_mono:" + key["boot_id"], "shards": [], "next_seq": 0,
                    "dropped": {}, "state": "active"}
        fd = os.open(directory / f"w-{writer_id}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "wb") as handle:
            handle.write(json_bytes(fragment) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return fragment


def _finalize(run_dir: str | Path, *, _load_json=None, _view_factory=None) -> dict[str, Any]:
    """Update only the manifest; never repair, compress, or mutate a writer file."""
    if _view_factory is None and (Path(run_dir)/'plan-b-capture.json').exists():
        from flagprof.offline.workspace import query_view
        from flagprof.offline.finish import finalize_bounded
        with query_view(run_dir) as view:return finalize_bounded(run_dir,view)
    from .reader import Reader
    read_json = load_json if _load_json is None else _load_json
    root = Path(run_dir)
    from ._proxy_lifecycle import lock_until, _DEADLINE
    with (root / ".finalize.lock").open("a+b") as lock, lock_until(lock, _DEADLINE.get()):
        manifest = read_json(root / "manifest.json")
        with Reader(root, retain=False) as reader:
            for _ in reader.stream():
                pass
        summaries = {}
        for fragment in writer_fragments(root):
            wid = str(fragment["writer_id"])
            from ._proxy_lifecycle import registered_proxies, effective_writer_state
            state = effective_writer_state(fragment, registered_proxies(root))
            report = reader.report["writers"].get(wid, {})
            summaries[wid] = {"state": state, "records": report.get("records", 0),
                              "dropped": report.get("dropped", 0), "tail_unknown": state != "closed"}
        manifest["writers"] = summaries
        # Importers own their O_EXCL trace descriptors; only the finalizer
        # copies the recorded whitelist into the manifest (PLAN/03 §4.8).
        # A partial registration remains visible rather than being repaired.
        registries, trace_errors = {}, []
        for path in sorted((root / "traces").glob("*.json")):
            try:
                descriptor = read_json(path)
                linker = descriptor["linker"]
                registry = linker["registry"]
                digest = hashlib.sha256(json_bytes(registry)).hexdigest()
                if digest != linker["registry_sha256"]:
                    raise ValueError("registry SHA mismatch")
                registries[digest] = registry
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as error:
                trace_errors.append({"trace": path.name, "reason": str(error)})
        if registries or trace_errors:
            collection = manifest.setdefault("collection", {})
            collection["linker_registries"] = registries
            collection["trace_errors"] = trace_errors
        from .view import StoreView
        from .captures import inspect_capture, capture_summary, log_counts
        with (StoreView if _view_factory is None else _view_factory)(root) as view:
            imports = view.traces()
        captures = []
        for path in sorted((root / "captures").glob("*/session.json")):
            try:
                registration = read_json(path.parent / "registration.json")
                if hashlib.sha256(path.read_bytes()).hexdigest() != registration['manifest_sha256']:
                    raise ValueError('capture manifest SHA mismatch')
                data = inspect_capture(path.parent, owner_alive=False if registration.get('owner_exited') else None)
                captures.append(capture_summary(data))
            except (OSError, ValueError, KeyError, TypeError) as error:
                captures.append(dict(session_id=path.parent.name, status='failed', reason=str(error)))
        for path in sorted((root/'attachments').glob('*/state.json')):
            attachment = read_json(path)
            if attachment.get('capture_status') == 'not_requested':
                captures.append(dict(session_id=attachment['id'], status='not_requested',
                                     label='未请求采集', windows=[]))
        manifest.setdefault('collection', {}).update(imports=imports, captures=captures, log_check=log_counts(captures))
        incomplete = any(r['status'] in ('partial', 'failed') for r in imports)
        incomplete |= any(c.get('status') == 'failed' or any(w['state'] != 'valid' for w in c.get('windows', [])) for c in captures)
        if reader.report["truncated"] or trace_errors or incomplete or any(x["state"] == "lost" for x in summaries.values()):
            manifest["status"] = "truncated"
        elif all(x["state"] == "closed" for x in summaries.values()):
            manifest["status"] = "complete"
        atomic_json(root / "manifest.json", manifest)
        return manifest


def finalize(run_dir: str | Path, *, _load_json=None, _view_factory=None) -> dict[str, Any]:
    from ._proxy_lifecycle import deadline_scope, wait_proxies
    with deadline_scope() as deadline:
        wait_proxies(run_dir, deadline)
        return _finalize(run_dir, _load_json=_load_json, _view_factory=_view_factory)
