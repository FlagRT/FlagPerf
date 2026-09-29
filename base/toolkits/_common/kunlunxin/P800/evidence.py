"""Raw subprocess evidence, timeout cleanup and deterministic hash indexing."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def reference(path, root):
    path, root = Path(path), Path(root)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def command(argv, directory, root, *, stem="microbenchmark", timeout=180, env=None, cwd=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stdout, stderr = directory / (stem + ".stdout"), directory / (stem + ".stderr")
    if stdout.exists() or stderr.exists():
        raise FileExistsError("refusing to overwrite existing command evidence: " + str(directory / stem))
    record = {"command": list(map(str, argv)), "timeout_s": timeout, "started_at": now(),
              "started_monotonic_s": time.monotonic(), "timed_out": False}
    with stdout.open("wb") as out, stderr.open("wb") as err:
        try:
            proc = subprocess.Popen(record["command"], stdout=out, stderr=err, env=env, cwd=cwd, start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
                rc = 124
            except BaseException:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
                raise
        except OSError as exc:
            err.write((str(exc) + "\n").encode())
            rc = 127
    record.update(returncode=rc, finished_at=now(), finished_monotonic_s=time.monotonic(),
                  stdout=reference(stdout, root), stderr=reference(stderr, root))
    record["duration_s"] = record["finished_monotonic_s"] - record["started_monotonic_s"]
    write_json(directory / (stem + ".command.json"), record)
    return record


def index(root):
    root = Path(root)
    paths = [p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()
             and p.name != "sha256-index.json"]
    write_json(root / "sha256-index.json", {"schema_version": 1, "scope": "all regular files under run root except this index",
                                          "files": [reference(p, root) for p in paths]})


def validate_index(root):
    root = Path(root)
    data = json.loads((root / "sha256-index.json").read_text())
    errors = []
    recorded = set()
    for item in data["files"]:
        if item["path"] in recorded:
            errors.append("duplicate:" + item["path"])
        recorded.add(item["path"])
        p = (root / item["path"]).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or sha256(p) != item["sha256"]:
            errors.append(item["path"])
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink() and p.name != "sha256-index.json"}
    errors += ["unindexed:" + name for name in sorted(actual-recorded)]
    return errors
