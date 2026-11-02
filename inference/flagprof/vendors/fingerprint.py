"""Portable source-closure identity for immutable, source-distributed formats.

Source bytes deliberately include comments. This conservative identity avoids
interpreter bytecode, absolute installation paths and filesystem timestamps.
Each format declares every local implementation dependency in its closure;
stdlib implementation identity belongs to the recorded execution environment.
Runtime monkey-patching without changing the installed source is unsupported.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
from typing import Sequence


def format_identity(version: str, sources: Sequence[str]) -> dict[str, str]:
    # Include this algorithm too: a change to hashing must not silently reuse
    # an old parser identity. Only content and ordered closure slots are hashed.
    paths = [Path(__file__), *(Path(source) for source in sources)]
    payload = dict(algorithm='source-closure-v1', version=version,
                   sources=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(version=version, sha256=digest)


def parser_identity(vendor: str, version: str, *, root: Path | None = None) -> dict:
    """Conservative static closure of shipped production source and data.

    Include both adapters and every shared module, including lazily selected
    registry/validation paths. This intentionally over-approximates a single
    import; a dynamic module list could silently omit unexercised branches.
    Supplied registry/configuration and raw source are inputs, not code bytes.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    paths = sorted(p for p in (root/'flagprof').rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.json')
                   and '__pycache__' not in p.parts)
    modules = [p.relative_to(root).as_posix() for p in paths]
    payload = dict(algorithm='parser-static-closure-v1', vendor=vendor, version=version,
                   sources=[(name,hashlib.sha256(p.read_bytes()).hexdigest())
                            for name,p in zip(modules,paths)])
    return dict(modules=modules,sha256=hashlib.sha256(json.dumps(
        payload,sort_keys=True,separators=(',',':')).encode()).hexdigest())
