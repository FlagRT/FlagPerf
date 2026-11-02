# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Reusable verification of unchanged, closed local Plan B stores.

Receipts are rebuildable work data, never Store evidence. Content is checked on
the first read; inode/size/mtime/ctime identities invalidate subsequent reuse.
The public Reader still verifies event shard bytes when building a page.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import sqlite3
import sys

from .bounded_verify import identity
from .limits import METADATA_BYTES, read_json, walk_files
from flagprof.store.manifest import atomic_json, json_bytes

VERSION = 1
_OPERATIONAL = {'.finalize.lock', 'plan-b-status.json'}


def store_identity(run):
    return {str(p.relative_to(run)): list(identity(p))
            for p in walk_files(run, max_files=256, max_entries=512)
            if str(p.relative_to(run)) not in _OPERATIONAL}


@lru_cache(maxsize=1)
def validator_identity():
    """A process uses its frozen implementation; a new process rebinds it."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.json') and '__pycache__' not in path.parts:
            digest.update(str(path.relative_to(root)).encode()+b'\0')
            digest.update(path.read_bytes())
    return dict(source_sha256=digest.hexdigest(), python=list(sys.version_info[:3]),
                sqlite=sqlite3.sqlite_version)


class VerificationCache:
    def __init__(self, run, root=None):
        self.run = Path(run).resolve()
        if root is None:
            from .workspace import locations
            root, _ = locations(self.run.parent, key='query-'+self.run.name)
        self.root = Path(root).resolve()
        if self.root.is_relative_to(self.run) or self.run.is_relative_to(self.root):
            raise ValueError('verification cache and Store run must be separate')
        key = hashlib.sha256(str(self.run).encode()).hexdigest()[:16]
        self.path = self.root/('verification-'+key+'.json')

    def binding(self):
        """An unavailable/oversized cache identity falls back to full validation."""
        try:
            registration = read_json(self.run/'plan-b-capture.json')
            bundle = Path(registration['bundle_path'])
            if registration.get('format_version') == 2:
                if bundle.is_absolute():return None
                bundle = self.run/bundle
            bundle = bundle.resolve()
            capture = {str(p.relative_to(bundle)): list(identity(p)) for p in
                       walk_files(bundle, max_files=16384, max_entries=32768)}
            result = dict(version=VERSION, run=str(self.run), validator=validator_identity(),
                          store=store_identity(self.run), capture_path=str(bundle), capture=capture)
            return result if len(json_bytes(result)) <= METADATA_BYTES//2 else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def load(self, binding):
        if binding is None:return None
        try:
            envelope = read_json(self.path)
            payload = envelope['payload']
            if envelope['sha256'] != hashlib.sha256(json_bytes(payload)).hexdigest():return None
            if payload['binding'] != binding:return None
            verified = payload['verified']
            if verified['status'] != 'complete' or verified['read_integrity'] != 'verified':return None
            if verified['reader_report']['read_integrity'] != 'verified':return None
            if any(type(verified[k]) is not int or verified[k] < 0 for k in ('count', 'source_records')):return None
            return verified
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def publish(self, binding, verified):
        if binding is None:return
        try:
            if read_json(self.run/'manifest.json').get('status') != 'complete':return
            if self.binding() != binding:return
            capture = Path(binding['capture_path'])
            if self.root.is_relative_to(capture) or capture.is_relative_to(self.root):return
            payload = dict(binding=binding, verified=verified)
            envelope = dict(payload=payload, sha256=hashlib.sha256(json_bytes(payload)).hexdigest())
            if len(json_bytes(envelope)) > METADATA_BYTES:return
            from .workspace import check_disk
            check_disk(self.root)
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_json(self.path, envelope)
        except (OSError, ValueError):
            # A read-only or missing cache never changes a successful validation.
            return
