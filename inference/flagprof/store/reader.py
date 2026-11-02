"""Offline and live readers share decoding, signature checks, and identities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping
import gzip
import hashlib
import json
import os

from .manifest import json_bytes, load_json, writer_fragments
from .schema import SchemaValidator, UnsupportedVersion, is_control


@dataclass(frozen=True)
class StoredRecord:
    run_id: str
    writer_id: int
    shard_index: int
    record: dict[str, Any]

    @property
    def identity(self) -> tuple[Any, ...]:
        record = self.record
        if "seq" in record:
            return ("event", self.run_id, self.writer_id, record["seq"])
        if record["t"] == "sig":
            return ("sig", record["sid"])
        if record.get("kind") == "drop_report":
            return ("drop_report", self.run_id, self.writer_id, self.shard_index)
        if record.get("kind") == "shard_header":
            return ("shard_header", self.run_id, self.writer_id, self.shard_index)
        return ("control", self.run_id, self.writer_id, json_bytes(record))

    def as_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "writer_id": self.writer_id, "shard_index": self.shard_index, "record": self.record}

    @property
    def source(self):
        """Persisted source locator, exposed without consumers reading ext."""
        ext=self.record.get('ext',{})
        uid,loc=ext.get('chrome.trace_uid'),ext.get('chrome.loc')
        return dict(trace_uid=uid,loc=loc) if uid is not None and loc is not None else None


@dataclass
class _Shard:
    writer_id: int
    index: int
    handle: BinaryIO | None = None
    read_lines: int = 0
    defaults: dict[str, Any] = field(default_factory=dict)
    signatures: set[str] = field(default_factory=set)
    incomplete: bool = False
    unreadable: bool = False
    last_seq: int | None = None
    descriptor: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    raw_handle: BinaryIO | None = None
    opened_identity: tuple = ()
    verified_stat: tuple = ()
    integrity: str = 'limited'
    prefix_digest: Any = field(default_factory=hashlib.sha256)


class StoreReadError(ValueError):
    """A strict public query cannot prove its requested record population."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__('store read failed: ' + '; '.join(e['reason'] for e in report.get('errors', [])))


def _stat_key(stat: os.stat_result) -> tuple:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _shard_identity(descriptor: dict[str, Any]) -> tuple:
    # A verified file alone does not prove the current completion declaration.
    return tuple(descriptor.get(k) for k in ('path', 'codec', 'sha256', 'records', 'sealed'))


class Reader:
    def __init__(self, run_dir: str | Path, *, preserve_unknown: bool = False, retain: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.manifest = load_json(self.run_dir / "manifest.json")
        if int(self.manifest["schema_version"].split(".")[0]) > 1:
            raise UnsupportedVersion("unsupported run schema major version: " + self.manifest["schema_version"])
        self.run_id = self.manifest["run_id"]
        self.preserve_unknown = preserve_unknown
        self.retain = retain
        self._stream_active = False
        self.validator = SchemaValidator()
        self._shards: dict[tuple[int, int], _Shard] = {}
        self._records: dict[tuple[Any, ...], StoredRecord] = {}
        # Insertion order is part of the legacy max(next_seq) tie semantics.
        self._drop_reports: dict[int, dict[tuple[Any, ...], StoredRecord]] = {}
        self._latest_drop: dict[int, StoredRecord] = {}
        self._fragments: dict[int, dict[str, Any]] = {}
        self._gaps: dict[int, list[tuple[int, int]]] = {}
        self._errors: list[dict[str, Any]] = []
        self._error_keys: set[str] = set()
        self.report: dict[str, Any] = {}

    def _error(self, shard: _Shard, reason: str) -> None:
        shard.integrity = 'failed'
        entry = {"writer_id": shard.writer_id, "shard_index": shard.index, "record_index": shard.read_lines, "reason": reason}
        key = json_bytes(entry).decode()
        if key not in self._error_keys:
            self._errors.append(entry)
            self._error_keys.add(key)

    def _registered_replacement(self, shard: _Shard) -> bool:
        """A published codec transition can unlink an open, unchanged inode.

        Reopen only the same registered shard's new location. Its full SHA and
        the already delivered prefix are checked by _open; ctime is not ignored.
        """
        if getattr(shard,'transition_retries',0)>=2:
            return False
        try:
            latest=load_json(self.run_dir/'writers'/f'w-{shard.writer_id}.json')
            current=next(d for d in latest['shards'] if d['index']==shard.index)
        except (OSError,ValueError,KeyError,StopIteration):
            return False
        if (current['path'],current['codec'])==(shard.descriptor['path'],shard.descriptor['codec']):
            return False
        self._close_shard(shard)
        shard.transition_retries=getattr(shard,'transition_retries',0)+1
        shard.descriptor=current
        shard.completed=False
        shard.opened_identity=None
        shard.verified_stat=None
        return True

    def _open(self, shard: _Shard) -> bool:
        path = self.run_dir / shard.descriptor["path"]
        codec = shard.descriptor["codec"]
        try:
            if codec not in ('none', 'gzip'):
                shard.unreadable = True
                self._error(shard, "unsupported codec: " + str(codec))
                return False
            raw = path.open('rb')
            shard.raw_handle = raw
            shard.opened_identity = _shard_identity(shard.descriptor)
            if shard.descriptor['sealed'] and shard.descriptor.get('sha256'):
                before = _stat_key(os.fstat(raw.fileno()))
                digest = hashlib.sha256()
                for block in iter(lambda: raw.read(1 << 20), b''):
                    digest.update(block)
                if before != _stat_key(os.fstat(raw.fileno())):
                    raise OSError('sealed shard changed during verification')
                if digest.hexdigest() != shard.descriptor['sha256']:
                    raise OSError('sealed shard SHA mismatch')
                raw.seek(0)
                shard.verified_stat = before
                shard.integrity = 'verified'
            else:
                shard.integrity = 'limited'
            handle = raw if codec == 'none' else gzip.GzipFile(fileobj=raw, mode='rb')
            shard.handle = handle
            # Across a codec/path transition byte offsets have no meaning.
            prefix = hashlib.sha256()
            for _ in range(shard.read_lines):
                line = handle.readline()
                if not line.endswith(b"\n"):
                    raise OSError("replacement shard has fewer committed records")
                prefix.update(line)
            if prefix.digest() != shard.prefix_digest.digest():
                raise OSError('previously read shard prefix changed')
            return True
        except FileNotFoundError:
            # Compression publishes its new path before unlinking the old one.
            if self._registered_replacement(shard):
                return Reader._open(self,shard)
            shard.unreadable = True
            self._error(shard, "registered shard file missing")
            return False
        except (OSError, EOFError) as exc:
            self._close_shard(shard)
            if self._registered_replacement(shard):
                return Reader._open(self,shard)
            shard.unreadable = True
            self._error(shard, str(exc))
            return False

    def _decode(self, raw: bytes, shard: _Shard) -> StoredRecord | None:
        value = json.loads(raw)
        if value.get("t") == "meta" and value.get("kind") == "shard_header":
            data = value.get("data", {})
            if data.get("writer_id") != shard.writer_id or data.get("shard_index") != shard.index:
                raise ValueError("header identity disagrees with writer fragment")
            shard.defaults = dict(data.get("defaults", {}))
        record = self.validator.decode(value, shard.defaults)
        unknown = record is None
        if self.preserve_unknown:
            record = dict(value)
            if "seq" in record:
                for k, v in shard.defaults.items():
                    record.setdefault(k, v)
        elif unknown:
            # Ignoring a future event type must not manufacture a seq gap.
            record = value
        if record["t"] == "meta" and record.get("kind") == "defaults":
            shard.defaults.update(record.get("data", {}))
        if record["t"] == "sig":
            shard.signatures.add(record["sid"])
        elif "seq" in record:
            for key in ("sid", "osid"):
                if key in record and record[key] not in shard.signatures:
                    raise ValueError("signature missing before reference in this shard: " + record[key])
            seq = record["seq"]
            if shard.last_seq is None:
                # Writer's previous shard gives a deterministic boundary reference.
                earlier = [s for (w, i), s in self._shards.items() if w == shard.writer_id and i < shard.index and s.last_seq is not None]
                previous = max(earlier, key=lambda x: x.index).last_seq if earlier else -1
            else:
                previous = shard.last_seq
            if seq <= previous:
                raise ValueError("non-increasing writer seq")
            if seq > previous + 1:
                self._gaps.setdefault(shard.writer_id, []).append((previous + 1, seq))
            shard.last_seq = seq
        if unknown and not self.preserve_unknown:
            return None
        return StoredRecord(self.run_id, shard.writer_id, shard.index, record)

    def _consume(self, shard: _Shard, updates: list[StoredRecord]) -> None:
        """Compatibility entrypoint for the registered bounded business reader."""
        updates.extend(self._consume_iter(shard))

    def _consume_iter(self, shard: _Shard) -> Iterator[StoredRecord]:
        if shard.unreadable:
            return
        identity = _shard_identity(shard.descriptor)
        if shard.completed or (shard.opened_identity and identity != shard.opened_identity):
            try:
                unchanged = (identity == shard.opened_identity and shard.integrity == 'verified' and
                             _stat_key((self.run_dir / shard.descriptor['path']).stat()) == shard.verified_stat)
                if shard.completed and unchanged:
                    return
            except OSError:
                pass  # _open reports missing artifacts through the same channel.
            self._close_shard(shard)
            shard.completed = False
        if shard.handle is None and not self._open(shard):
            return
        while True:
            offset = shard.handle.tell()
            try:
                raw = shard.handle.readline()
            except (OSError, EOFError) as exc:
                shard.unreadable = True
                self._error(shard, str(exc))
                break
            if not raw:
                break
            if not raw.endswith(b"\n"):
                shard.incomplete = True
                shard.handle.seek(offset)
                break
            shard.incomplete = False
            shard.read_lines += 1
            shard.prefix_digest.update(raw)
            try:
                stored = self._decode(raw, shard)
            except UnsupportedVersion:
                self.close()
                raise  # Never decode the following records under obsolete units.
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                self._error(shard, str(exc))
                continue
            if stored is None:
                continue
            key = stored.identity
            previous = self._records.get(key)
            if previous is not None and previous.record == stored.record:
                continue
            if previous is not None and stored.record["t"] == "sig":
                if stored.record.get("trunc") and not previous.record.get("trunc"):
                    continue
                if not stored.record.get("trunc") and not previous.record.get("trunc"):
                    raise ValueError("same sid has incompatible definitions")
            if self.retain or "seq" not in stored.record:
                self._records[key] = stored
            if stored.record.get("kind") == "drop_report":
                reports = self._drop_reports.setdefault(stored.writer_id, {})
                reports[key] = stored
                self._latest_drop[stored.writer_id] = max(reports.values(), key=lambda s: s.record["data"]["next_seq"])
            yield stored
        if shard.descriptor["sealed"]:
            if shard.verified_stat and _stat_key(os.fstat(shard.raw_handle.fileno())) != shard.verified_stat:
                if self._registered_replacement(shard):
                    yield from self._consume_iter(shard)
                    return
                self._error(shard, 'sealed shard changed during decoding')
                shard.unreadable = True
            expected = shard.descriptor["records"]
            if shard.read_lines == expected and not shard.incomplete:
                shard.completed = True
                self._close_shard(shard)
            elif shard.read_lines != expected:
                self._error(shard, f"sealed record count {shard.read_lines} differs from {expected}")

    def poll(self) -> list[StoredRecord]:
        """Return new identities and replacements (drop_report is cumulative)."""
        return list(self.stream())

    def stream(self) -> Iterator[StoredRecord]:
        """Decode one poll incrementally; report is committed only on exhaustion.

        Closing an iterator retains its delivered prefix and releases the guard.
        Control/signature/shard state remains retained even in streaming mode.
        """
        if self._stream_active:
            raise RuntimeError("only one active stream is allowed")
        self._stream_active = True
        try:
            self.manifest = load_json(self.run_dir / "manifest.json")
            for fragment in writer_fragments(self.run_dir):
                wid = fragment["writer_id"]
                self._fragments[wid] = fragment
                if fragment.get("initialization_incomplete"):
                    self._error(_Shard(wid, -1), "writer initial reservation is incomplete")
                for descriptor in sorted(fragment["shards"], key=lambda x: x["index"]):
                    key = (wid, descriptor["index"])
                    shard = self._shards.setdefault(key, _Shard(wid, descriptor["index"]))
                    shard.descriptor = descriptor
                    yield from self._consume_iter(shard)
            self._update_report()  # stream exhausted
        finally:
            self._stream_active = False

    def _update_report(self) -> None:
        writers = {}
        for wid, fragment in sorted(self._fragments.items()):
            state = fragment["state"]
            if state != "closed" and self.manifest.get("writers", {}).get(str(wid), {}).get("state") == "lost":
                state = "lost"
            latest = self._latest_drop.get(wid)
            data = latest.record["data"] if latest else {"dropped": {}, "next_seq": 0}
            reported = sum(data["dropped"].values())
            gaps = self._gaps.get(wid, [])
            visible_gaps = sum(end - start for start, end in gaps)
            after_report = sum(max(0, end - max(start, data["next_seq"])) for start, end in gaps)
            lower_bound = max(visible_gaps, reported + after_report)
            writers[str(wid)] = {"state": state, "records": sum(s.read_lines for (w, _), s in self._shards.items() if w == wid),
                                 "dropped": reported if state == "closed" else lower_bound,
                                 "drop_semantics": "exact" if state == "closed" else "lower_bound",
                                 "tail_unknown": state != "closed", "seq_gaps": visible_gaps,
                                 "reported_dropped": reported, "by_type": data["dropped"]}
        self.report = {"truncated": bool(self._errors) or any(s.incomplete or s.unreadable for s in self._shards.values()),
                       "errors": list(self._errors), "writers": writers,
                       "unknown_fields": self.validator.unknown_fields, "unknown_types": self.validator.unknown_types,
                       "unknown_enums": self.validator.unknown_enums}
        self.report['read_integrity'] = ('failed' if self.report['truncated'] else
            'verified' if self._shards and all(s.descriptor['sealed'] and s.integrity == 'verified'
                                             for s in self._shards.values()) else 'limited')
        self.report['verification_scope'] = 'registered event shards; active shards are read prefixes only'
        self.report['shards'] = [dict(writer_id=s.writer_id, index=s.index, path=s.descriptor['path'],
                                     sealed=s.descriptor['sealed'], integrity=s.integrity, read_lines=s.read_lines)
                                  for s in self._shards.values()]

    @property
    def records(self) -> list[StoredRecord]:
        if not self.retain:
            raise RuntimeError("records snapshot requires retain=True")
        # Each identity kind has fixed component types. Native tuple ordering
        # keeps writer seq numeric (2 before 10), also in derived compact data.
        return sorted(self._records.values(), key=lambda s: s.identity)

    def read(self) -> list[StoredRecord]:
        if not self.retain:
            raise RuntimeError("records snapshot requires retain=True")
        self.poll()
        return self.records

    def forget_handles(self) -> None:
        """Simulate a tail consumer reconnect; retain committed record counts."""
        for shard in self._shards.values():
            self._close_shard(shard)

    @staticmethod
    def _close_shard(shard: _Shard) -> None:
        if shard.handle is not None:
            shard.handle.close()
            shard.handle = None
        if shard.raw_handle is not None:
            shard.raw_handle.close()
            shard.raw_handle = None

    def close(self) -> None:
        self.forget_handles()

    def __enter__(self) -> Reader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class TailReader(Reader):
    """poll() streams changes; records is the current deduplicated snapshot."""


def read_compact(path: str | Path) -> list[StoredRecord]:
    result = []
    with Path(path).open("rb") as handle:
        for line in handle:
            value = json.loads(line)
            result.append(StoredRecord(value["run_id"], value["writer_id"], value["shard_index"], value["record"]))
    return result
