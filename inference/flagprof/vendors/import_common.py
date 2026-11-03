# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Offline import transaction: prepare, reserve, persist, barrier, completion."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

from flagprof.store.evidence import build_index
from flagprof.store.manifest import atomic_json, json_bytes, process_alive
from flagprof.store.writer import Writer, WriterConfig, EmitReceipt

RULES_VERSION = 't4-v2-20261011'


class ImportConflict(FileExistsError):
    """Content-addressed reservation conflict; never overwrite an attempt."""
    def __init__(self, code: str, path: Path):
        self.code = code
        self.path = path
        super().__init__(code + ': ' + str(path) + '; use a new analysis run/attempt; existing evidence is preserved')


def _conflict(path: Path, descriptor: dict, run: Path) -> ImportConflict:
    code = 'partial_retry_requires_new_run'
    try:
        old = json.loads(path.read_bytes())
        def occurrence(d):
            capture=d.get('capture') or {}
            return capture.get('session_id'),capture.get('capture_id'),capture.get('window')
        if old.get('sha256') != descriptor['sha256']:
            code = 'trace_uid_collision'
        elif occurrence(old) != occurrence(descriptor):
            code = 'content_occurrence_not_supported'
        elif any(json_bytes(old.get(k))!=json_bytes(descriptor.get(k)) for k in ('vendor','normalization','linker','native')) or json_bytes(old.get('identities',{}).get('parser'))!=json_bytes(descriptor.get('identities',{}).get('parser')):
            code = 'reinterpretation_requires_new_run'
        elif json.loads((run/old['completion_path']).read_bytes())['status']=='complete':
            code = 'already_exists'
    except (OSError,ValueError,KeyError,TypeError):
        pass
    return ImportConflict(code,path)


def importer_writer_config() -> WriterConfig:
    """One configuration for the worker and acceptance entrypoints."""
    return WriterConfig()


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path: Path, contents: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)


def import_fixture(source: str | Path, writer: Writer, *, vendor: str,
                   build_rules: Callable | None = None, device_fields: Callable | None = None,
                   registry: Mapping[str, Any], kind: str | None = None,
                   calls: Sequence[dict[str, Any]] | None = None, marker_prefix: str | None = None,
                   identities: Mapping[str, Any] | None = None, scope_config: Any = None,
                   capture: Mapping[str, Any] | None = None, taxonomy_version: str | None = None, flush_timeout: float = 120.0) -> dict[str, Any]:
    """Import a source once. A failed reservation stays occupied; retry in a new run.

    The historical callback keywords remain accepted for adapter compatibility;
    the production preparation entrypoint owns the rule family and normalization.
    """
    from .prepare import prepare_import
    if writer.fragment['kind'] != 'importer':
        raise ValueError('recorded traces require kind=importer')
    source = Path(source)
    contents = source.read_bytes()
    registry = json.loads(json_bytes(registry))
    prepared = prepare_import(contents, vendor=vendor, registry=registry, kind=kind, calls=calls,
                              marker_prefix=marker_prefix, identities=identities, scope_config=scope_config, taxonomy_version=taxonomy_version)
    uid = prepared.trace_uid
    derived = Path('derived') / uid
    files = {
        'ledger': ('ledger.jsonl', b''.join(json_bytes(r) + b'\n' for r in prepared.ledger)),
        'native': ('native.jsonl', b''.join(json_bytes(r) + b'\n' for r in prepared.native)),
        'summary': ('summary.json', json_bytes(dict(format_version=1, disposition=prepared.disposition,
            project=prepared.coverage, native=prepared.native_coverage, coverage_scope='prepared_events',
            unbound_calls=prepared.unbound_calls,format_assessment=prepared.format_assessment,
            native_states=dict(Counter(r['state'] for r in prepared.native)),
            native_reasons=dict(Counter(str(r.get('reason')) for r in prepared.native)))) + b'\n'),
        'index': ('evidence-index.json', json_bytes(build_index(contents, uid)) + b'\n'),
    }
    inventory = dict(prepared.inventory)
    inventory['raw_task_types'] = dict(sorted(Counter(str(r['task_type']) if r['task_type'] is not None else '<absent>'
                                                      for r in inventory['records']).items()))
    normalization = dict(prepared.normalization, calls_origin='argument' if calls is not None else 'default',
        calls_sha256=hashlib.sha256(json_bytes(prepared.normalization['calls'])).hexdigest(),
        calls_encoding='sorted compact UTF-8 JSON, no trailing newline')
    raw_path = Path('vendor_raw') / (uid + '.json')
    completion_path = Path('imports') / uid / 'completion.json'
    descriptor = dict(format_version=1, trace_uid=uid, vendor=vendor, source_path=str(source),
        sha256=prepared.sha256, raw_path=str(raw_path), imported_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        device_ids=sorted({e['dev'] for e in prepared.events}), rank=writer.fragment['rank'],
        inventory=inventory, linker=dict(rules_version=prepared.identities['parser'].get('rules',{}).get('version',RULES_VERSION), registry=registry,
            registry_sha256=hashlib.sha256(json_bytes(registry)).hexdigest(),
            registry_encoding='sorted compact UTF-8 JSON, no trailing newline'),
        clock_transform=dict(clock_valid=False, reason='fixture has no sourced transform to host_mono'),
        profiled_window=None, profiled_window_reason='fixture lacks measure/profiled window facts',
        writer_id=writer.writer_id, normalization=normalization, identities=prepared.identities,
        capture=json.loads(json_bytes(capture)) if capture is not None else None, disposition=prepared.disposition,
        native=dict(rules_version=prepared.native_rules_version, scope_config=prepared.scope_config),
        derived={key:dict(path=str(derived / name), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                          format_version=1) for key,(name,data) in files.items()}, completion_path=str(completion_path))
    descriptor_bytes = json_bytes(descriptor) + b'\n'
    trace_path = writer.run_dir / 'traces' / (uid + '.json')
    # O_EXCL is the transaction reservation. Never remove even an empty one.
    try:
        _write(trace_path, descriptor_bytes)
    except FileExistsError:
        raise _conflict(trace_path,descriptor,writer.run_dir) from None
    for path in (writer.run_dir / derived, (writer.run_dir / completion_path).parent):
        path.mkdir(parents=True, exist_ok=False)
        _sync_dir(path.parent)
        _sync_dir(writer.run_dir)
    _write(writer.run_dir / raw_path, contents)
    for name,data in files.values():
        _write(writer.run_dir / derived / name, data)
    receipts = []
    event_map = []
    barrier = writer.flush(flush_timeout)
    for ordinal,event in enumerate(prepared.events):
        event_map.append(dict(ordinal=ordinal, trace_uid=uid, loc=event['ext']['chrome.loc']))
        if not barrier:
            receipts.append(EmitReceipt(reason='barrier_unconfirmed'))
            continue
        payload = dict(event, ext={k:v for k,v in event['ext'].items() if k != 'chrome.raw_record'})
        receipt = writer.emit_with_receipt(payload)
        receipts.append(receipt)
        if (ordinal + 1) % writer.batch_limit == 0:
            barrier = writer.flush(flush_timeout)
    if barrier:
        barrier = writer.flush(flush_timeout)
    for item,receipt in zip(event_map, receipts):
        item.update(asdict(receipt))
        if not barrier:
            item.update(persisted=False, reason='barrier_unconfirmed')
    enqueued = sum(r.accepted for r in receipts)
    persisted = sum(r['persisted'] for r in event_map)
    dropped = len(receipts) - persisted
    seqs = [r.seq for r in receipts if r.seq is not None]
    status = 'complete' if barrier and not dropped else 'partial'
    if barrier:
        completion = dict(format_version=1, trace_uid=uid, descriptor_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
            derived_sha256={k:v['sha256'] for k,v in descriptor['derived'].items()}, writer_id=writer.writer_id,
            attempted=len(receipts), enqueued=enqueued, persisted=persisted, dropped=dropped,
            seq_range=[min(seqs),max(seqs)+1] if seqs else None, event_map=event_map, status=status,
            reasons=sorted({r['reason'] for r in event_map if r['reason']}))
        _write(writer.run_dir / completion_path, json_bytes(completion) + b'\n')
    return dict(trace_uid=uid, source_sha256=prepared.sha256, normalized_device_count=len(receipts), emitted=enqueued,
                persisted=persisted, dropped=dropped, status=status, barrier=barrier,
                summary=prepared.coverage, trace_path=str(trace_path))


def import_capture(directory: str | Path, writer: Writer, *, vendor: str,
                   exit_evidence: Mapping[str, Any] | None = None, **options: Any) -> list[dict[str, Any]]:
    """Copy an ended capture once and import its valid windows in this process.

    A launcher in another PID namespace must provide sourced exit evidence for
    the exact recorded process. The collection owner cannot import itself.
    """
    from flagprof.store.captures import inspect_capture
    directory = Path(directory)
    data = json.loads((directory / 'session.json').read_bytes())
    alive = process_alive(data.get('process'))
    if alive is not False:
        if (alive is True or not exit_evidence or not exit_evidence.get('source') or
                exit_evidence.get('process') != data.get('process') or exit_evidence.get('exited') is not True):
            raise ValueError('capture import requires a distinct process and sourced owner exit')
    session_id = data['session_id']
    if not isinstance(session_id,str) or not session_id or any(c not in '0123456789abcdef' for c in session_id):
        raise ValueError('invalid session id')
    # Validate every registered immutable artifact before copying anything.
    for window in data['windows']:
        for artifact in window['artifacts']:
            relative = Path(window['path']) / artifact['path']
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('invalid capture artifact path')
            raw = (directory / relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != artifact['sha256'] or len(raw) != artifact['bytes']:
                raise ValueError('capture artifact SHA mismatch')
    if any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('capture symlinks are not supported')
    target = writer.run_dir / 'captures' / session_id
    target.parent.mkdir(exist_ok=True)
    shutil.copytree(directory,target)
    for path in sorted(target.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    for path in sorted((p for p in target.rglob('*') if p.is_dir()),reverse=True):
        _sync_dir(path)
    _sync_dir(target)
    _sync_dir(target.parent)
    _sync_dir(writer.run_dir)
    manifest_sha = hashlib.sha256((target / 'session.json').read_bytes()).hexdigest()
    registration = dict(format_version=1,session_id=session_id,manifest_sha256=manifest_sha,
        process=data.get('process'),owner_exited=True,exit_evidence=dict(exit_evidence or
        dict(source='local process identity check',process=data.get('process'),exited=True)))
    atomic_json(target / 'registration.json',registration)
    results=[]
    for window in inspect_capture(target,owner_alive=False)['windows']:
        if window['effective_state'] != 'valid':
            continue
        for trace in window['validity']['traces']:
            if trace['path'] not in {a['path'] for a in window['artifacts']}:
                raise ValueError('trace is not a registered capture artifact')
            source = target / window['path'] / trace['path']
            if source.resolve().is_relative_to(target.resolve()) is False:
                raise ValueError('capture trace escapes directory')
            capture=dict(session_id=session_id,capture_id=window['capture_id'],window=window['index'],
                manifest_path=str((target/'session.json').relative_to(writer.run_dir)),
                manifest_sha256=manifest_sha,exit_evidence=registration['exit_evidence'])
            from .evidence import capture_identities
            identities=capture_identities(target,data,window,trace['path'],capture['manifest_path'],vendor)
            registry=options.get('registry')
            if registry is None:
                from .format import load_adapter
                registry = load_adapter(vendor).default_registry()
            supplied=dict(options,registry=registry,identities=identities,capture=capture)
            if vendor == 'thead':
                # The PPU window path persists the same device projection as
                # final captures. Never claim a scheduled window covers a
                # complete request.
                from tempfile import TemporaryDirectory
                from flagprof.offline.pipeline import prepare, import_prepared
                from flagprof.offline.device_analysis import capture_context
                from flagprof.offline.project import calls_from
                manifest=json.loads((writer.run_dir/'manifest.json').read_bytes())
                context_capture=dict(vendor=vendor,normal_close=False,occurrence=window['capture_id'],
                    trace=dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
                    metadata=dict(collection_run='.',source_run_id=manifest['run_id'],
                        process=data.get('process'),identities=identities,
                        requested_configuration=data.get('config',{})))
                analysis=capture_context(writer.run_dir,context_capture)
                analysis['clock']['sources']=[dict(path=str(source.relative_to(writer.run_dir)),
                    sha256=context_capture['trace']['sha256'])]
                for transform in analysis['clock'].get('project_transforms',[]):
                    transform['sources']=analysis['clock']['sources']
                analysis['window_identity']['window_kind']='scheduled_window'
                analysis['capture_scope']='scheduled_window_not_complete_request'
                calls=options.get('calls')
                if calls is None:calls=calls_from(writer.run_dir,manifest['run_id'])
                with TemporaryDirectory(prefix='thead-window-import-') as work:
                    prepared=prepare(source,Path(work)/'prepared',vendor=vendor,
                        registry=registry,identities=identities,calls=calls,
                        marker_prefix=options.get('marker_prefix','flagprof::op#'),
                        kind=options.get('kind'),scope_config=options.get('scope_config'),
                        taxonomy_version=options.get('taxonomy_version'),analysis_input=analysis)
                    results.append(import_prepared(source,writer,prepared,vendor=vendor,
                        registry=registry,calls_supplied=True,capture=capture))
            else:
                results.append(import_fixture(source,writer,vendor=vendor,**supplied))
    return results


def resolve_evidence(run: str | Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility single-item wrapper around the public read-only view."""
    from flagprof.store.view import StoreView
    with StoreView(run) as view:
        return view.resolve_many([reference])[0]
