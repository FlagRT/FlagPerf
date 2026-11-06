# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Public standard-library read-only view over persisted import facts.

No parsing rules run here. Project links and native relations stay separate;
legacy files retain evidence lookup without fabricated derived data.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import load_json
from .reader import Reader, StoreReadError


def _uid(value: str) -> str:
    if not isinstance(value,str) or len(value)!=16 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid trace uid')
    return value


def _signature(stat: os.stat_result) -> tuple:
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns


class StoreView:
    def __init__(self, run: str | Path) -> None:
        self.run = Path(run)
        self._raw: dict[str,dict] = {}
        self._derived: dict[tuple,Any] = {}
        self.sha_checks: dict[str,int] = {}
        self._closed = False
        self._bounded_view = None
        self._bounded_context = None
        self._analysis_verified = {}

    def iter_events(self, types=None, *, work=None, limits=None):
        """Strict streaming records with disk-backed signature/control state."""
        import sqlite3
        import tempfile
        from .analysis_read import AnalysisReader
        limits = limits or {}
        if work:
            Path(work).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='reader-', dir=work) as directory:
            db = sqlite3.connect(str(Path(directory)/'state.sqlite'))
            try:
                with AnalysisReader(self.run, db, **{k:limits[k] for k in ('record_bytes','cache_bytes','scratch_bytes') if k in limits}) as reader:
                    for stored in reader.stream():
                        if types is None or stored.record['t'] in types:
                            yield stored
                    self.analysis_read_report = reader.report
            finally:
                db.close()

    def probe_status(self, *, limits=None):
        """Attachment identities are retained separately; no assumed capability."""
        from .analysis_read import ResourceLimit
        limit = (limits or {}).get('record_bytes', 16 << 20)
        manifest_path=self.run/'manifest.json'
        if manifest_path.stat().st_size>min(limit,(limits or {}).get('cache_bytes',64 << 20)//4):
            raise ResourceLimit('resource_limit: manifest metadata')
        rows = []
        total = 0
        for path in sorted((self.run/'attachments').glob('*/state.json')):
            size=path.stat().st_size
            total+=size
            if len(rows) >= 256 or size > limit or total > (limits or {}).get('cache_bytes',64 << 20)//4:
                raise ResourceLimit('resource_limit: attachment metadata')
            state = load_json(path)
            rows.append(dict(id=state['id'], probes=state.get('probes', {}),
                effective_config=state.get('effective_config'), component=state.get('component'),
                route_table=state.get('route_table'), flaggems_identity=state.get('flaggems_identity'),
                op_kinds=state.get('op_kinds'), torch_version=state.get('torch_version'),
                oracle_identity=state.get('oracle_identity'), tensor=state.get('tensor'),
                tensor_boundaries=state.get('tensor_boundaries'), module_selection=state.get('module_selection'), route_unobserved=state.get('route_unobserved', 0),
                errors=state.get('errors', [])))
        return dict(status='available' if rows else 'unavailable', attachments=rows,
                    reason=None if rows else 'attachment_record_missing',
                    schema_version=load_json(manifest_path)['schema_version'])

    def tensor_status(self, *, verify=False):
        states=[s.get('tensor') for s in self.probe_status()['attachments'] if s.get('tensor')]
        result=dict(status='available' if states else 'unavailable',attachments=states,verified_blobs=0)
        if verify and states:
            from flagprof.tensor_data import verify_blob
            for stored in self.iter_events(types=('tensor',)):
                if 'blob' in stored.record:
                    verify_blob(self.run,stored.record)
                    result['verified_blobs']+=1
        return result

    def memory_status(self):
        """Normalized official memory completion; legacy Stores remain readable."""
        path = self.run/'memory.json'
        if not path.exists():
            return dict(status='unavailable', reason='memory_collection_not_recorded', requested=False)
        from flagprof.offline.limits import read_json
        from flagprof.offline.checkpoints import fileinfo
        state = read_json(path)
        if state.get('schema') != 'official-memory/1':
            raise ValueError('unsupported memory descriptor')
        if state.get('ledger'):
            if state['ledger']['path'] != 'memory-ledger.jsonl':
                raise ValueError('memory ledger path')
            if fileinfo(self.run/'memory-ledger.jsonl') != {k:state['ledger'][k] for k in ('bytes','sha256')}:
                raise ValueError('memory ledger changed')
        if state['status']=='complete' and state['normalized']+state['excluded']!=state['records']:
            raise ValueError('memory ledger conservation')
        return state

    def analysis_source(self):
        """Resolve only registered capture references, with capture/run binding."""
        from .lineage import read_lineage
        from .analysis_read import ResourceLimit
        registration = self.run/'plan-b-capture.json'
        if not registration.exists():
            return dict(status='unavailable', reason='capture_reference_missing')
        if registration.stat().st_size > 16 << 20:
            raise ResourceLimit('resource_limit: capture reference')
        ref = load_json(registration)
        directory = (self.run/ref['bundle_path']).resolve()
        from ..offline.bundle import load_bundle
        from ..offline.checkpoints import fileinfo
        if fileinfo(directory/'capture.json') != ref['envelope']:
            raise ValueError('capture reference digest mismatch')
        capture = load_bundle(directory)
        if capture != ref['capture']:
            raise ValueError('registered capture metadata mismatch')
        lineage = read_lineage(self.run)
        source = capture['metadata'].get('source_run_id')
        if not source or lineage.get('source_run_id') != source:
            raise ValueError('analysis source_run_id mismatch')
        relative = Path(capture['metadata']['collection_run'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('collection reference must be relative')
        collection = (directory/relative).resolve()
        if not collection.is_relative_to(directory) or load_json(collection/'manifest.json')['run_id'] != source:
            raise ValueError('collection run identity mismatch')
        return dict(status='available', source_run_id=source, collection_run=str(collection),
                    capture_occurrence=capture['occurrence'])

    def open_collection(self):
        source = self.analysis_source()
        if source['status'] != 'available':
            raise ValueError(source['reason'])
        return StoreView(source['collection_run'])

    def iter_native_relations(self, trace_uid=None, *, limits=None, work=None):
        """Verified bounded native rows, owner names and interpretation state."""
        from .native_read import rows
        from .analysis_read import ResourceLimit
        if trace_uid is None:
            for index,path in enumerate(sorted((self.run/'traces').glob('*.json'))):
                if index>=256:
                    raise ResourceLimit('resource_limit: traces')
                yield from rows(self,path.stem,limits=limits,work=work)
        else:
            yield from rows(self,_uid(trace_uid),limits=limits,work=work)

    def _bounded(self):
        if self._closed:raise ValueError('view is closed')
        if self._bounded_view is None and (self.run/'plan-b-capture.json').exists():
            from flagprof.offline.workspace import query_view
            self._bounded_context=query_view(self.run)
            self._bounded_view=self._bounded_context.__enter__()
        return self._bounded_view

    def page_device_events(self, *, offset=0, limit=1000):
        """Verified bounded page for Plan B runs."""
        view=self._bounded()
        if view is None:raise ValueError('bounded pages require a Plan B run')
        return view.page(offset,limit)

    def device_analysis_context(self):
        if len(self._analysis_verified)>1:
            contexts=[value['context'] for value in self._analysis_caches()]
            if all(c.get('status')=='available' for c in contexts):
                return dict(schema='device-analysis/2',status='available',source_clock='per_window',
                    clock=dict(clock_valid=False,status='per_window',reason='independent source clocks; no cross-window interval union'),
                    contexts=deepcopy(contexts),capture_scope='scheduled_windows_not_complete_request',
                    windows=[dict(w,source_clock=c['source_clock']) for c in contexts for w in c.get('windows',[])],
                    request_window_status='partial' if any(c.get('request_window_status')!='unavailable' for c in contexts) else 'unavailable')
        cached=self._analysis_cache()
        if cached is not None:return deepcopy(cached['context'])
        view=self._bounded()
        return view.device_analysis_context() if view else dict(schema='device-analysis/1',status='unavailable',reason='projection_not_recorded')

    def _analysis_cache(self):
        """Reuse the manifest/summary binding already checked by native reading."""
        if self._closed:raise ValueError('view is closed')
        if len(self._analysis_verified)!=1:return None
        value=next(iter(self._analysis_verified.values()))
        if any(_signature(path.stat())!=signature for path,signature in value['bindings']):
            raise ValueError('analysis metadata changed after read')
        return value

    def _analysis_caches(self):
        if self._closed:raise ValueError('view is closed')
        values=list(self._analysis_verified.values())
        for value in values:
            if any(_signature(path.stat())!=signature for path,signature in value['bindings']):
                raise ValueError('analysis metadata changed after read')
        return values

    def iter_device_events(self, *, trace_uid=None, category=None, projection='events'):
        """Verified bounded iterator for Plan B runs; no full result retention."""
        if projection=='analysis' and len(self._analysis_verified)>1:
            for value in self._analysis_caches():
                if trace_uid is not None and trace_uid!=value['trace_uid']:continue
                info=value['intervals']
                if info is None:continue
                path=self._path(info['path'],'derived');before=_signature(path.stat());digest=hashlib.sha256();size=0
                with path.open('rb') as source:
                    for line in iter(lambda:source.readline((16<<20)+1),b''):
                        if len(line)>16<<20:raise ValueError('analysis record byte limit')
                        digest.update(line);size+=len(line);row=json.loads(line)
                        if category is None or row['category']==category:yield row
                if size!=info['bytes'] or digest.hexdigest()!=info['sha256'] or _signature(path.stat())!=before:
                    raise ValueError('analysis intervals SHA mismatch')
            self._analysis_caches()
            return
        cached=self._analysis_cache() if projection=='analysis' else None
        if cached is not None:
            if trace_uid is not None and trace_uid!=cached['trace_uid']:raise ValueError('unknown trace uid')
            info=cached['intervals']
            if info is None:return
            path=self._path(info['path'],'derived');before=_signature(path.stat());digest=hashlib.sha256();size=0
            # Verify in the same streaming pass used by the analyzer. No raw
            # trace replay or event-map validation is introduced by this view.
            with path.open('rb') as source:
                for line in iter(lambda:source.readline((16<<20)+1),b''):
                    if len(line)>16<<20:raise ValueError('analysis record byte limit')
                    digest.update(line);size+=len(line);row=json.loads(line)
                    if category is None or row['category']==category:yield row
            if size!=info['bytes'] or digest.hexdigest()!=info['sha256'] or _signature(path.stat())!=before:
                raise ValueError('analysis intervals SHA mismatch')
            self._analysis_cache()
            return
        view=self._bounded()
        if view is None:raise ValueError('bounded iteration requires a Plan B run')
        yield from view.iter_device_events(trace_uid=trace_uid,category=category,projection=projection)

    def _path(self, relative: str, prefix: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!=prefix:
            raise ValueError('path must remain under '+prefix)
        target = self.run / path
        if not target.resolve().is_relative_to(self.run.resolve()/prefix) or target.is_symlink():
            raise ValueError('path escapes '+prefix)
        return target

    def _descriptor(self, uid: str) -> dict:
        if self._closed:
            raise ValueError('view is closed')
        data = load_json(self.run/'traces'/(_uid(uid)+'.json'))
        if data['trace_uid']!=uid or data['sha256'][:16]!=uid:
            raise ValueError('trace identity mismatch')
        if 'format_version' in data and data['format_version']!=1:
            raise ValueError('unsupported trace format')
        return data

    def _derived_value(self, descriptor: Mapping, name: str) -> Any:
        info=descriptor['derived'][name]
        path=self._path(info['path'],'derived')
        stat=_signature(path.stat())
        key=(str(path),info['sha256'],stat)
        if key not in self._derived:
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=info['sha256'] or len(data)!=info['bytes']:
                raise ValueError('derived SHA mismatch: '+name)
            result=([json.loads(line) for line in data.splitlines()] if path.suffix=='.jsonl' else json.loads(data))
            self._derived[key]=result
        return self._derived[key]

    def _completion(self, descriptor: Mapping) -> dict | None:
        path=self._path(descriptor['completion_path'],'imports')
        if not path.exists():
            return None
        result=load_json(path)
        uid=descriptor['trace_uid']
        if (result.get('format_version')!=1 or result.get('trace_uid')!=uid or
            result.get('writer_id')!=descriptor['writer_id'] or
            result.get('descriptor_sha256')!=hashlib.sha256((self.run/'traces'/(uid+'.json')).read_bytes()).hexdigest() or
            result.get('derived_sha256')!={k:v['sha256'] for k,v in descriptor['derived'].items()}):
            raise ValueError('completion identity/SHA mismatch')
        rows=result['event_map']
        if (result['attempted']!=len(rows) or result['enqueued']!=sum(r['accepted'] for r in rows) or
            result['persisted']!=sum(r['persisted'] for r in rows) or result['dropped']!=len(rows)-result['persisted'] or
            result['status'] not in ('complete','partial') or
            (result['status']=='complete' and result['dropped'])):
            raise ValueError('invalid completion counts/status')
        return result

    def _state(self, descriptor: Mapping, *, verify: bool = True) -> tuple[str,list[str]]:
        if 'format_version' not in descriptor:
            return 'legacy',[]
        completion=self._completion(descriptor)
        if verify:
            for name in descriptor['derived']:
                self._derived_value(descriptor,name)
            self._open_raw(descriptor['trace_uid'],dict(descriptor))
        if completion is None:
            return 'partial',['completion_missing']
        return completion['status'],completion['reasons']

    def traces(self, *, verify: bool = True) -> list[dict]:
        """Verify persisted evidence and the event population independently."""
        try:
            bounded=self._bounded()
            if bounded is not None:return bounded.traces()
        except (OSError,ValueError,KeyError,TypeError) as error:
            # Preserve the public trace-status contract on failed verification.
            path=next((self.run/'traces').glob('*.json'),None)
            return [dict(trace_uid=path.stem if path else 'unknown',status='failed',import_status='unavailable',
                         integrity='failed',read_integrity='failed',reasons=[str(error)],read_report=None)]
        rows=[]
        snapshot = None
        paths = sorted((self.run/'traces').glob('*.json'))
        if verify and paths:
            snapshot = self._read_snapshot()
        for path in paths:
            historical = 'unavailable'
            try:
                d=self._descriptor(path.stem)
                historical,_=self._state(d,verify=False)
                state,reasons=self._state(d,verify=verify)
                report = self._check_event_map(d, *snapshot) if snapshot is not None else None
                if report is not None and report['read_integrity'] == 'failed':
                    raise StoreReadError(report)
                rows.append(dict(trace_uid=d['trace_uid'],status=state,reasons=reasons,sha256=d['sha256'],
                    import_status=historical,
                    integrity='verified' if report and report['read_integrity']=='verified' and state!='legacy' else 'not_fully_verified',
                    read_integrity=report['read_integrity'] if report else 'limited',
                    read_report=report,
                    identities=d.get('identities',dict(status='unavailable',reason='legacy_store')),
                    capture=d.get('capture'),vendor=d['vendor'],writer_id=d['writer_id']))
            except (OSError,ValueError,KeyError,TypeError) as error:
                rows.append(dict(trace_uid=path.stem,status='failed',import_status=historical,
                    integrity='failed',read_integrity='failed',reasons=[str(error)],
                    read_report=error.report if isinstance(error,StoreReadError) else None))
        return rows

    def run_status(self) -> dict:
        from .status import run_status
        return run_status(self.run)

    def lineage(self) -> dict:
        from .lineage import read_lineage
        return read_lineage(self.run)

    def inspect(self, *, verify: bool = False) -> list[dict]:
        """Persisted facts only. Metadata mode never opens raw/native/ledger."""
        result=[]
        for row in self.traces(verify=verify):
            value=dict(row,import_status=row.get('import_status',row['status']))
            if row['status']=='failed':
                result.append(value);continue
            try:
                bounded=self._bounded()
                d=bounded.descriptor if bounded is not None else self._descriptor(row['trace_uid'])
                summary=bounded.summary() if bounded is not None else self._derived_value(d,'summary') if 'summary' in d.get('derived',{}) else {}
                assessment=summary.get('format_assessment',self._unavailable())
                groups=dict(precondition=[],scope=[],missing_evidence=[])
                for condition in assessment.get('conditions',[]):
                    state=condition['state']
                    if state=='unknown':groups['missing_evidence'].append(condition)
                    elif state=='fail':
                        scope_failure='allowed' in condition and condition['id']!='scope.host_device_pid_collision'
                        groups['scope' if scope_failure else 'precondition'].append(condition)
                evidence=d.get('identities',{}).get('collector',{}).get('evidence',{})
                disposition=summary.get('disposition',d.get('disposition',self._unavailable()))
                unknown=sum(g['count'] for g in assessment.get('unknown_groups',[]) if g['dimension']=='category')
                denominator=disposition.get('records')
                value.update(format_applicability=assessment,native_ownership=dict(
                    coverage=summary.get('native',self._unavailable()),states=summary.get('native_states',self._unavailable()),
                    reasons=summary.get('native_reasons',self._unavailable())),
                    project_linkage=summary.get('project',self._unavailable()),
                    coverage_scope=summary.get('coverage_scope','unavailable'),disposition=disposition,
                    unknown_categories=dict(count=unknown if 'unknown_groups' in assessment else None,
                        denominator=denominator,denominator_kind='raw source elements',
                        ratio=unknown/denominator if denominator and 'unknown_groups' in assessment else None,
                        groups=assessment.get('unknown_groups',[])),
                    reason_groups=groups,capture_completeness=evidence.get('capture_completeness',dict(status='unchecked')),
                    clock=summary.get('device_analysis',{}).get('clock',evidence.get('clock',dict(clock_valid=False,status='unavailable'))),
                    device_analysis=summary.get('device_analysis',dict(schema='device-analysis/1',status='unavailable',reason='projection_not_recorded')),
                    evidence=evidence,source=dict(path=d['raw_path'],sha256=d['sha256']))
            except (OSError,ValueError,KeyError,TypeError) as error:
                value.update(status='failed',integrity='failed',read_integrity='failed',reasons=[str(error)])
            result.append(value)
        return deepcopy(result)

    def _read_snapshot(self) -> tuple[list, dict]:
        try:
            with Reader(self.run, retain=False) as reader:
                records = [r for r in reader.stream() if "seq" in r.record and (
                    r.record.get('ext', {}).get('chrome.trace_uid') is not None or
                    r.record['t'] in ('kernel', 'memop') or
                    r.record['t'] == 'comm' and r.record.get("phase") == "device")]
                records.sort(key=lambda r: r.identity)
                report = deepcopy(reader.report)
            return records, report
        except (OSError, ValueError, KeyError, TypeError) as error:
            return [], dict(read_integrity='failed', truncated=True, writers={},
                            errors=[dict(reason=str(error))], verification_scope='registered event shards')

    def _check_event_map(self, descriptor: Mapping, records: list, report: dict) -> dict:
        report = deepcopy(report)
        if 'format_version' not in descriptor:
            if report['read_integrity'] != 'failed':
                report['read_integrity'] = 'limited'
            return report
        completion = self._completion(descriptor)
        if completion is None:
            return report
        uid = descriptor['trace_uid']
        actual = {(r.writer_id, r.record['seq']): r.record.get('ext', {})
                  for r in records if 'seq' in r.record and r.record.get('ext', {}).get('chrome.trace_uid') == uid}
        expected = {}
        failures = []
        for entry in completion['event_map']:
            if entry.get('trace_uid') != uid:
                failures.append('event mapping trace identity mismatch')
            if not entry['persisted']:
                continue
            key = (completion['writer_id'], entry['seq'])
            if key in expected:
                failures.append('duplicate persisted event mapping')
            expected[key] = str(entry['loc'])
            if key not in actual or str(actual[key].get('chrome.loc')) != str(entry['loc']):
                failures.append('persisted event mapping missing or mismatched: ' + str(key))
        if set(actual) != set(expected):
            failures.append('event population disagrees with completion')
        if failures:
            report.update(read_integrity='failed', truncated=True)
            report['errors'] += [dict(trace_uid=uid, reason=reason) for reason in failures]
        return report

    def query_device_events(self, *, trace_uid: str | None = None, category: str | None = None,
                            allow_partial: bool = False) -> dict:
        """Read one snapshot; partial data requires an explicit opt-in."""
        if trace_uid is not None:
            self._descriptor(trace_uid)
        records, report = self._read_snapshot()
        paths = [self.run/'traces'/(trace_uid+'.json')] if trace_uid else sorted((self.run/'traces').glob('*.json'))
        for path in paths:
            try:
                report = self._check_event_map(self._descriptor(path.stem), records, report)
            except (OSError, ValueError, KeyError, TypeError) as error:
                report.update(read_integrity='failed', truncated=True)
                report['errors'].append(dict(reason=str(error)))
        if report['read_integrity'] == 'failed' and not allow_partial:
            raise StoreReadError(report)
        result=[]
        for row in records:
            record=row.record
            kind=record.get('ktype','compute') if record['t']=='kernel' else 'memcpy' if record['t']=='memop' else 'comm' if record['t']=='comm' and record.get('phase')=='device' else None
            if kind is None:
                continue
            kind=kind if kind in ('compute','comm','memcpy','unknown') else 'unknown'
            source=dict(trace_uid=record.get('ext',{}).get('chrome.trace_uid'),loc=record.get('ext',{}).get('chrome.loc'))
            if trace_uid is not None and source['trace_uid']!=trace_uid or category is not None and kind!=category:
                continue
            result.append(dict(run_id=row.run_id,writer_id=row.writer_id,category=kind,source=source,
                               event={k:v for k,v in record.items() if k!='ext'}))
        return dict(rows=result, read_integrity=report['read_integrity'], report=report,
                    scope=dict(trace_uid=trace_uid, category=category, population='persisted_events'))

    def device_events(self, *, trace_uid: str | None = None, category: str | None = None) -> list[dict]:
        return self.query_device_events(trace_uid=trace_uid, category=category)['rows']

    def project_bindings(self, trace_uid: str) -> dict:
        d=self._descriptor(trace_uid)
        status,reasons=self._state(d)
        return dict(status=status,reasons=reasons,items=[dict(source=r['source'],writer_id=r['writer_id'],
                    seq=r['event']['seq'],link=r['event']['link']) for r in self.device_events(trace_uid=trace_uid)])

    @staticmethod
    def _unavailable() -> dict:
        return dict(status='unavailable',reason='legacy_store')

    def native_relations(self, trace_uid: str, *, loc: Any = None) -> dict:
        d=self._descriptor(trace_uid)
        if 'format_version' not in d:
            return self._unavailable()
        status,reasons=self._state(d)
        rows=self._derived_value(d,'native')
        return dict(status=status,reasons=reasons,items=deepcopy([r for r in rows if loc is None or str(r['loc'])==str(loc)]))

    def ledger(self, trace_uid: str, *, offset: int = 0, limit: int = 1000) -> dict:
        if type(offset) is not int or type(limit) is not int or offset<0 or limit<1:
            raise ValueError('invalid ledger page')
        bounded=self._bounded()
        if bounded is not None:
            if trace_uid!=bounded.descriptor['trace_uid']:raise ValueError('unknown trace uid')
            rows=bounded.derived_page('ledger',offset,limit);total=bounded.verified['source_records']
            return dict(status='complete',reasons=[],summary=bounded.descriptor['disposition'],total=total,offset=offset,
                        next_offset=offset+len(rows) if offset+len(rows)<total else None,items=rows)
        d=self._descriptor(trace_uid)
        if 'format_version' not in d:
            return self._unavailable()
        status,reasons=self._state(d)
        rows=self._derived_value(d,'ledger')
        return dict(status=status,reasons=reasons,summary=d['disposition'],total=len(rows),offset=offset,
                    next_offset=offset+limit if offset+limit<len(rows) else None,items=deepcopy(rows[offset:offset+limit]))

    def coverage(self, trace_uid: str) -> dict:
        d=self._descriptor(trace_uid)
        if 'format_version' not in d:
            return self._unavailable()
        status,reasons=self._state(d)
        summary=self._derived_value(d,'summary')
        completion=self._completion(d)
        return dict(status=status,reasons=reasons,scope=summary['coverage_scope'],project=deepcopy(summary['project']),
                    native=deepcopy(summary['native']),persistence={k:completion[k] for k in ('attempted','enqueued','persisted','dropped')} if completion else None)

    def evidence_index(self, trace_uid: str) -> dict:
        d=self._descriptor(trace_uid)
        if 'format_version' not in d:
            return self._unavailable()
        return dict(status=self._state(d)[0],index=deepcopy(self._derived_value(d,'index')))

    def format_assessment(self, trace_uid: str) -> dict:
        """Persisted interpretation limits, separate from import completion."""
        d=self._descriptor(trace_uid)
        if 'format_version' not in d or 'summary' not in d.get('derived',{}):
            return self._unavailable()
        summary=self._derived_value(d,'summary')
        if 'format_assessment' not in summary:
            return dict(status='unavailable',reason='format_assessment_not_recorded')
        status,reasons=self._state(d)
        return dict(status=status,reasons=reasons,assessment=deepcopy(summary['format_assessment']))

    def _open_raw(self, uid: str, descriptor: dict) -> dict:
        path=self._path(descriptor['raw_path'],'vendor_raw')
        if uid in self._raw:
            cached=self._raw[uid]
            if (cached['sha256']!=descriptor['sha256'] or cached['path']!=path or
                _signature(path.stat())!=cached['stat'] or _signature(os.fstat(cached['handle'].fileno()))!=cached['stat']):
                raise ValueError('raw evidence changed during reading session')
            return cached
        handle=path.open('rb')
        try:
            before=_signature(os.fstat(handle.fileno()))
            digest=hashlib.sha256()
            for block in iter(lambda:handle.read(1<<20),b''):
                digest.update(block)
            self.sha_checks[uid]=self.sha_checks.get(uid,0)+1
            if digest.hexdigest()!=descriptor['sha256'] or digest.hexdigest()[:16]!=uid:
                raise ValueError('raw evidence identity/SHA mismatch')
            if _signature(os.fstat(handle.fileno()))!=before or _signature(path.stat())!=before:
                raise ValueError('raw changed during SHA validation')
            cached=dict(handle=handle,stat=before,path=path,sha256=digest.hexdigest())
            if 'format_version' in descriptor:
                index=self._derived_value(descriptor,'index')
                if index['trace_uid']!=uid or index['raw_sha256']!=digest.hexdigest() or index['raw_bytes']!=before[2]:
                    raise ValueError('index identity mismatch')
                entries={str(e['loc']):e for e in index['entries']}
                if len(entries)!=len(index['entries']):
                    raise ValueError('duplicate index loc')
                cached['index']=entries
            else:
                handle.seek(0)
                data=json.load(handle)
                events=data if isinstance(data,list) else data['traceEvents']
                cached['legacy']={}
                for i,event in enumerate(events):
                    # T7's linear resolver selected the first matching loc.
                    # New imports reject duplicates; old stores retain behavior.
                    cached['legacy'].setdefault(str(event.get('_linker_loc',i)),event)
            self._raw[uid]=cached
            return cached
        except BaseException:
            handle.close()
            raise

    def resolve_many(self, refs: Sequence[Mapping]) -> list[dict]:
        result=[]
        descriptors={uid:self._descriptor(uid) for uid in dict.fromkeys(r['trace_uid'] for r in refs)}
        sources={uid:self._open_raw(uid,d) for uid,d in descriptors.items()}
        for ref in refs:
            uid,loc=ref['trace_uid'],str(ref['loc'])
            cached=sources[uid]
            if 'legacy' in cached:
                result.append(deepcopy(cached['legacy'][loc]))
                continue
            entry=cached['index'][loc]
            start,end=entry['start'],entry['end']
            if not (type(start) is int and type(end) is int and 0<=start<end<=cached['stat'][2]):
                raise ValueError('invalid evidence byte interval')
            cached['handle'].seek(start)
            record=json.loads(cached['handle'].read(end-start))
            if str(record.get('_linker_loc',entry['ordinal']))!=loc:
                raise ValueError('evidence loc mismatch')
            result.append(record)
        for uid,d in descriptors.items():
            self._open_raw(uid,d)  # Detect writes/replacement during this batch.
        return result

    def close(self) -> None:
        if self._bounded_context is not None:
            self._bounded_context.__exit__(None,None,None)
            self._bounded_context=None;self._bounded_view=None
        for cached in self._raw.values():
            cached['handle'].close()
        self._raw.clear()
        self._derived.clear()
        self._analysis_verified.clear()
        self._closed=True

    def __enter__(self) -> StoreView:
        return self

    def __exit__(self,*args: Any) -> None:
        if self._bounded_context is not None:
            self._bounded_context.__exit__(*args)
            self._bounded_context=None;self._bounded_view=None
        self.close()
