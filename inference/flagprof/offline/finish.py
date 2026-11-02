# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Bounded optional finalizer and verified paged query.

One-trace smoke contract; caps writer/shard/file cardinality before Reader opens.
The production finalizer accepts explicit bounded dependencies; no global patching.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
from .bounded_verify import Document,verify_open
from .disk import Sink,streaming,scratch_path
from flagprof.store.reader import Reader

class BoundedView:
    def __init__(self,run,work,*,cache_root=None):
        self.run=Path(run);self.work=Path(work);self.work.mkdir(exist_ok=True)
        self.pagers={};self.summary_value=None;self.pager=None;self.doc=None
        from .limits import preflight_store
        from .verification_cache import VerificationCache
        preflight_store(self.run)
        self.identities=self.snapshot()
        self.cache=VerificationCache(self.run,cache_root)
        self._cache_binding=self.cache.binding()
        self.verified=self.cache.load(self._cache_binding)
        self.verification_reused=self.verified is not None
        try:
            if self.verification_reused:
                self.doc=Document(self.work/'descriptor.sqlite');paths=list((self.run/'traces').glob('*.json'))
                if len(paths)!=1:raise ValueError('bounded view requires one trace')
                root,_,_=self.doc.parse(paths[0]);self.descriptor=self.doc.get(root)
            else:
                self.verified,self.doc=verify_open(self.run,self.work/'verify.sqlite')
                self.descriptor=self.doc.get(self.doc.descriptor_root)
                self.summary_value=self.doc.get(self.doc.summary_root)
            self.check()
            if self._cache_binding is not None and self.cache.binding()!=self._cache_binding:
                raise ValueError('verification inputs changed during open')
        except BaseException:
            self.close()
            raise
    def snapshot(self):
        from .verification_cache import store_identity
        return store_identity(self.run)
    def check(self):
        if self.snapshot()!=self.identities:raise ValueError('store changed after verification')
    def traces(self,verify=True):
        self.check();d=self.descriptor;v=self.verified
        return [dict(trace_uid=d['trace_uid'],status='complete',reasons=[],sha256=d['sha256'],import_status='complete',integrity='verified',
            read_integrity='verified',read_report=deepcopy(v['reader_report']),identities=d['identities'],capture=d['capture'],vendor=d['vendor'],writer_id=d['writer_id'])]
    def summary(self):
        self.check()
        if self.summary_value is None:
            root,_,_=self.doc.parse(self.run/self.descriptor['derived']['summary']['path'])
            self.summary_value=self.doc.get(root)
        return self.summary_value
    def inspect(self):
        # Share the public metadata schema without opening full trace arrays.
        from flagprof.store.view import StoreView
        view=StoreView(self.run);view._bounded_view=self
        try:return view.inspect()
        finally:view._bounded_view=None
    def build_pages(self):
        self.check();sink=Sink(self.work/'query.jsonl')
        with Reader(self.run,retain=False) as reader:
            for stored in reader.stream():
                r=stored.record;ext=r.get('ext',{})
                if ext.get('chrome.trace_uid')!=self.descriptor['trace_uid']:continue
                kind=r.get('ktype','compute') if r['t']=='kernel' else 'memcpy' if r['t']=='memop' else 'comm' if r['t']=='comm' and r.get('phase')=='device' else None
                if kind is None:continue
                if kind not in ('compute','comm','memcpy','unknown'):kind='unknown'
                sink.row(dict(run_id=stored.run_id,writer_id=stored.writer_id,category=kind,source=dict(trace_uid=ext['chrome.trace_uid'],loc=ext['chrome.loc']),event={k:v for k,v in r.items() if k!='ext'}))
            assert reader.report['read_integrity']=='verified'
        meta=sink.close();self.check()
        self.pager=streaming.JSONLPage(sink.path,scratch_path(self.work/'page.sqlite'),meta['sha256'])
    def page(self,offset,limit):
        self.check()
        if self.pager is None:self.build_pages()
        rows=self.pager.page(offset,limit);self.check();return rows
    def derived_page(self,name,offset,limit):
        self.check()
        if name not in ('ledger','native'):raise ValueError('unsupported derived page')
        if name not in self.pagers:
            info=self.descriptor['derived'][name]
            self.pagers[name]=streaming.JSONLPage(self.run/info['path'],scratch_path(self.work/(name+'-page.sqlite')),info['sha256'])
        result=self.pagers[name].page(offset,limit);self.check();return result
    def device_analysis_context(self):
        return deepcopy(self.summary().get('device_analysis',dict(schema='device-analysis/1',status='unavailable',reason='projection_not_recorded')))
    def iter_device_events(self,*,trace_uid=None,category=None,projection='events'):
        if trace_uid is not None and trace_uid!=self.descriptor['trace_uid']:raise ValueError('unknown trace uid')
        if projection not in ('events','analysis'):raise ValueError('unknown device projection')
        if projection=='analysis':
            self.check();info=self.descriptor['derived'].get('analysis_intervals')
            if info:
                from .disk import file_rows
                for row in file_rows(self.run/info['path']):
                    if category is None or row['category']==category:yield row
            self.check();return
        offset=0;limit=1000
        while True:
            try:rows=self.page(offset,limit)
            except ValueError as error:
                if str(error)!='max_page_bytes' or limit==1:raise
                limit=max(1,limit//2);continue
            if not rows:break
            for row in rows:
                if category is None or row['category']==category:yield row
            offset+=len(rows)
    def __enter__(self):return self
    def __exit__(self,*args):
        try:
            if args[0] is None:
                self.check()
                self.cache.publish(self._cache_binding,self.verified)
        finally:self.close()
        if args[0] is None:
            import shutil
            shutil.rmtree(self.work)
    def close(self):
        if self.pager:self.pager.close()
        for pager in self.pagers.values():pager.close()
        if self.doc:self.doc.c.close()

    def _accept_finalized_manifest(self):
        # finalize owns this one mutation. Every evidence file must stay fixed.
        current=self.snapshot()
        if {k:v for k,v in current.items() if k!='manifest.json'}!={k:v for k,v in self.identities.items() if k!='manifest.json'}:
            raise ValueError('store changed during finalize')
        self.identities=current
        binding=self.cache.binding()
        if self._cache_binding is not None and binding is not None:
            before=dict(self._cache_binding,store=current)
            self._cache_binding=binding if before==binding else None


def finalize_bounded(run,view):
    from contextlib import nullcontext
    from flagprof.store import manifest
    from .limits import read_json
    def load(path):
        path=Path(path)
        if path.parent==Path(run)/'traces':view.check();return view.descriptor
        return read_json(path)
    result=manifest.finalize(run,_load_json=load,_view_factory=lambda path:nullcontext(view))
    view._accept_finalized_manifest();view.check();return result
