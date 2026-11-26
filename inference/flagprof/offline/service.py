# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Bounded sealed imports with validated checkpoints and atomic run publication."""
from pathlib import Path
import hashlib
import os
import shutil
import sqlite3
import resource
import time
from . import pipeline
from .bundle import load_bundle, register
from .checkpoints import binding, validate, save_scan, save_prepared, load_prepared
from .finish import BoundedView, finalize_bounded
from .limits import read_json
from flagprof.progress import phase
from flagprof.store.manifest import create_run, atomic_json
from flagprof.store._offline_writer import offline_writer
from flagprof.vendors.import_common import importer_writer_config


def import_bundle(directory, *, store, run_id=None, options=None, calls_from_run=None):
    from .worker import on_disk
    directory, store = Path(directory).resolve(), Path(store).resolve()
    if store.is_relative_to(directory):
        raise ValueError('import output must be outside the sealed capture')
    return on_disk(store, _import_bundle, directory, store, run_id, options, calls_from_run)


def _import_bundle(directory, store, run_id, options, calls_from_run):
    from .disk import scratch_scope
    from .worker import active_directory
    from .workspace import checkpoint, private_name
    from flagprof.store.manifest import json_bytes
    directory, store = Path(directory), Path(store)
    temporary = active_directory()
    started = time.perf_counter(); timings = {}; options = dict(options or {})
    with phase("验证封存采集与导入身份"):
        capture = load_bundle(directory)
        source = directory/'sealed-trace/data/trace.json'
        if calls_from_run is not None:
            from .project import calls_from
            options.update(calls=calls_from(calls_from_run, capture['metadata'].get('source_run_id')), marker_prefix='flagprof::op#')
        if 'calls' not in options and capture['metadata'].get('collection_run'):
            from .checkpoints import checked_relative
            from .project import calls_from
            collection=directory/checked_relative(capture['metadata']['collection_run'])
            options['calls']=calls_from(collection,capture['metadata'].get('source_run_id'))
            options['marker_prefix']='flagprof::op#'
        if 'identities' not in options and capture['metadata'].get('identities'):
            options['identities'] = capture['metadata']['identities']
        from .device_analysis import capture_context
        options['analysis_input'] = capture_context(directory,capture)
        options['vendor'] = capture['vendor']
        options['profile_memory'] = capture['metadata'].get('requested_configuration',{}).get('profile_memory',False)
        from flagprof.vendors.native_common import ScopeConfig
        scope = options.get('scope_config')
        if scope is not None:
            options['scope_config'] = dict(scope)
        bind = binding(source, capture['occurrence'], options)
        bind['capture'] = capture
        key = hashlib.sha256(json_bytes(bind)).hexdigest()
        if scope is not None:
            options = dict(options, scope_config=ScopeConfig(**{k:tuple(v) if v is not None else None for k,v in scope.items()}))
    timings['validate_source_s'] = time.perf_counter()-started
    with checkpoint(store, key) as work, scratch_scope(temporary, work):
        reused = []
        t = time.perf_counter()
        derived, scan = work / 'prepared', work / 'scan'
        prepared = None
        if derived.is_dir():
            try:
                with phase("验证并读取已准备的导入检查点"):
                    prepared = load_prepared(derived, bind)
                reused.append('derived')
            except (ValueError, OSError, KeyError, TypeError, AttributeError, sqlite3.DatabaseError):
                shutil.rmtree(derived)
        if prepared is None:
            resume = None
            if scan.is_dir():
                try:
                    with phase('验证扫描检查点'):
                        validate(scan, bind, 'scan')
                    resume = scan / 'scan.sqlite'
                    reused.append('scan')
                except (ValueError, OSError, KeyError, TypeError, AttributeError, sqlite3.DatabaseError):
                    shutil.rmtree(scan)
            try:
                prepared = pipeline.prepare(source, derived, **options, resume_scan=resume,
                                            scan_checkpoint=lambda db: save_scan(db, scan, bind))
                with phase('保存已准备的导入检查点'):
                    save_prepared(prepared, bind)
            except BaseException:
                for stage in (derived, scan):
                    if stage.is_dir() and not (stage/'committed.json').is_file():
                        shutil.rmtree(stage)
                raise
        timings['prepare_or_restore_s'] = time.perf_counter()-t
        if run_id and (run_id == private_name() or run_id.startswith('.flagprof-import-')):
            raise ValueError('reserved run id')
        initial = create_run(temporary/'output', run_id=run_id, identity=dict(route='plan-b-final', capture_occurrence=capture['occurrence']))
        final_run = store / initial.name
        if final_run.exists():
            receipt = final_run/'plan-b-status.json'
            saved = read_json(receipt) if receipt.is_file() else {}
            if saved.get('status') != 'complete' or saved.get('import_key') != key:
                raise FileExistsError('run id belongs to another import: ' + str(final_run))
            with BoundedView(final_run, temporary/'existing') as view:
                if view.verified['status'] != 'complete':
                    raise ValueError('published run failed verification')
            return dict(saved, run=str(final_run), capture=str(directory))
        run = store / ('.flagprof-import-' + temporary.name)
        (temporary / 'staged-run').touch()
        initial.rename(run)
        from flagprof.monitor.reference import relocate
        relocate(directory/'monitor-ref.json', run/'monitor-ref.json')
        state = dict(schema='plan-b-import/1', import_key=key, status='incomplete', run=final_run.name, capture=os.path.relpath(directory, final_run),
                     reused=reused, timings=timings, throughput_gate='not_evaluated', performance_blocking=False,
                     performance_disposition='not_evaluated', latency_budget='pending')
        atomic_json(run/'plan-b-status.json', state)
        try:
            with phase('登记封存证据'):
                ref = register(run, directory, capture)
            from flagprof.store.lineage import register_lineage
            register_lineage(run,execution_id=capture['occurrence'],source_run_id=capture['metadata'].get('source_run_id'),
                             source_attempt_id=capture['occurrence'],attempt_id=final_run.name)
            t = time.perf_counter()
            with offline_writer(run, config=importer_writer_config()) as writer:
                imported = pipeline.import_prepared(source, writer, prepared, vendor=capture['vendor'],
                                                     registry=options.get('registry'), calls_supplied='calls' in options, capture=ref)
                from .memory import import_memory,finish_memory
                memory_started = time.perf_counter()
                memory = import_memory(directory, run, capture, writer=writer, prepared=prepared)
            timings['import_close_s'] = time.perf_counter()-t
            if imported['status'] != 'complete':raise ValueError('partial Store import')
            t = time.perf_counter()
            state['memory'] = finish_memory(run, memory, verify=False)
            timings['memory_import_verify_s'] = time.perf_counter()-memory_started
            view_work = temporary / 'verification'; t = time.perf_counter()
            with BoundedView(run, view_work) as view:
                timings['independent_verify_s'] = time.perf_counter()-t
                t = time.perf_counter()
                with phase('封存导入结果'):
                    final = finalize_bounded(run, view)
                timings['finalize_s'] = time.perf_counter()-t
                t = time.perf_counter()
                with phase('核验导入后第一页'):
                    first = view.page(0, 1)
                timings['first_page_s'] = time.perf_counter()-t
                if final['status'] != 'complete':raise ValueError('finalize incomplete')
                state.update(status='complete', count=view.verified['count'], source_records=view.verified['source_records'],
                             finalize=final['status'], first_page_count=len(first),
                             capture_delivery='sealed_after_normal_close',capture_loss='unknown',
                             format_applicability=view.summary().get('format_assessment',{}).get('status','unknown'))
        except BaseException as error:
            state['error'] = type(error).__name__ + ': ' + str(error)
            raise
        finally:
            timings['total_s'] = time.perf_counter()-started
            state['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            atomic_json(run/'plan-b-status.json', state)
        from .workspace import publish_run
        publish_run(run, final_run)
        return dict(state, run=str(final_run), capture=str(directory))


def inspect_run(run, *, offset=None, limit=100):
    from .worker import on_disk
    return on_disk(None, _inspect_run, run, offset, limit)


def _inspect_run(run, offset, limit):
    from .workspace import query_view
    with query_view(run) as view:
        if offset is None:
            from flagprof.store.status import run_status
            return dict(schema='inspect/2',run=run_status(run),traces=view.inspect(),count=view.verified['count'],
                        source_records=view.verified['source_records'],throughput_gate='not_evaluated',performance_blocking=False)
        return dict(schema='plan-b-page/1', offset=offset, limit=limit, total=view.verified['count'], rows=view.page(offset, limit))
