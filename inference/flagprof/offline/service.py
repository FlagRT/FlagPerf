"""Default bounded sealed import; each retry gets a new run."""
from pathlib import Path
from contextlib import contextmanager
import fcntl
import resource
import time
import uuid
from . import pipeline
from .bundle import load_bundle, register
from .checkpoints import binding, validate, save_scan, save_prepared, load_prepared
from .finish import BoundedView, finalize_bounded
from flagprof.store.manifest import create_run, atomic_json
from flagprof.store._offline_writer import offline_writer
from flagprof.vendors.import_common import importer_writer_config


@contextmanager
def lease(work):
    work = Path(work).resolve(); work.mkdir(parents=True, exist_ok=True)
    with (work/'.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield work


def import_bundle(directory, *, store, work=None, scratch=None, run_id=None, options=None):
    from .disk import scratch_scope
    from .workspace import locations, owned_scratch, check_capture_layout
    capture = load_bundle(directory)
    work,scratch=locations(store,work,scratch,key='import-'+capture['occurrence'])
    _check_scratch(scratch,store,work)
    check_capture_layout(directory, store, work, scratch)
    with owned_scratch(scratch) as owned, scratch_scope(owned,work):
        return _import_bundle(directory,store=store,work=work,run_id=run_id,options=options)


def _check_scratch(scratch,store,work):
    from .workspace import check_layout
    check_layout(store, work, scratch)


def _import_bundle(directory, *, store, work, run_id=None, options=None):
    started = time.perf_counter(); timings = {}; options = dict(options or {})
    directory = Path(directory).resolve(); capture = load_bundle(directory)
    source = directory/'sealed-trace/data/trace.json'
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
    bind = binding(source, capture['occurrence'], options)
    # Bind all capture metadata as well as the trace to recovery stages.
    bind['capture'] = capture
    timings['validate_source_s'] = time.perf_counter()-started
    with lease(work) as work:
        # Work must be outside Store: verification snapshots must not see scratch.
        store = Path(store).resolve()
        if work == store or work.is_relative_to(store) or store.is_relative_to(work):
            raise ValueError('work and store must be separate directory trees')
        # A failed pre-commit attempt remains untouched. A small durable pointer
        # identifies the latest attempt, including a commit made before death.
        from .limits import read_json
        from .checkpoints import checked_relative
        pointer=work/'stages.json'
        stages=read_json(pointer) if pointer.exists() else dict(scan='scan',prepared='prepared')
        scan, derived = (work/checked_relative(stages[k]) for k in ('scan','prepared'))
        reused = []; t = time.perf_counter()
        if (derived/'committed.json').exists():
            prepared = load_prepared(derived, bind); reused.append('derived')
        else:
            resume = None
            if (scan/'committed.json').exists():
                validate(scan, bind, 'scan'); resume = scan/'scan.sqlite'; reused.append('scan')
            elif scan.exists():
                scan = work/('scan-' + uuid.uuid4().hex)
            if derived.exists():derived = work/('prepared-' + uuid.uuid4().hex)
            atomic_json(pointer,dict(scan=scan.name,prepared=derived.name))
            prepared = pipeline.prepare(source, derived, **options, resume_scan=resume,
                                         scan_checkpoint=lambda db: save_scan(db, scan, bind))
            save_prepared(prepared, bind)
        timings['prepare_or_restore_s'] = time.perf_counter()-t
        run = create_run(store, run_id=run_id, identity=dict(route='plan-b-final', capture_occurrence=capture['occurrence'])).resolve()
        state = dict(schema='plan-b-import/1', status='incomplete', run=str(run), capture=str(directory),
                     reused=reused, timings=timings, throughput_gate='not_passed', performance_blocking=False,
                     performance_disposition='accepted_nonblocking_by_user', latency_budget='pending')
        atomic_json(run/'plan-b-status.json', state)
        try:
            ref = register(run, directory, capture)
            from flagprof.store.lineage import register_lineage
            register_lineage(run,execution_id=capture['occurrence'],source_run_id=capture['metadata'].get('source_run_id'),
                             source_attempt_id=capture['occurrence'],attempt_id=run.name)
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
            view_work = work/('view-' + uuid.uuid4().hex); t = time.perf_counter()
            with BoundedView(run, view_work) as view:
                timings['independent_verify_s'] = time.perf_counter()-t
                t = time.perf_counter(); final = finalize_bounded(run, view)
                timings['finalize_s'] = time.perf_counter()-t
                t = time.perf_counter(); first = view.page(0, 1)
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
        return state


def inspect_run(run, work=None, *, scratch=None, offset=None, limit=100):
    from .workspace import query_view
    with query_view(run,work=work,scratch=scratch) as view:
        if offset is None:
            from flagprof.store.status import run_status
            return dict(schema='inspect/2',run=run_status(run),traces=view.inspect(),count=view.verified['count'],
                        source_records=view.verified['source_records'],throughput_gate='not_passed',performance_blocking=False)
        return dict(schema='plan-b-page/1', offset=offset, limit=limit, total=view.verified['count'], rows=view.page(offset, limit))
