"""Explicit project calls and same-capture identity binding."""
import hashlib
import json
from pathlib import Path
from flagprof.store import StoreView
from flagprof.store.manifest import load_json
from .checkpoints import fileinfo


def calls_from(run, expected_source_run):
    run = Path(run)
    manifest = load_json(run/'manifest.json')
    if not expected_source_run or manifest['run_id'] != expected_source_run:
        raise ValueError('collection run_id must match capture source_run_id')
    result = []
    size = 0
    with StoreView(run) as view:
        for stored in view.iter_events(types=('op',)):
            r = stored.record
            if r.get('parent_corr'):
                continue
            row = dict(marker='flagprof::op#'+str(r['corr']), corr=r['corr'])
            size += len(json.dumps(row).encode())
            if size > 4 << 20:
                raise ValueError('project calls metadata budget exceeded')
            result.append(row)
    return result


def identity_from(source, identity_path):
    """Prove trace membership before accepting an otherwise valid identity file."""
    source, identity_path = Path(source).resolve(), Path(identity_path).resolve()
    selected = fileinfo(source)
    if identity_path.is_dir():
        from .bundle import load_bundle
        capture = load_bundle(identity_path)
        registered_trace=identity_path/'sealed-trace/data/trace.json'
        if not source.samefile(registered_trace):
            raise ValueError('selected trace is not the registered file of this capture instance')
        if selected != {k:capture['trace'][k] for k in ('bytes','sha256')}:
            raise ValueError('identity trace does not belong to selected capture')
        metadata = capture['metadata']
        identities = metadata.get('identities')
        if identities is None:
            raise ValueError('capture identities unavailable')
        return dict(identities=identities, identity_binding=dict(
            kind='final_capture', occurrence=capture['occurrence'], source_run_id=metadata.get('source_run_id'),
            trace=selected, source=fileinfo(identity_path/'capture.json')))
    state = load_json(identity_path)
    run=identity_path.parent.parent.parent
    relative=state.get('capture_relative')
    if relative:
        from .checkpoints import checked_relative
        capture_path=run/checked_relative(relative)
    else:
        capture_path = Path(state['capture'])
    from flagprof.store.captures import inspect_capture
    capture = inspect_capture(capture_path, owner_alive=False)
    matches = []
    for window in capture.get('windows', []):
        for trace in window.get('validity', {}).get('traces', []):
            # Follow only registered window outputs, never neighbor-name guesses.
            path = capture_path/window['path']/trace['path'] if isinstance(trace, dict) else capture_path/window['path']/trace
            if path.is_file() and source.samefile(path) and fileinfo(path) == selected:
                matches.append((window, path))
    if len(matches) != 1:
        raise ValueError('selected trace has no unique registered window binding')
    window, path = matches[0]
    identities = window.get('identities') or capture.get('identities')
    if identities is None:
        # Window identity assembly is provided by the existing importer helper.
        from flagprof.vendors.evidence import capture_identities
        identities = capture_identities(capture_path,capture,window,
            matches[0][1].relative_to(capture_path/ window['path']).as_posix(),
            str((capture_path/'session.json').relative_to(identity_path.parent.parent.parent)),
            state['format_vendor'])
    return dict(identities=identities, identity_binding=dict(kind='window_capture',
        attachment=state['id'], process=state['process'], source_run_id=load_json(run/'manifest.json')['run_id'],
        trace=selected, source=fileinfo(identity_path)))
