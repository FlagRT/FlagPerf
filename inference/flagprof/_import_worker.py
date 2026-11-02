from .store._offline_writer import offline_writer
"""Separate owner-exit handoff; never injected into the target process."""
import json
from pathlib import Path
import sys
from .store import Writer, StoreView
from .store.manifest import atomic_json, process_alive
from .vendors.import_common import import_capture, importer_writer_config


def main(run):
    run = Path(run)
    from .offline.limits import read_json
    # Bounded single-process entry; historical windowed artifacts stay on their
    # compatibility path. No trace or inventory arrays are loaded here.
    paths=[]
    for path in (run/'attachments').glob('*/state.json'):
        paths.append(path)
        if len(paths)>16:raise ValueError('attachment budget')
    states=[read_json(path) for path in paths]
    if states and all(s.get('capture_status')=='not_requested' for s in states):
        complete = all(s.get('status')=='closed' for s in states)
        atomic_json(run/'import-result.json', dict(status='complete' if complete else 'incomplete',
            capture_status='not_requested', captures=[], traces=[]))
        return 0 if complete else 2
    if any(s.get('capture_mode')=='final' for s in states):
        if len(states)!=1:raise ValueError('final handoff requires one attachment')
        state=states[0]
        from .offline.checkpoints import checked_relative
        from .offline.capture import seal_capture,finish_capture
        from .offline.service import import_bundle
        capture=run/checked_relative(state['capture_relative'])
        meta=read_json(capture/'producer.json')
        if process_alive(meta['process']) is not False:raise ValueError('producer must exit before handoff')
        meta['producer_exited_before_analysis']=True
        if not (capture/'capture-input.json').exists():seal_capture(capture,meta,state['format_vendor'])
        finish_capture(capture,timeout=1800)
        result=import_bundle(capture,store=run.parent/'analysis-runs')
        atomic_json(run/'import-result.json',result)
        return 0 if result['status']=='complete' else 2
    results = []
    with offline_writer(run, config=importer_writer_config()) as writer:
        for path in sorted((run/'attachments').glob('*/state.json')):
            state = json.loads(path.read_bytes())
            capture = Path(state['capture'])
            row = dict(attachment=state['id'], status='failed', collector_status=state.get('collector_status'))
            try:
                session = json.loads((capture/'session.json').read_bytes())
                if process_alive(session['process']) is not False:
                    row.update(status='pending_owner_exit')
                else:
                    expected = sum(len(w['validity']['traces']) for w in session['windows'] if w['state']=='valid')
                    imported = import_capture(capture, writer, vendor=state['format_vendor'])
                    row.update(expected=expected, imported=len(imported),
                               windows=[w['state'] for w in session['windows']],
                               status=('complete' if all(t['status']=='complete' for t in imported) else 'partial')
                                      if expected and len(imported)==expected else 'failed')
            except Exception as error:
                row['error'] = repr(error)
            results.append(row)
    with StoreView(run) as view:
        traces = view.traces()
    complete = bool(results) and all(r['status']=='complete' for r in results) and bool(traces) and all(t['status']=='complete' for t in traces)
    atomic_json(run/'import-result.json', dict(status='complete' if complete else 'incomplete', captures=results, traces=traces))
    return 0 if complete else 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
