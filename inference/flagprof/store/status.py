"""Run status from recorded facts and the shared live capture inspector."""
from pathlib import Path
import signal
from .captures import inspect_capture, capture_summary, log_counts
from .manifest import load_json, writer_fragments, process_alive


def validate_run(run):
    run=Path(run)
    try:
        manifest=load_json(run/'manifest.json')
        if not isinstance(manifest,dict) or not manifest.get('run_id'):
            raise ValueError('manifest must be an object with run_id')
        return manifest
    except (OSError,ValueError,TypeError) as error:
        hint='; looks like a store root, select a run directory' if run.is_dir() and any(run.glob('*/manifest.json')) else ''
        raise ValueError('not a run directory: '+str(run)+hint+'; '+str(error)) from error


def _exit(code):
    value=dict(returncode=code, status='available' if code is not None else 'unavailable')
    if isinstance(code,int) and code<0:
        try: value['signal']=signal.Signals(-code).name
        except ValueError: value['signal']='unknown'
    return value


def run_status(run):
    run=Path(run)
    manifest=validate_run(run)
    from ._proxy_lifecycle import attachments, registered_proxies, effective_writer_state
    states=attachments(run)
    proxies=registered_proxies(run, states)
    collection=manifest.get('collection',{})
    finalized='captures' in collection or manifest.get('status')!='running'
    live=not finalized and ((run/'attachments').exists() or (run/'capture-staging').exists())
    captures=collection.get('captures',[]) if finalized else []
    writers=manifest.get('writers',{}) if finalized else {}
    if live:
        for path in sorted((run/'attachments').glob('*/state.json')):
            try:
                attachment=load_json(path)
                if attachment.get('capture_status') == 'not_requested':
                    captures.append(dict(status='not_requested', label='未请求采集', windows=[]))
                else:
                    captures.append(capture_summary(inspect_capture(attachment['capture'])))
            except (OSError,ValueError,KeyError,TypeError) as error:
                captures.append(dict(status='unavailable', reason=str(error)))
        for fragment in writer_fragments(run):
            state=effective_writer_state(fragment, proxies)
            writers[str(fragment['writer_id'])]=dict(state=state, dropped=fragment.get('dropped',{}), tail_unknown=state!='closed')
    record={}
    record_error=None
    try: record=load_json(run/'record-status.json')
    except FileNotFoundError: pass
    except (OSError,ValueError) as error: record_error=str(error)
    result=dict(run_id=manifest['run_id'], state='finalized' if finalized else 'not_finalized' if live else 'unavailable',
                manifest_status=manifest.get('status','unavailable'), writers=writers, captures=captures,
                log_check=log_counts(captures), record_status=record.get('status','unavailable'),
                record_error=record_error,
                exits=dict(target=_exit(record.get('target_exit',{}).get('returncode')),
                           importer=_exit(record.get('importer_exit')), record=_exit(record.get('exit_code'))),
                capture_loss=dict(count=None,count_semantics='unknown',reason='collector_loss_not_certified'))

    if captures and all(c.get('status')=='not_requested' for c in captures):
        result['capture_loss'] = dict(count=0, count_semantics='not_requested', reason='not_requested')
    if proxies:
        result['writer_proxies']=[dict(id=s['id'], writer_mode=s.get('writer_mode'),
            writer_proxy=s.get('writer_proxy'), writer_failure=s.get('writer_failure')) for s in states if s.get('writer_proxy') or s.get('writer_failure')]
    elif any(s.get('writer_failure') for s in states):
        result['writer_proxies']=[dict(id=s['id'],writer_mode=s.get('writer_mode'),writer_failure=s['writer_failure']) for s in states if s.get('writer_failure')]
    return result
