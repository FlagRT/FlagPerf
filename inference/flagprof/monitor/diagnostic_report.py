"""Public task/rank diagnosis and resolvable evidence, independent of vendor SDKs."""
import hashlib
import json
from pathlib import Path
from .reading import status


def selection(run):
    run=Path(run);state=status(run);reference=state['reference']
    identity=reference.get('diagnostic_identity',{})
    process=reference.get('worker_proc_key');target=identity.get('run_id');group=None
    if (run/'group.json').exists():group=json.loads((run/'group.json').read_text())['group_id']
    elif (run/'manifest.json').exists() and not process:
        from flagprof.store.lineage import read_lineage
        target=read_lineage(run).get('source_run_id') or json.loads((run/'manifest.json').read_text()).get('run_id')
    def matches(row):
        return ((not target or row.get('run_id')==target) and (not process or row.get('proc_key')==process)
                and (not group or row.get('group_id')==group))
    return state,matches


def findings(run):
    state,matches=selection(run);root=state['path']
    if root is None:return
    for path in sorted((root/'findings').glob('*/*.json')):
        row=json.loads(path.read_text())
        if row.get('schema')!='business-finding/1' and not (row.get('rule')=='proc.exit' and row.get('run_id')):continue
        if matches(row):yield row


def coverage(run):
    """Observed progress coverage, counted by heartbeat rather than wall time."""
    from flagprof.store import StoreView
    state,matches=selection(run);root=state['path']
    if root is None:return
    rows={}
    with StoreView(root) as view:
        for event in view.iter_monitor_events(('health',)):
            detail=event.get('detail',{})
            if detail.get('scope')=='business_tick':
                for key,health in detail['sources'].items():
                    if key in rows and health!='ready':rows[key]['source_gap']+=1
                continue
            if detail.get('scope')!='business_input':continue
            source=detail['record'];data=source['detail'];identity=data['identity']
            if not matches(identity):continue
            row=rows.setdefault(identity['run_id'],dict(run_id=identity['run_id'],rank=identity['rank'],
                registered_execution=0,load_or_warmup_excluded=0,no_active_interval=0,source_gap=0,probe_gap=0))
            if data['action']=='gap':row['probe_gap']+=1
            if data['action']!='heartbeat':continue
            if detail.get('source_status','ready')!='ready':row['source_gap']+=1
            elif any(r['eligible'] for r in data['active'].values()):row['registered_execution']+=1
            elif data['active']:row['load_or_warmup_excluded']+=1
            else:row['no_active_interval']+=1
    yield from rows.values()


def evidence(run, identities):
    from flagprof.store import StoreView
    root=status(run)['path']
    if root is None:return
    pending={tuple(i) for i in identities}
    # Read the full selected journal: evidence may precede the target report window.
    with StoreView(root) as view:
        for event in view.iter_monitor_events(('health','counter')):
            identity=tuple(event['evidence_id'])
            if identity in pending:
                yield dict(identity=list(identity),record=event);pending.remove(identity)
            detail=event.get('detail',{})
            if detail.get('scope')=='business_input':
                identity=tuple(detail['source_identity'])
                if identity in pending:
                    yield dict(identity=list(identity),record=detail['record']);pending.remove(identity)
            if not pending:break


def panel(run):
    from flagprof.store import StoreView
    try:
        with StoreView(run) as view:
            monitor=view.monitor_status()
            if monitor['path'] is None:
                return dict(status='unavailable',findings=[],evidence=[],coverage=view.monitor_coverage(),
                    reason='diagnostic monitor unavailable: '+str(monitor['reference'].get('errors',[])))
            rows=list(view.iter_diagnostic_findings())
            refs={tuple(ref) for row in rows for ref in row['evidence']}
            records=list(view.iter_diagnostic_evidence(refs)) if refs else []
            missing=refs-{tuple(row['identity']) for row in records}
            coverage=view.monitor_coverage()
            progress=list(view.iter_diagnostic_coverage())
            rule_settings={}
            session=Path(monitor['path'])/'session.json'
            if session.is_file():
                saved=json.loads(session.read_text());config=saved.get('config',{});inventory=saved.get('inventory',{})
                if config.get('diagnosis') is not None:
                    from .rules import Engine
                    from .business import DEFAULTS
                    config_id=Engine(inventory,config).config_id
                    if any(row.get('config_version')==config_id for row in rows):
                        rule_settings[config_id]=dict(DEFAULTS,**config['diagnosis'])
        return dict(status='partial' if missing else 'available',findings=rows,evidence=records,
                    missing_evidence=[list(ref) for ref in sorted(missing)],coverage=coverage,progress_coverage=progress,rule_settings=rule_settings)
    except (OSError,ValueError,KeyError) as error:
        return dict(status='unavailable',findings=[],evidence=[],reason=str(error))


def anchor(identity):
    return 'diagnostic-'+hashlib.sha256(json.dumps(list(identity)).encode()).hexdigest()[:20]
