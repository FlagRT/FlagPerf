# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Evidence from launch manifests, per-rank component identities and native reports."""
import json
from pathlib import Path

from flagprof.report.group import validate_manifest
from flagprof.components.flaggems import gems_identity
from flagprof.report.performance import runner_rank
from flagprof.analyzers.workspace import AnalysisSet


def store_states(path):
    from flagprof.store import StoreView
    with StoreView(path) as view:
        source=AnalysisSet.open(view)
        try:
            return [s for member in source.views() for s in member.probe_status()['attachments']]
        finally:
            source.close()


def launch_result(plan, code):
    root=Path(plan['directory'])/'store'
    path=root/'group.json'
    group=validate_manifest(path)
    expect='captured' if plan['output_type']=='capture-group' else 'complete'
    if not code and group['status']!=expect:
        raise ValueError('launch 状态与预期不符 '+group['status'])
    stores, artifacts={},[dict(type='group',path=str(path))]
    for row in group['ranks']:
        rank=row['rank']
        if not code and row.get('status')!='complete':
            raise ValueError(f'launch rank {rank} 未完成')
        for field,kind in [('run','store'),('capture','capture')]:
            value=row.get(field)
            if not value:
                required=plan['output_type']==('group' if field=='run' else 'capture-group')
                if not code and required:
                    raise ValueError(f'launch rank {rank} 缺少 {kind}')
                continue
            dest=(root/value).resolve()
            if Path(value).is_absolute() or root not in dest.parents or not dest.is_dir():
                raise ValueError(f'launch rank {rank} {field} 路径无效 {value}')
            if field=='run':
                if dest in stores.values() or not (dest/'manifest.json').is_file():
                    raise ValueError(f'launch rank {rank} Store 不独立或缺失')
                stores[rank]=dest
            if field=='capture' and not (dest/'capture-input.json').is_file():
                if not code:
                    raise ValueError(f'launch rank {rank} capture 缺少封存回执')
                kind='capture-attempt'
            artifacts.append(dict(type=kind,path=str(dest),rank=rank))
    return group,stores,artifacts


def component_evidence(config, plan, role, stores, *, required):
    """Compare actual values with the experiment request before consuming its results."""
    components=config['components']
    gems=components['flaggems']
    requested='on' if gems['enabled'] else 'off'
    include=gems.get('include')
    include=json.loads(Path(include).read_text()) if isinstance(include,str) else include
    evidence=[]
    def observe(rank,path,identity,component,compiler,effective):
        switch,sha=gems_identity(identity.get('flaggems'),component)
        recorded_include=identity.get('include')
        settings=effective or {}
        if switch is None:
            switch=settings.get('flaggems')
        if recorded_include is None:
            recorded_include=settings.get('flaggems_include')
        reasons=[]
        def verify(field,want,actual):
            if actual!=want:
                reasons.append(f'{role} rank {rank} {field}: 请求 {want!r}, 观察 {actual!r}')
        if switch is not None or required and (gems['enabled'] or config['target']['type']=='builtin'):
            verify('flaggems',requested,switch)
        if switch is not None:
            verify('include',include or [],recorded_include or [])
        if settings.get('flaggems') is not None:
            verify('effective_config.flaggems',requested,settings['flaggems'])
        if settings.get('flaggems_include') is not None:
            verify('effective_config.flaggems_include',include or [],settings['flaggems_include'])
        compiler=compiler or identity.get('compiler') or {}
        if components['flagtree']!='inherit':
            verify('compiler','flagtree' if components['flagtree']=='on' else 'vendor',compiler.get('provider'))
        if components.get('vendor_compiler') and settings.get('vendor_compiler'):
            verify('vendor_compiler',components['vendor_compiler'],settings['vendor_compiler'])
        backend=config['hardware'].get('comm_backend')
        if config['hardware']['nproc']==2 and identity and (identity.get('comm_backend') or config['target']['type']=='builtin'):
            verify('comm_backend',backend,identity.get('comm_backend'))
        evidence.append(dict(subject='components',rank=rank,status='mismatch' if reasons else 'observed' if switch is not None else 'unavailable',
                    source=str(path),reasons=reasons,requested=dict(flaggems=requested,include=include,flagtree=components['flagtree'],
                                  vendor_compiler=components.get('vendor_compiler'),comm_backend=backend),
                    observed=dict(flaggems=switch,include=recorded_include,source_sha256=sha,compiler=compiler,
                                  communication=identity.get('communication'),comm_backend=identity.get('comm_backend'))))
    for rank, name in enumerate(plan['runners']):
        path=Path(name)
        if not path.is_file():
            continue
        value=json.loads(path.read_bytes())
        identity=value.get('oracle_identity') or {}
        if config['hardware']['nproc']==2 and runner_rank(value,identity,required=True)!=rank:
            raise ValueError(f'{role} rank {rank} runner rank identity mismatch')
        observe(rank,path,identity,value.get('component_identity'),None,value.get('effective_config'))
    for rank, path in stores.items():
        for state in store_states(path):
            identity=state.get('oracle_identity') or {}
            if state.get('effective_config') or identity:
                observe(rank,path,identity,state.get('flaggems_identity'),state.get('compiler_identity'),state.get('effective_config'))
    if required and (gems['enabled'] or components['flagtree']!='inherit' or config['target']['type']=='builtin'):
        seen={row['rank'] for row in evidence}
        for rank in sorted(set(range(config['hardware']['nproc']))-seen):
            reason=f'{role} rank {rank} 缺少组件身份，无法核对请求 flaggems={requested}'
            evidence.append(dict(subject='components',rank=rank,status='mismatch',source=plan['directory'],reasons=[reason]))
    return evidence


def report_rows(doc, path):
    rows=[]
    def add(subject,status,reasons=(),**details):
        rows.append(dict(subject=subject,status=status,reasons=list(reasons or []),source=str(path),**details))
    for key, panel in doc.get('results',{}).items():
        add(key,panel['status'],panel.get('notes'))
    for group in (doc.get('runner_summary') or {}).get('groups',[]):
        add('runner.'+group['scope'],group['status'],
            **{k:group.get(k) for k in ('rank','phase','shape','observations','mean_ns','median_ns','p95_ns')})
    if 'status' in doc:
        add(doc.get('schema','report'),doc['status'],doc.get('issues'))
    for pair in doc.get('pair_statuses',[]):
        add('oracle.'+pair['pair'],pair['status'],rank=pair.get('rank'))
    if doc.get('schema')=='device-comparison/1':
        for pair in doc['comparisons']:
            add('performance.comparison',pair['status'],pair.get('reasons'),left=pair['left'],right=pair['right'],
                differences=pair.get('differences',[]))
        # Preserve values, units and unavailable deltas at their native result path.
        for status,count in doc['metric_status_counts'].items():
            missing=doc['missing_deltas'][status]
            deltas={'delta':None,'relative_change':None} if count==missing else {}
            add('performance.metrics',status,metrics=count,missing_deltas=missing,table='metric_comparisons',**deltas)
    return rows
