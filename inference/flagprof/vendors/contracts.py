# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Versioned source-format data, independent of SDKs and vendor names."""
from __future__ import annotations
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def contract_path(module_file: str, version: str) -> Path:
    return Path(module_file).with_name('contracts') / ('v' + version + '.json')


def load_contract(module_file: str, version: str) -> dict[str, Any]:
    value = json.loads(contract_path(module_file, version).read_bytes())
    if not isinstance(value, dict) or value.get('schema_version') != 1 or value.get('format_version') != version:
        raise ValueError('unsupported source format contract')
    required = {'schema_version', 'format_version', 'revision', 'conditions', 'structural_conditions',
                'roles', 'support_scope', 'open_unknown', 'excluded_duplicate_names',
                'taxonomy_entries', 'known_unobserved_taxonomy', 'notes'}
    if set(value) != required:
        raise ValueError('missing or unknown executable contract field')
    if type(value['revision']) is not int or value['revision'] < 1:
        raise ValueError('revision must be a positive integer')
    if value['structural_conditions'] != ['device_complete_track']:
        raise ValueError('contract requires the device_complete_track structural condition')
    for rows in (value['conditions'], value['support_scope'].get('conditions')):
        if not isinstance(rows, list) or not rows:
            raise ValueError('contract conditions must be nonempty')
        identifiers = set()
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get('id'), str)
                    or row['id'] in identifiers or not isinstance(row.get('path'), str)
                    or not isinstance(row.get('allowed'), list) or not row['allowed']
                    or row.get('requirement') not in ('required', 'optional', 'conditional')):
                raise ValueError('invalid contract condition')
            identifiers.add(row['id'])
    if not isinstance(value['support_scope'].get('unchecked'), list):
        raise ValueError('support scope must declare unchecked limits')
    roles = value['roles']
    role_fields = {'device_categories', 'device_identity_signal', 'device_process_names',
                   'host_process_names', 'metadata_process_names', 'host_categories',
                   'expected_process_names', 'allowed_non_device_categories',
                   'structural_records', 'unknown_execution_process_names',
                   'communication_process_names', 'audit_device_structure'}
    if not isinstance(roles, dict) or set(roles) != role_fields:
        raise ValueError('invalid track role contract')
    for name in role_fields - {'device_identity_signal', 'audit_device_structure'}:
        if not isinstance(roles[name], list):
            raise ValueError('role collections must be lists')
    if roles['device_identity_signal'] not in ('category_or_device_stream', 'registered_process'):
        raise ValueError('unsupported device identity signal')
    if set(roles['host_categories']) & set(roles['allowed_non_device_categories']):
        raise ValueError('host categories and allowed device annotations must be disjoint')
    return value


def source_times_present(event: dict[str, Any]) -> bool:
    try:
        ts=Decimal(str(event.get('ts')));dur=Decimal(str(event.get('dur')))
        return ts.is_finite() and dur.is_finite() and dur>=0
    except (InvalidOperation,ValueError):
        return False


def closure(module_file: str, version: str) -> tuple[str, ...]:
    return (module_file,__file__,str(contract_path(module_file,version)))


def _get(value: dict, path: str) -> Any:
    for key in path.split('.'):
        if not isinstance(value,dict) or key not in value:return None
        value=value[key]
    return value


def identity_facts(identities: dict[str, Any]) -> dict[str, Any]:
    """Normalize sourced identities only; missing provenance never means a match."""
    hardware=identities.get('hardware',{});collector=identities.get('collector',{})
    h=hardware.get('facts',hardware) if hardware.get('source') else {}
    c=collector.get('facts',collector) if collector.get('source') else {}
    c=c.get('collector',c)
    configuration=collector.get('configuration',{}) if collector.get('source') else {}
    actual=configuration.get('actual',{})
    exporter=identities.get('exporter',identities.get('parser',{}).get('exporter',{}))
    if not isinstance(exporter,dict) or not exporter.get('source'):exporter={}
    observed=actual.get('values',{}) if actual.get('status')=='available' and (actual.get('source') or actual.get('sources')) else {}
    sdk={}
    sources=actual.get('sources',[])
    versions={row.get('sdk_version') for row in sources if isinstance(row.get('sdk_version'),str)}
    if sources and all(isinstance(row.get('sdk_version'),str) and row['sdk_version'] for row in sources) and len(versions)==1:
        sdk['version']=next(iter(versions))
    return dict(hardware=h,collector=c,exporter=exporter,actual=observed,sdk=sdk,
                execution=collector.get('execution',{}) if collector.get('source') else {})


def evaluate_conditions(conditions: list[dict], facts: dict) -> list[dict]:
    result=[]
    for condition in conditions:
        trigger=condition.get('when')
        value=_get(facts,condition['path']);state='unknown'
        if trigger:
            observed=_get(facts,trigger['path'])
            if observed is None:
                result.append(dict(condition,state='unknown',actual=value,reason='trigger_unknown'));continue
            if observed!=trigger['equals']:
                result.append(dict(condition,state='not_applicable',actual=value,reason='trigger_false'));continue
        if value is not None:
            state='pass' if any(type(value) is type(v) and value==v for v in condition['allowed']) else 'fail'
        result.append(dict(condition,state=state,actual=value,reason='evidence_missing' if state=='unknown' else 'value_match' if state=='pass' else 'outside_verified_values'))
    return result


def applicability(conditions: list[dict]) -> str:
    if not conditions:
        raise ValueError('cannot assess an empty condition list')
    relevant=[r for r in conditions if r.get('requirement','required')!='optional' and r['state']!='not_applicable']
    if any(r['state']=='fail' for r in relevant):return 'outside'
    if any(r['state']=='unknown' for r in relevant):return 'indeterminate'
    return 'within'


def _classify_tracks(events: list[dict], taxonomy: Any, context: Any) -> list[dict]:
    roles=taxonomy.contract['roles'];names=defaultdict(set);groups=defaultdict(list)
    for e in events:
        if e.get('name')=='process_name':names[e.get('pid')].add(e.get('args',{}).get('name'))
        groups[(e.get('pid'),e.get('tid'))].append(e)
    result=[]
    for (pid,tid),items in sorted(groups.items(),key=lambda pair:(str(pair[0][0]),str(pair[0][1]))):
        xs=[e for e in items if e.get('ph')=='X'];pn=names[pid]
        structural=next((r for r in roles.get('structural_records',[]) if xs and
            all(all(type(e.get(k)) is type(v) and e.get(k)==v for k,v in r['match'].items()) for e in xs)),None)
        declared = next((role for role, field in (('metadata','metadata_process_names'),
            ('device','device_process_names'),('host','host_process_names'))
            if pn and pn <= set(roles[field])), None)
        device_structure = (roles['device_identity_signal']=='category_or_device_stream'
                            and any(taxonomy.is_device(e,context) for e in xs))
        audit = roles['audit_device_structure']
        witnessed_conflict = (bool(xs) and declared in ('host','metadata') and bool(audit)
            and any(all(k in e.get('args',{}) for k in audit['args_required']) for e in xs))
        if not xs:role='metadata';basis='no_complete_events'
        elif device_structure:role='device';basis='declared_device_structure'
        elif witnessed_conflict:role='device';basis='witnessed_device_structure_conflict'
        elif pn and pn<=set(roles['metadata_process_names']):role='metadata';basis='derived_summary_track'
        elif pn and pn<=set(roles['device_process_names']):role='device';basis='declared_device_process'
        elif pn and pn<=set(roles['host_process_names']):role='host';basis='declared_host_process'
        elif structural:role=structural['role'];basis=structural['id']
        elif roles['host_categories'] and all(e.get('cat') in roles['host_categories'] for e in xs):role='host';basis='declared_host_categories'
        elif roles['allowed_non_device_categories'] and all(e.get('cat') in roles['allowed_non_device_categories'] for e in xs):
            role='metadata';basis='allowed_non_device_track'
        else:role='unrecognized';basis='no_declared_role'
        conflicts=[]
        if role=='device' and any(e.get('cat') in roles['host_categories'] for e in xs):
            conflicts.append('device_track_host_event')
        if (device_structure or witnessed_conflict) and declared is not None and declared!='device':
            conflicts.append('name_structure_conflict')
        result.append(dict(pid=pid,tid=tid,process_names=sorted(pn,key=str),role=role,basis=basis,
                           complete_events=len(xs),locs=[e.get('_linker_loc') for e in xs],conflicts=conflicts))
    return result


def bind_roles(events: list[dict], taxonomy: Any, context: Any) -> Any:
    """Build once per raw-source context, before normalization; no global cache."""
    context.tracks = _classify_tracks(events, taxonomy, context)
    context.role_index = {(r['pid'], r['tid']): r['role'] for r in context.tracks}
    return context


def track_roles(events: list[dict], taxonomy: Any, context: Any) -> list[dict]:
    # Return separate dictionaries: assessment may omit internal source locs.
    return [dict(r) for r in context.tracks]


def track_role(event: dict, context: Any) -> str:
    return context.role_index.get((event.get('pid'), event.get('tid')), 'unrecognized')


def device_scope(values: list, device_pids: set, host_pids: set, source: dict) -> dict:
    """Observed trace scope, not physical hardware or capture completeness."""
    known = sorted({v for v in values if type(v) is int and v >= 0})
    missing = sum(type(v) is not int or v < 0 for v in values)
    return dict(device_count=len(known) if not missing else None,
                device_ordinals=known if not missing else None,
                observed_devices=known, missing_device_identity=missing,
                host_device_pid_collision=bool(device_pids & host_pids), source=source)


def execution_tracks(devices: list, values: list, context: Any) -> dict:
    """Observed execution tracks per logical device; no one-track assumption."""
    by_device = defaultdict(set)
    for event,value in zip(devices,values):
        key = str(value) if type(value) is int and value >= 0 else 'unknown'
        if track_role(event,context) == 'device':
            by_device[key].add((event.get('pid'),event.get('tid')))
    return dict(execution_track_count=sum(t['role']=='device' and bool(t['complete_events']) for t in context.tracks),
                tracks_per_device={k:len(v) for k,v in sorted(by_device.items())})


def structure_conditions(contract: dict, tracks: list[dict]) -> list[dict]:
    present = any(t['role']=='device' and t['complete_events'] for t in tracks)
    return [dict(id=name, requirement='required', actual=present, expected=True,
                 state='pass' if present else 'fail',
                 source='raw trace track roles and complete-event counts',
                 reason='device execution track present' if present else 'no complete device execution track')
            for name in contract['structural_conditions']]


def role_conflicts(tracks: list[dict]) -> list[dict]:
    return [dict(id=kind, reason='device structure conflicts with declared host/metadata evidence',
                 pid=t['pid'], tid=t['tid'], count=t['complete_events'])
            for t in tracks for kind in t['conflicts']]


def assess(events: list[dict], identities: dict, ledger: list[dict], taxonomy: Any) -> dict[str,Any]:
    contract=taxonomy.contract;context=taxonomy.context(events)
    conditions=evaluate_conditions(contract['conditions'],identity_facts(identities))
    for row in conditions:
        row['source']={'hardware':identities.get('hardware',{}).get('source'),
            'collector':identities.get('collector',{}).get('source'),
            'actual':identities.get('collector',{}).get('configuration',{}),
            'sdk':identities.get('collector',{}).get('configuration',{}).get('actual',{}).get('sources',[]),
            'execution':identities.get('collector',{}).get('source'),
            'exporter':identities.get('exporter',identities.get('parser',{}).get('exporter',{}))}.get(row['path'].split('.')[0])
    tracks=track_roles(events,taxonomy,context)
    structural=structure_conditions(contract,tracks)
    conditions.extend(structural)
    scope=taxonomy.scope_facts(events,context)
    scope_conditions=evaluate_conditions(contract['support_scope']['conditions'],dict(scope=scope))
    for row in scope_conditions:row['source']=scope['source']
    conditions.extend(scope_conditions)
    failures=taxonomy.audit_dependencies(events,context)+role_conflicts(tracks)
    groups={};byloc={str(e['_linker_loc']):e for e in events}
    fmt=taxonomy.identity()
    def add(dimension,reason,event):
        label=taxonomy.task_type(event) or event.get('cat') or '<absent>'
        key=(dimension,str(label),reason)
        group=groups.setdefault(key,dict(dimension=dimension,label=label,reason=reason,format=fmt,count=0,known_time_ns=0,dur_missing=0,examples=[]))
        group['count']+=1
        try:
            duration=Decimal(str(event.get('dur')))
            if not duration.is_finite() or duration<0:raise ValueError('invalid duration')
            group['known_time_ns']+=int(duration*1000)
        except (InvalidOperation,ValueError):group['dur_missing']+=1
        if len(group['examples'])<3:group['examples'].append(dict(loc=event['_linker_loc']))
    for track in tracks:
        if track['role']=='unrecognized' and track['complete_events']:
            failures.append(dict(id='unrecognized_track',reason='unrecognized track contains complete events',pid=track['pid'],tid=track['tid'],count=track['complete_events']))
            for loc in sorted(track['locs'],key=lambda x:(type(x).__name__,str(x))):add('track','unrecognized_track',byloc[str(loc)])
    for row in ledger:
        event=byloc[str(row['loc'])]
        if row['category']=='unknown':add('category','unknown_category',event)
        if row['disposition']=='unknown':add('disposition',row['reason'] or 'unknown_disposition',event)
    for failure in failures:
        conditions.append(dict(id=failure['id'],requirement='required',state='fail',reason=failure['reason'],evidence=failure))
    warnings=list(failures)+[dict(id=r['id'],reason=r['reason'],actual=r['actual'])
                            for r in structural+scope_conditions if r['state']=='fail']
    for key in sorted(groups):
        group=groups[key]
        if group['dimension']=='category':
            warnings.append(dict(id='unknown_category',label=group['label'],count=group['count'],
                                 known_time_ns=group['known_time_ns'],dur_missing=group['dur_missing']))
    for track in tracks:track.pop('locs')
    return dict(format_version=1,format=fmt,status=applicability(conditions),conditions=conditions,tracks=tracks,
                warnings=sorted(warnings,key=lambda r:json.dumps(r,sort_keys=True)),
                unknown_groups=[groups[k] for k in sorted(groups)],
                support_scope=dict(observed=scope,unchecked=contract['support_scope']['unchecked']),
                time_semantics='sum of source activity durations; overlapping; not wall-clock',
                configuration=identities.get('collector',{}).get('configuration',dict(status='unavailable',reason='not_recorded')))
