# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Checkpointed, conservative cumulative discovery with a single run deadline."""
from copy import deepcopy
from pathlib import Path
import math
import shutil
import time
import yaml
from runtime.common import (read_json, write_json, digest, file_hash,
                            assert_source_snapshot, identity_differences)

# Calibration changes these constants only between sealed source versions.
BUDGET_MODE = 'fixed'
COST_MARGIN = 1.5
CHECKPOINT_SCHEMA = 1


def usable(result, require_hit=None):
    if result.get('status') != 'completed' or result.get('numerical_anomalies'):
        return False
    route = result.get('route', {})
    ranks = route.get('rank_function_calls', {})
    if result.get('parallelism') == 'tp':
        expected = {str(i) for i in range(result.get('world_size', len(result.get('ranks', {}))))}
        if not expected or set(ranks) != expected:
            return False
    if require_hit is None:
        return True
    return (route.get('actual_function_calls', {}).get(require_hit, 0) > 0 and
            all(c.get(require_hit, 0) > 0 for c in ranks.values()))


def normalized_candidates(candidates):
    out = []
    for row in candidates:
        out.append({'function': row['function'],
                    'aten_keys': sorted(set(row.get('aten_keys', []))),
                    'signatures': sorted({digest(s):s for s in row.get('signatures', [])}.values(), key=digest)})
    names = [r['function'] for r in out]
    if len(names) != len(set(names)):
        raise ValueError('duplicate preview candidate')
    return out


def checked_path(root, name):
    p = (root/name).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError('missing or external required resume evidence: ' + str(p))
    return p


def load_resume(source):
    source = Path(source).resolve()
    p = source/'preview/checkpoint.json'
    if not p.is_file():
        raise ValueError('no resumable checkpoint; schema 1/2 results need a fresh preview')
    envelope = read_json(p)
    state = envelope.get('state', {})
    if envelope.get('schema_version') != CHECKPOINT_SCHEMA or envelope.get('sha256') != digest(state):
        raise ValueError('invalid preview checkpoint schema or digest')
    if state.get('policy_schema') != 3 or not state.get('profiles'):
        raise ValueError('resume requires schema 3 preview progress')
    for name, expected in {**state['prepared_files'], **state['evidence_files']}.items():
        if file_hash(checked_path(source, name)) != expected:
            raise ValueError('resume evidence digest mismatch: '+name)
    for key, row in state['profiles'].items():
        if digest(row['identity']) != key or digest(row['candidates']) != row['candidate_digest']:
            raise ValueError('checkpoint identity or candidate digest mismatch')
        accepted = row['accepted']
        names = {r['function'] for r in row['candidates'] or []}
        if len(accepted) != len(set(accepted)) or not set(accepted) <= names:
            raise ValueError('invalid checkpoint accepted selection')
    return state


def import_resume(source, root, state):
    """Copy required small records. Tensor/trace provenance remains explicitly external."""
    source = Path(source).resolve()
    if digest(load_resume(source)) != digest(state):
        raise ValueError('resume checkpoint changed during import')
    target = root/'resume-input'; target.mkdir()
    prepared = Path(state['input_prepared'])
    for name in ['inputs.pt', 'samples.json']:
        relative = (prepared/name).as_posix()
        if relative not in state['prepared_files']:
            raise ValueError('resume archive is not sealed: '+relative)
        shutil.copyfile(checked_path(source, relative), target/name)
        if file_hash(target/name) != state['prepared_files'][relative]:
            raise ValueError('resume archive changed during copy')
    for name in state['evidence_files']:
        dest = root/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked_path(source, name), dest)
        if file_hash(dest) != state['evidence_files'][name]:
            raise ValueError('resume evidence changed during copy')
    write_json(root/'resume-source.json', {'source':str(source),
        'checkpoint_state_sha256':digest(state),
        'large_artifacts':'source references; this is not a standalone backup'})
    return target


class Budget:
    def __init__(self, seconds, timeout, profiles, mode=None, clock=None):
        self.clock = clock or time.monotonic
        self.started = self.clock(); self.seconds = seconds; self.timeout = timeout
        self.profiles = profiles; self.mode = mode or BUDGET_MODE
        self.cleanup = 0.0

    def remaining(self):
        return max(0.0, self.seconds - (self.clock()-self.started-self.cleanup))

    def estimate(self, key, phase, candidate=None):
        history = self.profiles[key]['timings']
        good = [r['seconds'] for r in history if r['phase'] == phase and r['completed'] and not r['timed_out']]
        if not good:
            good = [r['seconds'] for r in history if r['completed'] and not r['timed_out']]
        estimate = COST_MARGIN * max(good[-8:]) if good else max(1.0, self.seconds/4/max(1,len(self.profiles)))
        lower = [r['allowance'] for r in history if r['timed_out'] and
                 ((candidate is not None and r.get('candidate') == candidate) or
                  (candidate is None and r['phase'] == phase))]
        if lower: estimate = max(estimate, COST_MARGIN*max(lower))
        return min(self.timeout, max(1.0, estimate))

    def reserve(self):
        if self.mode == 'fixed' or any(not any(t['completed'] and not t['timed_out'] for t in p['timings']) for p in self.profiles.values()):
            return min(self.seconds*0.25, 2*len(self.profiles)*self.timeout)
        # Discovery may change the intersection. Reserve at most two workers per environment.
        return sum(2*self.estimate(k, 'final') for k in self.profiles)

    def room(self):
        return max(0.0, self.remaining()-self.reserve())


def run_profiles(cfg, root, contexts, invoke, progress, restored=None):
    unique = {c['key']:c for c in contexts.values()}
    root = Path(root); folder = root/'preview'; folder.mkdir(parents=True, exist_ok=True)
    request = {k:cfg['preview'].get(k,'off') for k in ['flaggems','flagtree','flagcx']}
    if restored and restored.get('environment_request') != request:
        raise ValueError('resume environment combination changed; run fresh preview')
    if restored and set(restored['profiles']) != set(unique):
        changes = {k:[identity_differences(p['identity'],read_json(c['prepared']/'identity.json'))
                     for p in restored['profiles'].values()] for k,c in unique.items()}
        write_json(root/'resume-mismatch.json', changes)
        raise ValueError('resume execution identity/environment mismatch; see resume-mismatch.json; run fresh preview')
    profiles = deepcopy(restored['profiles']) if restored else {}
    generation = restored['generation']+1 if restored else 0
    for index,(key,ctx) in enumerate(unique.items()):
        if key not in profiles:
            profiles[key] = {'identity':read_json(ctx['prepared']/'identity.json'), 'candidates':None,
                'candidate_digest':digest(None), 'accepted':[], 'excluded':{}, 'unknown':{},
                'attempted':[], 'inflight':None, 'timings':[], 'decisions':[], 'index':index}
    prepared_files = {}
    for ctx in unique.values():
        for name in ['inputs.pt','samples.json','identity.json','identity-key.json','analysis-identity.json']:
            p = ctx['prepared']/name
            if p.is_file(): prepared_files[p.relative_to(root).as_posix()] = file_hash(p)
    state = {'policy_schema':3, 'generation':generation, 'profiles':profiles,
        'input_prepared':next(iter(unique.values()))['prepared'].relative_to(root).as_posix(),
        'prepared_files':prepared_files, 'evidence_files':deepcopy(restored['evidence_files']) if restored else {},
        'source':str(cfg['preview'].get('resume_from') or ''), 'phase':'prepared',
        'environment_request':request, 'budget_mode':BUDGET_MODE, 'budget_seconds':cfg['preview']['budget_seconds']}
    inherited = {k:list(p['accepted']) for k,p in profiles.items()}
    budget = Budget(cfg['preview']['budget_seconds'],cfg['runtime']['timeout_seconds'],profiles)
    trials = []; skipped = []; counts = {k:0 for k in profiles}

    def save():
        for row in profiles.values():
            names = [c['function'] for c in row['candidates'] or []
                     if c['function'] not in row['accepted'] and c['function'] not in row['excluded']]
            interrupted = (row.get('inflight') or {}).get('candidate') or row.get('_interrupted_candidate')
            row['pending'] = ([interrupted] if interrupted in names else []) + [n for n in names if n != interrupted and n not in row['attempted']] + [n for n in names if n != interrupted and n in row['attempted']]
        assert_source_snapshot(cfg.get('_source_snapshot'))
        write_json(folder/'checkpoint.json', {'schema_version':CHECKPOINT_SCHEMA,'state':state,'sha256':digest(state)})

    def trial(key, phase, include, candidate=None, limit=None):
        room = budget.remaining() if phase in ['baseline','restore','final','common'] else budget.room()
        allowance = min(budget.timeout,room,limit if limit is not None else budget.timeout)
        if allowance < 1:
            return None
        row = profiles[key]; ctx = unique[key]
        label = f'g{generation}-p{row["index"]}-{len(trials):04}-{phase}'
        destination = folder/label
        state['phase'] = phase
        row['inflight'] = {'candidate':candidate,'phase':phase,'include':include,'trial':label}
        save()
        progress.phase = 'preview/'+label
        started = budget.clock()
        value = invoke(ctx['cfg'],root,'forward',destination,ctx['prepared'],include,True,allowance)
        elapsed = budget.clock()-started
        cleanup = float(value.get('cleanup_seconds',0))
        budget.cleanup += cleanup
        elapsed = max(0,elapsed-cleanup)
        if value.get('status') == 'completed':
            if value.get('identity_key') != key:
                raise ValueError('preview worker execution identity mismatch')
            if ctx['cfg']['runtime'].get('parallelism') == 'tp':
                expected = {str(i) for i in range(len(ctx['cfg']['runtime']['devices']))}
                ranks = value.get('ranks',{})
                if set(ranks) != expected or any(v.get('identity_key') != key for v in ranks.values()):
                    raise ValueError('preview missing rank or rank identity mismatch')
        timing = {'phase':phase,'candidate':candidate,'seconds':elapsed,'allowance':allowance,
            'completed':value.get('status')=='completed','timed_out':bool(value.get('timed_out'))}
        row['timings'].append(timing)
        name = f'preview/evidence/g{generation}/{label}.json'
        write_json(root/name, {'result':value,'timing':timing,'include':include,
                             'large_evidence_directory':str(destination.resolve())})
        state['evidence_files'][name] = file_hash(root/name)
        trials.append(dict(timing,environment=key,evidence=name))
        return value, name

    def valid(value, key, include=None):
        return usable(value) and (include is None or all(usable(value,n) for n in include))

    def finish():
        common = sorted(set.intersection(*(set(p['accepted']) for p in profiles.values())))
        entries = {}; verified = bool(common)
        for key,row in profiles.items():
            include = row['accepted']; final = trial(key,'final',include) if include else None
            ok = bool(final) and valid(final[0],key,include)
            common_trial = final if set(include) == set(common) else (trial(key,'common',common) if common else None)
            joint_ok = bool(common_trial) and bool(common) and valid(common_trial[0],key,common)
            verified = verified and ok and joint_ok
            unknown = [dict(c, **{k:v for k,v in row['unknown'].get(c['function'], {'reason':'budget_exhausted'}).items() if k!='function'})
                       for c in row['candidates'] or [] if c['function'] not in include and c['function'] not in row['excluded']]
            entries[key] = {'status':'verified' if ok and joint_ok else 'unverified',
                'selection_sha256':digest(common), 'identity':row['identity'],
                'accepted_include':include,'candidate_count':len(row['candidates'] or []),
                'compiler':row['identity'].get('compiler'),
                'stack':{k:v for k,v in unique[key]['stack'].items() if k!='flaggems'},
                'excluded':list(row['excluded'].values()),
                'unknown':unknown+[{'function':n,'reason':'not_in_joint_verified_selection'} for n in include if n not in common],
                'verification':common_trial[1] if common_trial else None,
                'final_verification':final[1] if final else None,
                'reused_final_verification':bool(final and common_trial is final),
                'inherited_include':inherited[key], 'new_include':[n for n in include if n not in inherited[key]]}
            row['inflight'] = None
        assert_source_snapshot(cfg.get('_source_snapshot'))
        policy = {'schema_version':3,'status':'verified' if verified else 'unverified',
            'include':common,'verified_selection_sha256':digest(common) if verified else None,
            'profiles':entries,'finite_error_policy':'report_only',
            'selection':'cumulative, order-dependent; current full-input per-rank verification; not accuracy acceptance',
            'analysis_key':cfg.get('_source_snapshot',{}).get('analysis_key')}
        (folder/'policy.yaml').write_text(yaml.safe_dump(policy,sort_keys=False,allow_unicode=True))
        report = {'mode':budget.mode,'seconds':budget.seconds,'charged_seconds':budget.clock()-budget.started-budget.cleanup,
            'cleanup_seconds':budget.cleanup,'trials':trials,'skipped':skipped,
            'estimated_minimum_next_budget_seconds':math.ceil(sum(budget.estimate(k,'baseline')+
                (budget.estimate(k,'restore') if p['accepted'] else 0)+2*budget.estimate(k,'final')+
                budget.estimate(k,'candidate') for k,p in profiles.items()))}
        write_json(folder/'budget.json',report)
        state['phase'] = 'verified' if verified else 'unverified'; save()
        return {'status':'completed' if verified else 'partial','policy':'preview/policy.yaml','verified':verified,
            'accepted':len(common),'excluded':sum(len(e['excluded']) for e in entries.values()),
            'unknown':sum(len(e['unknown']) for e in entries.values()),'policy_profiles':entries,
            'resume':{'source':state['source'] or None,'generation':generation,'inherited':inherited},'budget':report}

    save()
    # All profiles get their prerequisites before any profile consumes the search allowance.
    for key,row in profiles.items():
        pending = row['inflight']
        row['_interrupted_candidate'] = (pending.get('candidate') if pending else None) or row.get('_interrupted_candidate')
        baseline = trial(key,'baseline',None)
        if baseline is None:
            state['phase'] = 'budget_before_baseline'; save()
            return finish()
        if baseline[0].get('timed_out') and budget.remaining() < 1:
            skipped.append({'environment':key,'candidate':None,'reason':'budget_exhausted_during_baseline'})
            return finish()
        if not valid(baseline[0],key):
            raise RuntimeError('native preview baseline failed or has nonfinite outputs')
        candidates = normalized_candidates(baseline[0]['candidates'])
        if row['candidates'] is not None and sorted(candidates,key=lambda c:c['function']) != sorted(row['candidates'],key=lambda c:c['function']):
            raise ValueError('candidate mapping/signature changed; run a fresh preview')
        if row['candidates'] is None: row['candidates'] = candidates
        row['candidate_digest'] = digest(row['candidates'])
        row['inflight'] = None; save()
        if inherited[key]:
            restored_trial = trial(key,'restore',inherited[key])
            if restored_trial is None: return finish()
            if restored_trial[0].get('timed_out') and budget.remaining() < 1:
                skipped.append({'environment':key,'candidate':None,'reason':'budget_exhausted_during_restore'})
                return finish()
            if not valid(restored_trial[0],key,inherited[key]):
                raise RuntimeError('previous accepted set failed reverification; refusing to shrink it')
            row['inflight'] = None; save()
    queues = {}
    for key,row in profiles.items():
        names = [c['function'] for c in row['candidates'] if c['function'] not in row['accepted'] and c['function'] not in row['excluded']]
        interrupted = row.pop('_interrupted_candidate',None)
        queues[key] = ([interrupted] if interrupted in names else []) + [n for n in names if n != interrupted and n not in row['attempted']] + [n for n in names if n != interrupted and n in row['attempted']]
    allowance = budget.room()/len(profiles)
    spent = {k:0.0 for k in profiles}; deferred = {k:[] for k in profiles}
    # First equal shares; a second pass borrows unused shares without retrying a started decision.
    for borrow in [False,True]:
        if borrow: queues = deferred
        while any(queues.values()):
            for key,queue in queues.items():
                if not queue: continue
                name = queue.pop(0); row = profiles[key]
                room = budget.room() if borrow else min(budget.room(),max(0,allowance-spent[key]))
                estimate = budget.estimate(key,'candidate',name)
                if room < estimate:
                    if not borrow: deferred[key].append(name)
                    else: skipped.append({'environment':key,'candidate':name,'reason':'insufficient_candidate_and_verification_budget','estimated_seconds':estimate})
                    continue
                started = budget.clock(); proposed = row['accepted']+[name]
                result = trial(key,'candidate',proposed,name,room)
                if result is None: continue
                value, evidence = result
                decision = 'unknown'; reason = 'failure_not_confirmed_within_budget'
                if valid(value,key,proposed):
                    decision = 'accepted'; row['accepted'].append(name)
                elif usable(value): reason = 'not_observed'
                elif not value.get('timed_out'):
                    def extra(phase,selected):
                        remaining = budget.room() if borrow else min(budget.room(),max(0,allowance-spent[key]-(budget.clock()-started)))
                        return trial(key,phase,selected,name,remaining) if remaining >= budget.estimate(key,phase,name) else None
                    repeated = extra('repeat',proposed)
                    control = extra('control',row['accepted'] or None) if repeated else None
                    if repeated and control:
                        again = repeated[0]
                        resource = any(v.get('exit_code') in [-9,-15,137,143] or any(w in str(v.get('error','')).lower() for w in ['out of memory','oom','device unavailable']) for v in [value,again])
                        same = all(value.get(k)==again.get(k) for k in ['exception_type','exit_code','error','numerical_anomalies'])
                        if not usable(again) and usable(control[0]) and same and not resource and not again.get('timed_out'):
                            decision = 'excluded'
                            row['excluded'][name] = {'function':name,'reason':'reproduced_failure_in_cumulative_configuration',
                                'with_functions':list(row['accepted']),'trial':evidence,'repeat':repeated[1],'control':control[1],
                                'exclusion_scope':'historical cumulative configuration only; not global operator qualification'}
                        else: reason = 'unresolved_or_resource_failure'
                if decision == 'unknown': row['unknown'][name] = {'reason':reason,'trial':evidence}
                else: row['unknown'].pop(name,None)
                if name not in row['attempted']: row['attempted'].append(name)
                row['decisions'].append({'generation':generation,'function':name,'decision':decision,'trial':evidence,'with_functions':proposed})
                row['inflight'] = None; counts[key] += 1
                spent[key] += budget.clock()-started
                save()
    return finish()
