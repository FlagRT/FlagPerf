# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Bounded grouped discovery; a failed set never qualifies its members as failures."""
from runtime.common import digest
from runtime.preview_evidence import resource_failure

GROUP_SIZE = 8


def search(profiles, state, budget, trial, save, valid, usable, generation, skipped):
    summary = state.setdefault('search', {'reused_trials':[], 'scheduling':[]})
    summary['reused_trials'] = []
    summary['scheduling'] = []
    memo = {}

    def add(row, members, parent=None, depth=0):
        tree = row['search_tree']
        name = str(tree['next_id']); tree['next_id'] += 1
        tree['nodes'][name] = {'members':members,'parent':parent,'depth':depth,'status':'pending'}
        return name

    for row in profiles.values():
        tree = row.setdefault('search_tree', {'nodes':{},'queue':[],'next_id':0,'local_fallbacks':[]})
        remaining = [c['function'] for c in row['candidates']
                     if c['function'] not in row['accepted'] and c['function'] not in row['excluded']]
        present = set()
        queue = []
        for task in tree['queue']:
            node = tree['nodes'][task]
            members = [n for n in node['members'] if n in remaining and n not in present]
            if members:
                node.update(members=members,status='pending')
                present.update(members); queue.append(task)
        tree['queue'] = queue
        fresh = [n for n in remaining if n not in present and n not in row['attempted']]
        old = [n for n in remaining if n not in present and n in row['attempted']]
        for names in (fresh, old):
            for index in range(0,len(names),GROUP_SIZE):
                tree['queue'].append(add(row,names[index:index+GROUP_SIZE]))
    save()

    def complete(key, task, decision, value, evidence, metadata=None):
        row = profiles[key]; tree = row['search_tree']; node = tree['nodes'][task]
        before = list(row['accepted'])
        for name in node['members']:
            if name not in row['attempted']: row['attempted'].append(name)
            if decision == 'accepted':
                row['accepted'].append(name); row['unknown'].pop(name,None)
            elif decision == 'excluded':
                row['excluded'][name] = dict(metadata, function=name, with_functions=before,
                    exclusion_scope='historical cumulative configuration only; not global operator qualification')
                row['unknown'].pop(name,None)
            else:
                row['unknown'][name] = dict(metadata or {},trial=evidence,attempted=True,
                    last_stages={r:d['last_stage'] for r,d in value.get('preview_diagnostics',{}).items()})
            row['decisions'].append({'generation':generation,'function':name,'decision':decision,
                'trial':evidence,'with_functions':before+node['members'],
                'group':list(node['members']),'search_task':task})
        node.update(status='done',outcome=decision,trial=evidence,accepted_context=before)
        tree['queue'].remove(task); row['inflight'] = None
        save()

    def singleton(key, task, value, evidence):
        row = profiles[key]; node = row['search_tree']['nodes'][task]
        name = node['members'][0]; proposed = row['accepted']+[name]
        reason = 'confirmation_incomplete'; detail = {}; last = value
        if usable(value): reason = 'not_observed'
        elif value.get('timed_out'): reason = value.get('timeout_reason','worker_timeout')
        elif resource_failure(value): reason = 'resource_failure'
        elif value.get('preview_evidence_error'): reason = 'evidence_incomplete'
        else:
            def extra(phase, include):
                room = budget.room()
                cost = budget.estimate(key,phase,name,include=include)
                return trial(key,phase,include,name,room) if room >= cost else None
            repeated = extra('repeat',proposed)
            if repeated:
                again, repeat_evidence = repeated
                last = again; detail['repeat'] = repeat_evidence
                if resource_failure(again): reason = 'resource_failure'
                elif again.get('timed_out'): reason = again.get('timeout_reason','worker_timeout')
                elif again.get('preview_evidence_error'): reason = 'evidence_incomplete'
                else:
                    control = extra('control',row['accepted'] or None)
                    if control:
                        check, control_evidence = control
                        last = check; detail['control'] = control_evidence
                        same = all(value.get(k)==again.get(k) for k in
                                   ['exception_type','exit_code','error','numerical_anomalies'])
                        if (not usable(again) and valid(check,key,row['accepted']) and same
                                and not resource_failure(check) and not check.get('timed_out')):
                            complete(key,task,'excluded',last,evidence,dict(detail,trial=evidence,
                                reason='reproduced_failure_in_cumulative_configuration'))
                            return
                        reason = ('resource_failure' if resource_failure(check) else
                                  'evidence_incomplete' if check.get('preview_evidence_error') else
                                  check.get('timeout_reason','unresolved_failure'))
        complete(key,task,'unknown',last,evidence,dict(detail,reason=reason))

    def split(row, task, reason):
        tree = row['search_tree']; node = tree['nodes'][task]
        middle = len(node['members'])//2
        children = [add(row,part,task,node['depth']+1)
                    for part in (node['members'][:middle],node['members'][middle:])]
        index = tree['queue'].index(task)
        tree['queue'][index:index+1] = children
        node.update(status='split',children=children,split_reason=reason)

    blocked = set()
    while any(p['search_tree']['queue'] for p in profiles.values()):
        advanced = False
        for key,row in profiles.items():
            tree = row['search_tree']
            if not tree['queue'] or key in blocked: continue
            others = set().union(*(set(p['accepted']) for k,p in profiles.items() if k!=key))
            def priority(task):
                node = tree['nodes'][task]
                return (-node['depth'],
                        0 if others.intersection(node['members']) else 1,
                        0 if any(n not in row['attempted'] for n in node['members']) else 1,
                        int(task))
            task = min(tree['queue'],key=priority); node = tree['nodes'][task]
            group = node['members']; proposed = row['accepted']+group
            token = (key,digest(proposed))
            available = budget.room((key,proposed))
            active = sum(bool(p['search_tree']['queue']) and k not in blocked for k,p in profiles.items())
            share = available/max(1,active)
            estimate = budget.estimate(key,'group' if len(group)>1 else 'candidate',
                                       group[0] if len(group)==1 else None,group,proposed)
            if token in memo: estimate = 0.0
            # Shares only determine group size; a started worker is not cut at the share boundary.
            if len(group)>1 and estimate>share:
                split(row,task,'admission_size'); save(); advanced = True; continue
            summary['scheduling'].append({'environment':key,'task':task,'group':list(group),
                'estimated_seconds':estimate,'available_seconds':available,
                'verification_tasks':budget.verification_tasks((key,proposed))})
            if estimate>available:
                reason = ('global_budget_exhausted' if budget.remaining()<1 else
                          'search_allowance_exhausted' if available<1 else 'estimate_does_not_fit')
                skipped.append({'environment':key,'group':list(group),'reason':reason,'estimated_seconds':estimate})
                blocked.add(key)
                continue
            node.update(status='running',accepted_context=list(row['accepted']))
            save()
            result = memo.get(token)
            if result:
                summary['reused_trials'].append({'environment':key,'task':task,'trial':result[1]})
            else:
                result = trial(key,'group' if len(group)>1 else 'candidate',proposed,
                               group[0] if len(group)==1 else None,available,group=group)
                if result and not (result[0].get('timed_out') or resource_failure(result[0]) or
                                   result[0].get('preview_evidence_error')):
                    memo[token] = result
            if result is None:
                node['status']='pending'; blocked.add(key); save(); continue
            advanced = True
            value,evidence = result
            node['trial'] = evidence
            for name in group:
                if name not in row['attempted']: row['attempted'].append(name)
            if valid(value,key,proposed):
                complete(key,task,'accepted',value,evidence)
            elif len(group)==1:
                singleton(key,task,value,evidence)
            elif resource_failure(value) or value.get('timeout_reason') in ('global_budget_exhausted','search_allowance_exhausted'):
                complete(key,task,'unknown',value,evidence,{'reason':
                    'resource_failure' if resource_failure(value) else value['timeout_reason']})
            else:
                node['outcome']='failed'
                split(row,task,'failed_group')
                parent = tree['nodes'].get(node['parent'])
                if parent and all(len(tree['nodes'][child]['members'])>1 and
                                  tree['nodes'][child].get('outcome')=='failed' for child in parent['children']):
                    # Failure clusters say nothing about unrelated future groups.
                    tree['local_fallbacks'].append({'task':node['parent'],
                        'reason':'both_non_singleton_children_failed'})
                    for pending in list(tree['queue']):
                        item = tree['nodes'][pending]
                        ancestor = item['parent']
                        while ancestor is not None and ancestor != node['parent']:
                            ancestor = tree['nodes'][ancestor]['parent']
                        if ancestor == node['parent'] and len(item['members'])>1:
                            children=[add(row,[n],pending,item['depth']+1) for n in item['members']]
                            index=tree['queue'].index(pending); tree['queue'][index:index+1]=children
                            item.update(status='split',children=children,split_reason='dense_failure')
                row['inflight']=None; save()
        if not advanced: break
        # A changed selection/other environment's progress may change admission costs.
        blocked.clear()
    for key,row in profiles.items():
        for task in row['search_tree']['queue']:
            node = row['search_tree']['nodes'][task]
            node['status']='pending'
            for name in node['members']:
                row['unknown'].setdefault(name,{'reason':'estimate_does_not_fit',
                    'attempted':name in row['attempted'],'search_task':task})
    save()
