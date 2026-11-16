"""Persisted-input business rules; source event time and decision time stay distinct."""
from copy import deepcopy
from .rules import Engine as LegacyEngine, digest

DEFAULTS = dict(heartbeat_misses=10, progress_timeout_s=60, comm_timeout_s=60)


class Engine(LegacyEngine):
    def __init__(self, inventory, config, state=None):
        super().__init__(inventory, config, state)
        self.business = self.state.setdefault('business', dict(targets={}, episodes={}))
        self.settings = dict(DEFAULTS, **config.get('diagnosis', {}))

    def condition(self, rule, key, condition, now, source, target, *, onset=None, detail=None):
        identity = target['identity']
        episode_key = rule+'|'+key
        episode = self.business['episodes'].get(episode_key)
        if condition is True:
            if episode is None or episode['state']=='resolved':
                episode = dict(id=digest(['business-rules/1',self.config_id,episode_key,source]),
                    state='open', revision=0, normal=0, onset=onset if onset is not None else now,
                    first_detected=now, last_update=0)
                self.business['episodes'][episode_key] = episode
            changed = episode['state'] != 'open'
            episode.update(normal=0, state='open')
            if episode['revision'] and not changed and now-episode['last_update']<self.config['update_seconds']*1e9:
                return
        elif episode and episode['state'] != 'resolved':
            if condition is None:
                if episode['state']=='unknown':
                    return
                episode.update(state='unknown',normal=0)
            else:
                episode['normal'] += 1
                if episode['normal']<self.config['clear_observations']:
                    return
                episode['state']='resolved'
        else:
            return
        episode['revision'] += 1
        episode['last_update'] = now
        self.output.append(dict(schema='business-finding/1',rule_version='business-rules/1',
            config_version=self.config_id, id=episode['id'],revision=episode['revision'],
            rule=rule,state=episode['state'],target=identity['run_id'],run_id=identity['run_id'],
            rank=identity['rank'],group_id=identity.get('group_id'),proc_key=identity['proc_key'],
            window=[episode['onset'],now if episode['state']=='resolved' else None],
            first_detected=episode['first_detected'],observed_at=now,evidence=source,
            task_relative_ns=episode['onset']-target['started_ns'],detail=detail or {},
            suggested_action='inspect referenced task and rank evidence'))

    def feed(self, event):
        detail = event.get('detail', {})
        if detail.get('scope') not in ('business_input','business_tick'):
            rows=super().feed(event)
            identity=detail.get('diagnostic_identity')
            if identity and event.get('kind')=='process_exit' and detail.get('source')=='subprocess.wait':
                self.business.setdefault('exit_pending',{})[identity['run_id']]=True
            if identity:
                for row in rows:
                    if row['rule']=='proc.exit':
                        row.update(identity, termination_reason=detail.get('termination_reason','target_exit'))
            return rows
        self.output=[]
        now=event['ts']
        if detail['scope']=='business_input':
            source=detail['record'];data=source['detail']
            if data.get('version',1)!=1:raise ValueError('unsupported target diagnostic version')
            identity=data['identity'];key=identity['run_id']
            target=self.business['targets'].setdefault(key,dict(identity=identity,
                started_ns=data['started_ns'],active={},calls={},seen_calls={},completed_scopes={},completed_calls={},ended=False,last=None,period_ns=10**9))
            target['evidence']=[event['evidence_id'],detail['source_identity']]
            action=data['action']
            if action=='start':target['period_ns']=data['period_ns']
            elif action=='heartbeat':
                # New journals attach the source read result to each batch. Old
                # T17 inputs keep their recorded v1 behavior when absent.
                healthy=detail.get('source_status','ready')=='ready' and (
                    'source_status' not in detail or now-source['ts']<data['period_ns']*self.settings['heartbeat_misses'])
                for token in list(target['completed_scopes']):
                    self.condition('hang.no_progress',key+':'+token,False if healthy else None,now,target['evidence'],target)
                    if self.business['episodes']['hang.no_progress|'+key+':'+token]['state']=='resolved':
                        target['completed_scopes'].pop(token)
                for token in list(target['completed_calls']):
                    self.condition('comm.timeout',key+':'+token,False if healthy else None,now,target['evidence'],target)
                target.update(last=source['ts'],period_ns=data['period_ns'],active=data['active'],calls=data['calls'])
                self.condition('hang.heartbeat',key,False if healthy else None,now,target['evidence'],target)
                for token,row in data['active'].items():
                    valid=row['eligible']
                    expired=source['ts']-row['last_progress']>=self.settings['progress_timeout_s']*1e9
                    self.condition('hang.no_progress',key+':'+token,expired if valid and healthy else None,now,
                        target['evidence'],target,onset=row['last_progress'],detail=dict(scope=token,name=row['name']))
            elif action=='comm_enter':
                target['calls'][data['key']]=data['call']
                target['seen_calls'][data['key']]=target['evidence']
            elif action=='comm_return':
                target['calls'].pop(data['key'],None)
                target['completed_calls'][data['key']]=True
                self.condition('comm.timeout',key+':'+data['key'],False,now,target['evidence'],target)
            elif action=='leave':
                target['active'].pop(data['token'],None)
                if 'hang.no_progress|'+key+':'+data['token'] in self.business['episodes']:
                    target['completed_scopes'][data['token']]=True
                self.condition('hang.no_progress',key+':'+data['token'],False,now,target['evidence'],target)
            elif action=='finish':
                target['ended']=True
                target['active'].clear();target['calls'].clear()
            elif action=='oom':
                self.condition('oom.device',key+':'+str(detail['source_identity']),True,now,
                    target['evidence'],target,onset=source['ts'],detail=data)
            return deepcopy(self.output)
        health=detail['sources']
        for key,target in self.business['targets'].items():
            if target['ended']:
                continue
            source=[event['evidence_id']]+target.get('evidence',[])
            available=health.get(key)=='ready'
            if available and self.business.get('exit_pending',{}).get(key):
                target['ended']=True
                continue
            last=target['last'] if target['last'] is not None else target['started_ns']
            expired=now-last>=target['period_ns']*self.settings['heartbeat_misses']
            if not available or expired:
                self.condition('hang.heartbeat',key,True if available else None,
                    now,source,target,onset=last+target['period_ns']*self.settings['heartbeat_misses'])
            for call_key,call in target['calls'].items():
                known=call['role'] in ('model','control') and not call['asynchronous']
                pending=now-call['since']>=self.settings['comm_timeout_s']*1e9
                peers=[other for other in self.business['targets'].values()
                    if other['identity'].get('group_id')==target['identity'].get('group_id')
                    and other['identity']['rank']!=target['identity']['rank']]
                peers_ready=bool(peers) and all(health.get(o['identity']['run_id'])=='ready' for o in peers)
                entered=[o['identity']['rank'] for o in peers if call_key in o['seen_calls']]
                self.condition('comm.timeout',key+':'+call_key,
                    pending if available and known and peers_ready else None,now,source+[ev for o in peers for ev in o.get('evidence',[])],target,
                    onset=call['since'],detail=dict(call,peer_ranks_observed_in_call=entered,
                    observation='host_call_not_returned',device_completion='unavailable'))
            if not available or expired:
                for token in target['active']:
                    self.condition('hang.no_progress',key+':'+token,None,now,source,target)
        return deepcopy(self.output)


def engine(inventory, config, state=None):
    return (Engine if 'diagnosis' in config else LegacyEngine)(inventory,config,state)
