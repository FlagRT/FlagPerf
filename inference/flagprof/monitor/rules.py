"""Deterministic fact rules shared by online tailing and offline replay."""
import hashlib
import json

VERSION='monitor-rules/1'
RULES=('monitor.gap','monitor.call_hang','device.error','proc.exit','clock.thermal','clock.power_cap')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:32]


class Engine:
    def __init__(self, inventory, config, state=None):
        self.config=config
        self.config_id=digest(dict(version=VERSION,config=config,descriptors=inventory.get('descriptors',[]),
                                  source_identity=inventory.get('source_identity'),
                                  interval_ms=inventory.get('effective_interval_ms',config['interval_ms'])))
        self.descriptors={d['name']:d for d in inventory.get('descriptors',[])}
        self.fresh_ns=int(max(10,3*inventory.get('effective_interval_ms',config['interval_ms'])/1000)*1e9)
        self.state=state or dict(episodes={},previous={},watermarks={},clock=0,exit_ids={},hung={})
        self.output=[]

    def _condition(self, rule, entity, condition, timestamp, evidence, *, level='confirmed',target=None,stable_key=None,related=()):
        key=rule+'|'+entity+('|candidate' if level=='candidate' else '')
        episode=self.state['episodes'].get(key)
        if condition is True:
            if episode is None or episode['state']=='resolved':
                episode=dict(id=digest([VERSION,self.config_id,rule,entity,level,stable_key or evidence]),rule=rule,entity=entity,
                    target=target,state='open',window=[timestamp,None],revision=0,last_update=0,normal=0,level=level)
                self.state['episodes'][key]=episode
            episode['normal']=0
            changed=episode['state']!='open';episode['state']='open'
            if episode['revision'] and not changed and timestamp-episode['last_update']<self.config['update_seconds']*1e9:return
        elif episode and episode['state']!='resolved':
            if condition is None:
                if episode['state']=='unknown':return
                episode['state']='unknown';episode['normal']=0
            else:
                episode['normal']+=1
                if episode['normal']<self.config['clear_observations']:return
                episode['state']='resolved';episode['window'][1]=timestamp
        else:return
        episode['revision']+=1;episode['last_update']=timestamp
        self.output.append(dict(schema='monitor-finding/1',rule_version=VERSION,config_version=self.config_id,
            id=episode['id'],revision=episode['revision'],rule=rule,state=episode['state'],
            entity=entity,target=target or episode.get('target'),window=list(episode['window']),
            observed_at=timestamp,evidence=[evidence]+list(related),level=episode['level'],
            suggested_action={'monitor.gap':'inspect collector coverage and source availability',
                'monitor.call_hang':'inspect isolated API and collector restart evidence',
                'device.error':'inspect recorded device health evidence',
                'proc.exit':'inspect launcher exit status and target log',
                'clock.thermal':'inspect temperature and restriction interval',
                'clock.power_cap':'inspect power-domain and restriction interval'}[rule]))

    def feed(self, event):
        self.output=[]
        timestamp=event.get('ts')
        if timestamp is None:return []
        timestamp=max(timestamp+event.get('dur',0),self.state['clock']);self.state['clock']=timestamp
        evidence=event['evidence_id']; detail=event.get('detail',{})
        if event['t']=='counter':
            entity=event['dev'];telemetry=event.get('telemetry',{})
            for name,value in event['values'].items():
                descriptor=self.descriptors.get(name,{})
                sample_time=telemetry.get('sample_times',{}).get(name,event['ts'])
                valid=(value is not None and name not in event.get('miss',{}) and
                       0<=timestamp-sample_time<=self.fresh_ns and not descriptor.get('profiler_conflict') and
                       not telemetry.get('profiler_conflict'))
                key=entity+'|'+name
                if valid:self.state['watermarks'][key]=sample_time
                elif event.get('miss',{}).get(name)!='unsupported' and not descriptor.get('profiler_conflict'):
                    self.state['watermarks'].setdefault(key,timestamp)
                for reason,mask in descriptor.get('restriction_masks',{}).items():
                    rule={'thermal':'clock.thermal','power':'clock.power_cap'}.get(reason)
                    if rule:self._condition(rule,entity,bool(int(value)&mask) if valid else None,
                                            timestamp,evidence,level='candidate')
                for role in descriptor.get('roles',[]):
                    if not role.startswith('restriction.cumulative.'):continue
                    reason=role.rsplit('.',1)[-1]
                    rule={'thermal':'clock.thermal','power':'clock.power_cap'}.get(reason)
                    if rule is None:continue
                    previous=self.state['previous'].get(key)
                    condition=None
                    if valid and previous and 0<sample_time-previous['ts']<=self.fresh_ns and value>=previous['value']:
                        condition=value>previous['value']
                    self._condition(rule,entity,condition,timestamp,evidence,
                                    related=[previous['evidence']] if previous else [])
                    if valid:self.state['previous'][key]=dict(value=value,ts=sample_time,evidence=evidence)
                    else:self.state['previous'].pop(key,None)
            apis={row.get('api') for row in telemetry.get('reads',[])}
            for hung_entity,hung in self.state['hung'].items():
                matching_api=bool(hung_entity==entity and hung.get('source_api') and hung['source_api'] in apis)
                metrics=hung.get('metrics',[])
                matching_metrics=bool(hung_entity==entity and metrics and all(name in event['values'] for name in metrics))
                if (matching_api or matching_metrics) and event['values'] and all(v is not None for v in event['values'].values()):
                    self._condition('monitor.call_hang',hung_entity,False,timestamp,evidence)
        elif event['t']=='health':
            entity=detail.get('entity','node')
            if event['kind']=='hang' and detail.get('scope')=='telemetry_call':
                self.state['hung'][entity]=dict(metrics=detail.get('metrics',[]),source_api=detail.get('source_api'))
                self._condition('monitor.call_hang',entity,True,timestamp,evidence)
            elif event['kind'] in ('device_error','ecc','link_down'):
                self._condition('device.error',entity,True,timestamp,evidence)
            elif event['kind']=='process_exit' and detail.get('source')=='subprocess.wait':
                key=detail.get('target_id',entity)
                if key not in self.state['exit_ids']:
                    self.state['exit_ids'][key]=detail.get('returncode')
                    if detail.get('returncode') not in (None,0):
                        self._condition('proc.exit',entity,True,timestamp,evidence,target=key,stable_key=['launcher_exit',key])
            elif event['kind']=='heartbeat' and detail.get('scope')=='device_health':
                self._condition('device.error',entity,False if detail.get('healthy') else None,timestamp,evidence)
            elif event['kind']=='heartbeat' and detail.get('scope')=='monitor_gap':
                self._condition('monitor.gap','node',True,timestamp,evidence)
            elif event['kind']=='heartbeat' and detail.get('scope')=='monitor_tick':
                keys=list(self.state['watermarks'])
                stale=[key for key in keys if timestamp-self.state['watermarks'][key]>self.fresh_ns]
                self._condition('monitor.gap','node',bool(stale) if keys else timestamp-detail['started_raw_ns']>self.fresh_ns,
                                timestamp,evidence)
                for key in stale:
                    entity,name=key.split('|',1)
                    for role in self.descriptors.get(name,{}).get('roles',[]):
                        rule={'restriction.cumulative.thermal':'clock.thermal','restriction.cumulative.power':'clock.power_cap'}.get(role)
                        if rule:self._condition(rule,entity,None,timestamp,evidence)
                    for reason in self.descriptors.get(name,{}).get('restriction_masks',{}):
                        rule={'thermal':'clock.thermal','power':'clock.power_cap'}.get(reason)
                        if rule:self._condition(rule,entity,None,timestamp,evidence,level='candidate')
                    for rule in ('monitor.call_hang','device.error'):
                        self._condition(rule,entity,None,timestamp,evidence)
            if event['kind']=='process_exit':
                for collection in ('watermarks','previous'):
                    for key in list(self.state[collection]):
                        if key.startswith(entity+'|') or key.startswith(entity+'/'):self.state[collection].pop(key)
        return self.output
