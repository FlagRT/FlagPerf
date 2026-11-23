import unittest
from flagprof.monitor.rules import Engine
from flagprof.monitor.session import configuration


class RuleTests(unittest.TestCase):
    def test_call_hang_recovery_requires_same_device_and_two_valid_reads(self):
        engine=self.engine()
        event=dict(t='health',kind='hang',ts=10**9,evidence_id=['event','run',1,1],
            detail=dict(scope='telemetry_call',entity='chip:test',source_api='readMetric',metrics=['thermal']))
        self.assertEqual(engine.feed(event)[0]['rule'],'monitor.call_hang')
        for t in (2,3):
            wrong=self.point(t,1,reads=[dict(api='readMetric')]);wrong['dev']='chip:other'
            self.assertFalse(engine.feed(wrong))
        self.assertFalse(engine.feed(self.point(4,1,reads=[dict(api='readMetric')])) )
        result=engine.feed(self.point(5,1,reads=[dict(api='readMetric')]))
        self.assertTrue(any(r['rule']=='monitor.call_hang' and r['state']=='resolved' for r in result))

    def test_device_error_missing_is_unknown_and_normal_needs_two_observations(self):
        engine=self.engine()
        event=dict(t='health',kind='device_error',ts=1,evidence_id=['event','run',1,1],detail=dict(entity='chip:test',codes=[42]))
        self.assertEqual(engine.feed(event)[0]['rule'],'device.error')
        event.update(kind='heartbeat');event['detail'].update(scope='device_health',healthy=None)
        self.assertEqual(engine.feed(event)[0]['state'],'unknown')
        event['detail']['healthy']=True
        self.assertFalse(engine.feed(event));self.assertEqual(engine.feed(event)[0]['state'],'resolved')

    def test_instant_reason_is_candidate_and_stale_does_not_recover(self):
        engine=Engine(dict(descriptors=[dict(name='thermal',restriction_masks={'thermal':32})]),configuration())
        row=engine.feed(self.point(1,32))[0]
        self.assertEqual(row['rule'],'clock.thermal');self.assertEqual(row['level'],'candidate')
        stale=engine.feed(self.tick(20))
        self.assertTrue(any(r['rule']=='clock.thermal' and r['state']=='unknown' for r in stale))
        self.assertFalse(engine.feed(self.point(21,0)))
        resolved=engine.feed(self.point(22,0))[0];self.assertEqual(resolved['state'],'resolved')
        self.assertEqual(resolved['level'],'candidate')

    def engine(self):
        return Engine(dict(descriptors=[dict(name='thermal',roles=['restriction.cumulative.thermal'])]),configuration())

    def point(self,ts,value,**telemetry):
        return dict(t='counter',ts=ts*10**9,dev='chip:test',values={'thermal':value},
            miss={'thermal':'read_error'} if value is None else {},telemetry=telemetry,
            evidence_id=['event','run',1,ts])

    def tick(self,ts):
        return dict(t='health',kind='heartbeat',ts=ts*10**9,detail=dict(scope='monitor_tick',started_raw_ns=0),
                    evidence_id=['event','run',2,ts])

    def test_cumulative_occurrence_unknown_and_two_normal_observations(self):
        engine=self.engine()
        self.assertEqual(engine.feed(self.point(1,100)),[])
        opened=engine.feed(self.point(2,110));self.assertEqual(opened[0]['state'],'open')
        self.assertEqual(engine.feed(self.point(3,120)),[])
        unknown=engine.feed(self.point(4,None));self.assertEqual(unknown[0]['state'],'unknown')
        self.assertEqual(engine.feed(self.point(5,120)),[])
        self.assertEqual(engine.feed(self.point(6,120)),[])
        closed=engine.feed(self.point(7,120));self.assertEqual(closed[0]['state'],'resolved')
        self.assertEqual(closed[0]['id'],opened[0]['id'])

    def test_stale_reset_and_profiler_conflict_do_not_assert_restriction(self):
        for kind in ('stale','reset','conflict'):
            engine=self.engine();engine.feed(self.point(1,100))
            second=self.point(2,110,profiler_conflict=True) if kind=='conflict' else self.point(20,110) if kind=='stale' else self.point(2,10)
            self.assertFalse(engine.feed(second),kind)
        engine=self.engine();engine.feed(self.point(1,None))
        gap=engine.feed(self.tick(12));self.assertEqual(gap[0]['rule'],'monitor.gap')
        self.assertFalse(any(x['rule'].startswith('clock.') for x in gap))

    def test_only_launcher_exit_code_and_duplicate_receipts(self):
        engine=self.engine()
        event=dict(t='health',kind='process_exit',ts=1,evidence_id=['event','run',1,1],
                   detail=dict(entity='proc:test',target_id='target',source='pid_disappeared'))
        self.assertFalse(engine.feed(event))
        event['detail'].update(source='subprocess.wait',returncode=7)
        rows=engine.feed(event);self.assertEqual(rows[0]['rule'],'proc.exit')
        event['evidence_id']=['event','new-segment',1,1]
        event['ts']=65*10**9
        self.assertFalse(engine.feed(event))

    def test_restart_state_matches_uninterrupted_sequence(self):
        import copy
        full=self.engine();resumed=self.engine()
        events=[self.point(1,100),self.point(2,110),self.tick(20),self.point(21,110),self.point(22,110),self.point(23,110)]
        expected=[];actual=[]
        for event in events:expected+=copy.deepcopy(full.feed(event))
        for event in events[:3]:actual+=copy.deepcopy(resumed.feed(event))
        resumed=Engine(dict(descriptors=[dict(name='thermal',roles=['restriction.cumulative.thermal'])]),configuration(),copy.deepcopy(resumed.state))
        for event in events[3:]:actual+=copy.deepcopy(resumed.feed(event))
        self.assertEqual(actual,expected)

    def test_retired_process_device_metrics_do_not_make_monitor_gap(self):
        engine=self.engine()
        point=self.point(1,100);point['dev']='proc:test/chip:0';engine.feed(point)
        engine.feed(dict(t='health',kind='process_exit',ts=2*10**9,evidence_id=['event','run',1,2],
                         detail=dict(entity='proc:test',source='registration_end',exit_status='unknown')))
        point=self.point(20,100);engine.feed(point)
        self.assertFalse(engine.feed(self.tick(21)))
        self.assertFalse(any(k.startswith('proc:') for k in engine.state['watermarks']))


if __name__=='__main__':unittest.main()
