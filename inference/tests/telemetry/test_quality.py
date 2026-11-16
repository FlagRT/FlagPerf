import unittest
from flagprof.telemetry.quality import statistics, aggregate, energy, residual, violation, energy_per_token, low_frequency


class QualityTests(unittest.TestCase):
    def test_overlap_and_newest(self):
        # [0,10] at 2, [5,15] at 8 => 5*2 + 10*8, not two windows.
        result = statistics([dict(ts=10,value=2),dict(ts=15,value=8)],
                            dict(kind='window',window_ns=10),(0,20),5)
        self.assertEqual(result['covered_ns'],15)
        self.assertEqual(result['mean'],6)
        self.assertEqual(result['coverage'],.75)

    def test_instant_gap_and_null(self):
        result=statistics([dict(ts=0,value=2),dict(ts=5,value=None),dict(ts=20,value=8),dict(ts=25,value=4)],
                          dict(kind='instant'),(0,30),10)
        self.assertEqual(result['covered_ns'],15)
        self.assertAlmostEqual(result['mean'],14/3)
        self.assertEqual(result['miss'],{'invalid':1})

    def test_instant_silent_gap_stops_after_one_interval(self):
        result=statistics([dict(ts=0,value=2),dict(ts=20,value=8)],
                          dict(kind='instant'),(0,30),10)
        self.assertEqual(result['covered_ns'],20)
        self.assertEqual(result['coverage'],2/3)
        self.assertEqual(result['mean'],5)

    def test_source_timestamp_and_conflict_tail(self):
        points=[dict(ts=30,sample_ts=12,value=2),dict(ts=35,sample_ts=17,value=8),dict(ts=40,sample_ts=22,value=4)]
        descriptor=dict(kind='window',window_ns=10,profiler_conflict=True)
        result=statistics(points,descriptor,(0,30),5,[(5,13)])
        self.assertEqual(result['samples'],0)
        descriptor['profiler_conflict']=False
        result=statistics(points,descriptor,(0,30),5,[(5,13)])
        self.assertEqual(result['samples'],3)
        self.assertEqual(result['covered_ns'],20)

    def test_stage_distribution(self):
        a=statistics([dict(ts=i,value=1) for i in range(20)],{},(0,20),1)
        b=statistics([dict(ts=i,value=100) for i in range(2)],{},(0,2),1)
        result=aggregate([a,b])
        self.assertEqual(a['p95'],1)
        self.assertIsNone(b['p95'])
        self.assertEqual(result['p95'],100)
        self.assertEqual(result['mean'],10)

    def test_zero_reserved_negative_and_probe_state(self):
        args=dict(baseline=100,tolerance=2,interval_ns=10)
        points=[dict(ts=5,value=99)]
        self.assertEqual(residual(dict(ts=7,value=0,observed=True),points,**args)['bytes'],-1)
        self.assertEqual(residual(dict(ts=7,value=0,observed=False),points,**args)['reason'],'framework_unobserved')
        self.assertEqual(residual(dict(ts=7,value=0,observed=True),points,preexisting=True,**args)['reason'],'preexisting_usage')

    def test_energy_reset_gap_and_boundary(self):
        points=[dict(ts=0,value=10),dict(ts=10**9,value=20),dict(ts=2*10**9,value=5),dict(ts=3*10**9,value=15)]
        result=energy(points,(500000000,3*10**9),cumulative=True)
        self.assertEqual(result['joules'],15)
        self.assertEqual(result['resets'],1)
        self.assertEqual(result['coverage'],.6)
        result=energy(points[:2],(0,10**9))
        self.assertEqual(result['joules'],15)

    def test_complete_window_and_resolution(self):
        result=statistics([dict(ts=5,value=1),dict(ts=10,value=2)],dict(kind='window',window_ns=10),(0,10),5)
        self.assertEqual(result['status'],'insufficient_resolution')
        self.assertIsNone(result['mean'])

    def test_violation_bracket_not_stage_causality(self):
        points=[dict(ts=0,value=0),dict(ts=10,value=7),dict(ts=20,value=9)]
        self.assertEqual(violation(points,(0,10))['evidence'],'mechanism')
        self.assertEqual(violation(points,(5,15))['evidence'],'time_candidate')
        self.assertIsNone(violation(points,(5,15))['restricted_ns'])
        self.assertEqual(violation([dict(ts=0,value=9),dict(ts=10,value=1)],(0,10))['resets'],1)

    def test_energy_domain_denominator_and_gaps(self):
        value=dict(joules=120,coverage=1)
        self.assertEqual(energy_per_token(value,30,['chip:0'],{'chip:0'},False)['joules_per_token'],4)
        self.assertIsNone(energy_per_token(value,30,['chip:0','chip:1'],{'chip:0'},False)['joules_per_token'])
        value=dict(joules=60,coverage=.5)
        self.assertIsNone(energy_per_token(value,30,['chip:0'],{'chip:0'},True)['joules_per_token'])

    def test_pair_source_time_and_half_interval(self):
        args=dict(baseline=100,tolerance=2,interval_ns=10)
        point=[dict(ts=100,sample_ts=5,value=99)]
        self.assertEqual(residual(dict(ts=10,value=0,observed=True),point,**args)['bytes'],-1)
        self.assertEqual(residual(dict(ts=11,value=0,observed=True),point,**args)['reason'],'unpaired')

    def test_low_frequency_stops_at_gap(self):
        points=[dict(ts=0,value=50),dict(ts=5,value=60),dict(ts=10,value=None),dict(ts=15,value=94),dict(ts=20,value=96)]
        nominal=[dict(ts=i,value=100) for i in range(0,30,5)]
        result=low_frequency(points,nominal,5)
        self.assertEqual([r['window'] for r in result],[[0,10]])
        self.assertEqual(result[0]['samples'],2)
        self.assertTrue(all(r['evidence']=='time_candidate' for r in result))


if __name__=='__main__':unittest.main()
