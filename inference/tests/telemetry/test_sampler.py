import threading
import time
import unittest
from flagprof.telemetry.contract import Call, Observation, raw_clock
from flagprof.telemetry.sampler import Sampler


class SamplerTests(unittest.TestCase):
    def test_loop_delays_do_not_accumulate_into_sampling_drift(self):
        from unittest.mock import patch
        call=Call('a','chip:0','chip:0',('value',),None)
        sampler=Sampler([call]);sampler.epoch_ns=10**9;sampler.isolated.add('a')
        with patch('flagprof.telemetry.sampler.raw_clock',side_effect=[10**9,2020000000,4080000000]):
            sampler.tick();sampler.tick()
            self.assertEqual(sampler.next['a'],3*10**9)
            sampler.tick();self.assertEqual(sampler.next['a'],5*10**9)
        sampler.close()

    def test_result_queue_overflow_stops_instead_of_growing(self):
        sampler=Sampler([],queue_capacity=1)
        call=Call('a','chip:0','chip:0',('value',),None)
        for i in range(2):sampler._publish([(call,Observation('chip:0',{'value':i},i,i))])
        self.assertEqual(sampler.results.qsize(),1)
        with self.assertRaisesRegex(RuntimeError,'queue capacity'):sampler.tick()
        sampler.close()

    def test_hung_interface_then_device_isolation(self):
        release=threading.Event()
        def hang():
            release.wait(5)
            return Observation('chip:0',{'hang':1},raw_clock(),raw_clock())
        def good():return Observation('chip:1',{'ok':1},raw_clock(),raw_clock())
        sampler=Sampler([Call('hang1','0','chip:0',('hang',),hang),Call('hang2','0','chip:0',('hang',),hang),
                         Call('ok','1','chip:1',('ok',),good)],100,timeout=.04)
        health=[];points=[]
        try:
            end=time.monotonic()+.45
            while time.monotonic()<end:
                p,h=sampler.tick();points.extend(p);health.extend(h);time.sleep(.005)
            self.assertEqual([h['sev'] for h in health],['warn','error'])
            self.assertGreaterEqual(sum(p.entity=='chip:1' for p in points),3)
            self.assertTrue(any('isolated' in p.miss.values() for p in points))
        finally:sampler.close();release.set()

    def test_final_observation_brackets_target_completion(self):
        def read():return Observation('chip:0',{'power':1},raw_clock(),raw_clock())
        sampler=Sampler([Call('power','chip:0','chip:0',('power',),read)],1000,timeout=.1)
        try:
            sampler.tick();time.sleep(.01);sampler.tick(schedule=False)
            target_end=raw_clock()
            points=[point for batch,health in sampler.finish() for point in batch]
            self.assertTrue(any(p.start_ns>=target_end for p in points))
        finally:sampler.close()


if __name__=='__main__':unittest.main()
