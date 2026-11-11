# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Offline training coverage must reject missing, duplicated and failed phases."""
import sqlite3
import unittest
from flagprof.analyzers.training import analyze

class Work:
    def __init__(self, events):
        self.db=sqlite3.connect(':memory:');self.records=events;self.rows={}
    def events(self, types):
        for i, event in enumerate(self.records): yield 'run', 1, i, event
    def row(self,panel,key,row):self.rows.setdefault(panel,{})[key]=row
    def table(self,panel):return list(self.rows.get(panel,{}).values())

class TrainingAnalysisTests(unittest.TestCase):
    def fixture(self):
        phases=[('zero_grad',None),('forward',0),('backward',0),('forward',1),('backward',1),('optimizer',None),('step',None)]
        return [dict(t='span',dur=10,pid=1,tid=1,attrs=dict(training_schema='training/1',phase=phase,step=0,microstep=micro,accumulation=2,outcome=dict(status='complete'))) for phase,micro in phases]
    def run_events(self,events):
        work=Work(events);self.addCleanup(work.db.close)
        return analyze(work),work
    def test_complete(self):
        result,work=self.run_events(self.fixture())
        self.assertEqual(result[0].status,'ok')
        self.assertEqual(work.table('training')[0]['phase_counts']['backward'],2)
    def test_missing_backward_is_partial(self):
        events=self.fixture();events.pop(4)
        result,work=self.run_events(events)
        self.assertEqual(result[0].status,'partial')
        self.assertIn('backward:incomplete_microsteps',work.table('training')[0]['issues'])
    def test_duplicate_optimizer_is_partial(self):
        events=self.fixture();events.append(events[-2])
        result,_=self.run_events(events);self.assertEqual(result[0].status,'partial')
    def test_failed_phase_is_partial(self):
        events=self.fixture();events[1]['attrs']['outcome']['status']='failed'
        result,_=self.run_events(events);self.assertEqual(result[0].status,'partial')
    def test_no_training_does_not_add_panel(self):
        result,_=self.run_events([]);self.assertEqual(result,[])

if __name__=='__main__':unittest.main()
