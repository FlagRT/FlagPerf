"""Deterministic provider, available only through test process injection."""
from flagprof.telemetry.contract import Call,Metric,Observation,raw_clock


class Provider:
    def __init__(self,**_):
        self.metrics=[Metric('memory.used','B'),Metric('power.draw','W')]
        self.inventory=dict(source='test_fake',entities=[dict(entity='chip:0',power_domain='chip:0')],
                            idle_baselines={'chip:0':dict(bytes=100,tolerance=1)})

    def calls(self):
        def read():
            start=raw_clock()
            return Observation('chip:0',{'memory.used':200,'power.draw':10},start,raw_clock())
        return [Call('fake','chip:0','chip:0',('memory.used','power.draw'),read)]
