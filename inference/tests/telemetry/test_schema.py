import unittest
from flagprof.store.schema import SchemaValidator,SchemaError
from flagprof.store._validation import CachedValidator
from flagprof.telemetry.contract import Observation


class SchemaTests(unittest.TestCase):
    def test_new_and_old_counters(self):
        for cls in (SchemaValidator,CachedValidator):
            validator=cls()
            event=dict(t='counter',src='device',dev='chip:0',seq=0,ts=1,dur=2,values={'x':None},miss={'x':'future_reason'})
            self.assertEqual(validator.decode(event)['miss'],{'x':'other'})
            event['values']['x']=0
            with self.assertRaises(SchemaError):validator.decode(event)
            event['values']['x']=None
            event['miss']={}
            with self.assertRaises(SchemaError):validator.decode(event)
            event.pop('miss');event.pop('dur')
            self.assertIsNone(validator.decode(event)['values']['x'])
        with self.assertRaises(ValueError):Observation('chip:0',{'x':None},1,2,{})


if __name__=='__main__':unittest.main()
