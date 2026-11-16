import types
import unittest
from unittest.mock import patch
from flagprof.runtime import Attachment


class AllocatorTests(unittest.TestCase):
    def test_only_permitted_queries_and_zero(self):
        for vendor,backend in (('ascend','npu'),('nvidia','cuda')):
            reads=[]
            def allocated(device):reads.append(('allocated',device));return 0
            def reserved(device):reads.append(('reserved',device));return 0
            fake=types.SimpleNamespace(**{backend:types.SimpleNamespace(memory_allocated=allocated,memory_reserved=reserved)})
            events=[]
            handle=object.__new__(Attachment)
            handle.monitor_enabled=True;handle.state={'format_vendor':vendor,'device_identity':dict(status='available',entity='chip:0',logical_device=backend+':0')}
            handle.writer=types.SimpleNamespace(emit=events.append)
            with patch.dict('sys.modules',{'torch':fake}):handle.allocator_boundary('measure:begin')
            self.assertEqual(len(reads),2)
            event=events[0]
            self.assertEqual(event['values'],{'allocated_bytes':0,'reserved_bytes':0})
            self.assertTrue(event['ext']['telemetry']['observed'])
            self.assertEqual(event['ext']['producer'],'span_boundary')
            self.assertNotIn('memory',event['ext'])
            self.assertFalse(event['miss'])
            handle.monitor_enabled=False;handle.allocator_boundary('measure:end')
            self.assertEqual(len(events),1)


if __name__=='__main__':unittest.main()
