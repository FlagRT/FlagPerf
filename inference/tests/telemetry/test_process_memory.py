import ctypes as C
import os
import types
import unittest
from unittest.mock import patch
from flagprof.store.manifest import proc_key
from flagprof.telemetry.process_memory import observations
from flagprof.vendors.ascend.telemetry_process import Process as AscendProcess
from flagprof.vendors.nvidia.telemetry_process import Process as NVMLProcess, attach


class ProcessMemoryTests(unittest.TestCase):
    def test_unselected_device_does_not_create_target_memory_gap(self):
        key=proc_key();pid=os.getpid()
        rows=observations('chip:1',[],{pid:key},1,2,'fixture',target_devices={pid:('chip:0',)})
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0].entity,'chip:1')

    def test_installed_abi_layout_and_zero_memory(self):
        self.assertEqual((C.sizeof(AscendProcess),AscendProcess.used.offset),(16,8))
        self.assertEqual((C.sizeof(NVMLProcess),NVMLProcess.used.offset),(24,8))
        key=proc_key();rows=observations('chip:0',[(os.getpid(),0)],{os.getpid():key},1,2,'fixture')
        self.assertEqual(rows[1].values,{'process.device_memory':0})
        self.assertTrue(rows[1].entity.endswith('/chip:0'))

    def test_pid_reuse_and_unknown_process_do_not_gain_target_identity(self):
        key=proc_key();key['proc_start_ticks']-=1
        rows=observations('chip:0',[(os.getpid(),123),(99999,456)],{os.getpid():key},1,2,'fixture')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0].values['process.count'],2)
        self.assertEqual(rows[0].detail['process_table'][1]['host_pid'],99999)

    def test_sentinel_duplicate_and_absent_are_gaps(self):
        key=proc_key();pid=os.getpid()
        for rows,reason in [([(pid,2**64-1)],'unsupported'),([(pid,1),(pid,2)],'invalid'),([],'read_error')]:
            point=observations('chip:0',rows,{pid:key},1,2,'fixture')[1]
            self.assertIsNone(point.values['process.device_memory'])
            self.assertEqual(point.miss['process.device_memory'],reason)

    def test_nvml_enumeration_has_fixed_capacity_and_copies_registration(self):
        class Function:
            def __call__(self,handle,count,rows):
                self.capacity=C.cast(count,C.POINTER(C.c_uint)).contents.value
                C.cast(count,C.POINTER(C.c_uint)).contents.value=1
                rows[0].pid=os.getpid();rows[0].used=1234
                provider.targets={}  # Concurrent retirement cannot retag returned rows.
                return 0
        function=Function()
        provider=types.SimpleNamespace(lib=types.SimpleNamespace(nvmlDeviceGetComputeRunningProcesses_v3=function),
            metrics=[],_calls=[],unavailable={},targets={os.getpid():proc_key()})
        attach(provider,None,'chip:0')
        points=provider._calls[0].read()
        self.assertEqual(function.capacity,1024)
        self.assertEqual(points[1].values['process.device_memory'],1234)


if __name__=='__main__':unittest.main()
