import ctypes as C
from pathlib import Path
import types
import unittest
from unittest.mock import patch
from flagprof.vendors.nvidia.telemetry import Provider,Memory
from flagprof.vendors.ascend.telemetry_smi import parse as ascend_parse
from flagprof.vendors.nvidia.telemetry_smi import parse as nvidia_parse


class Function:
    def __init__(self,name,owner):self.name,self.owner=name,owner
    def __call__(self,*args):
        n=self.name
        if n=='nvmlInit_v2':return 0
        if n in ('nvmlSystemGetDriverVersion','nvmlSystemGetNVMLVersion','nvmlDeviceGetName','nvmlDeviceGetUUID'):
            args[-2].value=b'GPU-fake' if n.endswith('UUID') else b'fake';return 0
        obj=args[-1]._obj
        if n=='nvmlDeviceGetCount_v2':obj.value=1
        elif n=='nvmlDeviceGetHandleByIndex_v2':obj.value=123
        elif n=='nvmlDeviceGetCudaComputeCapability':args[-2]._obj.value=8;obj.value=0
        elif n=='nvmlDeviceGetMemoryInfo_v2':
            assert obj.version==C.sizeof(Memory)|(2<<24)
            obj.total=100;obj.used=0;obj.free=90;obj.reserved=10
        elif n=='nvmlDeviceGetUtilizationRates':return 3
        elif n=='nvmlDeviceGetViolationStatus':obj.reference=123;obj.duration=42
        elif n=='nvmlDeviceGetFieldValues':obj.nvmlReturn=3
        else:obj.value=10
        return 0


class Library:
    names={'nvmlInit_v2','nvmlSystemGetDriverVersion','nvmlSystemGetNVMLVersion','nvmlDeviceGetName','nvmlDeviceGetUUID',
           'nvmlDeviceGetCount_v2','nvmlDeviceGetHandleByIndex_v2','nvmlDeviceGetMemoryInfo_v2','nvmlDeviceGetUtilizationRates',
           'nvmlDeviceGetPowerUsage','nvmlDeviceGetEnforcedPowerLimit','nvmlDeviceGetTotalEnergyConsumption',
           'nvmlDeviceGetTemperature','nvmlDeviceGetClockInfo','nvmlDeviceGetMaxClockInfo',
           'nvmlDeviceGetCurrentClocksThrottleReasons','nvmlDeviceGetViolationStatus','nvmlDeviceGetFieldValues',
           'nvmlDeviceGetCudaComputeCapability'}
    def __getattr__(self,name):
        if name not in self.names:raise AttributeError(name)
        fn=Function(name,self);setattr(self,name,fn);return fn


class ProviderTests(unittest.TestCase):
    def test_nvml_abi_symbol_and_missing(self):
        with patch('ctypes.CDLL',return_value=Library()):
            provider=Provider(source='nvml')
            points=[call.read() for call in provider.calls()]
        memory=next(p for p in points if 'memory.used' in p.values)
        self.assertEqual(memory.values['memory.used'],0)
        self.assertFalse(memory.miss)
        util=next(p for p in points if 'nvidia.gpu.activity' in p.values)
        self.assertEqual(util.miss['nvidia.gpu.activity'],'unsupported')
        self.assertIn('nvmlDeviceGetCurrentClocksThrottleReasons',provider.inventory['symbols'].values())
        self.assertEqual(next(m for m in provider.metrics if m.name=='power.draw').kind,'instant')

    def test_real_ascend_fallback_and_truncation(self):
        root=Path(__file__).resolve().parents[1]
        raw=(root/'fixtures/telemetry/ascend/info.txt').read_text()
        result=ascend_parse(raw)
        self.assertEqual(len(result),16)
        self.assertEqual(result[0]['values']['memory.total'],64*1024**3)
        busy=ascend_parse((root/'fixtures/telemetry/ascend/busy-info.txt').read_text())
        self.assertEqual(len(busy),16)
        self.assertGreater(busy[13]['values']['memory.used'],0)
        for bad in (raw[:120],raw.replace('3154 / 65536','3154 / MISSING')):
            with self.assertRaises(ValueError):ascend_parse(bad)

    def test_nvidia_csv_and_unsupported(self):
        raw='0, GPU-fake, 0000:01:00.0, 0, 81920, 0, 0, 60, 400, N/A, 210, 1593\n'
        row=nvidia_parse(raw)[0]
        self.assertEqual(row['values']['memory.used'],0)
        self.assertEqual(row['miss'],{'temperature.chip':'unsupported'})
        with self.assertRaises(ValueError):nvidia_parse(raw.rsplit(',',1)[0])

    def test_real_nvidia_idle_busy_and_truncation(self):
        root=Path(__file__).resolve().parents[1]/'fixtures/telemetry/nvidia'
        idle=nvidia_parse((root/'idle.csv').read_text())[0]
        text=(root/'busy.csv').read_text();busy=nvidia_parse(text)[0]
        self.assertEqual(idle['uuid'],busy['uuid'])
        self.assertEqual(idle['values']['memory.used'],0)
        self.assertGreater(busy['values']['memory.used'],16<<30)
        self.assertGreater(busy['values']['nvidia.gpu.activity'],0)
        with self.assertRaises(ValueError):nvidia_parse(text[:60])

    def test_power_semantics_pre_ampere_and_newer(self):
        for capability,kind,window in [((7,5),'instant',None),((8,6),'window',1_000_000_000)]:
            lib=Library()
            def query(handle,major,minor):
                major._obj.value,minor._obj.value=capability
                return 0
            lib.nvmlDeviceGetCudaComputeCapability=query
            with patch('ctypes.CDLL',return_value=lib):provider=Provider(source='nvml')
            power=next(m for m in provider.metrics if m.name=='power.draw')
            self.assertEqual((power.kind,power.window_ns),(kind,window))


if __name__=='__main__':unittest.main()

class DCMIFunction:
    def __init__(self,name):self.name=name
    def __call__(self,*args):
        n=self.name
        if n=='dcmi_init':return 0
        if n=='dcmi_get_card_list':args[0]._obj.value=1;args[1][0]=6
        elif n=='dcmi_get_device_id_in_card':args[1]._obj.value=2
        elif n=='dcmi_get_device_type':args[-1]._obj.value=0
        elif n=='dcmi_get_device_logic_id':args[0]._obj.value=args[2]
        elif n=='dcmi_get_device_phyid_from_logicid':args[-1]._obj.value=12+args[0].value
        elif n in ('dcmi_get_driver_version','dcmi_get_dcmi_version'):args[0].value=b'26.1.1'
        return 0

class DCMILibrary:
    def __getattr__(self,name):
        value=DCMIFunction(name);setattr(self,name,value);return value

class DCMITests(unittest.TestCase):
    def test_power_once_for_two_ranks_and_type13_calibration(self):
        from flagprof.vendors.ascend.telemetry import DCMI
        with patch('ctypes.CDLL',return_value=DCMILibrary()):provider=DCMI(devices='12,13')
        self.assertEqual(sum('power.draw' in c.metrics for c in provider.calls()),1)
        card=next(e for e in provider.inventory['entities'] if e['entity']=='card:6')
        self.assertEqual(card['members'],['chip:12','chip:13'])
        self.assertTrue(next(m for m in provider.metrics if m.name=='ascend.npu.activity').profiler_conflict)
        self.assertTrue(next(m for m in provider.metrics if m.name=='ascend.aicore.activity').profiler_conflict)
        self.assertEqual(provider.inventory['minimum_interval_ms'],100)
        self.assertNotIn('type13_calibration', provider.inventory)


class PortableCalibrationTests(unittest.TestCase):
    def test_reuse_requires_same_topology_driver_and_hardware(self):
        import tempfile
        from unittest.mock import Mock
        from flagprof.monitor.calibration import calibrate
        inventory = dict(boot_id='boot', source='dcmi', calibration_rounds=3, source_identity={'sha256':'code'},
                         entities=[{'entity':'chip:1'}], descriptors=[], driver_version='v1')
        call = types.SimpleNamespace(group='fast', read=Mock())
        provider = types.SimpleNamespace(calls=lambda:[call])
        with tempfile.TemporaryDirectory() as root:
            measured = calibrate(root, 'ascend','agent', provider, inventory.copy())
            self.assertEqual(call.read.call_count, 3)
            saved = inventory.copy()
            self.assertEqual(calibrate(root, 'ascend','agent',provider,saved), measured)
            self.assertEqual(saved['calibration_status'], 'matched')
            self.assertEqual(call.read.call_count, 3)
            for change in ({'driver_version':'v2'}, {'entities':[{'entity':'chip:2'}]}):
                calibrate(root,'ascend','agent',provider,dict(inventory, **change))
            calibrate(root,'ascend','local',provider,inventory.copy())
            self.assertEqual(call.read.call_count, 12)

    def test_local_smi_probes_join_uuid_minor_and_pci(self):
        import os
        import stat
        import sys
        from flagprof.vendors.smi_identity import hardware_identity
        for prefix,model,tool in [('PPU','PPU-ZW810E','ppu-smi'),('XPU','P800','xpu-smi')]:
            record = (f'Driver Version : 1\nXPU-RT Version : 2\n{prefix} 0000:01:00.0\n'
                      f'Product Name : {model}\n{prefix} UUID : observed\n'
                      'Minor Number : 3\nBus Id : 0000:01:00.0\n')
            props = types.SimpleNamespace(uuid='GPU-observed',pci_bus_id='0000:01:00.0')
            cuda = types.SimpleNamespace(get_device_properties=lambda _:props,get_device_name=lambda _:model)
            kwargs = dict(command=tool,prefix=prefix,env_key='TEST_SMI_INVENTORY',node_pattern='test*',
                          model_names={model:model},runtime='XPU-RT Version' if prefix=='XPU' else None)
            node = Path('/dev/test3')
            with patch.dict(sys.modules,torch=types.SimpleNamespace(cuda=cuda)), patch.dict(os.environ, TEST_SMI_INVENTORY=''), \
                 patch('flagprof.vendors.smi_identity.subprocess.run',return_value=types.SimpleNamespace(stdout=record)), \
                 patch.object(Path,'glob',return_value=[node]), patch.object(Path,'stat', return_value=types.SimpleNamespace(st_mode=stat.S_IFCHR,st_rdev=os.makedev(42,3))):
                result=hardware_identity(0,**kwargs)
                self.assertEqual((result['model'],result['host_device_node']), (model,str(node)))
                props.pci_bus_id='0000:02:00.0'
                with self.assertRaisesRegex(ValueError,'PCI'):
                    hardware_identity(0,**kwargs)
