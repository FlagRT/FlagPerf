# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
import subprocess
import unittest
from unittest.mock import Mock, patch

from flagprof.telemetry.contract import ReadError
from flagprof.vendors.ascend import telemetry, telemetry_dsmi


class ContainerTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.lib = Mock(spec=['dsmi_get_device_count', 'dsmi_list_device', 'dsmi_get_phyid_from_logicid',
                              'dsmi_get_hbm_info', 'dsmi_get_device_utilization_rate'])

        def output(pointer, value):
            pointer._obj.value = value
            return 0

        def list_devices(ids, count):
            ids[:] = [1, 0]
            return 0

        def hbm(logical, pointer):
            self.assertEqual(logical, 0)
            pointer._obj.total, pointer._obj.used = 67108864, 0
            return 0

        self.lib.dsmi_get_device_count.side_effect = lambda p: output(p, 2)
        self.lib.dsmi_list_device.side_effect = list_devices
        self.lib.dsmi_get_phyid_from_logicid.side_effect = lambda i,p: output(p, {0:2, 1:4}[i])
        self.lib.dsmi_get_hbm_info.side_effect = hbm
        self.lib.dsmi_get_device_utilization_rate.side_effect = lambda i,k,p: output(p, 101)
        loader = patch.object(telemetry_dsmi.C, 'CDLL', return_value=self.lib)
        self.load = loader.start()
        self.addCleanup(loader.stop)

    def test_container_fallback_maps_and_samples_only_selected_physical_die(self):
        failure = subprocess.CompletedProcess([], 211, 'npu get board type failed. ret is -9005\n')
        with patch.object(telemetry, '_in_container', return_value=True), patch('subprocess.run', return_value=failure):
            provider = telemetry.Provider(devices='2')
        self.assertIn('211: npu get board type failed. ret is -9005', provider.inventory['initial_source_failure'])
        self.assertEqual(provider.inventory['source'], 'dsmi')
        device, = provider.inventory['entities']
        self.assertEqual((device['entity'], device['logical_id']), ('chip:2', 0))
        self.assertNotIn('power_domain', device)
        self.assertIn('temperature.chip', provider.inventory['unavailable'])
        memory, activity = [call.read() for call in provider.calls()]
        self.assertEqual(memory.values, {'memory.used': 0, 'memory.total': 64 << 30})
        self.assertEqual(memory.miss, {})
        self.assertEqual(activity.miss, {'ascend.aicore.activity': 'invalid'})
        self.lib.dsmi_get_hbm_info.side_effect = None
        self.lib.dsmi_get_hbm_info.return_value = 3
        self.assertEqual(provider.calls()[0].read().miss['memory.used'], 'read_error')

    def test_no_inferred_physical_identity(self):
        with self.assertRaisesRegex(ValueError, 'selected physical device missing'):
            telemetry_dsmi.Provider(devices='0')
        self.lib.dsmi_get_phyid_from_logicid.side_effect = None
        self.lib.dsmi_get_phyid_from_logicid.return_value = 3
        with self.assertRaisesRegex(ReadError, 'dsmi_get_phyid_from_logicid'):
            telemetry_dsmi.Provider()

    def test_explicit_smi_failure_and_successful_auto_do_not_switch(self):
        with patch.object(telemetry, '_in_container', return_value=True), \
             patch('flagprof.vendors.ascend.telemetry_smi.Provider') as smi:
            self.assertIs(telemetry.Provider(), smi.return_value)
            smi.side_effect = ReadError('read_error', '211')
            with self.assertRaisesRegex(ReadError, '211'):
                telemetry.Provider(source='npu-smi')
        self.load.assert_not_called()

    def test_topology_uses_visible_driver_when_smi_executable_is_missing(self):
        with patch.object(telemetry, '_in_container', return_value=True), patch('shutil.which', return_value=None):
            self.assertEqual(telemetry.default_topology(), 'local')
            self.load.side_effect = OSError('missing driver')
            self.assertEqual(telemetry.default_topology(), 'host')
