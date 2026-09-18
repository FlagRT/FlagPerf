"""Identity tests deliberately use reordered and non-contiguous device numbers."""
import copy
from datetime import datetime, timezone
import unittest

from device_mapping import device_identity, resolve_logical_devices, validate_device_set
from hardware_probe import validate_binding


class DeviceMappingTests(unittest.TestCase):
    def setUp(self):
        self.devices = [
            dict(host_physical_id=12, host_device_node='/dev/xpu10', container_node='/dev/xpu10',
                 device_minor=10, pci_bdf='0000:b6:00.0', uuid='11111111-1111-4111-8111-111111111111'),
            dict(host_physical_id=2, host_device_node='/dev/xpu3', container_node='/dev/xpu3',
                 device_minor=3, pci_bdf='0000:1c:00.0', uuid='22222222-2222-4222-8222-222222222222')]

    def test_no_eight_card_limit_or_single_digit_minor(self):
        validate_device_set(self.devices)
        raw = 'XPU 00000000:B6:00.0\n Minor Number : 10\n XPU UUID : GPU-11111111-1111-4111-8111-111111111111\n'
        self.assertEqual(device_identity(raw, '0000:b6:00.0')['host_device_node'], '/dev/xpu10')

    def test_runtime_order_does_not_define_requested_rank(self):
        resolved = resolve_logical_devices(self.devices, [d['uuid'] for d in reversed(self.devices)])
        self.assertEqual([d['host_physical_id'] for d in resolved], [12, 2])
        self.assertEqual([d['requested_rank'] for d in resolved], [0, 1])
        self.assertEqual([d['logical_device'] for d in resolved], [1, 0])

    def test_selection_reordering_changes_ranks_not_identity(self):
        observed = [d['uuid'] for d in self.devices]
        resolved = resolve_logical_devices(list(reversed(self.devices)), observed)
        self.assertEqual([d['logical_device'] for d in resolved], [1, 0])

    def test_missing_extra_wrong_and_duplicate_runtime_uuid(self):
        uuids = [d['uuid'] for d in self.devices]
        for observed in ([], uuids[:1], uuids+['unexpected'], [uuids[0]]*2, [uuids[0], 'unexpected']):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                resolve_logical_devices(self.devices, observed)

    def test_duplicate_host_identity_rejected(self):
        for field in ('host_physical_id', 'uuid', 'pci_bdf', 'host_device_node'):
            devices = copy.deepcopy(self.devices)
            devices[1][field] = devices[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_device_set(devices)

    def test_uuid_format_and_ambiguous_query_rejected(self):
        for uid in ('not-a-uuid', '111'):
            raw = f'XPU 00000000:B6:00.0\n Minor Number : 10\n XPU UUID : GPU-{uid}\n'
            with self.assertRaises(ValueError):
                device_identity(raw, '0000:b6:00.0')
        raw = 'XPU 00000000:B6:00.0\n Minor Number : 10\n Minor Number : 11\n XPU UUID : GPU-11111111-1111-4111-8111-111111111111\n'
        with self.assertRaises(ValueError):
            device_identity(raw, '0000:b6:00.0')

    def test_multi_binding_exact_nodes_and_visibility(self):
        binding = dict(schema_version=2, devices=self.devices, cuda_visible_devices='0,1',
                       reservation_reference='offline', reservation_start='2026-09-18T00:00:00Z',
                       reservation_end='2026-09-18T02:00:00Z')
        nodes = ['/dev/xpu10', '/dev/xpu3', '/dev/xpuctrl']
        now = datetime(2026, 9, 18, 1, tzinfo=timezone.utc)
        validate_binding(binding, {'CUDA_VISIBLE_DEVICES': '0,1'}, nodes, now)
        for env, mapped in [({'CUDA_VISIBLE_DEVICES': '12,2'}, nodes),
                            ({'CUDA_VISIBLE_DEVICES': '0,1'}, nodes+['/dev/xpu2'])]:
            with self.assertRaises(ValueError):
                validate_binding(binding, env, mapped, now)

    def test_empty_selection_and_negative_index_rejected(self):
        with self.assertRaises(ValueError):
            validate_device_set([])
        self.devices[0]['host_physical_id'] = -1
        with self.assertRaises(ValueError):
            validate_device_set(self.devices)


if __name__ == '__main__':
    unittest.main()
