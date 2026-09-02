# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import unittest
from pathlib import Path
from unittest import mock


BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS_DIR))

from drivers import ascend  # noqa: E402
from drivers import utils  # noqa: E402


class AscendAdapterTest(unittest.TestCase):
    def setUp(self):
        ascend._INITIALIZED = False

    def test_bootstrap_imports_torch_fl_for_ascend(self):
        flagos = mock.Mock()
        with mock.patch.object(ascend.importlib, "import_module") as importer:
            with mock.patch.object(ascend.torch, "flagos", flagos,
                                   create=True):
                utils.bootstrap_vendor("ascend/A3")
                utils.bootstrap_vendor("ascend/A3")

        importer.assert_called_once_with("torch_fl")

    def test_bootstrap_does_not_touch_other_vendors(self):
        with mock.patch.object(ascend, "initialize") as initialize:
            utils.bootstrap_vendor("nvidia/A100")
        initialize.assert_not_called()

    def test_device_is_explicit_flagos_local_rank(self):
        flagos = mock.Mock()
        with mock.patch.object(ascend, "initialize"):
            with mock.patch.object(ascend.torch, "flagos", flagos,
                                   create=True):
                with mock.patch.object(ascend.torch, "device",
                                       return_value="selected") as device:
                    selected = utils.accelerator_device("ascend/A3", 3)

        self.assertEqual(selected, "selected")
        flagos.set_device.assert_called_once_with(3)
        device.assert_called_once_with("flagos:3")

    def test_host_device_sync_uses_flagos(self):
        with mock.patch.object(ascend, "synchronize") as synchronize:
            utils.host_device_sync("ascend/A3")
        synchronize.assert_called_once_with()

    def test_multi_device_sync_keeps_control_barrier(self):
        with mock.patch.object(utils.torch.distributed, "barrier") as barrier:
            utils.multi_device_sync("ascend/A3")
        barrier.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
