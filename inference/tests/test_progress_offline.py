"""Verify the offline stages against the actual rows and checkpoint reused."""
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from flagprof.offline.bundle import make_bundle
from flagprof.offline.service import import_bundle
from flagprof.offline.workspace import private_name
from flagprof.progress import session
from flagprof.report import generate


class OfflineProgressTests(unittest.TestCase):
    def test_import_checkpoint_and_report_observe_the_actual_work(self):
        fixture = Path(__file__).parent / 'correlate/fixtures/t8.1/nvidia.json'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'raw'
            raw.mkdir()
            source = raw / 'trace.json'
            shutil.copyfile(fixture, source)
            capture, store = root / 'capture', root / 'store'
            make_bundle(raw, source, capture, vendor='nvidia',
                        metadata=dict(normal_close=True, target_returncode=0))

            def observe(function, *args, **kwargs):
                log = StringIO()
                with redirect_stderr(log), session('plain'):
                    result = function(*args, **kwargs)
                return result, log.getvalue()

            cold, first_log = observe(import_bundle, capture, store=store, run_id='cold')
            self.assertEqual(cold['status'], 'complete')
            self.assertEqual(cold['reused'], [])
            source_rows = len(json.loads(source.read_bytes())['traceEvents'])
            self.assertEqual(cold['source_records'], source_rows)
            self.assertIn('扫描源事件', first_log)
            self.assertIn('归一化源记录', first_log)
            self.assertIn(f'{source_rows}/{source_rows}', first_log)
            self.assertIn('写入 Store 设备事件', first_log)
            self.assertIn(f"{cold['count']}/{cold['count']}", first_log)

            warm, second_log = observe(import_bundle, capture, store=store, run_id='warm')
            self.assertEqual(warm['reused'], ['derived'])
            self.assertIn('验证并读取已准备的导入检查点', second_log)
            self.assertNotIn('归一化源记录', second_log)
            self.assertNotIn('扫描源事件', second_log)

            prepared = store / private_name() / 'checkpoints' / cold['import_key'] / 'prepared'
            (prepared / 'committed.json').write_text('{}')
            repaired, repair_log = observe(import_bundle, capture, store=store, run_id='repair')
            self.assertEqual(repaired['reused'], ['scan'])
            self.assertIn('恢复已验证的扫描索引', repair_log)
            self.assertNotIn('扫描源事件', repair_log)
            self.assertIn(f'{source_rows}/{source_rows}', repair_log)

            output = root / 'report'
            report, report_log = observe(generate, warm['run'], out=output)
            self.assertIn('perf.device', report['panels'])
            self.assertTrue((output / 'report.html').is_file())
            for label in ('读取 Store 事件', '读取原生调用关联', '分析面板 perf.device',
                          '导出机器结果与逐项明细', '渲染 HTML 和 Markdown 报告'):
                self.assertIn(label, report_log)
