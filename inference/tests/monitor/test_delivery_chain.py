import json
from pathlib import Path
import tempfile
import time
import sys
import unittest
from flagprof.monitor.commands import publish, release
from flagprof.monitor.consumer import replay
from flagprof.monitor.report import generate
from flagprof.record import record
from tests.monitor import test_agent


class DeliveryChainTests(unittest.TestCase):
    stop=staticmethod(test_agent.AgentTests.stop)
    start_agent=test_agent.AgentTests.start_agent

    def test_fake_full_delivery_and_online_replay_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);agent=self.start_agent(root/'node')
            body="from flagprof import current_attachment; import time; h=current_attachment(); h._monitor_bind([dict(status='available',entity='chip:0')]); time.sleep(.3); raise SystemExit(7)"
            code=record([sys.executable,'-c',body],store=root/'target',vendor='nvidia',preset='monitor',
                        config=dict(monitor_topology='agent',monitor_root=str(root/'node')))
            self.assertEqual(code,7)
            time.sleep(1.2);self.stop(agent)
            online=[json.loads(p.read_text()) for p in (root/'node/findings').glob('*/*.json')]
            replay(root/'node',root/'replay')
            offline=[json.loads(line) for line in (root/'replay/findings.jsonl').read_text().splitlines()]
            order=lambda row:(row['id'],row['revision'])
            self.assertEqual(sorted(online,key=order),sorted(offline,key=order))
            self.assertTrue(any(row['rule']=='proc.exit' for row in online))
            target=next((root/'target').glob('*/monitor-ref.json')).parent
            generate(root/'node',root/'report')
            publish(target,root/'delivery')
            result=release(root/'node',target/'monitor-ref.json',root/'delivery')
            self.assertEqual(result['status'],'released')
            self.assertTrue((target/'manifest.json').exists())
            self.assertGreater(result['restored_events'],0)


if __name__=='__main__':unittest.main()
