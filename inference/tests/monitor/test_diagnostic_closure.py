import json
from pathlib import Path
import tempfile
import unittest
from flagprof.monitor.commands import checkpoint_closure


class CheckpointClosureTests(unittest.TestCase):
    def test_transitive_checkpoint_inputs_and_missing_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'checkpoints').mkdir();(root/'targets/history').mkdir(parents=True)
            (root/'targets/history/target.json').write_text(json.dumps(dict(diagnostic_identity=dict(run_id='task'))))
            segments=[dict(id=name,status='closed') for name in ('early','middle','late')]
            state=dict(config={'diagnosis':{}},segments=segments)
            for name,refs in [('early',[]),('middle',[['event','early',0,2]]),('late',[['event','middle',0,4],['event','task',0,1]])]:
                (root/'checkpoints'/(name+'.json')).write_text(json.dumps(dict(engine=dict(evidence=refs))))
            rows,sources=checkpoint_closure(root,state,[segments[-1]])
            self.assertEqual([r['id'] for r in rows],['early','middle','late'])
            self.assertEqual(sources,{'task'})
            state['segments']=segments[1:]
            with self.assertRaisesRegex(ValueError,'referenced event source: early'):
                checkpoint_closure(root,state,[segments[-1]])

    def test_unknown_rules_cannot_replay_as_current_rules(self):
        from types import SimpleNamespace
        from flagprof.monitor.consumer import validate_rules
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'rules.json').write_text(json.dumps(dict(rule_version='monitor-rules/99',config_version='same')))
            with self.assertRaisesRegex(ValueError,'rule version'):
                validate_rules(root,SimpleNamespace(config_id='same'))
