"""Progress stays live during silent work and preserves ordered case events."""
from contextlib import redirect_stderr
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executors.progress import RunProgress


class ProgressTests(unittest.TestCase):
    def test_silent_benchmark_heartbeat_and_cleanup(self):
        output = io.StringIO()
        with redirect_stderr(output):
            with RunProgress("Benchmark", case="computation-FP16", interval=0.01) as progress:
                time.sleep(0.45)
                self.assertGreaterEqual(output.getvalue().count("elapsed="), 4)
                progress.finished(0)
        self.assertFalse(progress.thread.is_alive())
        self.assertIn("completed=1/1", output.getvalue())
        self.assertIn("container-exited (code=0)", output.getvalue())

    def test_ordered_cases_and_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            progress = RunProgress("Toolkit", events=path)
            output = io.StringIO()
            events = [dict(phase="running", completed=i, total=2,
                           case=case, index=i + 1, case_started=time.monotonic())
                      for i, case in enumerate(("FP16", "HBM"))]
            with redirect_stderr(output):
                progress._poll()  # Runner has not created the file yet.
                path.write_text(json.dumps(events[0]) + "\n" + json.dumps(events[1]))
                progress._poll()
                self.assertNotIn("HBM", output.getvalue())
                with path.open("a") as stream:
                    stream.write("\n")
                progress._poll()
            self.assertIn("case 1/2: FP16", output.getvalue())
            self.assertIn("case 2/2: HBM", output.getvalue())
            self.assertIn("completed=1/2", output.getvalue())

    def test_runner_publishes_actual_case_sequence(self):
        source = Path(__file__).resolve().parents[1] / "toolkits/_common/ascend/A3/evidence_runner.py"
        spec = importlib.util.spec_from_file_location("progress_evidence_runner", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "discovered_devices", return_value=[0]):
            runner = module.Runner(Path(tmp) / "evidence", ["first", "second"], False)
            with patch.object(module, "check_idle_devices"), \
                 patch.object(runner, "collect_environment"), \
                 patch.object(runner, "collect_topology"), \
                 patch.object(runner, "collect_health"), \
                 patch.object(runner, "collect_diagnostics"), \
                 patch.object(runner, "run_case") as run_case, \
                 redirect_stderr(io.StringIO()):
                runner.run()
            events = [json.loads(line) for line in (runner.root / "progress.jsonl").read_text().splitlines()]
            running = [event for event in events if event["phase"] == "running"]
            self.assertEqual([e["case"] for e in running], ["first", "second"])
            self.assertEqual([e["completed"] for e in running], [0, 1])
            self.assertTrue(all(e["total"] == 2 for e in running))
            self.assertEqual(run_case.call_count, 2)
            self.assertEqual(events[-1]["completed"], 2)

    def test_exception_propagates_and_stops_thread(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(TimeoutError):
                with RunProgress("Benchmark", case="FP16") as progress:
                    raise TimeoutError("test")
        self.assertFalse(progress.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
