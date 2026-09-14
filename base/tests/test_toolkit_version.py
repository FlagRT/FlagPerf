"""ToolBox minimum-version gate, without Docker or device execution."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executors import toolkit


class ToolBoxVersionTests(unittest.TestCase):
    def test_minimum_and_newer_versions(self):
        for version in ("26.1.0", "26.1.1", "26.2.0", "26.10.0", "27.0.0", "27.0.RC1"):
            with self.subTest(version=version):
                self.assertEqual(toolkit.validate_toolbox_version(f"version={version}\narch=aarch64\n"), version)

    def test_old_prerelease_and_unknown_versions(self):
        for info in ("version=7.2.RC1", "version=26.0.9", "version=26.1.RC1",
                     "", "version=unknown", "version=26.1", "version=26.1.0\nversion=27.0.0"):
            with self.subTest(info=info), self.assertRaises(RuntimeError):
                toolkit.validate_toolbox_version(info)

    def test_rejection_stops_before_preflight_or_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            latest = root / "toolbox" / "latest"
            latest.mkdir(parents=True)
            (latest / "ascend_toolbox_install.info").write_text("version=7.2.RC1\n")
            config = {"image": "test", "toolbox_host_path": str(latest.parent)}
            args = toolkit.parse_args([
                "--allow-privileged-root", "--allow-disruptive-dmi",
                "--npu-ids", "7", "--result-root", str(root / "results"),
            ])
            with patch.object(toolkit, "load_host_config", return_value=(root / "config", config)), \
                 patch.object(toolkit, "docker_inspect", return_value={"Id": "test"}), \
                 patch.object(toolkit, "validate_runtime_identity", return_value={}), \
                 patch.object(toolkit.grp, "getgrnam", return_value=SimpleNamespace(gr_gid=0)), \
                 patch.object(toolkit, "generate_report_safely"), \
                 patch.object(toolkit, "run_host_preflight") as preflight, \
                 patch.object(toolkit.subprocess, "run") as subprocess_run, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(toolkit.execute_toolkit(args), 2)
            preflight.assert_not_called()
            subprocess_run.assert_not_called()
            summary = json.loads(next((root / "results").glob("*/summary.json")).read_text())
            self.assertEqual(summary["failure_stage"], "host-environment")
            self.assertEqual(summary["status"], "failed")
            self.assertIn("7.2.RC1", summary["error"])


if __name__ == "__main__":
    unittest.main()
