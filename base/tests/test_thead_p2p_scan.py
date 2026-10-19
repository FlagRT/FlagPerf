"""Guard tests for the thead P2P single-node scan protocol (p2p-single-node-v1).

The scan protocol spans three repository-owned artifacts that must stay
consistent: the pinned configuration allowlist in runtime_requirements.json,
the calibration plan, and the qualification plan matrix. These tests fail
closed when a config is edited without re-pinning, when a plan cell drifts
from its pinned config, or when the calibration contract is loosened.
"""
import hashlib
import json
import unittest
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = BASE_DIR / "benchmarks" / "interconnect-P2P_intraserver" / "thead" / "PPU"
MESSAGE_BYTES_PER_MELEMENT = 1 << 22
CONTRACT_BOUNDS = {"Melements": (1, 256), "WARMUP": (1, 100), "ITERS": (1, 200000)}
KNOWN_SCOPES = {"formal-qualification", "bounded-smoke", "p2p-calibration-only", "p2p-qualification"}
PAIR_DEVICE_IDS = {"intracard-die-pair": [14, 15], "intercard-staged-pair": [7, 8]}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class TheadP2PScanProtocolTests(unittest.TestCase):
    def setUp(self):
        self.requirements = json.loads(
            (CASE_DIR / "runtime_requirements.json").read_text(encoding="utf-8"))
        self.allowlist = self.requirements["allowed_case_configs"]
        self.calibration = json.loads(
            (CASE_DIR / "p2p-calibration-plan.json").read_text(encoding="utf-8"))
        self.qualification = json.loads(
            (CASE_DIR / "p2p-qualification-plan.json").read_text(encoding="utf-8"))

    def test_scan_configs_are_allowlisted_with_matching_hashes(self):
        files = sorted(path.name for path in CASE_DIR.glob("case_config*.yaml"))
        pinned = sorted(Path(entry["path"]).name for entry in self.allowlist)
        self.assertEqual(files, pinned, "allowlist and directory must enumerate the same configs")
        for entry in self.allowlist:
            self.assertIn(entry["scope"], KNOWN_SCOPES)
            self.assertEqual(sha256_file(CASE_DIR / Path(entry["path"]).name), entry["sha256"],
                             "config bytes drifted from the pinned sha256: " + entry["path"])

    def test_scan_requires_explicit_override(self):
        self.assertIs(self.requirements["requires_case_override"], True)
        pinned = {Path(entry["path"]).name for entry in self.allowlist}
        self.assertIn("case_config.yaml", pinned, "formal path must stay allowlisted")
        self.assertIn("case_config.smoke.yaml", pinned, "smoke path must stay allowlisted")

    def test_scan_configs_respect_day6_contract_bounds(self):
        for entry in self.allowlist:
            name = Path(entry["path"]).name
            if not name.startswith("case_config.p2p-scan-"):
                continue
            config = load_config(CASE_DIR / name)
            for field, (low, high) in CONTRACT_BOUNDS.items():
                self.assertIsInstance(config[field], int, name + ": " + field)
                self.assertGreaterEqual(config[field], low, name + ": " + field)
                self.assertLessEqual(config[field], high, name + ": " + field)
            self.assertEqual(config["DTYPE"], "float32", name)
            self.assertEqual(config["DIST_BACKEND"], "cpu:gloo,cuda:nccl", name)
            self.assertEqual(config["IMPLEMENTATION"], "native-xpytorch", name)
            self.assertEqual(config["MODE"], "qualification", name)

    def test_qualification_plan_matrix_matches_pinned_configs(self):
        matrix = self.qualification["matrix"]
        self.assertEqual(len(matrix), 8)
        seen = set()
        for cell in matrix:
            name = Path(cell["config"]).name
            seen.add(name)
            config = load_config(CASE_DIR / name)
            self.assertEqual(cell["sha256"],
                             next(entry["sha256"] for entry in self.allowlist
                                  if Path(entry["path"]).name == name),
                             "matrix cell not pinned by the allowlist: " + name)
            self.assertEqual(cell["sha256"], sha256_file(CASE_DIR / name))
            self.assertEqual(cell["iterations"], config["ITERS"], name)
            self.assertEqual(cell["warmup"], config["WARMUP"], name)
            self.assertEqual(cell["message_bytes"],
                             config["Melements"] * MESSAGE_BYTES_PER_MELEMENT, name)
            self.assertEqual(cell["device_ids"], PAIR_DEVICE_IDS[cell["pair_id"]], name)
            sender, receiver = cell["device_ids"]
            self.assertEqual(cell["direction"], f"{sender}-to-{receiver}-only", name)
        self.assertEqual(len(seen), 8, "matrix cells must be eight distinct configs")

    def test_calibration_plan_pins_64gib_points_and_disclaims_formal(self):
        self.assertIs(self.calibration["formal_baseline_eligible"], False)
        self.assertEqual(self.calibration["measurement_contract"]["monitor"], "off")
        self.assertEqual(self.calibration["measurement_contract"]["repeats"], 1)
        self.assertEqual(len(self.calibration["size_points"]), 4)
        for point in self.calibration["size_points"]:
            name = Path(point["config"]).name
            config = load_config(CASE_DIR / name)
            payload = config["Melements"] * MESSAGE_BYTES_PER_MELEMENT * config["ITERS"]
            self.assertEqual(payload, self.calibration["measurement_contract"]["one_way_payload_bytes_per_point"],
                             name + ": calibration payload must be 64 GiB one-way")
            self.assertEqual(point["sha256"],
                             next(entry["sha256"] for entry in self.allowlist
                                  if Path(entry["path"]).name == name),
                             "calibration point not pinned by the allowlist: " + name)

    def test_calibration_evidence_pin_matches_plan_bytes(self):
        evidence = self.qualification["calibration_evidence"]
        self.assertEqual(evidence["path"],
                         "base/benchmarks/interconnect-P2P_intraserver/thead/PPU/"
                         "p2p-calibration-plan.json")
        self.assertEqual(evidence["sha256"], sha256_file(CASE_DIR / "p2p-calibration-plan.json"))

    def test_monitor_ab_order_covers_both_modes(self):
        monitor_ab = self.qualification["monitor_ab"]
        self.assertEqual(monitor_ab["order"], ["off", "on", "on", "off", "off", "on"])
        self.assertEqual(monitor_ab["cell"], "intracard-64m")
        cell = next(cell for cell in self.qualification["matrix"] if cell["cell"] == monitor_ab["cell"])
        self.assertEqual(cell["message_bytes"], 1 << 26)

    def test_windows_respect_contract_floor(self):
        contract = self.qualification["measurement_contract"]
        self.assertEqual(contract["minimum_measurement_seconds_per_run"],
                         self.requirements["qualification_min_measurement_seconds"])
        self.assertEqual(contract["warmup_iterations"], 20)
        self.assertEqual(contract["size_curve_monitor"], "on")
        self.assertIs(contract["automatic_workload_extension"], False)
        self.assertEqual(contract["size_curve_repeats"], 3)
        self.assertEqual(self.qualification["acceptance"]["per_cell_cv_pct_max"], 5.0)
        self.assertEqual(self.qualification["acceptance"]["monitor_median_delta_pct_max"], 5.0)


if __name__ == "__main__":
    unittest.main()
