# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Exercise the published checkout without hardware, Docker or sibling repos."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'base/vendors/ascend/torch_fl_2.10_flagcx'
CASE = ROOT / 'base/benchmarks/interconnect-P2P_intraserver/ascend'
DOCS = [ROOT / p for p in (
    'docs/README.md', 'docs/CHANGELOG.md', 'base/README.md', 'operation/README.md',
    'operation/vendors/ascend/README.md',
)] + list((ROOT / 'docs/ascend').glob('*.md'))


def run(*argv):
    result = subprocess.run(
        [sys.executable, *argv], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=30,
    )
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


class PublicDistributionTests(unittest.TestCase):
    def test_base_plans_are_usable_without_runtime(self):
        for domain in ('benchmark', 'toolkit'):
            with self.subTest(domain=domain):
                plan = run('base/run.py', domain, 'run', '--case',
                           'computation-FP16', '--npu-ids', '7', '--dry-run')
                self.assertIsInstance(plan, dict)

    def test_operation_catalog_and_vendor_plans(self):
        self.assertEqual(len(run('operation/run.py', 'list')), 52)
        for vendor in ('ascend', 'nvidia', 'cambricon', 'kunlunxin', 'iluvatar', 'metax'):
            with self.subTest(vendor=vendor):
                argv = ['operation/run.py', 'run', '--vendor', vendor,
                        '--device-ids', '0', '--case', 'abs', '--dry-run']
                if vendor != 'ascend':
                    argv += ['--image', 'example-vendor-runtime']
                plan = run(*argv)
                self.assertEqual(plan['vendor'], vendor)
                self.assertEqual(len(plan['tasks']), 1)

    def test_p2p_plans_need_no_private_experiment_directory(self):
        for script, count in (('run_p2p_calibration.py', 8),
                              ('run_p2p_qualification.py', 30)):
            with self.subTest(script=script):
                plan = run(str(PROFILE / script))
                self.assertEqual(plan['mode'], 'plan-only')
                self.assertEqual(plan['run_count'], count)

    def test_p2p_config_hashes_and_public_record_match(self):
        req = json.loads((CASE / 'runtime_requirements.json').read_text())
        for item in req['allowed_case_configs']:
            path = CASE.parent / item['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['sha256'])
        manifest = json.loads((PROFILE / 'image-manifest.json').read_text())
        ref = manifest['qualification_record']
        self.assertEqual(hashlib.sha256((PROFILE / ref['path']).read_bytes()).hexdigest(), ref['sha256'])

    def test_public_coverage_has_exact_denominators(self):
        summary = json.loads((ROOT / 'docs/ascend/operation-coverage.json').read_text())
        self.assertEqual(len(summary['coverage']), summary['applicable_combinations'])
        counts = {status: sum(row['status'] == status for row in summary['coverage'])
                  for status in ('passed', 'blocked', 'failed', 'partial')}
        for status, count in counts.items():
            self.assertEqual(count, summary['counts'][status])
        self.assertFalse(summary['hardware_revalidated_during_integration'])
        self.assertFalse(summary['raw_evidence_distributed'])

    def test_public_document_links_resolve_inside_checkout(self):
        for path in DOCS:
            prose = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
            for target in re.findall(r'\]\(([^)]+)\)', prose):
                if '://' in target or target.startswith('#'):
                    continue
                target = unquote(target.split('#', 1)[0])
                resolved = (path.parent / target).resolve()
                with self.subTest(document=str(path.relative_to(ROOT)), target=target):
                    self.assertTrue(resolved.is_relative_to(ROOT))
                    self.assertTrue(resolved.exists())

    def test_maintained_surface_has_no_development_stages_or_private_paths(self):
        paths = set(DOCS)
        for directory in (
            ROOT / 'base/vendors/ascend', CASE, ROOT / 'base/tests',
            ROOT / 'operation/runtime', ROOT / 'operation/vendors/ascend', ROOT / 'docs/ascend',
        ):
            paths.update(p for p in directory.rglob('*') if p.suffix in ('.py', '.json', '.yaml', '.md'))
        paths.update((ROOT / 'base/configs').glob('ascend*.yaml'))
        stage = re.compile(r'(?<![A-Za-z0-9#])(?:C[0-9](?:[-_/]|\b)|c[0-9][-_])')
        private = re.compile(r'/home/kzhang520|FlagPerf_advance|(?:\.\./)+personal/')
        for path in sorted(paths):
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertIsNone(stage.search(path.name))
                self.assertIsNone(stage.search(path.read_text()))
                self.assertIsNone(private.search(path.read_text()))


if __name__ == '__main__':
    unittest.main()
