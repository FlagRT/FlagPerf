# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Exercise the published checkout without hardware, Docker or sibling repos."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import unittest
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'base/vendors/ascend/torch_fl_2.10_flagcx'
CASE = ROOT / 'base/benchmarks/interconnect-P2P_intraserver/ascend'
DOCS = [ROOT / p for p in (
    'README.md', 'docs/README.md', 'docs/CHANGELOG.md', 'docs/base/base-case-doc.md',
    'base/README.md', 'operation/README.md',
    'operation/vendors/ascend/README.md',
    'operation/vendors/kunlunxin/README.md', 'operation/vendors/thead/README.md',
    'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/README.md',
    'base/toolkits/_common/kunlunxin/P800/README.md',
)] + list((ROOT / 'docs/ascend').glob('*.md')) + list((ROOT / 'docs/kunlunxin').glob('*.md')) + list((ROOT / 'docs/thead').glob('*.md'))
DOCS += list((ROOT / 'base/docs').glob('*.md'))
DOCS += list((ROOT / 'base/benchmarks').glob('*/kunlunxin/P800/README.md'))
DOCS += list((ROOT / 'base/toolkits').glob('*/kunlunxin/P800/README.md'))


def markdown_anchors(path):
    text = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8'), flags=re.S)
    anchors = set(re.findall(r'<a\s+(?:id|name)=[\"\']([^\"\']+)', text))
    seen = {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.M):
        heading = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', heading)
        heading = re.sub(r'<[^>]+>', '', heading).lower()
        slug = ''.join(c for c in heading if c.isalnum() or c in ' _-').replace(' ', '-')
        count = seen.get(slug, 0)
        anchors.add(slug + (f'-{count}' if count else ''))
        seen[slug] = count + 1
    return anchors


def run(*argv):
    result = subprocess.run(
        [sys.executable, *argv], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=30,
    )
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


class PublicDistributionTests(unittest.TestCase):
    def test_missing_host_dependency_has_actionable_cli_error(self):
        result = subprocess.run(
            [sys.executable, '-S', '-B', 'base/run.py', 'benchmark', 'run',
             '--case', 'computation-FP16', '--npu-ids', '7', '--dry-run'],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
            env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'},
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('base/requirements-host.txt', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_p800_base_matrix_is_usable_without_runtime(self):
        cases = sorted((ROOT / 'base/benchmarks').glob('*/kunlunxin/P800/runtime_requirements.json'))
        self.assertEqual(len(cases), 15)
        skipped = {'computation-FP64', 'computation-FP8', 'computation-TF32',
                   'interconnect-MPI_interserver', 'interconnect-P2P_interserver'}
        communication = {'interconnect-MPI_intraserver', 'interconnect-P2P_intraserver'}
        for path in cases:
            case = path.parents[2].name
            with self.subTest(case=case):
                argv = ['base/run.py', 'benchmark', 'run', '--config',
                        'base/configs/kunlunxin_p800_xpytorch29.yaml', '--case', case + ':P800',
                        '--physical-device-ids', '4,7' if case in communication else '4',
                        '--nproc-per-node', '2' if case in communication else '1',
                        '--timeout', '300', '--dry-run']
                if case == 'main_memory-capacity':
                    argv += ['--allow-high-risk-case']
                plan = run(*argv)
                self.assertEqual(plan['vendor'], 'kunlunxin')
                self.assertEqual(plan['applicability']['status'], 'skipped' if case in skipped else 'applicable')
                self.assertTrue(plan['runtime_lock']['image_manifest']['validated'])
                self.assertEqual(plan['runtime_lock']['image_manifest']['validation_status'], 'partial')

    def test_p800_toolkit_documented_entries_and_complete_plans(self):
        readmes = sorted(p for p in (ROOT / 'base/toolkits').glob('*/kunlunxin/P800/README.md')
                         if p.parents[2].name != '_common')
        self.assertEqual(len(readmes), 12)
        for path in readmes:
            case = path.parents[2].name
            blocks = re.findall(r'```bash\n(.*?)```', path.read_text(encoding='utf-8'), re.S)
            commands = [shlex.split(b.replace('\\\n', ' ')) for b in blocks if b.startswith('bash base/toolkits/')]
            self.assertEqual(len(commands), 1, case)
            argv = commands[0] + ['--dry-run']
            with self.subTest(case=case):
                proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=30,
                                      env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
                self.assertEqual(proc.returncode, 0, proc.stderr)
                plan = json.loads(proc.stdout)
                self.assertEqual(plan['cases'], [case])
                self.assertEqual(plan['excluded_physical_ids'], [])
                self.assertFalse(plan['settings']['allow_candidate_runtime'])
                self.assertTrue(plan['runtime_lock']['image_manifest']['validated'])
        for devices, logical, targets in (('4,7', 97, 485), ('0-7', 460, 2300)):
            plan = run('base/run.py', 'toolkit', 'run', '--config',
                       'base/configs/kunlunxin_p800_xpytorch29.yaml',
                       '--physical-device-ids', devices, '--repeat', '5', '--dry-run')
            self.assertEqual(sum(plan['logical_point_counts'].values()), logical)
            self.assertEqual(sum(plan['point_counts'].values()), targets)
            self.assertEqual(plan['excluded_physical_ids'], [])

    def test_base_plans_are_usable_without_runtime(self):
        for domain in ('benchmark', 'toolkit'):
            with self.subTest(domain=domain):
                plan = run('base/run.py', domain, 'run', '--case',
                           'computation-FP16', '--npu-ids', '7', '--dry-run')
                self.assertIsInstance(plan, dict)

    def test_operation_catalog_and_vendor_plans(self):
        self.assertEqual(len(run('operation/run.py', 'list')), 52)
        for vendor in ('ascend', 'nvidia', 'cambricon', 'kunlunxin', 'thead', 'iluvatar', 'metax'):
            with self.subTest(vendor=vendor):
                argv = ['operation/run.py', 'run', '--vendor', vendor,
                        '--device-ids', '0', '--case', 'abs', '--dry-run']
                if vendor != 'ascend':
                    argv += ['--image', 'example-vendor-runtime']
                if vendor == 'thead':
                    argv += ['--allow-candidate-runtime']
                plan = run(*argv)
                self.assertEqual(plan['vendor'], vendor)
                self.assertEqual(len(plan['tasks']), 1)

    def test_ppu_operation_documented_plans_need_only_stdlib(self):
        paths = [ROOT / 'operation/vendors/thead/README.md', ROOT / 'operation/README.md']
        plans = []
        for path in paths:
            blocks = re.findall(r'```bash\n(.*?)```', path.read_text(encoding='utf-8'), re.S)
            for block in blocks:
                for line in block.replace('\\\n', ' ').splitlines():
                    if 'operation/run.py run ' not in line or '--vendor thead' not in line:
                        continue
                    argv = shlex.split(line)
                    if '--dry-run' not in argv:
                        argv.append('--dry-run')
                    plan = run('-S', *argv[1:])
                    self.assertEqual(plan['runtime_profile'], 'torch210-ppu')
                    self.assertEqual(plan['vendor'], 'thead')
                    self.assertTrue(all(t['runtime_config']['runtime_profile'] == 'torch210-ppu' for t in plan['tasks']))
                    plans.append(plan['execution'])
        self.assertIn('docker', plans)
        self.assertIn('local', plans)

    def test_p800_operation_documented_plans_need_only_stdlib(self):
        paths = [ROOT / 'operation/vendors/kunlunxin/README.md',
                 ROOT / 'docs/kunlunxin/operation-design.md']
        plans = []
        for path in paths:
            blocks = re.findall(r'```bash\n(.*?)```', path.read_text(encoding='utf-8'), re.S)
            for block in blocks:
                for line in block.replace('\\\n', ' ').splitlines():
                    if 'operation/run.py run ' not in line:
                        continue
                    argv = shlex.split(line)
                    # Audit real-run examples as plans too; never start hardware.
                    if '--dry-run' not in argv:
                        argv.append('--dry-run')
                    plan = run('-S', *argv[1:])
                    self.assertEqual(plan['runtime_profile'], 'p800-xpytorch29')
                    self.assertTrue(all(t['runtime_profile'] == 'p800-xpytorch29' for t in plan['tasks']))
                    modes = {t['profiling_mode'] for t in plan['tasks']}
                    self.assertEqual(len(modes), 1)
                    plans.append((plan['execution'], len(plan['tasks']), modes.pop()))
        self.assertCountEqual(plans, [('docker', 1, 'off'), ('docker', 1, 'off'), ('local', 2, 'off'),
                                     ('docker', 728, 'off'), ('docker', 2, 'timeline')])

    def test_p800_operation_full_matrix_and_validation_boundary(self):
        lock = json.loads((ROOT / 'operation/vendors/kunlunxin/profiles/p800-xpytorch29.json').read_text())
        self.assertFalse(lock['validated'])
        self.assertEqual(lock['excluded_physical_ids'], [])
        for mode in ('docker', 'local'):
            with self.subTest(execution=mode):
                plan = run('-S', 'operation/run.py', 'run', '--vendor', 'kunlunxin',
                           '--runtime-profile', 'p800-xpytorch29', '--execution', mode,
                           '--device-ids', '1', '--oplib', 'both', '--dtype',
                           'FP32', 'FP16', 'BF16', 'INT32', 'INT16', 'BOOL', 'INT64', '--dry-run')
                tasks = plan['tasks']
                applicable = sum(t['applicable'] for t in tasks)
                self.assertEqual((len(tasks), applicable, len(tasks) - applicable), (728, 308, 420))
                self.assertEqual(len({(t['case'], t['dtype'], t['oplib']) for t in tasks}), 728)
                self.assertEqual(plan['runtime_profile'], 'p800-xpytorch29')
                self.assertEqual(plan['image'], lock['image'] if mode == 'docker' else None)
                self.assertTrue(all(t['runtime_config']['sharing_policy'] == 'exclusive-required' for t in tasks))

    def test_p800_operation_multidevice_sizes_and_shared_plans(self):
        for mode, devices in (('docker', '1,7'), ('local', '0,9')):
            with self.subTest(execution=mode):
                plan = run('-S', 'operation/run.py', 'run', '--vendor', 'kunlunxin',
                           '--runtime-profile', 'p800-xpytorch29', '--execution', mode,
                           '--device-ids', devices, '--case', 'mm', '--dtype', 'FP32',
                           '--oplib', 'both', '--size', 'M=7', '--size', 'M=13',
                           '--workload', 'smoke', '--allow-shared-device', '--dry-run')
                self.assertEqual(len(plan['tasks']), 8)
                self.assertEqual({t['case_config']['M'] for t in plan['tasks']}, {7, 13})
                self.assertEqual({t['device_id'] for t in plan['tasks']}, {int(d) for d in devices.split(',')})
                self.assertTrue(all(t['runtime_config']['sharing_policy'] == 'shared-authorized' for t in plan['tasks']))
                if mode == 'local':
                    self.assertTrue(all(t['device_binding']['status'] == 'externally-managed' for t in plan['tasks']))

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
            prose = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8'), flags=re.S)
            targets = re.findall(r'\]\(([^)]+)\)', prose) + re.findall(r'href=[\"\']([^\"\']+)', prose)
            for target in targets:
                if '://' in target or target.startswith('mailto:'):
                    continue
                filename, _, anchor = unquote(target.strip('<>')).partition('#')
                resolved = (path.parent / filename).resolve() if filename else path
                with self.subTest(document=str(path.relative_to(ROOT)), target=target):
                    self.assertTrue(resolved.is_relative_to(ROOT))
                    self.assertTrue(resolved.exists())
                    self.assertNotIn('result', resolved.relative_to(ROOT).parts)
                    self.assertNotIn('evidence', resolved.relative_to(ROOT).parts)
                    if anchor and resolved.suffix == '.md' and resolved.exists():
                        self.assertIn(anchor, markdown_anchors(resolved))

    def test_maintained_surface_has_no_development_stages_or_private_paths(self):
        paths = set(DOCS)
        for directory in (
            ROOT / 'base/vendors/ascend', CASE, ROOT / 'base/tests',
            ROOT / 'operation/runtime', ROOT / 'operation/vendors/ascend', ROOT / 'docs/ascend',
            ROOT / 'operation/vendors/kunlunxin', ROOT / 'operation/vendors/thead',
        ):
            paths.update(p for p in directory.rglob('*') if p.suffix in ('.py', '.json', '.yaml', '.md'))
        paths.update((ROOT / 'base/configs').glob('ascend*.yaml'))
        stage = re.compile(r'(?<![A-Za-z0-9#])(?<!\d:)(?:C[0-9](?:[-_/]|\b)|c[0-9][-_])')  # PCI BDF segments ('0000:C1:...') stay exempt via the digit-colon lookbehind; 'label:C1' stage markers are caught again
        private = re.compile(r'/home/kzhang(?:519|520)|/data1/flagperf-|FlagPerf_advance|(?:\.\./)+personal/')
        for path in sorted(paths):
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertIsNone(stage.search(path.name))
                self.assertIsNone(stage.search(path.read_text()))
                self.assertIsNone(private.search(path.read_text()))


if __name__ == '__main__':
    unittest.main()
