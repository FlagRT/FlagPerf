# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Adapted-model registry: lookup, resolution precedence and CLI defaults."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flagprof.training.registry import SCORES, load_registry, lookup, resolve, resolve_task


def write_registry(root, data):
    path = Path(root) / 'registry.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


class RegistryContractTests(unittest.TestCase):
    def test_shipped_registry_is_valid_and_covers_adapted_models(self):
        data = load_registry()
        entries = data['entries']
        for name, provenance in (('bge-m3', 'trained-verified'), ('Qwen3-Embedding-0.6B', 'trained-verified'),
                                 ('Qwen3-Reranker-0.6B', 'trained-verified'),
                                 ('bge-reranker-base', 'trained-verified')):
            self.assertEqual(entries[name]['provenance'], provenance, name)
        self.assertEqual(entries['Qwen3-Reranker-8B']['rerank_score'], 'causal_yesno')
        self.assertIn('128 GB', entries['Qwen3-Reranker-8B']['note'])
        self.assertEqual(entries['bge-m3']['pooling'], 'cls')
        self.assertEqual(entries['Qwen3-Embedding-0.6B']['pooling'], 'last')

    def test_load_rejects_bad_schema_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'schema'):
                load_registry(write_registry(directory, dict(schema='nope')))
            bad = dict(schema='model-registry-v1', entries={'m': dict(task='embedding', pooling='max')})
            with self.assertRaisesRegex(ValueError, 'pooling'):
                load_registry(write_registry(directory, bad))
            bad = dict(schema='model-registry-v1',
                       prefixes=[dict(prefix='', task='rerank', rerank_score='sequence_logit')])
            with self.assertRaisesRegex(ValueError, 'prefix'):
                load_registry(write_registry(directory, bad))

    def test_lookup_exact_then_longest_prefix(self):
        data = load_registry()
        self.assertEqual(lookup('/models/bge-m3', data)['pooling'], 'cls')
        self.assertEqual(lookup('hf:bge-m3', data)['matched'], 'bge-m3')
        self.assertEqual(lookup('/models/Qwen3-Reranker-0.6B/', data)['rerank_score'], 'causal_yesno')
        exact = lookup('/models/bge-reranker-base', data)  # exact entry beats any prefix
        self.assertEqual(exact['matched'], 'bge-reranker-base')
        self.assertEqual(exact['rerank_score'], 'sequence_logit')
        derived = lookup('/models/bge-reranker-XX-new-variant', data)  # prefix fallback
        self.assertEqual(derived['matched'], 'bge-reranker-')
        self.assertEqual(derived['rerank_score'], 'sequence_logit')
        self.assertIsNone(lookup('/models/chinese-roberta-wwm-ext', data))
        self.assertIsNone(lookup('/models/Some-Unknown-Model', data))

    def test_resolution_prefers_explicit_cli_value(self):
        data = load_registry()
        self.assertEqual(resolve('mean', 'bge-m3', kind='pooling', registry=data), ('mean', 'cli'))
        self.assertEqual(resolve(None, 'bge-m3', kind='pooling', registry=data), ('cls', 'registry'))
        self.assertEqual(resolve(None, 'Qwen3-Reranker-8B', kind='rerank_score', registry=data),
                         ('causal_yesno', 'registry'))
        self.assertEqual(resolve(None, '/models/unknown', kind='pooling', registry=data), (None, None))
        self.assertEqual(tuple(sorted(SCORES)), ('causal_yesno', 'sequence_logit'))

    def test_task_resolution_and_causal_lm_entries(self):
        data = load_registry()
        self.assertEqual(resolve_task('/models/bge-m3', data), 'embedding')
        self.assertEqual(resolve_task('hf:Qwen3-Reranker-8B', data), 'rerank')
        self.assertIsNone(resolve_task('/models/chinese-roberta-wwm-ext', data))
        causal = {name: entry for name, entry in data['entries'].items()
                  if entry.get('task') == 'causal-lm'}
        for name, entry in causal.items():
            self.assertIn(entry['provenance'], ('trained-verified', 'official-default'), name)
            self.assertNotIn('pooling', entry, name)
            self.assertNotIn('rerank_score', entry, name)

    def test_causal_lm_entry_resolves_task_without_extra_options(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = dict(schema='model-registry-v1', entries={
                'some-generative-rec': dict(task='causal-lm', provenance='official-default')})
            data = load_registry(write_registry(directory, custom))
            self.assertEqual(resolve_task('/models/some-generative-rec', data), 'causal-lm')


class CliDefaultTests(unittest.TestCase):
    def test_parser_defaults_leave_options_resolvable(self):
        from flagprof.training.__main__ import parser
        args = parser().parse_args(['--task', 'rerank', '--model', 'bge-reranker-base',
                                    '--package', '/pkg', '--outputs', 'out'])
        self.assertIsNone(args.rerank_score)
        args = parser().parse_args(['--task', 'embedding', '--model', 'bge-m3', '--outputs', 'out'])
        self.assertIsNone(args.pooling)
        self.assertIsNone(args.inputs)

    def test_ensure_generated_creates_and_reuses_corpus(self):
        import hashlib
        from flagprof.training.__main__ import ensure_generated
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / 'run'
            path = ensure_generated('corpus', str(outputs), 531)
            corpus = Path(path) / 'groups.jsonl'
            self.assertTrue(corpus.is_file())
            lines = corpus.read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines), 32)
            digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
            again = ensure_generated('corpus', str(outputs), 531)
            self.assertEqual(again, path)
            self.assertEqual(hashlib.sha256((Path(again) / 'groups.jsonl').read_bytes()).hexdigest(), digest)

    def test_ensure_generated_reuses_existing_package_marker(self):
        from flagprof.training.__main__ import ensure_generated
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / 'run'
            package = outputs.with_name(outputs.name + '-default-package')
            package.mkdir(parents=True)
            (package / 'manifest.json').write_text('{}', encoding='utf-8')
            self.assertEqual(ensure_generated('ctr-package', str(outputs), 531), str(package))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'training semantics require torch')
class ExecuteErrorPathTests(unittest.TestCase):
    def parse(self, extra):
        from flagprof.training.__main__ import parser
        return parser().parse_args(extra)

    def execute(self, extra):
        from flagprof.training.__main__ import execute
        execute(self.parse(extra))

    def test_ctr_error_lists_adapted_architectures(self):
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'ctr-run')
            with self.assertRaisesRegex(ValueError, 'dssm'):
                self.execute(['--task', 'ctr', '--package', directory, '--outputs', out])

    def test_unadapted_embedding_model_is_asked_for_pooling(self):
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'emb-run')
            with self.assertRaisesRegex(ValueError, 'not in'):
                self.execute(['--task', 'embedding', '--model', '/models/whatever-net',
                              '--inputs', directory, '--outputs', out])

    def test_unadapted_rerank_model_is_asked_for_score(self):
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'rerank-run')
            with self.assertRaisesRegex(ValueError, 'not in'):
                self.execute(['--task', 'rerank', '--model', '/models/whatever-net',
                              '--package', directory, '--outputs', out])

    def test_ddp_rejects_default_inputs_and_package(self):
        import os
        from unittest import mock
        environment = {'WORLD_SIZE': '2', 'RANK': '0', 'LOCAL_RANK': '0'}
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'ddp-run')
            with mock.patch.dict(os.environ, environment):
                with self.assertRaisesRegex(ValueError, 'explicit --inputs'):
                    self.execute(['--task', 'embedding', '--model', '/models/bge-m3', '--ddp', '--outputs', out])
                with self.assertRaisesRegex(ValueError, 'explicit --package'):
                    self.execute(['--task', 'ctr', '--ctr-arch', 'dlrm', '--ddp', '--outputs', out])


if __name__ == '__main__':
    unittest.main()
