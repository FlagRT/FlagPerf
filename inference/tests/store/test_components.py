# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Compiler selection and actual-provider counterexamples, without torch."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from flagprof.components import compiler
from flagprof.launch import launch


class ComponentTests(unittest.TestCase):
    def test_effective_configuration_can_roundtrip_through_worker(self):
        from flagprof.probes.config import resolve
        for preset in ('coverage','accuracy','perf','modules','perf-lite'):
            session,effective=resolve(preset,{'flagtree':'off','vendor_compiler':'/vendor-compiler'})
            again,normalized=resolve(preset,effective)
            self.assertEqual(again,session);self.assertEqual(normalized,effective)

    def bundle(self, root):
        (root/'triton').mkdir(parents=True)
        (root/'triton/__init__.py').write_text("__version__='fixture'\n")
        files={'triton/__init__.py':compiler.file_sha(root/'triton/__init__.py')}
        (root/'manifest.json').write_text(json.dumps(dict(source_image='sealed-fixture',files=files)))
        return root

    def test_explicit_selection_and_loaded_conflict(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(sys,'path',list(sys.path)), patch.dict(os.environ,{},clear=False):
            root=self.bundle(Path(tmp)/'vendor')
            compiler.select(dict(flagtree='off',vendor_compiler=str(root)),cache_root=Path(tmp)/'cache')
            self.assertEqual(sys.path[0],str(root))
            self.assertIn('compiler-off-',os.environ['TRITON_CACHE_DIR'])
            loaded=types.SimpleNamespace(__file__=str(Path(tmp)/'tree/triton/__init__.py'))
            with patch.dict(sys.modules,{'triton':loaded}):
                with self.assertRaisesRegex(ValueError,'already imported'):
                    compiler.select(dict(flagtree='off',vendor_compiler=str(root)))
            with patch.object(compiler,'distribution_entry',return_value=root/'triton/__init__.py'):
                compiler.select(dict(flagtree='on'),cache_root=Path(tmp)/'cache')
                self.assertIn('compiler-on-',os.environ['TRITON_CACHE_DIR'])
            (root/'triton/__init__.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'differs'):
                compiler.select(dict(flagtree='off',vendor_compiler=str(root)))

    def test_default_and_env_precedence(self):
        with patch.object(sys,'path',list(sys.path)):
            before=list(sys.path);compiler.select({})
            self.assertEqual(sys.path,before)
        with patch.dict(os.environ,{'FLAGPROF_VENDOR_COMPILER':'/env/bundle'}):
            self.assertEqual(compiler.bundle_path({}),Path('/env/bundle'))
            self.assertEqual(compiler.bundle_path({'vendor_compiler':'/cli/bundle'}),Path('/cli/bundle'))

    def test_installed_flagtree_does_not_label_vendor_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.bundle(Path(tmp)/'vendor')
            compiler._identity.cache_clear()
            with patch.object(compiler,'distribution_entry',return_value=Path(tmp)/'tree/triton/__init__.py'):
                value=compiler._identity(str(root/'triton/__init__.py'),'fixture')
            self.assertEqual(value['provider'],'vendor')
            compiler._identity.cache_clear()

    def test_conflicting_backend_fails_before_workers(self):
        with self.assertRaisesRegex(ValueError,'conflicts'):
            launch(['python','target.py'],store='/unused',flagcx='on',comm_backend='hccl')

    def test_compiler_declaration_is_specific_and_rank_is_required(self):
        from tests.store.test_oracle import OracleTests
        from flagprof.report import generate_oracle
        with tempfile.TemporaryDirectory() as tmp:
            owner=OracleTests();owner.root=Path(tmp)
            a=owner.fixture('a');b=owner.fixture('b')
            for run,provider,triton in ((a,'vendor','3.2'),(b,'flagtree','3.6')):
                path=run/'attachments/f/state.json';state=json.loads(path.read_text())
                state['oracle_identity'].update(compiler={'provider':provider},software={'torch':'same','triton':triton})
                path.write_text(json.dumps(state))
            declare={'differences':['compiler','software.triton']}
            generate_oracle(a,b,Path(tmp)/'ok',declare=declare)
            result=json.loads((Path(tmp)/'ok/result.json').read_text())['results']
            self.assertEqual(result['oracle.identity_diff']['status'],'ok')
            path=b/'attachments/f/state.json';state=json.loads(path.read_text())
            state['oracle_identity']['software']['torch']='changed';path.write_text(json.dumps(state))
            generate_oracle(a,b,Path(tmp)/'blocked',declare=declare)
            result=json.loads((Path(tmp)/'blocked/result.json').read_text())['results']
            self.assertEqual(result['oracle.identity_diff']['status'],'unavailable')
            for run,rank in ((a,0),(b,1)):
                path=run/'attachments/f/state.json';state=json.loads(path.read_text())
                state['oracle_identity'].update(rank=rank,parallelism={'world_size':2,'plan':'auto'})
                path.write_text(json.dumps(state))
            generate_oracle(a,b,Path(tmp)/'rank-blocked',declare={'differences':['compiler','software','rank']})
            result=json.loads((Path(tmp)/'rank-blocked/result.json').read_text())['results']
            self.assertEqual(result['oracle.identity_diff']['status'],'unavailable')


if __name__=='__main__':unittest.main()
