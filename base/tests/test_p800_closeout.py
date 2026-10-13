"""Regression for bugs discovered by the September 30 hardware acceptance."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from benchmarks.day6_contract import allreduce_timed_coefficient,capacity_release_verified

class CloseoutRegression(unittest.TestCase):
    def test_allreduce_inputs_remain_zero_for_two_to_eight_ranks(self):
        import torch
        base=torch.arange(1024,dtype=torch.float32).remainder_(251).add_(9)
        for size in range(2,9):
            inputs=[base*allreduce_timed_coefficient(rank,size) for rank in range(size)]
            reduced=torch.stack(inputs).sum(0)
            self.assertTrue(torch.equal(reduced,torch.zeros_like(base)),size)
            for _ in range(100):reduced*=size
            self.assertTrue(torch.isfinite(reduced).all(),size)
        # The prior two-rank-only expression is detectably wrong at eight ranks.
        self.assertFalse(torch.equal(base*sum([1]+[-1]*7),torch.zeros_like(base)))

    def test_capacity_release_never_compares_bytes_to_mib(self):
        self.assertFalse(capacity_release_verified(96000,4096*(1<<20)))
        self.assertFalse(capacity_release_verified(96000,96000))
        self.assertTrue(capacity_release_verified(96000,91200*(1<<20)))
        self.assertFalse(capacity_release_verified(96000,91200*(1<<20)-1))
        self.assertFalse(capacity_release_verified(None,96000*(1<<20)))

    def test_provider_import_in_clean_container_python_path(self):
        # CPU tests normally insert base/ and hide this startup regression.
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,'-S','-B','-c',
            'from base.vendors.kunlunxin.provider import binding_records; assert callable(binding_records)'],
            cwd=BASE.parent,env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
