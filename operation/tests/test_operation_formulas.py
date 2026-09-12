# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Check production formulas and report contracts without Torch or devices."""
import ast
import contextlib
import io
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_formula(op, elements=1, shape=(1, 1)):
    path = ROOT / "benchmarks" / op / "main.py"
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "build_case")
    # Execute the real count setup and lambda, excluding tensor allocation
    # and benchmark execution. A minimal tensor double supplies numel().
    statements = [n for n in main.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id in
                          {"elements", "op2flops"} for t in n.targets)]
    namespace = {"a": SimpleNamespace(numel=lambda: elements), "shape": shape}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace["op2flops"]


def load_result_functions():
    path = ROOT / "benchmarks/drivers/calculate.py"
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"cal_perf", "print_result"}]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace


class FormulaTests(unittest.TestCase):
    def test_additive_work_per_element(self):
        for op in ("add", "sub", "rsub"):
            for elements in (1, 6, 1048576):
                with self.subTest(op=op, elements=elements):
                    count = load_formula(op, elements=elements)
                    self.assertEqual(count(1), elements)
                    self.assertEqual(count(0), 0)
                    self.assertEqual(count(5), 5 * elements)
                    self.assertEqual(count(10), 2 * count(5))
                    self.assertEqual(load_formula(op, elements=2 * elements)(5),
                                     2 * count(5))

    def test_triu_counts_all_output_elements_linearly(self):
        for rows, cols in ((1, 1), (1, 7), (7, 1), (4, 4), (3, 8), (8, 3)):
            with self.subTest(shape=(rows, cols)):
                count = load_formula("triu", shape=(rows, cols))
                self.assertEqual(count(1), rows * cols)
                self.assertEqual(count(0), 0)
                self.assertEqual(count(5), 5 * rows * cols)
                self.assertEqual(count(10), 2 * count(5))
                self.assertEqual(load_formula("triu", shape=(2 * rows, cols))(5),
                                 2 * count(5))

    def test_conversions_and_existing_report_patterns(self):
        functions = load_result_functions()
        tree = ast.parse((ROOT / "helper/render.py").read_text())
        patterns = ast.literal_eval(next(
            n.value for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "regex_dict"
                    for t in n.targets)))
        for op in ("add", "sub", "rsub", "triu"):
            with self.subTest(op=op):
                count = load_formula(op, elements=2_000_000_000,
                                     shape=(40000, 50000))
                # 2e9 operations / 100us = 20 TFLOPS; 50us = 40 TFLOPS.
                result = functions["cal_perf"](0.0001, 0.00005, count, "100")
                self.assertEqual(result, (100, 50, 10000, 20000, 20, 40, 20, 40))
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    functions["print_result"](
                        SimpleNamespace(oplib="nativetorch", dataformat="FP32"),
                        op, *result, True, 150, 100)
                expected = {"tflops": "20.0", "kernel_clock": "40.0",
                            "fu_cputime": "20.0%", "kerneltime": "40.0%",
                            "cpu_time": "100.0", "kernel_time": "50.0",
                            "cpu_ops": "10000.0", "kernel_ops": "20000.0"}
                for key, value in expected.items():
                    match = re.search(patterns[key], output.getvalue())
                    self.assertIsNotNone(match, key)
                    self.assertEqual(match.group(1), value, key)


if __name__ == "__main__":
    unittest.main()
