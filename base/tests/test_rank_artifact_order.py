"""Two-digit rank artifact ordering in the bounded executor (11/16-rank surfaces)."""
import tempfile, unittest
from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from executors.bounded_benchmark import rank_artifact_names


class RankArtifactOrderTests(unittest.TestCase):
 def test_two_digit_ranks_sort_numerically(self):
  with tempfile.TemporaryDirectory() as d:
   out = Path(d)
   for r in range(12):
    (out / ('metric-rank-%d.json' % r)).write_text('{}')
   got = rank_artifact_names(out, 'metric-rank-', 12)
   self.assertEqual(got, ['metric-rank-%d.json' % r for r in range(12)])
   lexicographic = sorted(path.name for path in out.glob('metric-rank-*.json'))
   self.assertNotEqual(lexicographic, got, 'test lost its two-digit regression bite')

 def test_missing_or_extra_rank_breaks_exact_match(self):
  for mode in ['missing', 'extra']:
   with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
    out = Path(d)
    for r in range(11):
     if r != 3:
      (out / ('metric-rank-%d.json' % r)).write_text('{}')
    if mode == 'extra':
     (out / 'metric-rank-99.json').write_text('{}')
    self.assertNotEqual(rank_artifact_names(out, 'metric-rank-', 11),
                        ['metric-rank-%d.json' % r for r in range(11)])

 def test_single_digit_order_unchanged(self):
  with tempfile.TemporaryDirectory() as d:
   out = Path(d)
   for r in range(8):
    (out / ('correctness-rank-%d.json' % r)).write_text('{}')
   self.assertEqual(rank_artifact_names(out, 'correctness-rank-', 8),
                    ['correctness-rank-%d.json' % r for r in range(8)])


if __name__ == '__main__':
 unittest.main()
