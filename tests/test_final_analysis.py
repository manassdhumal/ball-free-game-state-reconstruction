import unittest
import os
from src.validation import final_analysis as fa
from pathlib import Path


class TestFinalAnalysis(unittest.TestCase):
    def test_load_canonical_and_counts(self):
        df = fa.load_canonical_summaries(require_n_runs=60)
        self.assertEqual(df['run_id'].nunique(), 60)
        self.assertEqual(len(df), 180)

    def test_run_analysis_outputs(self):
        out = fa.run_analysis()
        self.assertIn('n_runs', out)
        self.assertEqual(out['n_runs'], 60)
        odir = Path('results') / 'metrics' / 'final_analysis'
        self.assertTrue((odir / 'final_statistical_report.json').exists())
        self.assertTrue((odir / 'final_statistical_summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
