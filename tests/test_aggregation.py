import os
import json
import tempfile
import shutil
import unittest
from pathlib import Path


def write_csv(path: Path, conditions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('condition,value\n')
        for c in conditions:
            f.write(f"{c},1\n")


class TestAggregator(unittest.TestCase):
    def run_agg_in_tmp(self, setup_fn, write_integrity=False):
        tmp = tempfile.mkdtemp()
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            setup_fn()
            from scripts import aggregate_robustness_summary as agg
            return agg.aggregate(write_integrity=write_integrity)
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmp)

    def test_one_valid_canonical(self):
        def setup():
            metrics = Path('results/metrics/robustness_expansion')
            runs = Path('results/runs/robustness_expansion')
            write_csv(metrics / 'robustness_benchmark_exp_0_60s_seed_1_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
            (runs / 'robustness_benchmark_exp_0_60s_seed_1_sev_mild_tactical_scores.csv').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_1_sev_mild_tactical_scores.csv').write_text('a')
            (runs / 'robustness_benchmark_exp_0_60s_seed_1_sev_mild_config.json').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_1_sev_mild_config.json').write_text('{}')
        integrity = self.run_agg_in_tmp(setup, write_integrity=False)
        self.assertEqual(integrity['valid_unique_runspecs'], 1)
        self.assertEqual(integrity['duplicate_runs'], 0)
        self.assertEqual(integrity['invalid_runspecs'], 0)
        self.assertEqual(integrity['valid_condition_rows'], 3)

    def test_canonical_plus_legacy_duplicate(self):
        def setup():
            metrics = Path('results/metrics/robustness_expansion')
            runs = Path('results/runs/robustness_expansion')
            write_csv(metrics / 'robustness_benchmark_exp_0_60s_seed_2_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
            (runs / 'robustness_benchmark_exp_0_60s_seed_2_sev_mild_tactical_scores.csv').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_2_sev_mild_tactical_scores.csv').write_text('a')
            (runs / 'robustness_benchmark_exp_0_60s_seed_2_sev_mild_config.json').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_2_sev_mild_config.json').write_text('{}')
            # legacy duplicate summary nested
            nested = Path('results/metrics/robustness_expansion/robustness_expansion')
            write_csv(nested / 'robustness_benchmark_exp_0_60s_seed_2_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
        integrity = self.run_agg_in_tmp(setup, write_integrity=False)
        self.assertEqual(integrity['valid_unique_runspecs'], 1)
        self.assertEqual(integrity['duplicate_runs'], 1)
        self.assertEqual(integrity['valid_condition_rows'], 3)
        # file count does not equal valid runs
        summary_files = len(list(Path('results/metrics').rglob('*summary.csv')))
        self.assertNotEqual(summary_files, integrity['valid_unique_runspecs'])

    def test_missing_one_condition(self):
        def setup():
            metrics = Path('results/metrics/robustness_expansion')
            runs = Path('results/runs/robustness_expansion')
            write_csv(metrics / 'robustness_benchmark_exp_0_60s_seed_3_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED'])
            (runs / 'robustness_benchmark_exp_0_60s_seed_3_sev_mild_tactical_scores.csv').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_3_sev_mild_tactical_scores.csv').write_text('a')
            (runs / 'robustness_benchmark_exp_0_60s_seed_3_sev_mild_config.json').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_3_sev_mild_config.json').write_text('{}')
        integrity = self.run_agg_in_tmp(setup, write_integrity=False)
        self.assertEqual(integrity['valid_unique_runspecs'], 0)
        self.assertEqual(integrity['invalid_runspecs'], 1)

    def test_duplicate_condition_rows(self):
        def setup():
            metrics = Path('results/metrics/robustness_expansion')
            runs = Path('results/runs/robustness_expansion')
            # duplicate CLEAN rows
            write_csv(metrics / 'robustness_benchmark_exp_0_60s_seed_4_sev_mild_summary.csv', ['CLEAN','CLEAN','NOISE-MITIGATED'])
            (runs / 'robustness_benchmark_exp_0_60s_seed_4_sev_mild_tactical_scores.csv').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_4_sev_mild_tactical_scores.csv').write_text('a')
            (runs / 'robustness_benchmark_exp_0_60s_seed_4_sev_mild_config.json').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_4_sev_mild_config.json').write_text('{}')
        integrity = self.run_agg_in_tmp(setup, write_integrity=False)
        self.assertEqual(integrity['valid_unique_runspecs'], 0)
        self.assertEqual(integrity['invalid_runspecs'], 1)

    def test_malformed_config_json(self):
        def setup():
            metrics = Path('results/metrics/robustness_expansion')
            runs = Path('results/runs/robustness_expansion')
            write_csv(metrics / 'robustness_benchmark_exp_0_60s_seed_5_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
            (runs / 'robustness_benchmark_exp_0_60s_seed_5_sev_mild_tactical_scores.csv').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_5_sev_mild_tactical_scores.csv').write_text('a')
            (runs / 'robustness_benchmark_exp_0_60s_seed_5_sev_mild_config.json').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_5_sev_mild_config.json').write_text('{ malformed json')
        integrity = self.run_agg_in_tmp(setup, write_integrity=False)
        self.assertEqual(integrity['valid_unique_runspecs'], 0)
        self.assertEqual(integrity['invalid_runspecs'], 1)

    def test_report_entry_mismatch(self):
        def setup():
            metrics = Path('results/metrics/robustness_expansion')
            runs_dir = Path('results/runs/robustness_expansion')
            runs_dir.mkdir(parents=True, exist_ok=True)
            rp = runs_dir / 'full_run_report.json'
            rp.write_text(json.dumps({'runs': [{'run_id': 'exp_nonexistent_seed_1_sev_mild', 'status': 'ok'}]}))
        integrity = self.run_agg_in_tmp(setup, write_integrity=False)
        rf = integrity['report_filesystem_discrepancies']
        self.assertTrue(len(rf.get('in_report_not_filesystem', [])) > 0)

    def test_two_legacy_duplicates(self):
        def setup():
            metrics = Path('results/metrics/robustness_expansion')
            runs = Path('results/runs/robustness_expansion')
            write_csv(metrics / 'robustness_benchmark_exp_0_60s_seed_6_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
            (runs / 'robustness_benchmark_exp_0_60s_seed_6_sev_mild_tactical_scores.csv').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_6_sev_mild_tactical_scores.csv').write_text('a')
            (runs / 'robustness_benchmark_exp_0_60s_seed_6_sev_mild_config.json').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_6_sev_mild_config.json').write_text('{}')
            nested = Path('results/metrics/robustness_expansion/robustness_expansion')
            write_csv(nested / 'robustness_benchmark_exp_0_60s_seed_6_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
            nested2 = Path('results/metrics/robustness_expansion/robustness_expansion/robustness_expansion')
            write_csv(nested2 / 'robustness_benchmark_exp_0_60s_seed_6_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
        integrity = self.run_agg_in_tmp(setup, write_integrity=False)
        self.assertEqual(integrity['valid_unique_runspecs'], 1)
        self.assertEqual(integrity['duplicate_runs'], 1)
        self.assertTrue(integrity['noncanonical_artifacts'] >= 1)

    def test_different_seeds_remain_separate(self):
        def setup():
            metrics = Path('results/metrics/robustness_expansion')
            runs = Path('results/runs/robustness_expansion')
            # seed 7
            write_csv(metrics / 'robustness_benchmark_exp_0_60s_seed_7_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
            (runs / 'robustness_benchmark_exp_0_60s_seed_7_sev_mild_tactical_scores.csv').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_7_sev_mild_tactical_scores.csv').write_text('a')
            (runs / 'robustness_benchmark_exp_0_60s_seed_7_sev_mild_config.json').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_7_sev_mild_config.json').write_text('{}')
            # seed 8
            write_csv(metrics / 'robustness_benchmark_exp_0_60s_seed_8_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
            (runs / 'robustness_benchmark_exp_0_60s_seed_8_sev_mild_tactical_scores.csv').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_8_sev_mild_tactical_scores.csv').write_text('a')
            (runs / 'robustness_benchmark_exp_0_60s_seed_8_sev_mild_config.json').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_8_sev_mild_config.json').write_text('{}')
        integrity = self.run_agg_in_tmp(setup, write_integrity=False)
        self.assertEqual(integrity['valid_unique_runspecs'], 2)

    def test_different_severities_remain_separate(self):
        def setup():
            metrics = Path('results/metrics/robustness_expansion')
            runs = Path('results/runs/robustness_expansion')
            # mild
            write_csv(metrics / 'robustness_benchmark_exp_0_60s_seed_9_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
            (runs / 'robustness_benchmark_exp_0_60s_seed_9_sev_mild_tactical_scores.csv').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_9_sev_mild_tactical_scores.csv').write_text('a')
            (runs / 'robustness_benchmark_exp_0_60s_seed_9_sev_mild_config.json').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_9_sev_mild_config.json').write_text('{}')
            # moderate
            write_csv(metrics / 'robustness_benchmark_exp_0_60s_seed_9_sev_moderate_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
            (runs / 'robustness_benchmark_exp_0_60s_seed_9_sev_moderate_tactical_scores.csv').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_9_sev_moderate_tactical_scores.csv').write_text('a')
            (runs / 'robustness_benchmark_exp_0_60s_seed_9_sev_moderate_config.json').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_9_sev_moderate_config.json').write_text('{}')
        integrity = self.run_agg_in_tmp(setup, write_integrity=False)
        self.assertEqual(integrity['valid_unique_runspecs'], 2)

    def test_deterministic_output(self):
        def setup():
            metrics = Path('results/metrics/robustness_expansion')
            runs = Path('results/runs/robustness_expansion')
            write_csv(metrics / 'robustness_benchmark_exp_0_60s_seed_10_sev_mild_summary.csv', ['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'])
            (runs / 'robustness_benchmark_exp_0_60s_seed_10_sev_mild_tactical_scores.csv').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_10_sev_mild_tactical_scores.csv').write_text('a')
            (runs / 'robustness_benchmark_exp_0_60s_seed_10_sev_mild_config.json').parent.mkdir(parents=True, exist_ok=True)
            (runs / 'robustness_benchmark_exp_0_60s_seed_10_sev_mild_config.json').write_text('{}')
        a = self.run_agg_in_tmp(setup, write_integrity=False)
        b = self.run_agg_in_tmp(setup, write_integrity=False)
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
