# Report Table Specifications

All tables should preserve the runspec as the primary experimental unit. Generated CSVs are derived artifacts and remain subject to the repository result-storage policy.

## Table 1 - Dataset and Experimental Configuration

- **Source:** `results/metrics/final_analysis/experimental_design.csv`, `configs/robustness_experiment.yaml`
- **Columns:** window labels, seeds, severities, conditions, counts, and configuration parameters
- **Interpretation:** Defines the frozen population: 60 unique runspecs and 180 condition observations across 4 windows, 5 seeds, 3 severities, and 3 conditions.
- **Caveat:** Seeds are repeated degradation realizations, not independent matches; the benchmark uses one Metrica match.

## Table 2 - Tracking Quality by Condition

- **Source:** `results/metrics/final_analysis/tracking_quality_by_condition.csv`
- **Columns:** `condition`, `tracking_count`, `tracking_mean`, `tracking_median`, `tracking_std`, `tracking_min`, `tracking_max`, and confidence-interval columns
- **Interpretation:** Compares tracking missingness or the primary tracking-quality metric across conditions.
- **Caveat:** Baseline Metrica missingness can reflect off-field or substituted players and should not automatically be read as detector failure.

## Table 3 - Ball-Free Possession Performance

- **Source:** `results/metrics/final_analysis/possession_by_condition.csv`
- **Columns:** `condition`, `possession_count`, `possession_mean`, `possession_median`, `possession_std`, range, and CI columns
- **Interpretation:** Reports event-based possession precision, recall, and F1 summaries by condition; highlight F1 means 0.49199, 0.23451, and 0.41328.
- **Caveat:** Metrica events are post-hoc reference annotations, not perfect possession ground truth; results depend on the 1.0-second matching tolerance.

## Table 4 - Tactical Robustness Relative to CLEAN

- **Source:** `results/metrics/final_analysis/tactical_robustness_by_condition.csv`
- **Columns:** condition-level tactical MAE, median absolute error, correlation/agreement fields where available, and counts
- **Interpretation:** Measures sensitivity of TAS to degradation relative to CLEAN; mean MAE is 0.00000, 0.15635, and 0.10888 by condition.
- **Caveat:** CLEAN is a reference for comparison, not tactical ground truth; zero CLEAN error is expected from the reference construction.

## Table 5 - Performance by Severity

- **Source:** `results/metrics/final_analysis/possession_by_severity.csv`, `results/metrics/final_analysis/tactical_by_severity.csv`
- **Columns:** severity, condition, sample count, primary possession/tactical metrics, and uncertainty summaries
- **Interpretation:** Shows how controlled mild, moderate, and severe stress changes downstream performance.
- **Caveat:** Severity labels describe this synthetic configuration only and are not universal tracker-quality categories.

## Table 6 - Window Robustness

- **Source:** `results/metrics/final_analysis/robustness_by_window.csv`
- **Columns:** window, condition, sample count, possession F1, tactical MAE, and recovery or tracking metrics as available
- **Interpretation:** Assesses whether the aggregate effect is consistent across the four 60-second windows.
- **Caveat:** Four windows from one match do not establish full-match or cross-match generalization.

## Table 7 - Seed Robustness

- **Source:** `results/metrics/final_analysis/robustness_by_seed.csv`
- **Columns:** seed, condition, sample count, possession F1, tactical MAE, and recovery metrics as available
- **Interpretation:** Shows variation across seeds 72-76.
- **Caveat:** Seeds vary the synthetic degradation process; they are not independent data sources.

## Table 8 - Mitigation Recovery

- **Source:** `results/metrics/final_analysis/mitigation_recovery.csv`, `results/metrics/final_analysis/mitigation_recovery_by_seed.csv`
- **Columns:** run identifier, metric, clean, degraded, mitigated, absolute improvement, recovery fraction
- **Interpretation:** Quantifies movement from degraded toward the clean reference. Mean possession F1 improvement is 0.17877, with recovery fraction 0.69432.
- **Caveat:** Recovery above or below 100% can occur in individual runs; it does not mean mitigation reconstructs the true ball or guarantees clean-level output.

## Table 9 - Hypothesis Evaluation

- **Source:** `results/metrics/final_analysis/final_statistical_report.json`, `docs/final_research_findings.md`
- **Columns:** hypothesis, operational comparison, estimate, uncertainty/effect summary, decision, caveat
- **Interpretation:** H1, H2, and H3 are supported within the frozen benchmark.
- **Caveat:** Decisions are scoped to synthetic degradation, the tested Metrica source, and heuristic downstream tasks; they do not establish universal robustness or real-world GSR success.
