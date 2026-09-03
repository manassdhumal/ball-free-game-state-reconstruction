# Robustness Statistical Validation (Step 74)

This document describes the experiment grid, metrics, and reproducible output
locations for the Step 74 robustness expansion. It is a companion to the
code in `src/validation/robustness_experiment.py` and
`configs/robustness_experiment.yaml`.

Key design points
- Conditions: `CLEAN`, `SYNTHETIC-DEGRADED`, `NOISE-MITIGATED` (identical inference contract)
- Windows: 0–60s, 300–360s, 600–660s, 900–960s
- Seeds: 72, 73, 74, 75, 76
- Severities: mild, moderate, severe
- Event matching tolerance: 1.0s (25 frames at 25 FPS)

Outputs
- results/metrics/robustness_expansion/ : CSV/JSON summaries per run
- results/runs/robustness_expansion/ : per-run tactical scores and resolved configs
- results/figures/robustness_expansion/ : figures

Reproducibility record (to be appended by the runner)
- match_id, start_frame, end_frame, duration, seed, severity, degradation parameters,
  mitigation parameters, possession parameters, tactical parameters, orientation convention,
  event matching tolerance

"""
