# Project Status: Ball-Free Game State Reconstruction

## 1. Project Overview & Current State
- **Project Name:** Ball-Free Game State Reconstruction (GSR)
- **Current Milestone:** Phase 6 - Original-Scope Completion: SoccerNet-GSR Perception, Broadcast Validation, Tactical Pass Modeling & Final Robustness Benchmark
- **Current Step:** Step 82 - Tactical Pass Model (COMPLETE); Step 83 is next and not started.
- **Data Integrity Policy:** Raw tracking and event data in `data/raw/` (including `metrica/` and `skillcorner/`) remains strictly untouched, read-only, and gitignored. All intermediate/synthetic/mitigated outputs are generated into `data/interim/` and figures into `results/figures/`.

---

## 2. Completed Milestones & Components
- **Canonical Schemas:**
  - `docs/canonical_tracking_schema.md` (v0.2.0): Standard tall tabular representation (`match_id`, `frame`, `timestamp`, `player_id`, `team`, `x`, `y`, `confidence`, `visible`).
  - `docs/canonical_event_schema.md`: Unified match event schema.
- **Data Parsers & Test Suites:**
  - `src/data/metrica_parser.py`: Ingestion of Metrica tracking CSVs into canonical player and ball DataFrames with strict integrity validation.
  - `tests/test_metrica_parser.py`: Unit tests for canonical tracking validation and synchronization checks.
  - `src/data/metrica_event_parser.py`: Ingestion of Metrica raw events.
  - `tests/test_metrica_event_parser.py`: Unit tests for event schema validation.
  - `src/data/skillcorner_parser.py`: Ingestion of SkillCorner Open Data JSONL + match metadata into canonical tracking schema (v0.2.0) with normalized coordinates and ball separation.
  - `tests/test_skillcorner_parser.py`: Unit tests for SkillCorner parser against real match data (match 1886347) and mocked edge cases (19 tests).
- **Documentation & Notes:**
  - `docs/canonical_tracking_schema.md` (v0.2.0): Standard tall tabular representation.
  - `docs/canonical_event_schema.md`: Unified match event schema.
  - `docs/skillcorner_data_notes.md`: Exhaustive 25-point dataset specification for SkillCorner Open Data (Australian A-League 2024/2025).
  - `docs/noise_mitigation.md`: Technical documentation defining algorithms, mathematical formulations, metrics, and failure modes.
- **Exploratory Data Analysis & Figures:**
  - `notebooks/01_data_exploration/01_metrica_tracking_quality.ipynb`: Quality audit, coverage per player, and missingness distribution.
  - `notebooks/01_data_exploration/02_metrica_motion_anomalies.ipynb`: Frame-to-frame displacement, velocity thresholding, and anomaly classification.
  - `results/figures/skillcorner_detection_rate_by_match.png`: Detection vs extrapolation rates across 10 matches.
  - `results/figures/skillcorner_visible_players_timeline.png`: Active/detected/extrapolated player count timeline for match 1886347.
- **Synthetic Degradation Framework & Validation:**
  - `src/noise/degradation.py`: Two-tier degradation engine (observation & identity level) with parameter validation and severity tiers (`clean`, `mild`, `moderate`, `severe`).
  - `tests/test_degradation.py`: 21 unit tests (all passing).
  - `notebooks/03_noise_analysis/01_synthetic_degradation.ipynb`: Degradation demonstration notebook.
  - `notebooks/03_noise_analysis/02_degradation_parameter_validation.ipynb`: Parameter validation study.
- **Noise Mitigation Framework (Completed):**
  - `src/noise/interpolation.py`: Linear trajectory interpolation bounded by configurable `max_gap`, provenance tracking via internal `imputed` column, and strict non-extrapolation bounds.
  - `src/noise/smoothing.py`: Low-pass trajectory filtering supporting Centered Moving Average and Savitzky-Golay polynomial smoothing on contiguous visible segments.
  - `tests/test_interpolation.py`: 7 unit tests (all passing).
  - `tests/test_smoothing.py`: 6 unit tests (all passing).
  - `notebooks/03_noise_analysis/02_noise_mitigation.ipynb`: Mitigation experiment notebook.
  - `results/figures/`: 6 trajectory and jitter mitigation analysis plots.
- **Task-Driven Robustness Benchmark (Completed):**
  - `src/validation/robustness_benchmark.py`: Reproducible CLEAN → SYNTHETIC-DEGRADED → NOISE-MITIGATED downstream benchmark with strict tracking-only inference and post-hoc event evaluation.
  - `configs/robustness_benchmark.yaml`, `tests/test_robustness_benchmark.py`, `docs/robustness_benchmark.md`, and `notebooks/05_tactical_validation/02_robustness_benchmark.ipynb`.
  - Verified Metrica Game 1 frames 1–1500 with seed 72; summary metrics, configuration, TAS values, and SVG comparison stored under `results/`.
- **Infrastructure:**
  - `scripts/setup_gsr_colab.sh`: Google Colab automated setup script.

### Current Research State

- **Research foundation:** Phases 0-5 are complete and form the validated downstream research foundation: canonical ingestion, tracking-quality analysis, noise/degradation and mitigation, ball-free possession/event inference, tactical scoring, and the controlled robustness benchmark with final statistical analysis.
- **Primary remaining research gap:** Genuine broadcast-video perception through SoccerNet-GSR, including real player detections/tracking and GS-HOTA evaluation.
- **Secondary remaining research gaps:** Independent action-spotting validation, explicit-outcome pass-completion probability validation, and a real-broadcast robustness tier.
- **Current blocker:** Authorized SoccerNet-GSR data and matching labels, complete perception checkpoints/runtime, and an approved GPU execution environment are not currently available.

---

## Current Experimental Result

The current quantitative robustness benchmark is **validated but not the final complete benchmark** for the original project scope.

- **Step 72:** Verified single-window Metrica baseline using one 60-second window and one degradation seed. These results are historical/preliminary context.
- **Steps 74-75:** Expanded and statistically analyzed the primary controlled Metrica robustness benchmark across 4 windows, 5 seeds, 3 severities, and 3 conditions. The final aggregate contains 60 unique runspecs and 180 condition observations.
- These results evaluate controlled Metrica conditions: `CLEAN`, `SYNTHETIC-DEGRADED`, and `NOISE-MITIGATED`.
- They do not constitute genuine SoccerNet-GSR perception evaluation and do not establish the final real-broadcast robustness claim.

The controlled benchmark remains intact as the validated development result. A genuine broadcast/GSR condition, GS-HOTA evaluation, independent event validation, pass modeling, counterfactual ranking, and final end-to-end validation remain outstanding.

### Step 81 Closeout - COMPLETE

Step 81 evaluated player-only possession transitions against available Metrica event annotations. The exact setup used Sample Game 1 frames 1-1500 (0-60 seconds at 25 FPS), fixed existing ball-free inference, and post-hoc labels only. Supported mappings were `pass_candidate` to `PASS`, `turnover_candidate` to `BALL LOST`/`BALL OUT`, and `recovery_candidate` to `RECOVERY`; `CHALLENGE` and `SET PIECE` were excluded.

One-to-one matching was evaluated at ±0.20, ±0.50, and ±1.00 seconds. Aggregate TP/FP/FN were `(4,52,24)`, `(6,50,22)`, and `(11,45,17)`, with F1 `0.095238`, `0.142857`, and `0.261905`; the strongest aggregate result was F1 `0.261905` at ±1.00 seconds. Outputs are isolated under `results/step81/`. Focused tests passed 5/5 and the full suite passed 175 tests with 11 expected skips. This is controlled annotated player/event validation, not SoccerNet-GSR or real-broadcast performance; GSR remains BLOCKED / INCOMPLETE and required for original-scope completion.

### Step 82 Closeout - COMPLETE

Step 82 added player-only hypothetical pass candidate generation, interpretable geometry/pressure/support/space features, a deterministic regularized logistic model with an honest heuristic fallback, and counterfactual ranking. The exact experiment used Metrica Sample Game 1 frames 1-1500 (0-60 seconds at 25 FPS), `home_attacks_x1=true`, default candidate distance bounds 0.02-0.70, and fixed ranking weights from `configs/step82_tactical_pass_model.yaml`.

The canonical Metrica event schema exposes 15 PASS source/receiver mappings in this window but no explicit completion outcome. Therefore reliable binary completion labels are 0, the learned model falls back to the fixed heuristic, and ROC-AUC, PR-AUC, Brier, log loss, calibration, and completion ablations are explicitly not estimable. Ranking was evaluated only against those identifiable annotated receivers: 15 states and 149 candidates. The fallback pass ranking achieved MRR 0.4440, NDCG 0.5738, top-1 agreement 0.2667, and top-3 inclusion 0.5333. The nearest-teammate baseline was strongest (MRR 0.6478, top-1 0.5333); most-forward and TAS-style baselines each had MRR 0.3313 and top-1 0.1333. These are target-ranking results, not completion claims.

Implementation is in `src/tactics/pass_candidates.py`, `src/tactics/pass_probability.py`, `src/tactics/pass_ranking.py`, and `src/validation/run_step82.py`; documentation is `docs/step82_tactical_pass_model.md`; configuration is `configs/step82_tactical_pass_model.yaml`; isolated outputs are under `results/step82/`. Focused Step 82 tests passed 5/5 and the full suite passed 180 tests with 11 expected skips. SoccerNet-GSR remains BLOCKED / INCOMPLETE and is not falsely claimed complete.

### Step 80A Closeout - COMPLETE

Step 80A completed a controlled broadcast-specific noise analysis using Metrica Sample Game 1 as a proxy. It covered candidate camera-cut/discontinuity evidence, frame-level missingness, bounded short-gap repair, coordinate jitter, apparent discontinuity, and track fragmentation. All eight configured mitigation methods were compared: none, interpolation, moving average, Savitzky-Golay, causal Kalman-style filtering, and interpolation combined with each filter.

The exact experiment used the 0-60 second window, mild/moderate/severe degradation, and seeds 72 and 73. All 48/48 unique conditions completed with finite metrics and no duplicates or missing rows. `interpolation_kalman` had the best mean possession F1 (`0.419522`) and mean TAS MAE relative to CLEAN (`0.110128`). Candidate cuts are auditable evidence signals, not ground-truth shot-boundary labels; inference remained ball-free and TAS retained the existing definition.

Profiling identified repeated per-frame spatial graph construction as the bottleneck. Per-condition frame-graph caching reduced representative runtime from `0.680111 s` to `0.446889 s` (`1.522x`), with exactly equal possession outputs, TAS maximum absolute difference `0.0`, and tracking-metric difference `0.0`. Atomic checkpoint/resume support was implemented and verified.

Step 80A artifacts are isolated under `results/step80/`, including the ablation table, candidate-cut table, summary statistics, checkpoint metadata, and five required figures. Focused tests passed 9/9; the full regression suite passed 170 tests with 11 expected skips. Genuine SoccerNet-GSR perception remains blocked/incomplete and mandatory for original-scope completion.

---

## 3. Key Empirical Findings (Baselines & Mitigation Performance)

### A. Dataset Baselines
1. **Metrica Sample Game 1:** 25 FPS ($dt = 0.04\text{ s}$), 145,006 frames, 28 unique players, coordinates normalized in $[0.0, 1.0]$. Systemic half-time reset identified at frame 71269. Baseline missingness (10.71%) corresponds strictly to off-field substitution bench periods.
2. **SkillCorner Open Data (10 Matches):** 10 FPS ($dt = 0.10\text{ s}$), ~57kâ€“72k frames per match, 29â€“32 unique players per match. Completeness 100% on-pitch (57.7% detected on broadcast camera, 42.3% extrapolated).

### B. Mitigation Performance Summary Table
| Severity | Processing Stage | Missing % | RMSE | MAPE | Velocity Jitter ($\sigma_{\Delta v}$) | Accel Spike % ($>0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean** | Raw Reference | 10.71% | 0.0000 | 0.0000 | 7.4380 | 99.97% |
| **Mild** | Raw Degraded | 13.30% | 0.0020 | 0.0013 | 7.4398 | 99.98% |
| **Mild** | Interpolated (`max_gap=10`) | **11.28%** | 0.0679 | 0.0105 | 7.4004 | 97.70% |
| **Mild** | Smoothed (SavGol $W=7$) | 13.30% | 0.2959 | 0.2671 | **2.4589** (-66.9%) | 99.94% |
| **Mild** | **Pipeline A (Interp + SavGol)** | **11.28%** | 0.3019 | 0.2729 | **2.4689** (-66.8%) | 99.90% |
| **Moderate** | Raw Degraded | 28.48% | 0.0079 | 0.0040 | 7.4291 | 99.98% |
| **Moderate** | Interpolated (`max_gap=10`) | **21.60%** | 0.1347 | 0.0398 | 7.2338 | 91.15% |
| **Moderate** | Smoothed (SavGol $W=7$) | 28.48% | 0.2789 | 0.2426 | **2.6329** (-64.5%) | 99.94% |
| **Moderate** | **Pipeline A (Interp + SavGol)** | **21.60%** | 0.3087 | 0.2762 | **2.4084** (-67.6%) | 99.54% |
| **Severe** | Raw Degraded | 66.25% | 0.0253 | 0.0115 | 7.4682 | 99.97% |
| **Severe** | Interpolated (`max_gap=10`) | **57.88%** | 0.2032 | 0.0909 | 6.9375 | 79.74% |
| **Severe** | Smoothed (SavGol $W=7$) | 66.25% | 0.2336 | 0.1770 | **3.6019** (-51.8%) | 99.96% |
| **Severe** | **Pipeline A (Interp + SavGol)** | **57.88%** | 0.3213 | 0.2821 | **2.3156** (-69.0%) | 99.92% |

### C. Key Scientific Insights
1. **Pipeline A (`Interp -> SavGol`) Superiority:** Interpolating short micro-gaps prior to polynomial filtering bridges fragmented tracks, allowing the Savitzky-Golay filter to operate over longer continuous spans and reducing velocity jitter by up to 69%.
2. **Jitter Suppression:** Both Moving Average ($W=5$) and Savitzky-Golay ($W=7, p=2$) significantly suppress high-frequency noise. Savitzky-Golay is preferred due to its superior preservation of trajectory curvature.
3. **Linear Interpolation Trade-Off:** Linear interpolation is highly effective for short gaps ($\le 10$ frames / $0.40\text{ s}$), recovering 2.0â€“8.4% of total match observations. However, for sharp curvilinear turns, linear interpolation produces chords that underestimate peak curvature.

---

## 4. Parameter Selection & Recommendations
- **Interpolation:** `max_gap = 10` frames ($0.40\text{ s}$ at 25 FPS) â€” optimal trade-off between gap recovery and geometric fidelity.
- **Smoothing:** Savitzky-Golay with `window_length = 7`, `polyorder = 2` ($0.28\text{ s}$) â€” optimal balance between jitter removal and peak velocity/acceleration retention.

---

## 5. Master Changelog
- **Step 67:** Refined severe `gap_start_prob` from 0.015 â†’ 0.010 for strict monotonicity and bounded missingness (~66%). All 21 tests in `tests/test_degradation.py` passing.
- **Step 68:** Ingested SkillCorner Open Data (10 matches, 2024/2025 A-League). Created `docs/skillcorner_data_notes.md`, `src/data/skillcorner_parser.py`, and `tests/test_skillcorner_parser.py`.
- **Step 69:** Implemented trajectory smoothing (`src/noise/smoothing.py`) and missing-track linear interpolation (`src/noise/interpolation.py`). Created `docs/noise_mitigation.md`, `tests/test_interpolation.py`, and `tests/test_smoothing.py`.
- **Step 70:** Implemented ball-free spatial graph and heuristic possession baseline (`src/possession/`). Created `docs/ball_free_possession_baseline.md`, tests, and `01_ball_free_possession_baseline.ipynb`. Validated on first 5 minutes of Metrica Game 1.
- **Step 71:** Implemented tactical scoring baseline (`src/tactics/`). Created `docs/tactical_scoring_baseline.md`, tests, and `01_tactical_scoring_baseline.ipynb`. Validated features (team geometry, pressure, options) and Tactical Advantage Score (TAS).
- **Step 72:** Implemented the reproducible task-driven robustness benchmark (src/validation/robustness_benchmark.py) across CLEAN, SYNTHETIC-DEGRADED, and NOISE-MITIGATED conditions. Verified the 60-second Metrica Game 1 subset (frames 1–1500, seed 72), emitted compact result artifacts, and passed 140 tests (11 skipped for absent SkillCorner data).

- **Step 74 (Robustness Expansion):** Executed the expansion grid (4 windows × 5 seeds × 3 severities). A non-destructive provenance audit confirms a complete canonical fileset: 60 unique runspecs and 180 validated condition rows. Legacy nested `robustness_expansion` artifacts were detected and classified as non-canonical duplicates; they have been preserved and not removed. Full test suite passed locally (159 tests, 11 skipped). Final statistical analysis has not been completed — see `docs/robustness_statistical_validation.md` for next steps. 

- **Step 75 — Final Statistical Analysis & Research Findings:** Completed analysis of 60 unique runspecs and 180 condition observations across 4 windows, 5 seeds, 3 severities, and 3 conditions. The final aggregate results support the main possession degradation finding, the tactical robustness finding, and incomplete but meaningful mitigation recovery. Severity, window, and seed analyses were generated and reviewed, with limitations documented in `docs/final_research_findings.md`. No real-world GSR validation was performed; GSR remains BLOCKED / INCOMPLETE and is required for completion of the original perception and real-broadcast validation scope.

- **Step 76 — SoccerNet-GSR Data Access, GPU Recovery & Perception Readiness Audit:** Audited the checked-out GSR repository at commit `1c958345`, the official example prediction archive, local data directories, model-weight locations, and both Python environments. No SoccerNet-GSR source video, annotations, complete model weights, installed TrackLab runtime, or verifiable local NVIDIA/CUDA environment is available. Objective 1 (real perception adaptation and per-frame GSR output) remains NOT COMPLETE; no GS-HOTA result exists. GSR remains BLOCKED / INCOMPLETE and is required for completion of the original perception and real-broadcast validation scope. The exact recovery route, one-sequence baseline command, canonical mapping, storage policy, and GPU recommendation are documented in `docs/gsr_recovery_plan.md` and `docs/gpu_environment.md`. Next actionable step: obtain authorized access to one validation sequence and labels in persistent GPU storage.

- **Step 77 — SoccerNet-GSR Access & GPU Execution Environment:** Rechecked official access guidance, local data/search paths, the GSR checkout, dependency declarations, model locations, and GPU availability. Authorized validation data, labels, source video, complete weights, installed runtime, and a local NVIDIA/CUDA environment remain unavailable; the public access page was not reachable from the audit environment. No sequence ID was invented and no inference or GS-HOTA evaluation was run. Objective 1 remains INCOMPLETE and GSR remains BLOCKED / INCOMPLETE and is required for completion of the original perception and real-broadcast validation scope. Access requirements, sequence metadata checklist, storage constraints, pinned GPU setup, checkpoint requirements, one-sequence command, and success criteria are documented in `docs/gsr_access_checklist.md` and `docs/gsr_gpu_execution_plan.md`. Next actionable step: obtain authorized access to one validation sequence and matching labels in persistent GPU storage.

---

## 6. Current Roadmap

### Phase 6 — GSR Perception Completion

#### Step 77 — SoccerNet-GSR Access & GPU Execution Environment

**Status:** BLOCKED

**Current immediate action:** Obtain authorized access to one SoccerNet-GSR validation sequence and matching labels in approved persistent GPU storage.

#### Step 78 — Real One-Sequence GSR Baseline Inference

- Run the official TrackLab/sn-gamestate baseline on one verified validation sequence.
- Persist configuration, environment, provenance, logs, outputs, and tracker state.
- Convert verified GSR output into the canonical tracking schema.
- Preserve source sequence identity, FPS, orientation, team mapping, and coordinate evidence.

#### Step 79 — Perception Evaluation & Adaptation

- Evaluate genuine GSR output using the intended GS-HOTA evaluation path.
- Analyze detector/tracker/calibration failure modes on broadcast footage.
- Implement one targeted adaptation only where justified by evidence.
- Compare baseline versus adapted perception quantitatively.
- Do not claim improvement without measured evidence.

### Phase 7 — Broadcast Noise and Event Validation

#### Step 80A — Broadcast-Specific Noise Analysis and Optimization - COMPLETE

- Completed the controlled Metrica broadcast-noise proxy analysis across 48/48 conditions.
- Compared all configured mitigation methods and evaluated tracking quality, ball-free possession F1, and TAS MAE relative to CLEAN.
- Added auditable candidate camera-cut/discontinuity evidence, bounded repair, paired statistics, frame-graph caching, and checkpoint/resume support.
- Genuine SoccerNet-GSR perception remains mandatory for original-scope completion and remains blocked.

#### Step 81 — Controlled Ball-Free Event Evaluation - COMPLETE

- Evaluated player-only possession transitions against available Metrica event annotations.
- Used one-to-one matching at ±0.20, ±0.50, and ±1.00 seconds with per-type and aggregate TP/FP/FN, precision, recall, F1, and timing error.
- Produced isolated machine-readable metrics, tolerance analysis, summary JSON, and figures under `results/step81/`.
- This does not claim SoccerNet-GSR or real-broadcast performance; genuine GSR remains required and incomplete.

#### Step 82 — Pass Completion Model

**Status:** NEXT / NOT STARTED

### Phase 8 — Tactical Decision Modeling

#### Step 82 — Pass Completion Model

- Extend the current Tactical Advantage Score baseline with an interpretable pass-completion probability model.
- Use player geometry, pressure, support, options, and motion-derived features.
- Establish a reproducible train/validation/evaluation protocol.
- Compare the new model against the existing TAS baseline.

#### Step 83 — Counterfactual Pass Ranking

- Generate candidate passes from the player-only game state.
- Rank counterfactual passing options using the pass-completion/tactical model.
- Evaluate ranking agreement using appropriate quantitative metrics and, where required, human sanity checks.
- Do not use ball detection as an input to the ball-free inference pipeline.

### Phase 9 — Final Robustness Benchmark

#### Step 84 — Three-Tier Robustness Benchmark

Extend the existing validated robustness benchmark to:

1. `CLEAN`
2. `SYNTHETIC-DEGRADED`
3. `REAL BROADCAST / GSR OUTPUT`

Keep the existing Metrica CLEAN/SYNTHETIC-DEGRADED/NOISE-MITIGATED benchmark intact as the validated controlled development result. Add genuine broadcast/GSR output as the real-world condition once available and compare downstream possession and tactical stability across controlled and real conditions.

#### Step 85 — Cross-Dataset Validation

- Use Metrica as the primary controlled development/reference dataset.
- Use SkillCorner as complementary external tracking-domain validation.
- Use SoccerNet-GSR as the genuine broadcast perception source.
- Document coordinate conventions, temporal resolution, team mapping, and comparability limitations.

#### Step 86 — Final Statistical Analysis

- Re-run or update statistical analysis only after real GSR/broadcast experiments are complete.
- Report appropriate means, variability, confidence intervals, effect sizes, and robustness comparisons.
- Clearly distinguish controlled benchmark findings from real-broadcast findings.
- Freeze final quantitative results only after all required experiments are reproducible.
