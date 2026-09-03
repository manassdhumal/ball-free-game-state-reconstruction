# Project Status: Ball-Free Game State Reconstruction

## 1. Project Overview & Current State
- **Project Name:** Ball-Free Game State Reconstruction (GSR)
- **Current Milestone:** Phase 1/2 - Multi-Source Tracking Ingestion, Synthetic Degradation, & Noise Mitigation Framework
- **Current Step:** Step 74 - Robustness Experiment Expansion (results collected, audit recorded)
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

---

## 6. Next Steps & Phase 2 Roadmap
1. **Next: Expand robustness conditions after data verification**
   - Convert and validate a real GSR output (or another canonical source), align its orientation, then add it as a condition without changing the downstream metric contract.
