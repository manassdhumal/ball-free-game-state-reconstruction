# Project Status: Ball-Free Game State Reconstruction

## 1. Project Overview & Current State
- **Project Name:** Ball-Free Game State Reconstruction (GSR)
- **Current Milestone:** Phase 1/2 - Multi-Source Tracking Ingestion & Baseline Analysis
- **Current Step:** Step 68 - SkillCorner Open Data Ingestion & Analysis (Completed & Pushed - Commit `2929233`)
- **Active Workspace:** `/Users/sunilsaraf/.gemini/antigravity/scratch/ball-free-game-state-reconstruction`
- **Data Integrity Policy:** Raw tracking and event data in `data/raw/` (including `metrica/` and `skillcorner/`) remains strictly untouched, read-only, and gitignored. All intermediate/synthetic outputs are generated into `data/interim/` and figures into `results/figures/`.

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
  - `docs/canonical_tracking_schema.md` (v0.2.0): Standard tall tabular representation (`match_id`, `frame`, `timestamp`, `player_id`, `team`, `x`, `y`, `confidence`, `visible`).
  - `docs/canonical_event_schema.md`: Unified match event schema.
  - `docs/skillcorner_data_notes.md`: Exhaustive 25-point dataset specification for SkillCorner Open Data (Australian A-League 2024/2025).
- **Exploratory Data Analysis & Figures:**
  - `notebooks/01_data_exploration/01_metrica_tracking_quality.ipynb`: Quality audit, coverage per player, and missingness distribution.
  - `notebooks/01_data_exploration/02_metrica_motion_anomalies.ipynb`: Frame-to-frame displacement, velocity thresholding, and anomaly classification.
  - `results/figures/skillcorner_detection_rate_by_match.png`: Detection vs extrapolation rates across 10 matches.
  - `results/figures/skillcorner_visible_players_timeline.png`: Active/detected/extrapolated player count timeline for match 1886347.
- **Synthetic Degradation Framework & Validation:**
  - `src/noise/__init__.py`: Package initialization exporting the degradation API.
  - `src/noise/degradation.py`: Two-tier degradation engine (observation & identity level), configurable zero-mean coordinate jitter (`distribution="gaussian"`), geometric gap duration model, deterministic track fragmentation (`<pid>_frag_<seg>`), same-team identity switches, composition pipeline (`degrade_tracking`), and structured metadata generation. Refined in Step 67 with severe `gap_start_prob: 0.010`.
  - `tests/test_degradation.py`: Complete test suite covering all 12 test categories with 21 unit tests (all passing).
  - `notebooks/03_noise_analysis/01_synthetic_degradation.ipynb`: Degradation demonstration notebook.
  - `notebooks/03_noise_analysis/02_degradation_parameter_validation.ipynb`: Parameter validation study evaluating monotonicity, tier separation, and evidence review tables.
- **Infrastructure:**
  - `scripts/setup_gsr_colab.sh`: Google Colab automated setup script.

---

## 3. Key Empirical Findings (Metrica & SkillCorner Baselines)
1. **Metrica Sample Game 1:** 25 FPS ($dt = 0.04\text{ s}$), 145,006 frames, 28 unique players, coordinates normalized in $[0.0, 1.0]$. Systemic half-time reset identified at frame 71269.
2. **SkillCorner Open Data (10 Matches):** 10 FPS ($dt = 0.10\text{ s}$), ~57k–72k frames per match, 29–32 unique players per match.
   - **Coordinate System:** Metric (meters), origin $(0, 0)$ at pitch center. Pitch lengths vary by match ($104\text{m}$, $105\text{m}$, $106\text{m}$), width $68\text{m}$.
   - **Detection vs. Extrapolation:** On-pitch completeness is 100% (22 players per active frame); ~57.7% of player observations are directly detected on broadcast camera (`is_detected=True`), and ~42.3% are extrapolated off-screen (`is_detected=False`).
   - **Coordinate Normalization:** Mapped to canonical $[0, 1]$ via $(x + L/2)/L$ and $(y + W/2)/W$ using match-specific metadata.

---

## 4. Test Results Summary (71 / 71 Tests Passing)
- **Metrica Tracking Parser (`tests/test_metrica_parser.py`):** 11/11 tests passed.
- **Metrica Event Parser (`tests/test_metrica_event_parser.py`):** 20/20 tests passed.
- **Synthetic Degradation Engine (`tests/test_degradation.py`):** 21/21 tests passed.
- **SkillCorner Tracking Parser (`tests/test_skillcorner_parser.py`):** 19/19 tests passed.

---

## 5. Master Changelog (Recent Steps)
- **Step 67:** Refined severe `gap_start_prob` from 0.015 → 0.010 for strict monotonicity and bounded missingness (~66%). All 21 tests in `tests/test_degradation.py` passing.
- **Step 68:** Downloaded and ingested SkillCorner Open Data (10 matches, 2024/2025 A-League). Created `docs/skillcorner_data_notes.md`, `src/data/skillcorner_parser.py`, and `tests/test_skillcorner_parser.py`. All 19 tests passing (71/71 full suite). Committed and pushed (`commit 2929233`).

---

## 6. Next Steps & Phase 2 Roadmap
1. **Step 69: Trajectory Smoothing & Interpolation Module (`src/noise/smoothing.py`)**
   - Implement spline interpolation for short tracking gaps and Savitzky-Golay / Kalman filtering for coordinate jitter.
   - Evaluate smoothing efficacy on both degraded Metrica and raw extrapolated SkillCorner trajectories.
2. **Step 70: Ball-Free Spatial Graph & Possession Baseline (`src/possession/`)**
   - Dynamic Voronoi partitioning and player spatial graph generation.
   - Heuristic candidate possession scoring without ball detection.
3. **Step 71: Tactical Scoring & Counterfactual Evaluation (`src/tactics/`)**
   - Pass success probability and counterfactual receiver ranking.
4. **Step 72: Task-Driven Robustness Benchmark Pipeline (`src/validation/`)**
   - End-to-end evaluation across Clean $\rightarrow$ Synthetic $\rightarrow$ SkillCorner $\rightarrow$ SoccerNet-GSR.
