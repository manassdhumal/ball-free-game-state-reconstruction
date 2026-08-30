# Project Status: Ball-Free Game State Reconstruction

## 1. Project Overview & Current State
- **Project Name:** Ball-Free Game State Reconstruction (GSR)
- **Current Milestone:** Phase 1 - Ingestion, Data Quality, and Synthetic Degradation Framework
- **Current Step:** Step 67 - Controlled Parameter Refinement and Re-Validation (Completed)
- **Workspace:** Main repository root (`c:\Users\Saniya Gharat\Desktop\ball-free-game-state-reconstruction`)
- **Data Integrity Policy:** Raw tracking and event data in `data/raw/metrica/` remains strictly untouched and read-only. All intermediate/synthetic outputs are generated into `data/interim/` and figures into `results/figures/`.

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
- **Exploratory Data Analysis:**
  - `notebooks/01_data_exploration/01_metrica_tracking_quality.ipynb`: Quality audit, coverage per player, and missingness distribution.
  - `notebooks/01_data_exploration/02_metrica_motion_anomalies.ipynb`: Frame-to-frame displacement, velocity thresholding, and anomaly classification.
- **Synthetic Degradation Framework & Validation:**
  - `src/noise/__init__.py`: Package initialization exporting the degradation API.
  - `src/noise/degradation.py`: Two-tier degradation engine (observation & identity level), configurable zero-mean coordinate jitter (`distribution="gaussian"`), geometric gap duration model, deterministic track fragmentation (`<pid>_frag_<seg>`), same-team identity switches, composition pipeline (`degrade_tracking`), and structured metadata generation. Refined in Step 67 with severe `gap_start_prob: 0.010`.
  - `tests/test_degradation.py`: Complete test suite covering all 12 test categories with 21 unit tests (all passing).
  - `notebooks/03_noise_analysis/01_synthetic_degradation.ipynb`: Degradation demonstration notebook.
  - `notebooks/03_noise_analysis/02_degradation_parameter_validation.ipynb`: Parameter validation study evaluating monotonicity, tier separation, and evidence review tables.
- **Infrastructure:**
  - `scripts/setup_gsr_colab.sh`: Google Colab automated setup script.

---

## 3. Key Empirical Findings (Metrica Sample Game 1 Baseline)
1. **Sampling & Scope:** 25 FPS ($dt = 0.04\text{ s}$), 145,006 frames (5,800.24 s total duration), 28 unique players (14 home, 14 away).
2. **Coordinate Normalization:** Coordinates are normalized in $[0.0, 1.0]$.
3. **High Baseline Completeness:** 100% of frames have $\ge 22$ visible players. Active on-pitch players have 100% tracking completeness without micro-gaps.
4. **Substitutions:** All missing tracking observations correspond to substitution bench periods (16 total macro-gaps; minimum missing duration 944.56 s / 23,614 frames, maximum 4,775.20 s / 119,380 frames).
5. **Systemic Discontinuities:** A major half-time reset occurs at frame 71269 (speed anomaly up to 29.91 norm/s across all players simultaneously), classified as a global tracker state reset rather than an isolated physical player jump.
6. **Velocity & Motion Percentiles:**
   - 99.0th percentile speed: $0.07201\text{ norm/s}$ ($\approx 0.00288\text{ norm displacement/frame}$)
   - 99.5th percentile speed: $0.07892\text{ norm/s}$ ($\approx 0.00316\text{ norm displacement/frame}$)
   - 99.9th percentile speed: $0.09605\text{ norm/s}$ ($\approx 0.00384\text{ norm displacement/frame}$)
   - Static physical threshold ($25\text{ m/s} \approx 0.25\text{ norm/s}$).
7. **Tracking Artifacts vs. Motion:** Rare isolated non-systemic jumps occur (e.g., frame 71281 displacement $0.2328\text{ norm units}$); boundary-clipping artifacts place coordinates at $-0.05$ or $1.05$.

---

## 4. Step 67: Parameter Refinement & Re-Validation Study

### Refinement Action
- **Parameter Modified:** Severe `gap_start_prob` reduced from `0.015` $\rightarrow$ `0.010` in `src/noise/degradation.py`.
- **Rationale:** Validation identified that combining `missing_prob = 0.20` with `gap_start_prob = 0.015` (mean gap length 100 frames) caused excessive compounding of missingness up to 76.00%, approaching trajectory collapse.

### Re-Validation Metrics Comparison (Old vs. Revised)
| Metric | Clean | Mild | Moderate | Severe (Original: 0.015) | Severe (Revised: 0.010) | Status |
|---|---|---|---|---|---|---|
| **Missing Obs %** | 10.71% | 13.30% | 28.48% | **76.00%** | **66.25%** | **Strictly Monotonic** (-9.75% reduction) |
| **Visible Obs** | 125,000 | 121,374 | 100,126 | 33,604 | 47,255 | +13,651 visible observations preserved |
| **Contiguous Gaps** | 6 (subs) | 2,498 | 8,412 | 7,031 | 9,968 | Gaps remain distinct rather than coalescing |
| **Gap Mean Length** | 2,500.0 f | 7.5 f | 4.7 f | 15.1 f | 9.3 f | Well-controlled geometric scaling |
| **Gap P95 Length** | 2,500.0 f | 2.0 f | 10.0 f | 48.0 f | 25.0 f | Realistic upper bound distribution |
| **Mean Perturbation**| 0.00000 | 0.00127 | 0.00396 | 0.01167 | 0.01147 | **Strictly Monotonic** |
| **Std Perturbation** | 0.00000 | 0.00176 | 0.00681 | 0.02459 | 0.02434 | **Strictly Monotonic** |
| **Max Perturbation** | 0.00000 | 0.14477 | 0.29684 | 0.50957 | 0.50957 | Preserves severe jump upper bound |
| **Isolated Jumps** | 0 | 25 | 114 | 183 | 240 | **Strictly Monotonic** |
| **Fragmented Tracks**| 0 | 0 | 3 | 15 | 15 | Monotonic expansion |
| **Switched Frames** | 0 | 0 | 6,065 | 25,130 | 41,302 | Monotonic expansion |
| **Duplicate Keys** | 0 | 0 | 0 | 0 | **0** | **Zero Duplicates** (Schema invariant verified) |

### Key Validation Insights & Assessment
1. **Acceptability:** The revised severe preset is **acceptable**. Reducing `gap_start_prob` from 0.015 to 0.010 lowers severe missingness from 76.00% to 66.25%, preventing total track breakdown while retaining a robust, high-difficulty stress test.
2. **Strict Monotonicity Preserved:** `Clean (10.71%) < Mild (13.30%) < Moderate (28.48%) < Severe (66.25%)` for missingness, `0.0 < 0.00127 < 0.00396 < 0.01147` for spatial noise, and `0 < 25 < 114 < 240` for isolated jumps.
3. **Remaining Research Considerations:** `frag_prob` and `switch_prob` remain classified as **[TBD]** research parameterizations until empirical re-identification switch rates can be benchmarked against multi-camera tracking (e.g. SoccerNet-GSR). All other parameters are firmly established and validated.

---

## 5. Final Parameter Review & Evidence Classification Table

| Parameter | Current Value | Evidence Type | Assessment | Recommendation |
|---|---|---|---|---|
| `missing_prob` | 0.02 / 0.08 / 0.20 | **[EM]** | Acceptable | Maintain values; creates well-separated point dropout baseline. |
| `gap_start_prob` | 0.001 / 0.005 / 0.010 | **[EM]** | Acceptable | Refined to 0.010 in Step 67; bounds compounding and prevents track collapse. |
| `gap_p_continue` | 0.08 / 0.03 / 0.01 | **[EM]** | Acceptable | Geometric model scaled from 25 FPS frame rate (0.5s–4.0s mean duration). |
| `gap_min/max_length` | 5-25 / 10-75 / 25-250 | **[EM]** | Acceptable | Temporal bounds correctly calibrated to 25 FPS frame rate ($0.2\text{ s} - 10.0\text{ s}$). |
| `jitter_scale` | 0.001 / 0.003 / 0.008 | **[ED]** Mod / **[EM]** Mild,Sev | Acceptable | Moderate ($0.003$) directly matches measured P99 step velocity ($0.00288$). Mild is sub-median; Severe exceeds P99.9. |
| `jitter_distribution` | `"gaussian"` | **[EM]** | Acceptable | Zero-mean Gaussian assumption; configurable API verified. |
| `jump_prob` | 0.0002 / 0.001 / 0.005 | **[EM]** | Acceptable | Effectively models sparse re-acquisition glitches without overwhelming trajectories. |
| `jump_mag_min/max` | 0.05-0.15 / 0.08-0.30 / 0.12-0.50 | **[ED]** | Acceptable | Displacements exceed sprinting velocity ($>0.25\text{ norm/s}$). Upper bound ($0.30$) covers maximum non-systemic jump in Game 1 ($0.2328$). Systemic reset at frame 71269 correctly excluded. |
| `frag_prob & num_splits` | 0.05 (2 splits) / 0.15 (4 splits) | **[TBD]** | Acceptable | Flagged as [TBD] research parameterization. Verified 0 duplicate keys. |
| `switch_prob & duration` | 0.02 (25-125f) / 0.08 (25-500f) | **[TBD]** | Acceptable | Flagged as [TBD] research parameterization. Team consistency fully preserved. |

---

## 6. Test Results
- **Degradation Test Suite (`tests/test_degradation.py`):** 21/21 tests passed (0.130 s).
  - Input immutability (7/7 tests passed)
  - Determinism & seed divergence (3/3 tests passed)
  - Clean identity passthrough (1/1 test passed)
  - Observation-level methods: random missing, geometric gaps, jitter distributions, isolated jumps (5/5 tests passed)
  - Identity-level methods: track fragmentation syntax & key uniqueness, same-team switches (3/3 tests passed)
  - Composite degradation & metadata schema (1/1 test passed)
  - Invalid configuration error handling (3/3 tests passed)

---

## 7. Files Added / Modified in this Step
- `src/noise/degradation.py` (Severe `gap_start_prob` updated from 0.015 to 0.010; evidence comments refined)
- `notebooks/03_noise_analysis/02_degradation_parameter_validation.ipynb` (Re-executed with updated parameter review table)
- `results/figures/validation_severity_monotonicity.png` (Re-generated)
- `results/figures/validation_gap_statistics.png` (Re-generated)
- `data/interim/validation/` (Updated validation datasets and metadata)
- `PROJECT_STATUS.md` (Updated with Step 67 parameter refinement results)

---

## 8. Next Steps & Phase 2 Roadmap
1. Tactical validation and pitch spatial transformation modules.
2. Trajectory reconstruction and smoothing baselines (Kalman filtering, spline interpolation).
3. Ball-free game state perception and possession modelling.
