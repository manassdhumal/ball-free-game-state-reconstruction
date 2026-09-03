# Ball-Free Game State Reconstruction — Roadmap

## Research Goal

Reconstruct player-centric game state from broadcast-style tracking without relying on ball detection, then measure how tracking uncertainty affects downstream possession and tactical interpretation.

## Completed Milestones

Phases 0-5 are complete as controlled/development work and form the validated downstream research foundation.

### Phase 0 — Project Infrastructure

- Repository structure
- Git/Drive/Colab workflow
- Canonical tracking schema
- Canonical event schema

### Phase 1 — Multi-Source Tracking

- Metrica ingestion and validation
- Metrica event ingestion
- SkillCorner ingestion and compatibility analysis

### Phase 2 — Noise Analysis and Mitigation

- Tracking-quality analysis
- Anomaly analysis
- Synthetic degradation
- Interpolation
- Smoothing
- Parameter validation

### Phase 3 — Ball-Free Possession

- Spatial graph
- Player-only possession scoring
- Temporal persistence
- Possession/event candidate generation
- Post-hoc event evaluation
- Anti-leakage tests

### Phase 4 — Tactical Interpretation

- Interpretable tactical features
- Tactical Advantage Score
- Tactical validation

### Phase 5 — Task-Driven Robustness Benchmark

- CLEAN baseline
- SYNTHETIC-DEGRADED
- NOISE-MITIGATED
- Possession robustness evaluation
- Tactical-score stability evaluation

## Current Phase

### Phase 6 - Original-Scope Completion

SoccerNet-GSR perception is currently blocked by data-access and GPU/runtime constraints. It is not optional: genuine GSR output is required to complete the original broadcast-perception objective and real-broadcast validation scope.

## Current Experimental Result

The current quantitative robustness benchmark is validated but is not the final complete benchmark for the original project scope.

### Historical Step 72 Single-Window Baseline

Verified Step 72 benchmark:

- Metrica Game 1, frames 1–1500 (60 seconds at 25 FPS)
- Degradation seed: `72`
- Orientation convention: `home_attacks_x1=true`
- `CLEAN` possession F1: `0.4923`
- `SYNTHETIC-DEGRADED` possession F1: `0.2054`
- `NOISE-MITIGATED` possession F1: `0.4494`
- `SYNTHETIC-DEGRADED` TAS MAE relative to CLEAN: `0.1241`
- `NOISE-MITIGATED` TAS MAE relative to CLEAN: `0.0929`

These historical/preliminary results evaluate robustness relative to the CLEAN reference; they do not establish ground-truth tactical advantage and are not genuine SoccerNet-GSR perception results.

### Steps 74-75 Expanded Controlled Benchmark

Step 74 expanded the Metrica experiment across four temporal windows, five degradation seeds, and three severity levels. Step 75 completed the statistical analysis of 60 unique runspecs and 180 condition observations across `CLEAN`, `SYNTHETIC-DEGRADED`, and `NOISE-MITIGATED`. This is the primary validated controlled robustness result for the current project.

The Steps 74-75 results are based on controlled Metrica conditions. They do not establish the final real-broadcast robustness claim and do not constitute SoccerNet-GSR perception evaluation.

### Controlled Benchmark vs Final Benchmark

**Current validated controlled benchmark:** The existing Metrica-based `CLEAN` -> `SYNTHETIC-DEGRADED` -> `NOISE-MITIGATED` benchmark remains the validated controlled robustness experiment and must not be discarded or redesigned unnecessarily.

**Final original-scope benchmark:** The eventual final benchmark will extend the controlled result with genuine broadcast-derived GSR output as the real-world condition, only after GSR perception output has been converted and validated.

## Historical Milestone Record

### Step 72 - Task-Driven Robustness Benchmark - COMPLETE

The reproducible Metrica benchmark was implemented across `CLEAN`, `SYNTHETIC-DEGRADED`, and `NOISE-MITIGATED` conditions and verified on the initial 60-second, seed-72 subset.

### Step 74 - Robustness Experiment Expansion - COMPLETE

The verified Step 72 experiment was expanded beyond one 60-second window and one degradation seed.

Objectives:

- Evaluate multiple temporal windows from Metrica Game 1.
- Evaluate multiple degradation seeds.
- Evaluate multiple degradation severities.
- Compare `CLEAN`, `SYNTHETIC-DEGRADED`, and `NOISE-MITIGATED`.
- Measure tracking-quality degradation.
- Measure possession precision, recall, and F1.
- Measure tactical-score stability relative to CLEAN.
- Report mean, median, standard deviation, and confidence intervals where appropriate.
- Produce degradation-severity-versus-performance plots.
- Verify that the observed recovery from noise mitigation is consistent rather than a single-seed effect.

The expanded experiment is the primary quantitative controlled robustness result for the project.

### Step 75 - Final Statistical Analysis - COMPLETE

The controlled benchmark was statistically analyzed with run-level means, variability summaries, confidence intervals, paired effects, severity analysis, window analysis, seed analysis, mitigation recovery, report tables, and figures. These findings remain controlled Metrica results rather than genuine GSR or real-broadcast validation.

### Step 76 - GSR Perception Readiness Audit - COMPLETE

The checked-out SoccerNet-GSR repository, local assets, model locations, runtime requirements, GPU availability, storage policy, and canonical output route were audited. No genuine GSR data, complete model assets, usable local GPU runtime, or GS-HOTA result was found.

### Step 77 - GSR Access and GPU Execution Environment - CURRENT / BLOCKED

No authorized SoccerNet-GSR validation sequence, matching labels, usable NVIDIA/CUDA environment, complete runtime, or required pretrained assets are currently available. No GSR inference has been run and no GS-HOTA score exists. Objective 1 remains incomplete.

Current immediate action: obtain authorized access to one SoccerNet-GSR validation sequence and matching labels in approved persistent GPU storage.

## Phase 6-9 Roadmap

### Phase 6 - GSR Perception Completion

#### Step 77 - SoccerNet-GSR Access & GPU Execution Environment

**Status:** BLOCKED

**Current immediate action:** Obtain authorized access to one SoccerNet-GSR validation sequence and matching labels in approved persistent GPU storage.

**Required prerequisites:**

- Genuine validation video/frame sequence.
- Matching `Labels-GameState.json`.
- Verified sequence metadata and dataset/version provenance.
- Usable NVIDIA GPU environment and pinned Python/runtime.
- Required TrackLab/sn-gamestate dependencies.
- Required detector, ReID, OCR, calibration, and other model assets.

No inference should be claimed before these prerequisites exist.

#### Step 78 - Real One-Sequence GSR Baseline Inference

- Execute the official TrackLab/sn-gamestate baseline on one verified validation sequence.
- Preserve environment, configuration, provenance, logs, tracker state, and raw GSR output.
- Convert verified output into the canonical tracking schema.
- Record sequence identity, FPS, team mapping, orientation, coordinate convention, and projection evidence.

#### Step 79 - Perception Evaluation & Adaptation

- Evaluate genuine GSR tracking using GS-HOTA.
- Analyze detector, tracker, calibration, and broadcast-specific failure modes.
- Implement one targeted adaptation only if evidence justifies it.
- Compare baseline and adapted perception quantitatively.
- Do not claim improvement without measured evidence.

### Phase 7 - Broadcast Noise and Event Validation

#### Step 80 - Broadcast-Specific Noise Analysis

- Analyze camera-cut fragmentation, frame jitter, missingness, and identity fragmentation.
- Compare appropriate smoothing and mitigation approaches, including moving average, Kalman-style filtering, and Savitzky-Golay where appropriate.
- Quantify effects on downstream geometry and tracking stability.

#### Step 81 - SoccerNet Action Spotting Validation

- Validate ball-free possession and event inference against compatible independent action-spotting labels.
- Report precision, recall, and F1.
- Preserve the ball-free and anti-leakage constraints.
- Do not use future-frame or ball information as inference inputs.

### Phase 8 - Tactical Decision Modeling

#### Step 82 - Pass Completion Model

Extend the current TAS baseline with an interpretable pass-completion probability model using player-only pressure, support, forward options, spatial geometry, and motion-derived features. Establish reproducible train/validation/evaluation procedures and compare the model with the existing TAS baseline.

#### Step 83 - Counterfactual Pass Ranking

- Generate candidate passes from player-only game state.
- Estimate and rank candidate pass outcomes.
- Evaluate ranking quality quantitatively and include human sanity checks where required.
- Do not introduce ball detection into the inference pipeline.

### Phase 9 - Final Robustness and Cross-Dataset Validation

#### Step 84 - Three-Tier Robustness Benchmark

Final conditions:

1. `CLEAN`
2. `SYNTHETIC-DEGRADED`
3. `REAL BROADCAST / GSR OUTPUT`

Keep the existing Metrica controlled benchmark intact. Add the real-broadcast condition only after genuine GSR perception output has been validated. Evaluate tracking quality, possession precision/recall/F1, tactical-score stability, and downstream degradation/recovery while distinguishing synthetic degradation from real broadcast noise.

#### Step 85 - Cross-Dataset Validation

- Use Metrica as the primary controlled development/reference dataset.
- Use SkillCorner as complementary external tracking-domain validation.
- Use SoccerNet-GSR as the genuine broadcast perception and real-world validation source.
- Document coordinate systems, temporal resolution, team mapping, orientation, and comparability limitations.

#### Step 86 - Final Statistical Analysis

Only after the real-broadcast experiments are complete:

- Update aggregate statistics.
- Calculate appropriate uncertainty intervals and effect sizes.
- Compare controlled and real-broadcast conditions.
- Freeze final quantitative results only after all required experiments are reproducible.
- Finalize the complete reproducibility record.

## Current Research Status

### Completed

- Project infrastructure.
- Multi-source tracking ingestion.
- Metrica event ingestion.
- SkillCorner compatibility and ingestion.
- Tracking noise and degradation framework.
- Interpolation and smoothing.
- Ball-free possession baseline.
- Tactical Advantage Score baseline.
- Controlled robustness benchmark.
- Expanded robustness experiment.
- Statistical analysis of the controlled benchmark.

### Incomplete

- Genuine SoccerNet-GSR perception.
- GS-HOTA evaluation.
- Broadcast-specific perception and noise validation.
- Independent action-spotting validation.
- Pass-completion probability model.
- Counterfactual pass ranking.
- Real-broadcast robustness condition.
- Final end-to-end validation.

### Current Blocker

Authorized SoccerNet-GSR data and matching labels, an approved GPU execution environment, and the required pretrained assets.

## Final Research Deliverables

- Methodology
- Implementation
- Experiments
- Robustness benchmark
- Ablation/parameter analysis
- Limitations
- Reproducibility documentation
