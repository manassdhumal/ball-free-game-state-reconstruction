# Ball-Free Game State Reconstruction — Roadmap

## Research Goal

Reconstruct player-centric game state from broadcast-style tracking without relying on ball detection, then measure how tracking uncertainty affects downstream possession and tactical interpretation.

## Completed Milestones

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

## Current Experimental Result

Verified Step 72 benchmark:

- Metrica Game 1, frames 1–1500 (60 seconds at 25 FPS)
- Degradation seed: `72`
- Orientation convention: `home_attacks_x1=true`
- `CLEAN` possession F1: `0.4923`
- `SYNTHETIC-DEGRADED` possession F1: `0.2054`
- `NOISE-MITIGATED` possession F1: `0.4494`
- `SYNTHETIC-DEGRADED` TAS MAE relative to CLEAN: `0.1241`
- `NOISE-MITIGATED` TAS MAE relative to CLEAN: `0.0929`

These results evaluate robustness relative to the CLEAN reference; they do not establish ground-truth tactical advantage.

## Next Milestone

### Step 74 — Robustness Experiment Expansion & Statistical Validation

Expand the verified Step 72 experiment beyond a single 60-second window and single degradation seed.

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

The expanded experiment becomes the primary quantitative robustness result for the project.

## Deferred / Optional

### Step 73 — Real GSR Validation

Status: BLOCKED

No provenance-verifiable real GSR tracking sequence with sufficient source metadata is currently available.

Required before proceeding:

- Genuine GSR run output
- Source sequence/video identity
- Frame rate
- Team mapping
- Coordinate/orientation evidence
- Confidence or detection-quality information
- Valid coordinate/projection policy

Do not fabricate, simulate, or assume GSR output exists.

### Future GSR Benchmark Condition

Add GSR as a fourth benchmark condition only after canonical conversion and validation succeed.

Do not redesign the existing benchmark around the unavailable GSR condition.

## Final Research Deliverables

- Methodology
- Implementation
- Experiments
- Robustness benchmark
- Ablation/parameter analysis
- Limitations
- Reproducibility documentation
