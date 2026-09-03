# Ball-Free Game State Reconstruction

## Abstract

This project studies whether a player-centric football game state can support downstream possession and tactical reasoning when ball coordinates are unavailable. It implements a canonical tracking representation for heterogeneous sources, a lightweight player-only spatial graph, a heuristic ball-free possession baseline, and an interpretable Tactical Advantage Score (TAS). Robustness is evaluated by applying controlled synthetic tracking degradation and a deterministic interpolation-plus-smoothing mitigation pipeline to the same Metrica tracking windows. The final analysis contains 60 unique runspecs and 180 condition observations across four windows, five seeds, three severities, and three conditions: CLEAN, SYNTHETIC-DEGRADED, and NOISE-MITIGATED. Mean possession F1 was 0.49199, 0.23451, and 0.41328 respectively; tactical MAE relative to CLEAN was 0.00000, 0.15635, and 0.10888. Mitigation improved possession F1 by 0.17877 on average. The results support degradation sensitivity and meaningful but incomplete mitigation recovery within this benchmark. They do not establish real-world GSR performance, universal robustness, or ground-truth tactical advantage.

## 1. Introduction

### 1.1 Problem Statement

Broadcast-derived tracking can provide player locations while the ball is missing, unreliable, or intentionally excluded. The problem is to reconstruct useful player-centric game-state signals without allowing ball coordinates or future labels to enter inference.

### 1.2 Motivation

A player-only representation is useful for studying geometry, team structure, and robustness of downstream inference. It also makes the information boundary explicit: the system must distinguish what can be inferred from player arrangement from what requires direct ball observation.

### 1.3 Research Question

Does controlled tracking degradation harm ball-free possession inference and tactical interpretation, and does interpolation/smoothing mitigation recover downstream performance?

### 1.4 Objectives

- Ingest tracking sources into a validated canonical player schema.
- Quantify missingness and motion anomalies.
- Infer possession and candidate events using player tracking only.
- Compute an interpretable tactical score from player geometry.
- Evaluate how synthetic degradation changes downstream outputs.
- Measure recovery after bounded interpolation and smoothing.

### 1.5 Contributions

1. A common player-centric tracking representation for Metrica and SkillCorner-compatible inputs.
2. A lightweight player-only spatial graph and heuristic possession baseline.
3. An interpretable tactical scoring baseline based on pressure, support, forward options, and defensive spread.
4. A task-driven robustness benchmark connecting tracking quality to possession and tactical-score stability.
5. A run-level statistical analysis of degradation and mitigation across 60 canonical runspecs.

These are engineering and empirical contributions. The project does not claim a learned state estimator, a ground-truth tactical model, real-world GSR validation, or universal robustness.

## 2. Background

### 2.1 Broadcast Football Tracking

Single-camera broadcast tracking is affected by camera framing, occlusion, detection failures, identity fragmentation, and extrapolation. The project therefore treats visibility, confidence, coordinates, and identity as explicit data semantics rather than assuming complete observations.

### 2.2 Player-Centric Game State

The canonical state is a tall table with one player observation per frame. It preserves match, time, identity, team, normalized coordinates, confidence, and visibility, allowing downstream grouping by frame, team, and player.

### 2.3 Ball-Free Reasoning

The possession baseline uses spatial relations among visible players and short-term movement continuity. The ball is deliberately absent from inference, so outputs are heuristic proxies and candidate events rather than direct ball possession labels.

### 2.4 Tracking Noise and Missing Data

Degradation is represented at observation and identity levels, including coordinate perturbation, gaps, and identity effects. Missingness is not automatically detector failure: in the Metrica baseline, inactive or substituted players can also produce missing rows.

### 2.5 Tactical Interpretation

TAS summarizes geometric conditions around the inferred possessor. It is an interpretable stability baseline, not expected goals, expected threat, or ground-truth tactical advantage.

## 3. System Architecture

### 3.1 End-to-End Pipeline

The pipeline is documented in [system_architecture.md](system_architecture.md). Inference consumes canonical player tracking only. Metrica event annotations enter after prediction for post-hoc evaluation.

### 3.2 Data Sources

Metrica Sample Game 1 supplies the primary benchmark tracking and event data. SkillCorner Open Data is ingested and documented for compatibility analysis, but real SkillCorner match validation is not available in the local test environment. SoccerNet-GSR is not a benchmark condition.

### 3.3 Canonical Tracking Representation

The canonical schema uses `match_id`, `frame`, `timestamp`, `player_id`, `team`, `x`, `y`, `confidence`, and `visible`. Coordinates are normalized to pitch space where the source mapping supports it; ball data is kept separate.

### 3.4 Noise Modeling

Synthetic degradation creates controlled tracking perturbations at configured severity levels. It is an evaluation stressor, not a claim that the generated process exactly reproduces every broadcast tracker failure.

### 3.5 Noise Mitigation

The configured mitigation applies bounded linear interpolation to short internal gaps, followed by Savitzky-Golay smoothing on contiguous visible segments. Leading/trailing gaps are not extrapolated, long gaps remain missing, and imputed rows retain provenance.

### 3.6 Spatial Graph

Each visible player is a node. Vectorized Euclidean distances provide nearest-teammate distance, nearest-opponent distance, local density, and optional directed KNN edges. The graph is a lightweight dataclass rather than a network library graph.

### 3.7 Ball-Free Possession

Seven normalized heuristic features are combined with configurable weights. Temporal persistence, switch margin, and minimum score threshold reduce frame-to-frame oscillation. Possession transitions generate pass, turnover, and recovery candidates.

### 3.8 Tactical Scoring

TAS combines pressure, forward options, teammate support, and opposing-team defensive spread. Attacking direction is an explicit configuration parameter.

### 3.9 Robustness Evaluation

The benchmark runs CLEAN, SYNTHETIC-DEGRADED, and NOISE-MITIGATED conditions from the same canonical windows. Tracking quality, event-matching metrics, and tactical stability are aggregated at the runspec level.

## 4. Data and Preprocessing

### 4.1 Metrica Dataset

The final benchmark uses Metrica Sample Game 1 at 25 FPS over four 60-second windows: 0-60 s, 300-360 s, 600-660 s, and 900-960 s. The source tracking and events remain read-only under the repository data policy.

### 4.2 SkillCorner Dataset

The repository documents ten Australian A-League 2024/2025 SkillCorner Open Data matches at 10 FPS. Its native data includes detected and extrapolated player positions and separate ball data. Local real-data tests are skipped when the match archive is absent.

### 4.3 Coordinate Conventions

Metrica coordinates are normalized to the canonical pitch convention. The benchmark uses `home_attacks_x1=true` for its configured windows and does not silently infer orientation changes. SkillCorner metric coordinates are normalized using match-specific pitch dimensions during ingestion.

### 4.4 Data Quality

Quality metrics include missingness, visible-player completeness, coordinate validity, interpolation fraction, and large-motion anomaly counts/rates. Missingness must be interpreted alongside roster and visibility semantics.

### 4.5 Event Data for Post-Hoc Evaluation

Metrica event labels are used only after possession candidates have been produced. Greedy one-to-one temporal matching maps predicted candidates to configured reference event types at a 1.0-second tolerance.

## 5. Methodology

### 5.1 Tracking Representation

All downstream modules receive canonical player rows. Invalid or unavailable coordinates use explicit missing values, and ball rows remain separate.

### 5.2 Synthetic Degradation

The experiment uses the configured mild, moderate, and severe degradation levels with seeds 72-76. Each degradation is paired with its clean source run and retains a resolved configuration record.

### 5.3 Interpolation

Only internal gaps at or below `max_gap=10` frames are linearly interpolated. This is bounded recovery, not extrapolation or identity correction.

### 5.4 Smoothing

The benchmark mitigation uses Savitzky-Golay smoothing with window length 7 and polynomial order 2 on contiguous segments. It does not fill missing observations by itself.

### 5.5 Spatial Graph Construction

Visible player coordinates are used to calculate pairwise distances, team-neighbor relations, local density, and optional KNN edges. Ball columns, if present in an input frame, are ignored.

### 5.6 Ball-Free Possession Inference

Frame-local geometry is combined with movement continuity and a temporal persistence rule. The default configuration uses a 0.15 switch margin, five-frame persistence, and a minimum score threshold of 0.10.

### 5.7 Tactical Advantage Score

The score weights pressure at 35%, forward options at 35%, support at 15%, and defensive disorganization at 15%. A higher score indicates a more favorable geometric state according to this heuristic, not an externally validated tactical outcome.

### 5.8 Anti-Leakage Design

Inference uses only player tracking fields and does not read ball coordinates, event labels, future frames, or future possession. Metrica events are post-hoc evaluation data only.

## 6. Experimental Design

### 6.1 Research Hypotheses

- **H1:** Tracking degradation reduces ball-free possession performance.
- **H2:** Tracking degradation increases tactical instability relative to CLEAN.
- **H3:** Interpolation and smoothing recover part of the degraded downstream performance.

### 6.2 Experimental Conditions

The final population contains 60 unique runspecs and 180 condition observations: 60 CLEAN, 60 SYNTHETIC-DEGRADED, and 60 NOISE-MITIGATED. CLEAN is a within-run robustness reference, not tactical ground truth.

### 6.3 Temporal Windows

Four 60-second windows are evaluated: 0-60 s, 300-360 s, 600-660 s, and 900-960 s.

### 6.4 Degradation Seeds

Seeds 72, 73, 74, 75, and 76 are treated as independent stochastic realizations of the configured degradation process. They are not independent matches.

### 6.5 Degradation Severities

Mild, moderate, and severe settings test a controlled range of tracking stress. The levels are defined by the experiment configuration and should not be generalized to universal industry severity categories.

### 6.6 Evaluation Metrics

Possession uses precision, recall, and F1 for temporally matched candidate events. Tactical robustness uses mean absolute TAS error relative to CLEAN. Tracking metrics summarize missingness and validity.

### 6.7 Statistical Methodology

The runspec is the experimental unit. Condition summaries report mean, median, standard deviation, range, and bootstrap 95% confidence intervals where available. Paired effects compare conditions within the same run, avoiding frame-level pseudo-replication as the primary inferential unit.

## 7. Results

### 7.1 Tracking Robustness

Tracking quality worsens under SYNTHETIC-DEGRADED and improves partially after mitigation. The exact condition summaries are in `tracking_quality_by_condition.csv`.

### 7.2 Ball-Free Possession

Mean possession F1 is 0.49199 for CLEAN, 0.23451 for SYNTHETIC-DEGRADED, and 0.41328 for NOISE-MITIGATED. The degraded-minus-clean change is negative, and mitigation does not fully restore the CLEAN reference.

### 7.3 Tactical Robustness

Mean tactical MAE is 0.00000 for CLEAN, 0.15635 for SYNTHETIC-DEGRADED, and 0.10888 for NOISE-MITIGATED. The zero CLEAN value is a reference-comparison consequence, not evidence of tactical truth.

### 7.4 Severity Analysis

The severity tables and figures show how tracking and downstream metrics vary across mild, moderate, and severe settings. Interpret these as responses to configured synthetic stress levels.

### 7.5 Window Analysis

Window-level results quantify whether the observed effects are concentrated in a particular 60-second interval. They do not establish full-match generalization.

### 7.6 Seed Analysis

Seed-level results show variation across the five degradation realizations. Seeds provide repeated stress realizations, not new matches or independent populations.

### 7.7 Noise-Mitigation Recovery

Mean possession F1 improved by 0.17877 from degraded to mitigated conditions, corresponding to a recovery fraction of 0.69432 of the observed degradation drop. Tactical MAE also decreased from 0.15635 to 0.10888.

## 8. Discussion

### 8.1 Interpretation of H1

H1 is supported within the benchmark: controlled tracking degradation is associated with a substantial drop in event-based ball-free possession F1.

### 8.2 Interpretation of H2

H2 is supported within the benchmark: degraded player geometry produces larger TAS differences relative to the clean reference.

### 8.3 Interpretation of H3

H3 is supported within the benchmark: mitigation recovers meaningful performance, but the mitigated means remain below the CLEAN reference.

### 8.4 Why Mitigation Helps

Interpolation restores short internal gaps and gives smoothing contiguous local trajectories to process. This can reduce fragmentation and high-frequency coordinate variation before downstream geometry is calculated.

### 8.5 Cases Where Mitigation Is Incomplete

Long gaps, identity errors, sharp turns, and information lost through severe degradation cannot be reliably reconstructed by bounded linear interpolation and local smoothing. Smoothing can also attenuate real movement changes.

## 9. Limitations

- The final robustness experiment uses one Metrica match and one downstream heuristic implementation.
- Degradation is synthetic and parameter-specific.
- Metrica event labels are imperfect proxies and evaluation depends on a 1.0-second matching tolerance.
- CLEAN is a reference condition, not tactical ground truth; its tactical MAE is zero by construction.
- Possession and TAS are heuristic baselines, not learned or externally calibrated models.
- SkillCorner compatibility is documented, but the local real-data suite is skipped without the archive.
- No real SoccerNet-GSR output was validated. GSR remains BLOCKED / OPTIONAL.

## 10. Reproducibility

The repository setup, data policy, configuration, test command, canonical-result policy, and final-analysis entry point are documented in [reproducibility.md](reproducibility.md). The final derived tables and figures are under `results/metrics/final_analysis/` and `results/figures/final_analysis/` when present locally; generated results remain governed by `.gitignore`.

## 11. Conclusion

This project demonstrates a complete, auditable player-only baseline for reconstructing selected game-state signals and testing their sensitivity to tracking quality. Across 60 canonical runspecs and 180 condition observations, synthetic degradation reduced possession performance and increased tactical-score error, while mitigation recovered a meaningful but incomplete share of the loss. The strongest conclusion is therefore benchmark-specific robustness evidence, not universal validity or real-world GSR success.

## References

1. Project canonical tracking schema: `docs/canonical_tracking_schema.md`.
2. Project canonical event schema: `docs/canonical_event_schema.md`.
3. Ball-free possession baseline: `docs/ball_free_possession_baseline.md`.
4. Tactical scoring baseline: `docs/tactical_scoring_baseline.md`.
5. Noise mitigation specification: `docs/noise_mitigation.md`.
6. Task-driven robustness benchmark: `docs/robustness_benchmark.md`.
7. Robustness statistical validation: `docs/robustness_statistical_validation.md`.
8. SkillCorner Open Data notes: `docs/skillcorner_data_notes.md`.
9. SkillCorner Open Data repository: https://github.com/SkillCorner/opendata.
