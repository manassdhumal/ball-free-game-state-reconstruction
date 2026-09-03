# System Architecture

## End-to-End Pipeline

```text
Broadcast / Tracking Data
        |
        v
Canonical Tracking
        |
        v
Quality Analysis
        |
        v
Synthetic Degradation
        |
        v
Interpolation + Smoothing
        |
        v
Spatial Graph
        |
        v
Ball-Free Possession
        |
        v
Tactical Features
        |
        v
Tactical Score
        |
        v
Robustness Evaluation
        |
        v
Statistical Analysis
```

## Data Boundaries

```text
Metrica Events ----------------------> POST-HOC EVALUATION ONLY

Ball Coordinates --------------------> NOT USED BY POSSESSION/Tactical INFERENCE
```

The possession and tactical stages consume canonical player rows only: match identity, frame/time, player identity, team, coordinates, confidence, and visibility. Event annotations are loaded only after predictions are complete to calculate temporal matching metrics. Ball coordinates are stored separately when available and are excluded from the inference whitelist.

## Component Responsibilities

- **Ingestion:** Convert Metrica and compatible tracking sources into the canonical tall schema.
- **Quality analysis:** Measure missingness, coordinate validity, visibility completeness, and motion anomalies.
- **Synthetic degradation:** Apply configured, seeded tracking perturbations for controlled stress testing.
- **Interpolation and smoothing:** Recover bounded internal gaps and reduce local coordinate jitter without extrapolating boundary gaps.
- **Spatial graph:** Represent visible players as nodes with pairwise, teammate, opponent, density, and optional KNN relations.
- **Ball-free possession:** Score players from geometry and movement continuity, then apply temporal persistence and infer candidate transitions.
- **Tactical features and score:** Calculate team geometry and possessor context, then combine them into an interpretable TAS.
- **Robustness evaluation:** Compare condition-level tracking, possession, and tactical outputs using the runspec as the experimental unit.
- **Statistical analysis:** Aggregate the frozen canonical population across windows, seeds, severities, and conditions.

## Verified Final Population

The final analysis covers 60 unique runspecs and 180 condition observations across four windows, five seeds, three severities, and CLEAN, SYNTHETIC-DEGRADED, and NOISE-MITIGATED conditions. This architecture describes the completed benchmark; it does not include a real GSR condition.
