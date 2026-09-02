# Task-Driven Robustness Benchmark

## Research question

How much does controlled degradation of player tracking alter downstream, ball-free possession inference and Tactical Advantage Score (TAS) output?  This is an evaluation of robustness, not a new possession or tactical model.

## Design

The configured pipeline is:

```text
canonical player tracking → spatial graph → ball-free possession → tactical features → TAS → metrics
```

`configs/robustness_benchmark.yaml` is the complete, explicit run contract. It is JSON-formatted YAML so it also runs in the project virtual environment without PyYAML. The initial verified subset is Metrica Sample Game 1 frames 1–1500 (60 seconds at 25 FPS), not the full match. The fixed degradation seed is `72`.

Three conditions are run from the same clean canonical subset:

- `CLEAN`: original Metrica player tracking.
- `SYNTHETIC-DEGRADED`: clean tracking passed to the existing moderate synthetic degradation framework.
- `NOISE-MITIGATED`: degraded tracking after existing short-gap linear interpolation (`max_gap=10`) and Savitzky–Golay smoothing (`window=7`, `polyorder=2`).

The `home_attacks_x1: true` orientation setting is passed unchanged to tactical extraction in every condition. Consequently, no coordinate flip can be mistaken for a robustness effect. Runs should not cross half-time unless the configuration and orientation policy are updated deliberately.

## Leakage boundary

Inference accepts a strict canonical-player whitelist only: match ID, frame, time, player/team IDs, coordinates, confidence, and visibility. Ball coordinates and event fields are ignored before both possession and TAS processing. Metrica annotations are loaded and used only after condition predictions are complete, for post-hoc temporal matching.

## Metrics

Each condition summary contains the following fields.

- Tracking quality: missingness rate; mean visible-player completeness relative to the CLEAN roster; coordinate-validity rate among visible samples (finite and in normalized pitch bounds); interpolation fraction; and count/rate of consecutive-frame displacement anomalies above `0.05` normalized pitch units.
- Possession candidates: unique inferred pass/turnover/change frames are greedily matched one-to-one to `PASS`, `BALL LOST`, `BALL OUT`, and `RECOVERY` Metrica references. The configured tolerance is ±1.0 second (25 frames), with predicted/reference/matched counts, precision, recall, F1, and matched temporal-error statistics.
- Tactical stability: absolute per-frame TAS difference from CLEAN, mean and median absolute score error, Pearson correlation with CLEAN (when defined), top-`k` frame agreement, and mean clean-possession-segment TAS error.

CLEAN is a **reference state**, not ground-truth tactical truth. TAS has no external ground truth here; stability measures the sensitivity of this baseline to tracking quality.

## Outputs and reproducibility

The command below writes only compact derived artifacts—no raw or intermediate tracking data:

```powershell
.venv\Scripts\python.exe -m src.validation.robustness_benchmark --config configs\robustness_benchmark.yaml
```

- `results/metrics/`: flattened condition table (CSV) and nested metrics (JSON).
- `results/runs/`: resolved configuration plus per-frame TAS values needed to reproduce the plot.
- `results/figures/`: TAS comparison plot (PNG when matplotlib is installed; otherwise a dependency-free SVG).

The stochastic degradation has an explicit seed; all remaining components are deterministic. Changing the subset, seed, degradation severity, mitigation, possession parameters, tolerance, orientation, or output identifier requires changing the configuration.

## Limitations

The tracking and possession baselines remain heuristic and sparse/missing tracking can change inferred player identities and possession candidates. Metrica event labels are imperfect proxies for possession changes, and matching statistics depend on the selected tolerance. Interpolation does not restore long gaps; smoothing cannot repair identity fragmentation. Results are restricted to the configured early first-half interval and do not establish full-match generalization.

## Future GSR condition

The condition loop operates on canonical player-tracking DataFrames, so a verified SoccerNet-GSR output can be added as a named fourth condition without altering downstream scoring or metrics. It must first be converted to the canonical schema, have its coordinate orientation documented/aligned to `home_attacks_x1`, and be verified as an actual local match output. This repository currently has GSR setup/model resources and an example archive, but no verified canonical GSR tracking output for this Metrica run; it is therefore not included or claimed as a result.
