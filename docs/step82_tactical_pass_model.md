# Step 82 — Tactical Pass Model

## Research question

Can a lightweight, interpretable player-only state representation generate plausible pass options, estimate completion probability when reliable outcomes exist, and rank counterfactual options without ball coordinates?

## Scope and data

The reproducible run uses Metrica Sample Game 1, frames 1–1500 (0–60 seconds, 25 FPS), with `home_attacks_x1=true`. Events are loaded only after player-only candidate construction. The 15 PASS events with non-null source and receiver IDs support ranking against an annotated target. The canonical Metrica event schema has no explicit completion/success field, so the number of reliable binary completion labels is **zero**.

## Relation to TAS

The existing Tactical Advantage Score remains a team-level heuristic based on pressure, support, forward options, and defensive spread. Step 82 reuses those ideas but keeps three quantities separate:

- **Pass-completion probability:** the probability model output for one hypothetical source-target option.
- **Tactical value:** a transparent combination of forward progress, target space, support, pressure, and distance.
- **Counterfactual ranking:** the ordering of available hypothetical options by probability plus tactical value.

No result claims that the observed player selected the highest-ranked option.

## Candidate definition

For a single current tracking snapshot, candidates are distinct same-team player pairs with visible players and configurable distance bounds. The default maximum distance is 0.70 normalized pitch units and the minimum is 0.02. Each candidate stores frame/time, source and target IDs and coordinates, and its feature vector. Candidates are hypothetical decision options, not claims about attempted passes.

## Features

The player-only feature vector contains pass distance, forward progress, pass angle, source pressure, target pressure/space, support count near the target, opponent obstruction near the source-target segment, team width, and team length. Pressure and space are nearest-opponent distances. The feature function reads only the supplied current snapshot; it does not read ball columns, event columns, outcomes, or future frames.

## Probability model

`src/tactics/pass_probability.py` provides a deterministic regularized logistic model with standardized features and a fixed iteration count. It also provides a fixed monotonic heuristic baseline. The fitter requires configurable minimum positive and negative counts; otherwise it returns `heuristic_fallback` rather than manufacturing a learned result. In this run, no binary outcomes were available, so logistic coefficients, ROC-AUC, PR-AUC, Brier score, log loss, and calibration error are not estimable.

## Split and leakage boundary

The configuration records a chronological development/evaluation split for future labeled data. The current run has no labels and therefore has zero train/evaluation examples. Candidate construction and ranking accept only player state and a model. Event rows are used post hoc for identifiable source/receiver ranking evaluation. Event labels, ball coordinates, future positions, post-pass state, and outcomes are not feature inputs.

## Counterfactual ranking

The decision score is a configured weighted sum of fallback/model probability and transparent tactical terms. The default weights are success 0.55, progress 0.20, space 0.15, support 0.10, pressure penalty 0.10, and distance penalty 0.05. The implementation returns the probability, tactical value, and final decision score separately.

## Ranking evaluation and baselines

Only the 15 identifiable PASS source/receiver mappings are evaluated, and only when the annotated receiver is among generated candidates. This produced 15 ranked states and 149 candidates. Results:

| Method | MRR | NDCG | Top-1 | Top-3 |
|---|---:|---:|---:|---:|
| Pass model fallback | 0.4440 | 0.5738 | 0.2667 | 0.5333 |
| Nearest teammate | **0.6478** | **0.7292** | **0.5333** | **0.7333** |
| Most-forward teammate | 0.3313 | 0.4879 | 0.1333 | 0.2667 |
| Existing TAS-style baseline | 0.3313 | 0.4879 | 0.1333 | 0.2667 |

These metrics measure agreement with the annotated target, not pass completion or rejection of absent alternatives. The nearest-teammate baseline is the strongest result in this small sample; Step 82 does not claim an improvement over it.

## Ablation

The requested geometry, geometry-plus-pressure, geometry-plus-pressure-plus-support/space, and full-model ablations are represented in `ablation_results.csv`. Their completion metrics are unavailable because there are zero reliable completion labels; no numeric ablation superiority is claimed.

## Artifacts and reproducibility

Run:

```powershell
.venv\Scripts\python.exe src\validation\run_step82.py
```

Configuration: `configs/step82_tactical_pass_model.yaml`.

Metrics and the candidate table are under `results/step82/metrics/`; calibration, distribution, ranking, feature-contribution, and counterfactual figures are under `results/step82/figures/`. The focused test module is `tests/test_step82_tactical_pass_model.py`.

## Limitations and interpretation

This is one Metrica match and 60 seconds. Metrica source/receiver annotations are usable for target-ranking evaluation, but they do not establish completion. No learned completion model was trained or validated because no defensible binary completion labels were available. The heuristic fallback probability is a principled decision-support score, not a calibrated empirical completion probability. A larger dataset with explicit pass outcomes is required before reporting probability quality or claiming learned-model value. These controlled results do not establish real-broadcast performance; SoccerNet-GSR remains required and incomplete.