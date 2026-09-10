# Step 83 — Counterfactual Tactical Decision Analysis

## Research question

For an observed player-only game state, do ranked hypothetical pass options provide useful tactical decision information beyond simple heuristics, and how stable are those rankings under realistic tracking perturbations?

This is model-based tactical decision support from player-only state. It is not player-intent reconstruction, optimality proof, coaching-effectiveness evidence, or real-broadcast/GSR validation.

## Data and scope

The deterministic experiment uses Metrica Sample Game 1, frames 1–1500 (0–60 seconds at 25 FPS), with `home_attacks_x1=true`. Fifteen PASS events have defensible source and receiver identities, yielding 15 evaluated states and 149 clean hypothetical candidates, or 9.93 candidates per state. Event annotations select the evaluation states and targets only after player-only rankings are constructed. They are not scoring features.

Step 82 provides the heuristic pass-completion proxy because the canonical Metrica events contain no reliable binary completion outcome. Step 83 therefore evaluates target-ranking agreement and perturbation stability, not calibrated completion probability.

## Counterfactual definition and ranking

For an observed snapshot $S_t$, each candidate is a hypothetical action $a=(source,target)$ evaluated as a player-only option $S_t(a)$. Only the target choice changes conceptually; no future state is simulated and no claim is made that the action occurred.

The scorer reuses Step 82 features and keeps the components interpretable:

- **Probability/proxy:** Step 82's fixed heuristic fallback, because no defensible completion labels exist.
- **Tactical value:** forward progression, target space, support, target pressure, and distance.
- **Decision score:** a configurable weighted combination of the proxy and tactical value, followed by deterministic target-ID tie-breaking.

The balanced Step 83 profile uses success `0.45`, progress `0.22`, space `0.16`, support `0.12`, pressure `0.10`, and distance `0.05`. This is a decision-support score, not a probability of actual success.

## Actual versus counterfactual evaluation

When the annotated receiver is available among generated candidates, the analysis reports its rank, percentile, top-1 agreement, top-3 inclusion, and reciprocal rank. It answers “how highly did the model rank the observed target?” It does not call the target optimal and does not treat absent candidates as rejected alternatives.

Clean ranking results over 15 states:

| Method | MRR | Top-1 | Top-3 | Mean rank |
|---|---:|---:|---:|---:|
| Nearest teammate | **0.6478** | **0.5333** | **0.7333** | 3.13 |
| Most-forward teammate | 0.3313 | 0.1333 | 0.2667 | 4.40 |
| Existing TAS-style baseline | 0.3313 | 0.1333 | 0.2667 | 4.40 |
| Step 82 fallback | 0.4440 | 0.2667 | 0.5333 | 4.07 |
| Step 83 balanced decision score | 0.4107 | 0.2000 | 0.5333 | 4.13 |

The nearest-teammate baseline is strongest on this small target-ranking sample. Step 83 does not claim an improvement over that baseline.

## Perturbation design and stability

The bounded experiment uses fixed seeds `83`, `184`, and `285`, with three perturbation families and mild/moderate/severe settings:

- Coordinate jitter: Gaussian standard deviations `0.001`, `0.003`, and `0.008` normalized pitch units, matching the existing noise-analysis scale.
- Player dropout: `2%`, `8%`, and `20%` visible-player dropout.
- Availability jitter: the corresponding jitter plus half-rate dropout.

For each state and method, candidates are regenerated and compared to the clean ranking using top-1 retention, top-3 retention, Spearman correlation, Kendall correlation, mean rank displacement, score variance, and decision flips.

For the Step 83 balanced score:

| Perturbation | Severity | Top-1 retention | Top-3 retention | Spearman | Mean displacement | Flip rate |
|---|---|---:|---:|---:|---:|---:|
| Coordinate jitter | Mild | 1.0000 | 1.0000 | 0.9895 | 0.1467 | 0.0000 |
| Coordinate jitter | Moderate | 1.0000 | 1.0000 | 0.9787 | 0.2548 | 0.0000 |
| Coordinate jitter | Severe | 0.9333 | 1.0000 | 0.9294 | 0.6281 | 0.0667 |
| Player dropout | Mild | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Player dropout | Moderate | 0.7143 | 1.0000 | 0.9398 | 0.7071 | 0.3333 |
| Player dropout | Severe | 0.6364 | 1.0000 | 0.9310 | 1.2229 | 0.5333 |
| Availability jitter | Mild | 1.0000 | 1.0000 | 0.9895 | 0.1467 | 0.0000 |
| Availability jitter | Moderate | 0.6667 | 1.0000 | 0.9787 | 0.4504 | 0.3333 |
| Availability jitter | Severe | 0.5714 | 1.0000 | 0.8664 | 0.9976 | 0.4667 |

The overall balanced flip rate averaged over the nine perturbation conditions is `0.1926`. This is a controlled sensitivity result, not a claim of robustness to every broadcast failure mode.

## Baseline stability and decision flips

Stability and flip tables include all five methods. Overall flip rates were: nearest teammate `0.1333`, most-forward `0.0667`, TAS `0.0667`, Step 82 fallback `0.2296`, and Step 83 balanced `0.1926`. Dropout produces substantially more action changes than coordinate jitter; at severe dropout the balanced method flips `0.5333` of evaluated states.

## Weight sensitivity

Three fixed profiles were compared without tuning on evaluation targets:

| Profile | MRR | Top-1 | Top-3 | Retention vs balanced | Spearman vs balanced |
|---|---:|---:|---:|---:|---:|
| Probability-heavy | 0.4784 | 0.2667 | 0.6000 | 0.6667 | 0.8761 |
| Balanced | 0.4107 | 0.2000 | 0.5333 | 1.0000 | 1.0000 |
| Tactical-heavy | 0.3440 | 0.1333 | 0.3333 | 0.7333 | 0.9553 |

The ranking is meaningfully sensitive to the probability/tactical tradeoff. The probability-heavy profile performs best on this target-ranking sample, but this cannot establish completion-model superiority because no completion labels exist.

## Scenario analysis

Four actual observed states were selected deterministically from the evaluated states: highest source pressure, greatest target space, greatest forward opportunity, and greatest average pass-line obstruction. The machine-readable scenario table records the top three actual candidates and their proxy, tactical value, and final score. These examples illustrate feature tradeoffs only; they are not statistical proof.

## Anti-leakage and temporal causality

The scorer accepts only the current player snapshot and model/configuration. Tests and summary metadata verify that ball coordinates, event labels, outcomes, and future frames do not affect current ranking. Perturbations are applied to the current snapshot only. No post-pass state or future simulation is used.

## Artifacts and reproducibility

Run:

```powershell
.venv\Scripts\python.exe src\validation\run_step83.py
```

Configuration: `configs/step83_counterfactual_decision.yaml`.

Metrics are under `results/step83/metrics/`:
`counterfactual_candidates.csv`, `actual_target_ranking.csv`, `ranking_stability.csv`, `decision_flip_rates.csv`, `weight_sensitivity.csv`, `baseline_comparison.csv`, `scenario_analysis.csv`, and `step83_summary.json`.

Figures are under `results/step83/figures/`: ranking stability, decision flips, baseline comparison, weight sensitivity, and observed counterfactual scenarios.

## Limitations and conclusion

The experiment supports a narrow conclusion: player-only counterfactual rankings provide interpretable target-ranking information, but the nearest-teammate heuristic remains stronger on this small observed-target sample, and recommended actions can change under moderate/severe missingness. Coordinate jitter is comparatively benign. The method is therefore a controlled tactical decision-support prototype, not an optimal decision engine.

Limitations are the 15-state single-match sample, lack of reliable completion outcomes, hypothetical actions, ambiguity between model preference and actual player choice, dependence on heuristic features, and controlled Metrica rather than real-broadcast data. SoccerNet-GSR remains required and incomplete.