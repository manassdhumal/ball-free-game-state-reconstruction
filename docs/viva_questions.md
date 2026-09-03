# Viva Questions and Evidence-Based Answers

## 1. Why is there no ball detection?

The project deliberately studies whether useful player-centric signals can be inferred when ball coordinates are unavailable or unreliable. Ball coordinates are excluded from possession and tactical inference by contract.

## 2. Why use player geometry?

Relative player positions encode spacing, pressure, support, density, and forward options. These signals are incomplete but provide an interpretable baseline under the ball-free constraint.

## 3. How is possession inferred?

Visible players receive a weighted heuristic score from opponent proximity, teammate proximity, local density, centrality, movement continuity, forward progress, and team compactness. Temporal persistence and a switch margin reduce unstable changes.

## 4. Why use heuristic scoring?

The project prioritizes an auditable baseline whose inputs and failure modes can be inspected. The weights are configurable and intuition-based, not learned or optimized as a claimed final model.

## 5. Why not use a GNN or TGN?

A learned temporal graph model would be a valid future upgrade, but it would add modeling complexity and training requirements. This project first establishes a deterministic graph and a leakage-tested baseline for robustness analysis.

## 6. What is synthetic degradation?

It is a controlled, seeded perturbation of clean player tracking that introduces configured observation and identity-level stress. It is an evaluation mechanism, not a claim to reproduce all real tracker errors.

## 7. Why interpolation?

Short internal gaps break trajectory continuity. Bounded linear interpolation can restore those gaps while preserving observed endpoints. Long and boundary gaps are deliberately not extrapolated.

## 8. Why smoothing?

Coordinate jitter can create unstable velocities and geometry. Savitzky-Golay smoothing reduces local high-frequency variation while operating only on contiguous visible segments.

## 9. How is robustness measured?

The benchmark compares tracking quality, event-based possession metrics, and tactical-score stability across conditions. The runspec, not the individual frame, is the primary experimental unit.

## 10. Why use CLEAN as the reference?

Every degraded and mitigated run is generated from a corresponding clean run, making within-run comparisons possible. CLEAN is a robustness reference, not a tactical ground-truth condition.

## 11. Why is CLEAN not tactical ground truth?

TAS has no external tactical labels in this study. The clean tactical output is used as the comparison reference, so CLEAN tactical MAE is zero by construction.

## 12. What is F1?

F1 is the harmonic mean of precision and recall: $F1 = 2PR/(P+R)$. Here it evaluates temporally matched possession-event candidates against Metrica reference events.

## 13. Why run-level statistics?

Frames within a run are temporally dependent and would create pseudo-replication if treated as independent observations. Run-level aggregation gives each configured runspec one experimental unit.

## 14. Why multiple seeds?

Different seeds test whether results depend on one stochastic degradation realization. The five seeds are repeated stress realizations, not independent matches.

## 15. Why multiple severities?

Severity analysis tests whether downstream sensitivity changes across a controlled range of degradation settings. The labels are configuration-specific, not universal categories.

## 16. What does H1 mean?

H1 states that tracking degradation reduces ball-free possession performance. It is supported: mean F1 falls from 0.49199 CLEAN to 0.23451 degraded.

## 17. What does H2 mean?

H2 states that degradation increases tactical instability relative to CLEAN. It is supported: mean tactical MAE rises from 0.00000 to 0.15635.

## 18. What does H3 mean?

H3 states that mitigation recovers part of the downstream loss. It is supported: possession F1 improves by 0.17877 to 0.41328, although it remains below CLEAN.

## 19. Why only one Metrica match?

The final benchmark is a controlled expansion of one validated Metrica source. This provides repeated windows and degradation realizations but limits cross-match and league generalization.

## 20. Why is GSR not included?

No provenance-verifiable local GSR tracking output has been validated for this benchmark. GSR is therefore BLOCKED / OPTIONAL rather than presented as an unsupported result.

## 21. What are the main weaknesses?

The main weaknesses are synthetic rather than real degradation, one benchmark match, heuristic downstream models, imperfect event references, tolerance dependence, and incomplete recovery under severe information loss.

## 22. How would you improve the system?

Use learned temporal or graph models, calibrated tracking uncertainty, stronger identity handling, multiple validated matches, and external tactical or possession annotations while preserving the anti-leakage contract.

## 23. How would real-time deployment change the design?

Latency, causal filtering, online identity management, missing-data decisions, and compute budgets would matter. Centered smoothing may need to be replaced or adapted because it uses future samples within its window.

## 24. How would you validate on another league?

Convert the source to the canonical schema, document orientation and pitch dimensions, verify identities and visibility semantics, rerun the same frozen metric contract, and report results separately before pooling.

## 25. What is the main research contribution?

The main contribution is an auditable player-only robustness benchmark that connects controlled tracking degradation to downstream possession and tactical-score sensitivity, then quantifies incomplete recovery after mitigation.

## 26. Are Metrica event labels used by the predictor?

No. They are used only after predictions are finalized for post-hoc temporal matching and evaluation.

## 27. Are ball coordinates ever present in the repository?

They may exist in source or auxiliary structures, but the possession and tactical inference modules use a strict player-tracking whitelist and ignore ball columns.

## 28. Does mitigation guarantee clean-level performance?

No. The aggregate mitigated F1 of 0.41328 remains below the CLEAN mean of 0.49199, and individual runs vary.

## 29. Does the result prove causality?

It supports the directional hypotheses under a controlled synthetic experiment. It does not establish causality beyond that design or transfer to real broadcast systems.

## 30. What should be said about tactical performance?

TAS robustness is measured as deviation from CLEAN, not accuracy against tactical ground truth. The tactical result should be described as stability of an interpretable heuristic score.
