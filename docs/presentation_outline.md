# Final-Year Project Presentation Outline

Target duration: 10-15 minutes. Keep each slide to 3-5 visible points.

## 1. Title

- Ball-Free Game State Reconstruction
- Player-centric inference from tracking
- Robustness under synthetic degradation

**Recommended visual:** system architecture diagram.

**Speaker note:** State immediately that the project studies player-only inference and does not claim real-world GSR validation.

## 2. Problem

- Ball detections may be unavailable or unreliable.
- Tracking contains gaps, jitter, and identity issues.
- Downstream possession and tactical signals can amplify tracking errors.

**Recommended visual:** clean versus degraded player-tracking example.

**Speaker note:** Frame the work as an auditable baseline and robustness study.

## 3. Motivation

- Player geometry still contains team-structure information.
- A strict information boundary makes leakage testable.
- Robustness should be measured at the task level, not only at the coordinate level.

**Recommended visual:** player-only pitch geometry sketch.

**Speaker note:** Explain why a deliberately limited system is scientifically useful.

## 4. Research Question

- Does degradation reduce ball-free possession performance?
- Does it increase tactical-score instability?
- Does interpolation and smoothing recover performance?

**Recommended visual:** H1/H2/H3 flow.

**Speaker note:** Define CLEAN as a reference condition, not tactical ground truth.

## 5. System Overview

- Canonical tracking
- Quality analysis and synthetic degradation
- Mitigation, spatial graph, possession, tactical score
- Robustness and statistical analysis

**Recommended visual:** [system_architecture.md](system_architecture.md) diagram.

**Speaker note:** Mark Metrica events as post-hoc evaluation only and ball coordinates as excluded from inference.

## 6. Data Sources

- Metrica Sample Game 1 is the final benchmark source.
- Four 60-second windows are analyzed.
- SkillCorner ingestion is documented for compatibility.
- GSR is blocked and not included.

**Recommended visual:** source-to-canonical schema diagram.

**Speaker note:** Mention the one-match limitation clearly.

## 7. Ball-Free Possession

- Visible players become graph nodes.
- Geometry and movement continuity produce heuristic scores.
- Persistence prevents rapid frame-level switching.
- Events are candidate transitions, evaluated post-hoc.

**Recommended visual:** spatial graph with teammate/opponent relations.

**Speaker note:** Explain that no ball coordinate, event label, or future frame enters inference.

## 8. Tactical Scoring

- Pressure
- Forward options
- Teammate support
- Defensive spread

**Recommended visual:** TAS feature diagram or tactical feature plot.

**Speaker note:** Emphasize interpretability and the absence of external tactical ground truth.

## 9. Robustness Experiment

- 60 unique runspecs
- 180 condition observations
- 4 windows x 5 seeds x 3 severities
- CLEAN, SYNTHETIC-DEGRADED, NOISE-MITIGATED

**Recommended visual:** experiment grid/table.

**Speaker note:** The runspec, not the frame, is the primary experimental unit.

## 10. Results

- Possession F1: 0.49199 -> 0.23451 -> 0.41328
- Tactical MAE: 0.00000 -> 0.15635 -> 0.10888
- Possession improvement after mitigation: 0.17877

**Recommended visual:** `possession_f1_vs_severity.png` and `tactical_mae_vs_severity.png`.

**Speaker note:** Say the arrows represent CLEAN, degraded, mitigated conditions in that order.

## 11. Key Findings

- H1 supported within the benchmark.
- H2 supported within the benchmark.
- H3 supported with incomplete recovery.
- Effects vary across severity, windows, and seeds.

**Recommended visual:** `possession_degraded_vs_mitigated.png` or `mitigation_by_window.png`.

**Speaker note:** Avoid causal or universal robustness language.

## 12. Limitations

- One Metrica match
- Synthetic, parameter-specific degradation
- Heuristic possession and TAS
- Imperfect post-hoc event reference
- CLEAN is not tactical ground truth

**Recommended visual:** limitation matrix.

**Speaker note:** Treat limitations as part of the result, not a footnote.

## 13. GSR Future Work

- GSR remains BLOCKED / OPTIONAL.
- A verified output requires provenance, schema conversion, and orientation evidence.
- No GSR result is claimed here.

**Recommended visual:** validation checklist, not a fabricated result.

**Speaker note:** Do not imply that repository model resources equal validated GSR output.

## 14. Conclusion

- Player-only geometry supports an interpretable baseline.
- Tracking degradation materially affects downstream tasks.
- Mitigation recovers part, not all, of the loss.
- The benchmark is reproducible and explicitly bounded.

**Recommended visual:** final pipeline plus three headline numbers.

**Speaker note:** Close with the scoped contribution: evidence about robustness of this baseline under this benchmark.
