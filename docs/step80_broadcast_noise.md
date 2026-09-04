# Step 80: Broadcast-Specific Noise Analysis

Step 80 is a separate ablation and does not replace or rewrite the validated Step 72/74/75 outputs. It uses the existing Metrica Sample Game 1 canonical tracking and event files. Events are loaded only for post-hoc possession evaluation; ball coordinates are never passed to possession or tactical inference.

## Failure modes

The analysis distinguishes frame-level missingness, short bounded gaps, coordinate jitter, identity fragmentation, apparent discontinuity, and camera-cut/discontinuity events. A missing or fragmented track is not treated as a camera cut by itself. A candidate cut requires evidence at a frame boundary from simultaneous high displacement and/or a large visible-track count change.

## Candidate cut signal

`detect_candidate_cuts` emits one row per frame with `frame`, `candidate_cut`, `cut_score`, visible counts, matched-track count, high-motion fraction, and mean displacement. For boundary $t$, high-motion fraction is the fraction of matched `(team, player_id)` tracks whose displacement from $t-1$ to $t$ exceeds `displacement_threshold`. The score is:

$$s_t = 0.65 h_t + 0.35 \min(c_t, 1)$$

where $h_t$ is high-motion fraction and $c_t$ is the relative visible-count change. `candidate_cut` is true when $h_t$ reaches `min_support_fraction`, or when the count change is large and motion support reaches half that threshold. This is an auditable candidate signal, not independently validated shot-boundary ground truth.

## Mitigation conditions

The ablation compares no mitigation, bounded interpolation, centered moving average, centered Savitzky-Golay, causal constant-velocity Kalman-style filtering, and interpolation combined with each of those filters. Interpolation uses `max_gap=10` frames and requires observed points on both sides. Longer, leading, and trailing gaps are not extrapolated. Moving average and Savitzky-Golay are offline symmetric smoothers and are split at candidate boundaries. The Kalman-style filter uses only current and prior observations, resets after a gap, and never fills missing rows.

## Metrics and protocol

The default configuration evaluates Metrica Sample Game 1, the 0-60 second window, severities `mild`, `moderate`, and `severe`, and seeds 72 and 73. Outputs are written only below `results/step80/`. Tracking metrics include missingness, unresolved gap count/mean/max gap length, adjacent trajectory continuity, mean and P95 displacement, velocity stability (standard deviation of successive speed changes), acceleration stability, and candidate count. Downstream metrics are ball-free possession precision/recall/F1 against Metrica events and TAS MAE relative to the CLEAN inference output using the existing TAS definition.

The machine-readable table is `results/step80/metrics/step80_ablation.csv`. Summary means, medians, standard deviations, and paired differences versus `none` (with approximate 95% confidence intervals) are stored in `step80_summary.json`. Figures include tracking quality, possession F1, TAS MAE, the candidate-cut timeline, and a labeled CLEAN/DEGRADED/MITIGATED trajectory example. A frame without an immediately preceding frame is not compared as an adjacent boundary.

## Limitations

Synthetic degradation is a controlled proxy, not broadcast annotation. Track fragmentation and identity switches are represented by the existing degradation framework but cannot be independently distinguished from a camera event using tracking alone. TAS MAE is stability relative to CLEAN, not tactical ground truth. A candidate cut may be a coordinated tracking failure or a real rapid scene-wide change, so it must not be reported as a confirmed camera cut.