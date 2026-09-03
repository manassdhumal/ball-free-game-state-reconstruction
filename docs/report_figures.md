# Report Figure Guide

| Filename | Purpose | X-axis | Y-axis | Expected interpretation | Report section |
|---|---|---|---|---|---|
| `possession_f1_vs_severity.png` | Show possession F1 across stress levels | Severity | Possession F1 | Degradation lowers event-based ball-free possession performance; mitigation partially recovers it | 7.2, 7.4 |
| `tactical_mae_vs_severity.png` | Show tactical instability across stress levels | Severity | Tactical MAE relative to CLEAN | Degradation increases TAS error; mitigation reduces but does not eliminate it | 7.3, 7.4 |
| `tracking_quality_vs_severity.png` | Link tracking quality to severity and condition | Severity | Tracking-quality metric, typically missingness | Synthetic stress worsens tracking quality and mitigation changes the observed quality profile | 7.1, 7.4 |
| `possession_degraded_vs_mitigated.png` | Compare paired possession outputs | Run or paired condition | Possession F1 | Mitigation moves many runs upward from degraded performance, with incomplete recovery | 7.2, 7.7 |
| `tactical_degraded_vs_mitigated.png` | Compare paired tactical error outputs | Run or paired condition | Tactical MAE | Mitigation reduces tactical error for the aggregate benchmark, with run-level variation | 7.3, 7.7 |
| `mitigation_by_window.png` | Show recovery variation by window | Window | Recovery metric/fraction | Recovery is examined across four temporal windows rather than one interval | 7.6, 7.7 |
| `mitigation_by_seed.png` | Show recovery variation by seed | Seed | Recovery metric/fraction | Recovery is examined across five degradation realizations | 7.6, 7.7 |
| `tracking_vs_possession.png` | Relate tracking quality to possession performance | Tracking-quality metric | Possession F1 | Poorer tracking quality is associated with weaker possession performance within this benchmark; association is not universal causality | 7.1, 8.1 |
| `tracking_vs_tactical_instability.png` | Relate tracking quality to tactical error | Tracking-quality metric | Tactical MAE/instability | Tracking degradation is associated with greater TAS deviation from CLEAN within the tested data | 7.1, 8.2 |

## Presentation Notes

- Use captions that name the condition and benchmark scope.
- State that CLEAN is the robustness reference, not tactical ground truth.
- Do not present a figure as validation of real GSR output.
- Do not regenerate these figures during report preparation.
