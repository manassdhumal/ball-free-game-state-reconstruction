# Technical Demo Plan

Target duration: 3-5 minutes. Use an existing Metrica window and existing derived outputs. Do not run a new experiment during the demo.

## 1. Load a Metrica Window

- Load one of the configured 60-second windows from canonical Metrica tracking.
- Display frame range, FPS, orientation convention, and player count.
- State that raw data is read-only.

## 2. Show Player Tracking

- Plot visible home and away player coordinates on normalized pitch space.
- Keep ball coordinates hidden from the inference view.
- Point out explicit visibility and missing-coordinate semantics.

## 3. Show the Clean Spatial Graph

- Build or display a frame graph for one clean frame.
- Highlight nodes, nearest teammates, nearest opponents, density, and optional KNN edges.
- Explain that the graph uses player geometry only.

## 4. Show Synthetic Degradation

- Load the existing degraded output for the same configured run.
- Compare missingness, visible-player completeness, and motion anomalies.
- Label this as synthetic degradation, not a real tracker output.

## 5. Show Mitigation

- Display the existing interpolation-plus-Savitzky-Golay result.
- Highlight short-gap recovery and smoother contiguous trajectories.
- Note that long gaps are not extrapolated and imputed rows retain provenance.

## 6. Show Ball-Free Possession

- Run the existing possession inference on the selected condition or load its derived output.
- Display inferred possessor/team and candidate transitions.
- Explicitly state: Metrica events are not used during inference; they are used only later for post-hoc evaluation.

## 7. Show Tactical Score

- Display pressure, support, forward options, defensive spread, and TAS for the selected frame.
- Explain that TAS is an interpretable geometric proxy, not ground-truth tactical advantage.

## 8. Show Robustness Comparison

- Open the existing final-analysis summary or figure.
- Present the frozen aggregate values:
  - Possession F1: CLEAN 0.49199, degraded 0.23451, mitigated 0.41328.
  - Tactical MAE: CLEAN 0.00000, degraded 0.15635, mitigated 0.10888.
- Close with the scope: 60 runspecs, 180 condition observations, one Metrica match, no real GSR validation.

## Demo Safety Rules

- Do not expose ball coordinates in an inference panel.
- Do not imply event labels were available to the predictor.
- Do not regenerate canonical results.
- Do not present GSR resources or setup files as GSR validation.
