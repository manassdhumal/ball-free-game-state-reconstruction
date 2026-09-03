# Ball-Free Game State Reconstruction

A computer-vision research project for reconstructing selected football game-state signals from player tracking without using ball coordinates during possession or tactical inference. The system covers Metrica and SkillCorner-compatible ingestion, canonical player tracking, tracking-quality analysis, synthetic degradation, bounded interpolation and smoothing, a player-only spatial graph, heuristic possession/event inference, interpretable tactical scoring, and task-driven robustness evaluation.

## Final Result

The verified Step 75 analysis contains 60 unique runspecs and 180 condition observations across 4 windows, 5 seeds, 3 severities, and 3 conditions. Mean possession F1 is 0.49199 for `CLEAN`, 0.23451 for `SYNTHETIC-DEGRADED`, and 0.41328 for `NOISE-MITIGATED`. Mean tactical MAE relative to `CLEAN` is 0.00000, 0.15635, and 0.10888 respectively. Mitigation improves mean possession F1 by 0.17877.

These are benchmark-specific results from one Metrica match. `CLEAN` is a robustness reference, not tactical ground truth; Metrica events are used only for post-hoc evaluation; and ball coordinates are excluded from inference.

## Pipeline

```text
Tracking sources -> canonical tracking -> quality analysis -> degradation
-> interpolation/smoothing -> spatial graph -> ball-free possession
-> tactical features/TAS -> robustness evaluation -> statistical analysis
```

## Reproducibility

Start with [docs/reproducibility.md](docs/reproducibility.md), [docs/final_project_report.md](docs/final_project_report.md), and the frozen outputs under `results/metrics/final_analysis/` when available locally. Run the test suite with:

```powershell
.venv\Scripts\python.exe -m unittest discover tests -v
```

## Repository Structure

- `src/data/`: Metrica and SkillCorner ingestion
- `src/noise/`: degradation, interpolation, and smoothing
- `src/possession/`: spatial graph and ball-free possession baseline
- `src/tactics/`: tactical features and TAS
- `src/validation/`: benchmark and final statistical analysis
- `docs/`: methodology, report, presentation, demo, viva, and reproducibility material
- `configs/`: benchmark and expanded experiment contracts
- `tests/`: validation and anti-leakage tests

## Limitations and GSR Status

The final robustness result uses one Metrica match, synthetic parameter-specific degradation, heuristic downstream models, and imperfect post-hoc event references. SkillCorner compatibility is documented, but local real-data tests are skipped when the archive is unavailable. SoccerNet-GSR remains **BLOCKED / OPTIONAL**: no provenance-verifiable GSR output is included or claimed.
