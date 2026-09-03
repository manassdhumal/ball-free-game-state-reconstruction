# Reproducibility Guide

## Repository Setup

From the repository root:

```powershell
cd C:\Users\manas\OneDrive\Desktop\ball-free-game-state-reconstruction
```

The project uses a local virtual environment at `.venv`. Activate it if desired, or call its interpreter directly:

```powershell
.venv\Scripts\python.exe --version
```

The repository contains a `pyproject.toml` for the separate `soccernet-gamestate` resource package, but the project scripts and tests should be run from the repository root with the project virtual environment.

## Dependencies

The core code uses Python, NumPy, pandas, and standard-library modules. Matplotlib is used by optional figure-generation utilities when installed. The current final analysis and frozen result files already exist locally; report preparation does not require regenerating figures.

## Data Requirements

Raw and intermediate datasets are governed by `.gitignore` and are not committed. The Metrica benchmark expects the Sample Game 1 tracking and event files under:

```text
data/raw/metrica/data/Sample_Game_1/
```

The configured Metrica paths are recorded in `configs/robustness_benchmark.yaml` and `configs/robustness_experiment.yaml`.

## Metrica Data Location and Benchmark

The primary benchmark uses `metrica_sample_game_1` at 25 FPS with four 60-second windows: 0-60 s, 300-360 s, 600-660 s, and 900-960 s. The orientation convention is `home_attacks_x1=true` for the configured benchmark windows.

Metrica events are used only after predictions are complete for post-hoc temporal matching. They are not inputs to possession or tactical inference.

## SkillCorner Limitation

The repository includes a SkillCorner parser and detailed compatibility notes for ten Australian A-League 2024/2025 matches. The local real-data test cases are skipped when match 1886347 is not available. SkillCorner compatibility documentation is not evidence of final benchmark validation.

## Experiment Configuration

The original single-run benchmark configuration is in `configs/robustness_benchmark.yaml`. The expanded grid configuration is in `configs/robustness_experiment.yaml` and specifies:

- windows: `[0, 60]`, `[300, 360]`, `[600, 660]`, `[900, 960]`
- seeds: `72`, `73`, `74`, `75`, `76`
- severities: `mild`, `moderate`, `severe`
- conditions: CLEAN, SYNTHETIC-DEGRADED, NOISE-MITIGATED
- interpolation: `max_gap=10`
- smoothing: Savitzky-Golay, window 7, polynomial order 2
- event matching tolerance: 1.0 seconds

The verified final population is 60 unique runspecs and 180 condition observations.

## Step 74 Experiment Setup

Step 74 expanded the earlier single-window, single-seed benchmark across four windows, five seeds, and three severity levels. It generated canonical per-run summaries and resolved configurations under the robustness expansion result directories. Legacy nested artifacts remain preserved and are not to be deleted or renamed.

## Canonical Result Policy

Raw, intermediate, model, and generated result files are ignored by default. Do not alter canonical result files during report preparation. Do not delete, move, or rename legacy result artifacts. Final-analysis CSVs, JSON files, and figures are derived evidence and should be read as locally available outputs under the existing storage policy.

## Final Analysis Entry Point

The final analysis source is [src/validation/final_analysis.py](../src/validation/final_analysis.py). It reads validated canonical summaries and writes derived outputs under `results/metrics/final_analysis/` and `results/figures/final_analysis/`. During the report-preparation phase, use the existing outputs and do not rerun the experiment or regenerate canonical results.

## Test Command

```powershell
.venv\Scripts\python.exe -m unittest discover tests -v
```

The verified current result is 161 tests passed and 11 skipped. Skips are associated with unavailable local real SkillCorner data.

## GSR Status

SoccerNet-GSR remains **BLOCKED / OPTIONAL**. No provenance-verifiable local GSR tracking output has been validated for this benchmark, so no GSR condition or real-world GSR performance claim is included.
