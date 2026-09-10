# SoccerNet-GSR Reproducibility & Export Protocol

## 1. One-Sequence Execution Policy

To prevent unbounded compute usage and ensure reproducibility:
1. Every GSR execution must target exactly **one** confirmed validation sequence (e.g. `SNGS-04`).
2. Large-scale multi-sequence runs are prohibited until single-sequence verification passes.

## 2. Canonical Tracking Schema Mapping

The raw TrackLab output must be transformed strictly to **Canonical Tracking Schema (v0.2.0)**:

| Canonical Column | Data Type | Source Mapping |
| :--- | :--- | :--- |
| `match_id` | String | Format: `gsr_valid_<sequence_id>` |
| `frame` | Integer | Source frame index |
| `timestamp` | Float | `frame / fps` (seconds) |
| `player_id` | String | Source track ID string (e.g. `"10"`) |
| `team` | String | Controlled vocabulary: `"home"`, `"away"`, or `"referee"` |
| `x` | Float | Pitch-space longitudinal coordinate in $[0.0, 1.0]$ |
| `y` | Float | Pitch-space lateral coordinate in $[0.0, 1.0]$ |
| `confidence` | Float | Detection/tracking confidence in $[0.0, 1.0]$ |
| `visible` | Boolean | True if directly detected and $conf > 0.1$ |

## 3. Artifact Packaging Specification

All transferable GSR artifacts are packaged into `results/gsr_kaggle/`:
```text
results/gsr_kaggle/
├── environment.json            # Hardware and OS audit
├── run_manifest.json           # Pre-execution metadata & commit SHAs
├── tracking_summary.csv        # Missingness, track length, coordinate bounds
├── logs/
│   └── inference_<seq>.log     # Full stdout/stderr capture
├── exports/
│   ├── gsr_tracking_canonical.csv
│   └── manifest.json           # SHA256 checksums & row counts
└── figures/
    └── player_trajectories.png # Trajectory visualization
```
