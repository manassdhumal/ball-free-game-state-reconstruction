
## Local Application

The preserved Lovable frontend is in `frontend/`; the FastAPI adapter is in `backend/`. Python research modules under `src/` and historical results under `results/step80/` through `results/step83/` remain the source of truth.

Prerequisites: Python with the repository `.venv`, Node.js/npm, and the backend requirements in `backend/requirements.txt`.

Start the backend:

```powershell
.\start_backend.ps1
```

Start the frontend in a second terminal:

```powershell
.\start_frontend.ps1
```

The API is `http://localhost:8000` and its documentation is `http://localhost:8000/docs`. The frontend is `http://localhost:5173`. Set `VITE_API_BASE_URL` in `frontend/.env.local` when using another API origin; see `frontend/.env.example`. Current-frame pass ranking uses the existing heuristic fallback and is not a pass-completion probability; possession timeline materialization remains pending.

See [docs/frontend_backend_architecture.md](docs/frontend_backend_architecture.md) for the API list, data flow, result mapping, GSR boundary, and troubleshooting.
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

## Video Perception & Game State Reconstruction Studio

The platform supports end-to-end video game-state reconstruction directly from uploaded match footage (`.mp4`, `.mov`, `.avi`):
- **Web Studio (`/broadcast`):** Drag-and-drop video upload, camera perspective presets, interactive 4-point pitch calibration, and dual-view video playback synchronized with an animated 2D tactical pitch radar.
- **Offline CLI (`scripts/process_video.py`):**
  ```powershell
  .venv\Scripts\python.exe scripts/process_video.py --video data/samples/sample_match_clip.mp4 --export-video
  ```
- **Three-Tier Robustness Benchmark (Step 84):** Evaluates stability across Clean ground tracking, Synthetic-Degraded tracking, and Real Video Perception tracking:
  ```powershell
  .venv\Scripts\python.exe -m unittest tests/test_three_tier_benchmark.py
  ```

## Repository Structure

- `src/data/`: Metrica and SkillCorner ingestion
- `src/noise/`: Degradation, bounded interpolation, and Savitzky-Golay smoothing
- `src/possession/`: Spatial graph and ball-free possession baseline
- `src/tactics/`: Tactical Advantage Score (TAS) and counterfactual pass ranking
- `src/vision/`: Video tracking (YOLOv8 + ByteTrack), 2D pitch homography, and jersey classification
- `src/validation/`: Benchmark, statistical analysis, and cross-dataset validation
- `backend/`: FastAPI adapter with match data and video processing endpoints
- `frontend/`: Interactive React/Vite dashboard with live 2D pitch radar and Video Studio
- `scripts/`: Offline processing, sample generation, and verification scripts
- `docs/`: Technical specifications, architecture, and reproducibility reports
- `tests/`: Automated unit tests and anti-leakage contracts

