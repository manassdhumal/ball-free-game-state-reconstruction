# Frontend and Backend Architecture

## Application paths

- `frontend/` contains the preserved Lovable React/TanStack application.
- `backend/` contains the FastAPI adapter and API schemas.
- `src/` remains the research source of truth. API services import those modules; they do not duplicate them.
- `results/step80` through `results/step83` are read-only cached experiment artifacts.

## Startup

From the repository root, run `./start_backend.ps1` and `./start_frontend.ps1` in separate PowerShell terminals, or run:

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
cd ..\frontend
npm install
npm run dev -- --host localhost --port 5173
```

The API is at `http://localhost:8000`, Swagger docs are at `/docs`, and the browser app is at `http://localhost:5173`.

## Data flow

The frontend uses `frontend/src/api/` for HTTP calls. FastAPI reads canonical Metrica tracking/events through the existing parsers in `src/data/`, calls the existing ball-free possession/tactical/pass modules for bounded frame requests, and reads static Step 80-83 summaries from `results/` without rerunning experiments.

## API surface

The adapter exposes health, matches, game state, players and trajectories, possession, events and metrics, tactical state, pass candidates/ranking, robustness summaries, and GSR status under `/api`. Pass ranking is a current-frame Step 82 heuristic fallback and is labelled as such; it is not a calibrated completion model. Possession timeline materialization and predicted event rows remain pending. Unsupported or missing data is returned as an explicit availability state or a structured 404; no values are fabricated.

## Results mapping and provenance

Step 80 tracking/robustness, Step 81 event evaluation, Step 82 ranking, and Step 83 counterfactual stability files are read from their existing `metrics/` directories. Responses include dataset, sequence, frame range, FPS, method, or experiment provenance where the source provides it. GSR remains blocked until authorized data, labels, weights, and a runnable environment exist.

## Troubleshooting

- `Backend unavailable`: start FastAPI on port 8000 and check `http://localhost:8000/api/health`.
- CORS errors: use the documented localhost ports; development CORS permits `http://localhost:5173`.
- `Data not available`: use a frame within the selected match range. The API intentionally bounds frame requests and does not preload full matches into the browser.