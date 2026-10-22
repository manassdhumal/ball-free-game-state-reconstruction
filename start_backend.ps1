Set-Location $PSScriptRoot
& .venv\Scripts\python.exe -m uvicorn backend.app:app --reload --reload-dir backend --reload-dir src --port 8000