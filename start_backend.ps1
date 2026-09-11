Set-Location $PSScriptRoot
& .venv\Scripts\python.exe -m uvicorn backend.app:app --reload --port 8000