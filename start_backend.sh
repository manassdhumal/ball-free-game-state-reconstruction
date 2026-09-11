#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
python3 -m uvicorn backend.app:app --reload --port 8000
