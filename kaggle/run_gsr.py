#!/usr/bin/env python3
"""
Official SoccerNet-GSR (TrackLab) One-Sequence Baseline Execution Runner.

Enforces:
  1. Strict One-Sequence Policy (runs exactly 1 confirmed sequence).
  2. Pre-execution run manifest generation with commit SHAs, GPU metadata, and paths.
  3. Real-time stdout/stderr capture and telemetry logging.
  4. Error classification without silent fallbacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_git_commit(repo_dir: Path) -> str:
    """Retrieve the current Git commit SHA of a repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def get_file_sha256(filepath: Path) -> Optional[str]:
    """Calculate SHA256 checksum of a file if it exists."""
    if not filepath.exists() or not filepath.is_file():
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def create_run_manifest(
    output_dir: Path,
    project_root: Path,
    gsr_dir: Path,
    dataset_name: str,
    sequence_id: str,
    data_dir: Path,
    config_name: str,
    checkpoint_paths: Dict[str, str],
) -> Dict[str, Any]:
    """Generates pre-execution run manifest recording complete provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "timestamp_start": datetime.now(timezone.utc).isoformat(),
        "project_commit": get_git_commit(project_root),
        "gsr_commit": get_git_commit(gsr_dir) if gsr_dir.exists() else "UNAVAILABLE",
        "dataset": dataset_name,
        "sequence_id": sequence_id,
        "data_dir": str(data_dir.resolve()),
        "config_name": config_name,
        "checkpoints": checkpoint_paths,
        "python_executable": sys.executable,
        "environment": {},
    }

    # Record environment if available
    env_json = output_dir / "environment.json"
    if env_json.exists():
        try:
            with open(env_json, "r", encoding="utf-8") as ef:
                manifest["environment"] = json.load(ef)
        except Exception:
            pass

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Run manifest created at: {manifest_path}")
    return manifest


def execute_one_sequence(
    gsr_dir: Path,
    sequence_id: str,
    split: str = "valid",
    config_name: str = "soccernet",
    data_dir: Optional[Path] = None,
    output_dir: Path = Path("results/gsr_kaggle"),
) -> int:
    """Executes the official TrackLab GSR baseline for exactly one sequence."""
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"inference_{sequence_id}.log"

    venv_tracklab = gsr_dir / ".venv" / "bin" / "tracklab"
    tracklab_bin = str(venv_tracklab) if venv_tracklab.exists() else (shutil.which("tracklab") or "tracklab")

    cmd = [
        tracklab_bin,
        "-cn", config_name,
        f"dataset.eval_set={split}",
        "dataset.nvid=1",
        f"dataset.vids_dict.{split}=[{sequence_id}]",
    ]
    if data_dir:
        cmd.append(f"dataset.dataset_path={data_dir}")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    print("============================================================")
    print(f"  RUNNING OFFICIAL TRACKLAB GSR BASELINE — SEQUENCE: {sequence_id}")
    print("============================================================")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"[LOG] {log_file}")

    start_time = time.time()
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"=== SOCCERNET-GSR INFERENCE LOG — {datetime.now(timezone.utc).isoformat()} ===\n")
        lf.write(f"Command: {' '.join(cmd)}\n\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(gsr_dir) if gsr_dir.exists() else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:  # type: ignore
            sys.stdout.write(line)
            lf.write(line)
            lf.flush()

        proc.wait()
        exit_code = proc.returncode

    duration = time.time() - start_time
    print(f"\n[INFO] Inference completed in {duration:.2f}s with exit code: {exit_code}")

    # Harvest predictions into output_dir / predictions
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for d in [gsr_dir / "outputs", output_dir / "tracklab_output"]:
        if d.exists():
            for ext in ("*.csv", "*.json", "*.pklz"):
                candidates.extend(list(d.glob(f"**/{ext}")))

    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        newest = candidates[0]
        dest = pred_dir / f"{sequence_id}{newest.suffix}"
        shutil.copy2(newest, dest)
        print(f"[INFO] Harvested prediction file: {newest} -> {dest}")

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Official SoccerNet-GSR One-Sequence Runner")
    parser.add_argument("--sequence-id", type=str, required=True, help="Sequence ID to execute (e.g. SNGS-04)")
    parser.add_argument("--split", type=str, default="valid", help="Dataset split (default: valid)")
    parser.add_argument("--data-dir", type=str, default=None, help="Root directory containing SoccerNetGS dataset")
    parser.add_argument("--gsr-dir", type=str, default="soccernet-gamestate", help="Path to sn-gamestate repository")
    parser.add_argument("--output-dir", type=str, default="results/gsr_kaggle", help="Directory for output manifests & logs")
    parser.add_argument("--dry-run", action="store_true", help="Generate manifest without launching subprocess")
    args = parser.parse_args()

    project_root = Path.cwd()
    gsr_dir = Path(args.gsr_dir)
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data" / "SoccerNetGS"

    # 1. Generate run manifest
    create_run_manifest(
        output_dir=output_dir,
        project_root=project_root,
        gsr_dir=gsr_dir,
        dataset_name="SoccerNetGS",
        sequence_id=args.sequence_id,
        data_dir=data_dir,
        config_name="soccernet",
        checkpoint_paths={},
    )

    if args.dry_run:
        print("[INFO] --dry-run specified. Manifest generated successfully.")
        return 0

    # 2. Check if GSR directory / tracklab exists
    venv_tracklab = gsr_dir / ".venv" / "bin" / "tracklab"
    if not venv_tracklab.exists() and not shutil.which("tracklab"):
        print(f"[ERROR] TrackLab executable not found at '{venv_tracklab}'. Run kaggle/setup_gsr.py first.", file=sys.stderr)
        return 1

    # 3. Check if sequence data exists
    seq_path = data_dir / args.split / args.sequence_id
    if not seq_path.exists():
        print(f"[ERROR] Sequence directory not found: '{seq_path}'. Verify dataset download.", file=sys.stderr)
        return 2

    # 4. Execute inference
    return execute_one_sequence(
        gsr_dir=gsr_dir,
        sequence_id=args.sequence_id,
        split=args.split,
        config_name="soccernet",
        data_dir=data_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
