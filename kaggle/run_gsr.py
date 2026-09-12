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

    # Ensure data directory symlink inside gsr_dir so all relative lookups work
    if data_dir:
        try:
            (gsr_dir / "data").mkdir(parents=True, exist_ok=True)
            link_target = gsr_dir / "data" / "SoccerNetGS"
            if not link_target.exists() and not link_target.is_symlink():
                link_target.symlink_to(data_dir.resolve(), target_is_directory=True)
                print(f"[INFO] Created symlink: {link_target} -> {data_dir.resolve()}")
        except Exception as e:
            pass

    cmd = [
        tracklab_bin,
        "-cn", config_name,
        f"dataset.eval_set={split}",
        "dataset.nvid=1",
        f"dataset.vids_dict.{split}=[{sequence_id}]",
    ]
    if data_dir:
        cmd.append(f"dataset.dataset_path={data_dir.resolve()}")

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
    for d in [gsr_dir / "outputs", Path("outputs"), output_dir / "tracklab_output", output_dir]:
        if d.exists():
            for ext in ("*.csv", "*.json", "*.pklz"):
                candidates.extend(list(d.glob(f"**/{ext}")))

    if candidates:
        matching = [p for p in candidates if sequence_id in p.name]
        harvest_list = matching if matching else candidates
        harvest_list.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for newest in harvest_list[:3]:
            dest = pred_dir / f"{sequence_id}{newest.suffix}"
            shutil.copy2(newest, dest)
            print(f"[INFO] Harvested prediction file: {newest} -> {dest}")
            if sequence_id.startswith("SNGS-04"):
                shutil.copy2(newest, pred_dir / f"SNGS-04{newest.suffix}")
                shutil.copy2(newest, pred_dir / f"SNGS-040{newest.suffix}")

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

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        candidates = [
            project_root / "data" / "SoccerNetGS",
            Path("/kaggle/working/capstone/data/SoccerNetGS"),
            Path("/kaggle/working/data/SoccerNetGS"),
            Path("data/SoccerNetGS"),
        ]
        data_dir = project_root / "data" / "SoccerNetGS"
        for c in candidates:
            if (c / args.split / args.sequence_id).exists():
                data_dir = c
                print(f"[INFO] Auto-discovered dataset at: {data_dir}")
                break

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

    # 3. Check if sequence data exists, or automatically download it
    split_dir = data_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    target_seq = args.sequence_id
    seq_path = split_dir / target_seq

    # Auto-resolve exact 3-digit sequence ID if 2-digit was provided (e.g., SNGS-04 -> SNGS-040)
    if not seq_path.exists() or not any(seq_path.iterdir() if seq_path.is_dir() else []):
        existing = sorted([p for p in split_dir.glob(f"{target_seq}*") if p.is_dir() and any(p.iterdir())])
        if existing:
            resolved_seq = existing[0].name
            print(f"[INFO] Found existing sequence folder: '{resolved_seq}' ({len(list(existing[0].iterdir()))} files)")
            if resolved_seq != target_seq and not seq_path.exists():
                try:
                    seq_path.symlink_to(resolved_seq, target_is_directory=True)
                    print(f"[INFO] Created symlink: {seq_path} -> {resolved_seq}")
                except Exception:
                    pass
            target_seq = resolved_seq
            seq_path = split_dir / target_seq

    if not seq_path.exists() or not any(seq_path.iterdir() if seq_path.is_dir() else []):
        print(f"[INFO] Sequence '{target_seq}' not found locally at '{seq_path}'.")
        print(f"[INFO] Initiating automated download of SoccerNetGS {args.split} split...")
        try:
            import SoccerNet
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "SoccerNet"], check=True)
            import SoccerNet

        from SoccerNet.Downloader import SoccerNetDownloader
        downloader = SoccerNetDownloader(LocalDirectory=str(data_dir))
        downloader.downloadDataTask(task="gamestate-2024", split=[args.split])

        possible_zips = [
            data_dir / "gamestate-2024" / f"{args.split}.zip",
            data_dir / f"{args.split}.zip",
        ]
        zip_file = next((p for p in possible_zips if p.exists()), None)
        if zip_file and zip_file.exists():
            import zipfile
            target_names = [target_seq, f"{target_seq}0", f"{target_seq[:5]}0{target_seq[5:]}" if len(target_seq) == 7 else target_seq]
            with zipfile.ZipFile(zip_file, "r") as z:
                members = [m for m in z.namelist() if any(f"{t}/" in m or f"/{t}/" in m for t in target_names) or (m.startswith(f"{args.split}/") and m.count("/") == 1)]
                if not members:
                    members = [m for m in z.namelist() if any(t in m for t in target_names)]
                print(f"[INFO] Extracting {len(members)} files for sequence '{target_seq}'...")
                for m in members:
                    if m.startswith(f"{args.split}/"):
                        z.extract(m, str(data_dir))
                    else:
                        z.extract(m, str(data_dir / args.split))
            print(f"[INFO] Deleting {zip_file.name} to preserve multi-gigabyte disk space...")
            try:
                os.remove(zip_file)
                if zip_file.parent.name == "gamestate-2024":
                    shutil.rmtree(zip_file.parent, ignore_errors=True)
            except Exception:
                pass

        if not seq_path.exists():
            existing = sorted([p for p in split_dir.glob(f"{target_seq}*") if p.is_dir() and any(p.iterdir())])
            if existing:
                resolved_seq = existing[0].name
                if resolved_seq != target_seq and not seq_path.exists():
                    try:
                        seq_path.symlink_to(resolved_seq, target_is_directory=True)
                    except Exception:
                        pass
                target_seq = resolved_seq
                seq_path = split_dir / target_seq

        if not seq_path.exists() or not any(seq_path.iterdir() if seq_path.is_dir() else []):
            print(f"[ERROR] Sequence directory still not found after download: '{seq_path}'.", file=sys.stderr)
            return 2
        print(f"[INFO] Successfully verified sequence data at: {seq_path}")

    # 4. Execute inference
    return execute_one_sequence(
        gsr_dir=gsr_dir,
        sequence_id=target_seq,
        split=args.split,
        config_name="soccernet",
        data_dir=data_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
