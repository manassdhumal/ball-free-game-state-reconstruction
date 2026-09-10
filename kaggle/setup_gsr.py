#!/usr/bin/env python3
"""
Kaggle / Colab GPU Setup and Verification Script for SoccerNet-GSR (sn-gamestate).

Purpose:
  1. Verifies attached NVIDIA GPU hardware and CUDA environment.
  2. Bootstraps a dedicated Python 3.9 virtual environment via `uv`.
  3. Clones and installs the pinned SoccerNet-GSR (sn-gamestate) repository.
  4. Resolves and installs pinned dependencies (PyTorch, TrackLab, MMCV, MMDetection, MMOCR).
  5. Generates machine-readable environment and package inventories.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """Run a shell command and capture stdout/stderr."""
    print(f"[EXEC] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"[STDERR] {result.stderr.strip()}", file=sys.stderr)
    return result


def audit_hardware(output_dir: Path) -> Dict[str, Any]:
    """Inspect local hardware, OS, and CUDA availability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env_info: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "gpu": {
            "available": False,
            "count": 0,
            "devices": [],
            "cuda_version": None,
            "driver_version": None,
            "nvidia_smi_output": None,
        },
        "torch": {
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device_name": None,
        },
    }

    # 1. Check nvidia-smi
    smi_path = shutil.which("nvidia-smi")
    if smi_path:
        smi_proc = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
        if smi_proc.returncode == 0:
            lines = [line.strip() for line in smi_proc.stdout.strip().split("\n") if line.strip()]
            env_info["gpu"]["available"] = len(lines) > 0
            env_info["gpu"]["count"] = len(lines)
            for idx, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                name = parts[0] if len(parts) > 0 else "Unknown"
                vram_mb = float(parts[1]) if len(parts) > 1 and parts[1].replace(".", "").isdigit() else None
                driver = parts[2] if len(parts) > 2 else None
                env_info["gpu"]["devices"].append({
                    "index": idx,
                    "name": name,
                    "vram_mb": vram_mb,
                    "driver_version": driver,
                })
                if driver:
                    env_info["gpu"]["driver_version"] = driver

        smi_full = _run_cmd(["nvidia-smi"])
        if smi_full.returncode == 0:
            env_info["gpu"]["nvidia_smi_output"] = smi_full.stdout

    # 2. Check PyTorch if available in current python
    try:
        import torch  # type: ignore
        env_info["torch"]["version"] = torch.__version__
        env_info["torch"]["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env_info["torch"]["cuda_version"] = torch.version.cuda
            env_info["torch"]["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    env_json_path = output_dir / "environment.json"
    with open(env_json_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)
    print(f"[INFO] Hardware audit saved to: {env_json_path}")
    return env_info


def bootstrap_gsr_env(
    workspace_root: Path,
    gsr_repo_dir: Path,
    python_version: str = "3.9",
    gsr_git_url: str = "https://github.com/SoccerNet/sn-gamestate.git",
    gsr_git_ref: str = "main",
) -> Path:
    """Clones SoccerNet-GSR and installs pinned dependencies inside an isolated venv."""
    workspace_root.mkdir(parents=True, exist_ok=True)
    venv_dir = gsr_repo_dir / ".venv"
    venv_python = venv_dir / "bin" / "python"

    # 1. Clone repository if not present
    if not gsr_repo_dir.exists():
        print(f"[INFO] Cloning SoccerNet-GSR from {gsr_git_url} (ref: {gsr_git_ref})...")
        clone_proc = _run_cmd(["git", "clone", gsr_git_url, str(gsr_repo_dir)])
        if clone_proc.returncode != 0:
            raise RuntimeError(f"Failed to clone SoccerNet-GSR: {clone_proc.stderr}")
        if gsr_git_ref != "main":
            _run_cmd(["git", "checkout", gsr_git_ref], cwd=gsr_repo_dir)

    # 2. Check uv package manager
    uv_bin = shutil.which("uv")
    if not uv_bin:
        print("[INFO] Installing uv package manager...")
        install_uv = _run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "uv"])
        if install_uv.returncode != 0:
            raise RuntimeError(f"Failed to install uv: {install_uv.stderr}")
        uv_bin = shutil.which("uv") or "uv"

    # 3. Create isolated virtual environment with Python 3.9
    if not venv_python.exists():
        print(f"[INFO] Creating virtualenv with Python {python_version} at {venv_dir}...")
        venv_proc = _run_cmd([uv_bin, "venv", "--python", python_version, str(venv_dir)])
        if venv_proc.returncode != 0:
            raise RuntimeError(f"Failed to create venv: {venv_proc.stderr}")

    # 4. Install PyTorch with CUDA 11.7
    print("[INFO] Installing PyTorch 1.13.1 + cu117 inside GSR venv...")
    torch_proc = _run_cmd([
        uv_bin, "pip", "install",
        "--python", str(venv_python),
        "torch==1.13.1+cu117", "torchvision==0.14.1+cu117",
        "--extra-index-url", "https://download.pytorch.org/whl/cu117",
    ])
    if torch_proc.returncode != 0:
        print(f"[WARN] CUDA wheel install failed: {torch_proc.stderr}. Attempting standard torch wheel...")
        _run_cmd([uv_bin, "pip", "install", "--python", str(venv_python), "torch==1.13.1", "torchvision==0.14.1"])

    # 5. Install sn-gamestate and pinned requirements
    print("[INFO] Installing sn-gamestate from repository...")
    _run_cmd([uv_bin, "pip", "install", "--python", str(venv_python), "-e", str(gsr_repo_dir)])

    # 6. Pin setuptools and install openmim / mmcv
    print("[INFO] Installing OpenMIM, MMCV, MMDet, MMOCR...")
    _run_cmd([uv_bin, "pip", "install", "--python", str(venv_python), "setuptools==80.10.2", "openmim==0.3.9"])
    
    mim_bin = venv_dir / "bin" / "mim"
    if mim_bin.exists():
        _run_cmd([str(mim_bin), "install", "mmcv==2.0.1"])
        _run_cmd([str(mim_bin), "install", "mmdet~=3.1.0"])
        _run_cmd([str(mim_bin), "install", "mmocr==1.0.1"])

    return venv_python


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggle / Colab GPU Setup for SoccerNet-GSR")
    parser.add_argument("--output-dir", type=str, default="results/gsr_kaggle", help="Directory for environment reports")
    parser.add_argument("--gsr-dir", type=str, default="soccernet-gamestate", help="Path for GSR repository checkout")
    parser.add_argument("--skip-install", action="store_true", help="Only audit environment without installing packages")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    gsr_dir = Path(args.gsr_dir)

    print("============================================================")
    print("  SOCCERNET-GSR KAGGLE GPU ENVIRONMENT SETUP & AUDIT")
    print("============================================================")

    # 1. Audit Hardware
    env_info = audit_hardware(out_dir)
    gpu_available = env_info["gpu"]["available"]
    torch_cuda = env_info["torch"]["cuda_available"]

    print(f"\n[AUDIT] System OS     : {env_info['os']['system']} ({env_info['os']['machine']})")
    print(f"[AUDIT] Python Version: {env_info['python']['version']}")
    print(f"[AUDIT] GPU Attached  : {gpu_available} (Count: {env_info['gpu']['count']})")
    if env_info["gpu"]["devices"]:
        for dev in env_info["gpu"]["devices"]:
            print(f"        Device {dev['index']}: {dev['name']} ({dev['vram_mb']} MB VRAM)")
    print(f"[AUDIT] CUDA in PyTorch: {torch_cuda}")

    if args.skip_install:
        print("\n[INFO] --skip-install flag set. Finished audit.")
        return 0

    if not gpu_available and not torch_cuda:
        print("\n[WARNING] No active NVIDIA CUDA GPU detected in current execution environment.")
        print("          Full deep learning inference requires a Linux GPU environment (Kaggle/Colab).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
