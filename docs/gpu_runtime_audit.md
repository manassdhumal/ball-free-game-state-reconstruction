# GPU Runtime & Environment Audit

## 1. Executive Summary

This document records the hardware, CUDA, and environment audit for the **SoccerNet Game State Reconstruction (GSR)** perception baseline on branch `gpu-gsr-kaggle`.

## 2. Hardware Environments Comparison

| Attribute | Local Development Environment | Kaggle / Colab GPU Environment |
| :--- | :--- | :--- |
| **Operating System** | macOS (Darwin Kernel, arm64) | Linux (Ubuntu x86_64) |
| **GPU Hardware** | None (Apple Silicon Unified Memory) | NVIDIA Tesla T4 / P100 / A100 |
| **CUDA Driver** | N/A | CUDA >= 11.7 (Driver >= 515) |
| **PyTorch Stack** | `torch` CPU only (Python 3.12) | `torch==1.13.1+cu117` (Python 3.9) |
| **GSR Perception Execution** | **BLOCKED** (No CUDA / Linux runtime) | **SUPPORTED** (via `kaggle/` framework) |
| **Downstream Ball-Free Modeling** | **FULL SUPPORT** (All 81 steps passing) | **SUPPORTED** |

## 3. Dependency Compatibility Matrix

SoccerNet-GSR (`sn-gamestate` version `0.2.0`) requires a tightly coupled computer vision stack:

- **Python:** `>=3.9,<3.10`
- **PyTorch:** `1.13.1+cu117`
- **MMCV:** `2.0.1` (installed via `openmim`)
- **MMDetection:** `~=3.1.0`
- **MMOCR:** `1.0.1`
- **TrackLab:** `1.3.24`
- **Setuptools:** Pinned to `80.10.2` to preserve `pkg_resources` compatibility.

## 4. Execution Workflow

The repository provides automated scripts under `kaggle/`:
- `kaggle/setup_gsr.py`: Verifies GPU attachment, installs Python 3.9 venv, and builds dependencies.
- `kaggle/run_gsr.py`: Enforces the **One-Sequence Policy**, capturing stdout/stderr and telemetry.
- `kaggle/export_gsr.py`: Converts raw TrackLab predictions into Canonical Tracking Schema v0.2.0 with SHA256 manifests.
- `kaggle/gsr_validation.ipynb`: Rerunnable 12-stage notebook.
