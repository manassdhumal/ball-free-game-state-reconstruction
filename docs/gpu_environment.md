# GPU Environment for SoccerNet-GSR

## Recommended Supported Environment

The checked-out `soccernet-gamestate` package declares Python `>=3.9,<3.10` and pins the core stack around:

- `sn-gamestate 0.2.0`
- `tracklab==1.3.24`
- `torch==1.13.1`
- `soccernet==0.1.55`
- `mmcv==2.0.1` installed through OpenMIM
- `mmocr==1.0.1`
- `mmdet~=3.1.0`
- `lightning==2.0.9`
- ReID packages declared in `pyproject.toml`

Use Linux with a CUDA-capable NVIDIA GPU for the baseline. The repository's `scripts/setup_gsr_colab.sh` is designed for a fresh Colab Linux runtime and creates a dedicated Python 3.9 environment with `uv`.

## Local Verification Result

The current Windows machine is not a usable GSR inference environment:

- `nvidia-smi` is not available.
- `nvcc` is not available.
- No NVIDIA GPU was verifiable through the command-line audit.
- The project `.venv` uses Python 3.12.1 and has no PyTorch or TrackLab.
- `soccernet-gamestate/.venv` uses Python 3.9.13 but is only a scaffold; PyTorch, TrackLab, sn-gamestate, TrackEval, MMCV, MMOCR, and MMDetection are not installed.
- No model weights beyond small calibration support arrays are bundled.

No model was loaded and no inference was started.

## Best Practical Execution Route

1. **Institutional GPU/HPC:** preferred for persistent datasets, model checkpoints, and reproducible batch jobs.
2. **Colab GPU:** recommended first practical route for one validation sequence, using Google Drive for persistent storage.
3. **Kaggle:** fallback when a compatible CUDA image and authorized dataset access are available.
4. **Cloud GPU:** viable when cost, disk, and data-retention controls are explicit.

The immediate recommendation is institutional Linux GPU/HPC where available; otherwise use Colab GPU for the first one-sequence validation run.

## Colab Setup

The existing setup script does not download the dataset or execute inference. It creates the GSR environment and verifies core imports after installation:

```bash
bash /content/capstone/scripts/setup_gsr_colab.sh
```

The script expects the GSR checkout at `/content/soccernet-gamestate` and the project checkout at `/content/capstone`. It uses `/content/soccernet-gamestate/.venv` with Python 3.9 and does not store secrets in the repository.

After the environment is built, the official baseline command is:

```bash
MPLBACKEND=Agg /content/soccernet-gamestate/.venv/bin/tracklab -cn soccernet
```

For the first bounded test, set the validation split and one confirmed sequence using Hydra overrides:

```bash
MPLBACKEND=Agg /content/soccernet-gamestate/.venv/bin/tracklab -cn soccernet \
  dataset.eval_set=valid dataset.nvid=1 \
  'dataset.vids_dict.valid=[SNGS-04]'
```

Replace `SNGS-04` only after confirming that the sequence exists in the downloaded validation directory. Do not begin until the dataset, labels, and weights are present.

## Dataset and Persistent Storage

The baseline expects an absolute `data_dir` containing:

```text
data/SoccerNetGS/
  train/
  valid/
  test/
  challenge/
```

Use a persistent Drive or institutional path rather than ephemeral notebook storage:

```text
Google Drive / institutional storage/
  SoccerNetGS/                 # raw dataset and labels
  gsr-models/                  # downloaded checkpoints
  gsr-runs/                    # raw prediction and tracker-state outputs
  canonical-gsr/               # validated converted tables
```

Keep these locations outside Git. The project repository should contain only code, configuration, documentation, and tests.

## VRAM Considerations

The baseline configuration uses batch sizes of 8 for detection and jersey recognition, 64 for ReID/tracking, and 1 for pitch/calibration. These are starting values, not guaranteed requirements. Reduce module-specific batch sizes if CUDA out-of-memory errors occur; increase them only after observing GPU utilization and memory headroom. The repository does not contain a verified VRAM benchmark for this exact environment, so do not promise a minimum card size without testing the chosen model stack.

The GSR FAQ reports roughly 12 minutes per video on GPU for the full baseline, with jersey-number recognition among the slower stages. This makes one-sequence validation the appropriate first execution scope.

## Verification Commands

Run these after environment creation and dependency installation:

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
nvidia-smi
python -c "import tracklab; print('tracklab import ok')"
python -c "import sn_gamestate; print('sn_gamestate import ok')"
python -c "import trackeval; print('trackeval import ok')"
```

A readiness check must report Python, PyTorch, CUDA runtime, GPU model and memory, TrackLab version, sn-gamestate version, and model-weight paths. Import success alone is not evidence of valid model loading.

## Troubleshooting

- **Python mismatch:** use a dedicated Python 3.9 environment; the project Python 3.12 environment is not compatible with the declared GSR package range.
- **TrackLab command missing:** install the GSR package in the active environment with `uv pip install -e .` or use the provided Colab setup, then verify the environment's `tracklab` executable.
- **MMCV build/install failure:** install PyTorch first and use the environment's `mim install mmcv==2.0.1` path as documented by the checkout.
- **CUDA unavailable:** confirm the runtime is an NVIDIA GPU instance, then check driver compatibility with `nvidia-smi`; do not attempt full inference on CPU.
- **Out-of-memory:** lower detector, ReID, tracker, and OCR batch sizes in the Hydra configuration.
- **Missing data:** confirm `data_dir` is absolute and contains `SoccerNetGS/valid` plus versioned labels.
- **Missing weights:** download the official baseline states/checkpoints to persistent storage and record their source/version; do not substitute the example prediction ZIP.
- **GS-HOTA mismatch:** use the pinned TrackLab 1.3.24-compatible environment and verify the evaluator against matching labels before recording any metric.
- **Output provenance unclear:** stop and preserve the resolved Hydra configuration, sequence ID, commit hashes, package versions, and weight identifiers before downstream conversion.

## Security and Reproducibility

Do not store dataset passwords, account details, access tokens, PATs, or credentials in notebooks, logs, configuration files, or Git. Use the hosting platform's secret mechanism or enter credentials interactively outside the repository. No secrets are required or recorded by this guide.
