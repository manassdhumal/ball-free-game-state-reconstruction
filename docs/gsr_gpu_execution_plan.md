# Step 77 - GSR GPU Execution Plan

## Status

This is a setup and execution-readiness plan only. No GSR inference has been run, no dataset or checkpoint was downloaded, and no Step 74 or Step 75 result was rerun.

## Environment

Use a dedicated Linux GPU environment with Python 3.9.x. The checked-out package declares:

- `sn-gamestate==0.2.0`
- `tracklab==1.3.24`
- `torch==1.13.1`
- `soccernet==0.1.55`
- `mmcv==2.0.1` through OpenMIM
- `mmdet~=3.1.0`
- `mmocr==1.0.1`
- `openmim==0.3.9`
- `lightning==2.0.9`
- declared ReID packages and calibration plugin

Use the exact dependency declarations in `soccernet-gamestate/pyproject.toml`. TrackLab is external to the checkout and must be installed by the package environment.

### GPU requirements

Use an NVIDIA CUDA-capable GPU under Linux. The configuration starts with detector/OCR batch size 8, ReID/tracking batch size 64, and pitch/calibration batch size 1; lower these values if memory is insufficient. The repository does not establish a precise minimum VRAM requirement, so determine it with a bounded one-sequence test rather than promising a card size.

The GSR FAQ reports approximately 12 minutes per video on GPU for the full baseline and identifies jersey recognition as a slow stage. This supports a one-sequence first run.

## Storage

Keep raw data, labels, checkpoints, tracker states, and raw predictions in persistent storage outside Git:

```text
<persistent storage>/SoccerNetGS/        # raw validation data and labels
<persistent storage>/gsr-models/         # baseline model weights
<persistent storage>/gsr-runs/           # configs, tracker states, predictions, logs
<persistent storage>/canonical-gsr/       # validated conversion output
```

The TrackLab configuration expects an absolute `data_dir` with:

```text
<data_dir>/SoccerNetGS/
  train/
  valid/
  test/
  challenge/
```

Do not put dataset passwords, access tokens, or credentials in these paths' Git-tracked configuration.

## Bootstrap: Colab Route

The project already contains `scripts/setup_gsr_colab.sh`. It is reusable for a fresh Linux/Colab runtime and is designed to:

1. install or reuse `uv`;
2. create `/content/soccernet-gamestate/.venv` with Python 3.9;
3. install the checked-out GSR package from `pyproject.toml`;
4. pin setuptools;
5. install MMCV through the environment's `mim` command;
6. perform import/CUDA checks.

It does **not** mount Drive, download the dataset, download checkpoints, or run inference. Mount persistent storage and place the authorized dataset/checkpoints outside the repository before using TrackLab.

A representative bootstrap sequence is:

```bash
# In a fresh Linux/Colab runtime, after cloning both repositories.
!git clone https://github.com/SoccerNet/sn-gamestate /content/soccernet-gamestate
!git clone https://github.com/<team>/ball-free-game-state-reconstruction /content/capstone
!bash /content/capstone/scripts/setup_gsr_colab.sh
```

Replace the project repository URL with the team's authorized repository location. Do not place authentication material in the command, notebook, or repository.

For institutional HPC, use the same Python 3.9 and dependency pins in a module/conda/uv environment, with dataset and checkpoints on persistent storage.

## Environment Verification

Run after installation, before loading a model:

```bash
python --version
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
python -c "import tracklab; print('tracklab import ok')"
python -c "import sn_gamestate; print('sn_gamestate import ok')"
python -c "import trackeval; print('trackeval import ok')"
```

Record Python, PyTorch, CUDA runtime, GPU model, total/free GPU memory, TrackLab version, sn-gamestate commit, and package lock state. The verification must pass before any model-loading check.

## Model-Loading Check

After the environment and weights are installed, perform a minimal, bounded model-loading check using the baseline configuration. Confirm that the detector, ReID, calibration, OCR/jersey, team, and tracker components resolve their configured assets. Do not process a dataset during this check and do not substitute the example prediction ZIP for weights.

The checked-out configuration references:

- detector: `yolo_ultralytics` through the external TrackLab configuration;
- ReID: `sn_gamestate.reid.prtreid_api.PRTReId`, with `${model_dir}/reid/prtreid-soccernet-baseline.pth.tar`;
- tracker: `bpbreid_strong_sort` through the external TrackLab configuration;
- jersey recognition: MMOCR module;
- pitch/calibration: `nbjw_calib` by default, with calibration support assets under `${model_dir}/calibration`;
- team and team-side modules: KMeans embeddings and mean position;
- tracker state: saved to `states/${experiment_name}.pklz` unless disabled.

The detector, ReID, OCR, and tracker weights are not bundled in the current checkout. TrackLab/GSR documentation indicates that baseline weights may be downloaded automatically on first use; the exact downloaded files and checksums must be recorded after authorized setup.

## One-Sequence Execution

The baseline command from the checked-out GSR README is:

```bash
uv run tracklab -cn soccernet
```

The default configuration uses `eval_set: valid`, `nvid: 1`, `dataset_path: ${project_dir}/data/SoccerNetGS`, and an empty explicit validation-video list. Once actual validation metadata is available, select the observed sequence ID, not an invented identifier:

```bash
uv run tracklab -cn soccernet \
  dataset.eval_set=valid \
  dataset.nvid=1 \
  'dataset.vids_dict.valid=[<VERIFIED_SEQUENCE_ID>]'
```

Required values to verify before execution:

- `dataset.eval_set=valid`
- `dataset.nvid=1`
- `dataset.vids_dict.valid` contains one ID observed in downloaded validation metadata
- `data_dir` points to an absolute persistent dataset location
- `model_dir` points to persistent baseline weights
- `state.load_file=null` for a fresh baseline run
- default pipeline enabled, including detector, ReID, tracker, calibration, OCR, aggregation, team, and team-side modules
- `eval_tracking=true` only after matching labels are available

Do not run this command until the access checklist is complete.

## Intended Data Flow

```text
SoccerNet validation video/frames
        |
        v
TrackLab / sn-gamestate
        |
        v
detections
        |
        v
ReID / tracking
        |
        v
pitch calibration
        |
        v
team / jersey / role processing
        |
        v
GSR prediction
        |
        v
GS-HOTA evaluation
        |
        v
canonical player tracking
```

The future canonical conversion must preserve the existing player-only downstream contract. Ball data remains separate and is not passed to possession or tactical inference.

## Persistence and Provenance

For the one-sequence run, persist:

- source sequence ID, split, and video/frame metadata;
- dataset and label version;
- source video/frame checksums outside Git;
- GSR repository commit;
- TrackLab and package versions;
- resolved Hydra configuration;
- model checkpoint filenames and checksums;
- tracker state path;
- raw prediction ZIP/JSON and visualization path;
- GS-HOTA evaluator configuration and output;
- canonical conversion mapping and validation report.

A prediction is not ready for downstream use until its identities, teams, roles, coordinates, frame mapping, and provenance are validated.

## Success Criteria

Step 77 remains incomplete until all of the following are true:

1. Authorized access to one validation sequence exists.
2. Matching ground-truth labels exist.
3. A real GPU environment is available.
4. The GSR runtime installs successfully.
5. Required model weights are available.
6. One sequence loads successfully.
7. Baseline inference completes.
8. Prediction output is persisted.
9. Provenance is documented.
10. GS-HOTA evaluation runs or is demonstrably ready against matching labels.

No criterion is currently satisfied beyond source-code/configuration readiness.
