# Step 76 - SoccerNet-GSR Recovery Plan

## Scope and Status

Step 76 is an investigation and readiness audit for the missing upstream perception stage:

```text
Broadcast video -> SoccerNet-GSR / TrackLab perception -> real player tracks -> canonical tracking
```

The existing possession, tactical, degradation, mitigation, and Step 70-75 result pipelines are complete and are not modified by this audit. Objective 1, adapting SOTA perception components to produce per-frame player coordinates, remains **NOT COMPLETE**.

## Current GSR Checkout

- **Path:** `soccernet-gamestate/`
- **Remote:** `https://github.com/SoccerNet/sn-gamestate.git`
- **Branch:** `main`
- **Commit:** `1c958345067218297d221e45e1a6405f975f83e0`
- **Package metadata:** `sn-gamestate 0.2.0`
- **Declared TrackLab dependency:** `tracklab==1.3.24`
- **Declared Python range:** `>=3.9,<3.10`
- **Declared related dependencies:** PyTorch 1.13.1, MMDetection 3.1.x, MMOCR 1.0.1, MMCV 2.0.1 via OpenMIM, EasyOCR, ReID packages, and SoccerNet 0.1.55.

The checkout contains the GSR-specific modules and Hydra configurations, but TrackLab itself is not vendored in this repository.

## Asset Audit

| Artifact | Local path | Classification | Finding |
|---|---|---|---|
| Official example submission archive | `soccernet-gamestate/examples_predictions/SoccerNetGS-test.zip` | D. Official example/submission prediction | 69,319,587 bytes; 49 JSON members under `tracklab/`; contains submission-style prediction files, not local inference provenance |
| Calibration normalization arrays | `soccernet-gamestate/pretrained_models/calibration/mean.npy`, `std.npy` | G. Supporting calibration assets | 152 bytes each; not a complete perception model checkpoint |
| Calibration reference image | `soccernet-gamestate/pretrained_models/calibration/Radar.png` | G. Supporting asset | 2,155 bytes; not a prediction or source video |
| GSR source code/configuration | `soccernet-gamestate/sn_gamestate/` | G. Code/configuration | Includes GSR-specific modules, but not the external TrackLab runtime or downloaded dataset |
| Metrica tracking/events | `data/raw/metrica/` | B. Existing non-GSR source data | Project benchmark input only; not SoccerNet-GSR video or annotation data |

No local SoccerNetGS dataset directory, GSR source video, GSR annotation archive, locally generated prediction directory, tracker state, or detector/ReID/MMOCR checkpoint was found. The bundled ZIP is explicitly described by `ChallengeRules.md` as an example submission for the test set. It is therefore not evidence of a local inference run and cannot substitute for a provenance-verifiable prediction generated from accessible source video.

## Data Access Findings

### Accessible now

- Publicly checked-out GSR code, configuration, documentation, and dependency declarations.
- The official example submission ZIP, which is prediction-format reference material only.
- Small calibration support arrays and image.
- Existing Metrica data used by the downstream benchmark.

### Not accessible locally

- SoccerNet-GSR `train`, `valid`, `test`, or `challenge` dataset directories.
- Source videos required for genuine perception inference.
- Ground-truth `Labels-GameState.json` files.
- A locally generated GSR prediction or tracker state.
- Complete baseline model weights.

The checked-out documentation describes dataset retrieval through the `SoccerNetDownloader` or TrackLab's automatic download, with a target structure of `data/SoccerNetGS/{train,valid,test,challenge}`. The local documentation does not establish that an account, NDA, password, or token is required, so this audit does not invent an access requirement. Any current access request must be completed through the official SoccerNet channel and must not place credentials in this repository.

## GPU and Environment Audit

### Local Windows environment

- `nvidia-smi`: unavailable as a command.
- `nvcc`: unavailable as a command.
- Project `.venv`: Python 3.12.1, NumPy 2.5.2, pandas 3.0.5; PyTorch and TrackLab are not installed.
- `soccernet-gamestate/.venv`: Python 3.9.13 scaffold; `sn-gamestate`, TrackLab, PyTorch, TrackEval, MMCV, MMOCR, and MMDetection are not installed.
- No CUDA environment variables or Colab/Kaggle runtime indicators were present.
- No local NVIDIA GPU was verifiable from this Windows environment.

### Recommended route

1. **Institutional GPU/HPC:** preferred for repeatable long inference and persistent storage.
2. **Colab GPU:** practical first route for one validation sequence, using the existing `scripts/setup_gsr_colab.sh` and Google Drive for data, weights, and outputs.
3. **Kaggle:** possible alternative if the required dependencies and dataset access can be configured.
4. **Cloud GPU:** viable but requires explicit storage and cost controls.

The best immediate practical route is a dedicated Linux GPU environment, preferably institutional HPC; Colab GPU is the fastest low-friction fallback for a one-sequence smoke-sized readiness test. No GPU inference was started in Step 76.

## Baseline Inference Plan

The official baseline configuration is `soccernet-gamestate/sn_gamestate/configs/soccernet.yaml`. Its default pipeline is:

```text
bbox_detector -> reid -> track -> pitch -> calibration
-> jersey_number_detect -> tracklet_agg -> team -> team_side
```

The declared dataset defaults are one video (`nvid: 1`) from the `valid` split, with `dataset_path: ${project_dir}/data/SoccerNetGS`. The baseline command documented by the checkout is:

```bash
uv run tracklab -cn soccernet
```

For a deliberately bounded first run after access and environment setup, select one known validation sequence and keep `nvid=1`:

```bash
uv run tracklab -cn soccernet dataset.eval_set=valid dataset.nvid=1 dataset.vids_dict.valid="['SNGS-04']"
```

`SNGS-04` is an example identifier from the repository documentation; replace it only with a sequence confirmed to exist in the downloaded validation split. Do not run this command until dataset access, model weights, and the Linux GPU environment are ready.

Expected outputs are under the Hydra run directory `outputs/sn-gamestate/<date>/<time>/`, including a tracker state at the configured `states/sn-gamestate.pklz` location relative to the run and a visualization video under the run's visualization output. A successful future test must persist the resolved config, source sequence ID, package commits, model-weight identifiers, prediction files, and validation checks.

## Adaptation Plan

Adaptation is intentionally not implemented in Step 76.

| Area | Baseline | Candidate adaptation | Reason | Expected effect | Evaluation |
|---|---|---|---|---|---|
| Detector | `yolo_ultralytics` with configured batch size | Tune confidence/NMS and batch size after profiling one sequence | Broadcast player scale and occlusion can change recall/precision and throughput | Fewer missed players or false detections without changing downstream contract | Detection recall/precision where labels permit; GS-HOTA after full output |
| Re-identification | `prtreid` with SoccerNet baseline weights | Compare calibrated confidence/association thresholds or a documented compatible ReID checkpoint | Identity continuity affects tracking and jersey/team aggregation | Fewer identity switches and more stable IDs | AssA/ID components and GS-HOTA |
| Calibration | `nbjw_calib` pitch/calibration modules | Compare calibration backend only after baseline output is validated | Pitch projection error directly changes top-view coordinates | Better localization in pitch space | Localization similarity and GS-HOTA |
| Jersey/team modules | MMOCR, voting aggregation, KMeans team, mean-position side | Tune thresholds or replace one module with a documented compatible alternative | Attribute errors are strictly penalized by GS-HOTA | Better role, jersey, and team attributes | Attribute agreement and GS-HOTA |

A baseline run must be completed and archived before any adaptation is called a comparison. No adaptation result exists yet.

## GS-HOTA Evaluation Plan

The GSR challenge submission format is a ZIP containing one JSON per video. Each prediction contains a top-level `predictions` list with `category_id`, `image_id`, `track_id`, `supercategory`, `confidence`, `attributes` (`role`, `jersey`, `team`), and `bbox_pitch` with `x_bottom_middle` and `y_bottom_middle` in meters. Ground truth is supplied in the dataset's `Labels-GameState.json` files.

The checked-out `sn_gamestate/configs/soccernet.yaml` selects `eval: gs_hota`, but its local `gs_hota.yaml` delegates to TrackLab's TrackEval evaluator and requests `CLEAR`, `HOTA`, and `Identity`. The exact evaluator behavior must be verified in the installed TrackLab/TrackEval version after environment installation. The official repository notes that TrackLab 1.3.24 addresses a broken GS-HOTA evaluation path.

Future evaluation sequence:

1. Install the pinned Linux GPU environment.
2. Download and verify the dataset version, including `Labels-GameState.json` version >= 1.3.
3. Run one validation sequence and preserve the prediction JSON and resolved config.
4. Run the configured evaluation path or official evaluator against the matching validation labels.
5. Record GS-HOTA and component metrics only from actual output.

No GS-HOTA result has been executed or claimed in this repository.

## Canonical Output Contract

The future conversion from genuine GSR predictions to the existing downstream schema must preserve the current contract:

| Canonical field | Future GSR source | Units/semantics |
|---|---|---|
| `match_id` | SoccerNet sequence ID plus dataset split | Stable string identifier |
| `frame` | `image_id` or source frame index after documented mapping | Integer frame index |
| `timestamp` | Source video FPS and frame index | Seconds from sequence start |
| `player_id` | Stable GSR `track_id`, optionally combined with sequence ID | String; do not use jersey as identity |
| `team` | `attributes.team` | Explicit mapping to canonical team labels; document left/right versus home/away carefully |
| `x`, `y` | `bbox_pitch.x_bottom_middle`, `bbox_pitch.y_bottom_middle` | Meters in SoccerNet pitch coordinates; convert only with documented pitch/orientation transform |
| `confidence` | Prediction `confidence` | Numeric confidence in the source's declared range |
| `visible` | Presence of a valid prediction in the frame | Boolean; distinguish absent detection from source extrapolation |

Ball data remains separate and is not added to the player table. The future parser must document source field, units, coordinate system, identity mapping, team mapping, confidence semantics, and orientation transformation before any downstream comparison.

## Storage Plan

```text
RAW source video/dataset:       Google Drive or institutional storage
MODEL checkpoints:              Google Drive or institutional storage
GSR raw predictions/states:     Google Drive or institutional storage
Canonical processed output:     data/processed/ or persistent Drive location
Git:                            code, configuration, documentation, tests only
```

The existing `.gitignore` policy should remain unchanged. Raw data, intermediate data, model checkpoints, generated results, temporary smoke outputs, and legacy artifacts must not be committed.

## Exact Blockers

1. No accessible local SoccerNet-GSR dataset or source video.
2. No local GSR annotations for validation.
3. No complete detector, ReID, jersey, calibration, or TrackLab model-weight set.
4. No installed TrackLab/GSR runtime in either local environment.
5. No verifiable local NVIDIA GPU/CUDA runtime.
6. No provenance-verifiable local GSR inference output.

## Next Execution Steps

1. Obtain authorized access to the SoccerNet-GSR dataset and download one validation sequence plus matching labels to persistent GPU storage.
2. Build the pinned Python 3.9/Linux GPU environment and download the baseline weights.
3. Run exactly one validation sequence with the bounded command above.
4. Archive the output provenance and validate player coordinates/identities before designing any adaptation.
5. Convert only validated output to the existing canonical schema.
6. Evaluate GS-HOTA and then connect the resulting canonical player table to the unchanged downstream pipeline.
