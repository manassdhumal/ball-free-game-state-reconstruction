"""Step 80 broadcast-noise analysis and mitigation ablation.

This module composes the existing degradation, bounded interpolation, and
ball-free benchmark functions.  It adds an auditable candidate
camera-cut/discontinuity signal and a causal constant-velocity alpha-beta
filter.  Candidate cuts are evidence signals, not ground truth labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from src.noise.interpolation import interpolate_trajectory
from src.noise.smoothing import smooth_trajectory
from src.validation.robustness_benchmark import (
    _json_safe,
    _repo_root,
    load_config,
    load_metrica_subset,
    possession_change_metrics,
    run_inference_condition,
    tactical_stability_metrics,
    tracking_quality_metrics,
)
from src.noise.degradation import degrade_tracking


METHODS = (
    "none",
    "interpolation",
    "moving_average",
    "savgol",
    "kalman",
    "interpolation_moving_average",
    "interpolation_savgol",
    "interpolation_kalman",
)


def _required_columns(df: pd.DataFrame) -> None:
    required = {"frame", "team", "player_id", "x", "y", "visible"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tracking is missing required columns: {sorted(missing)}")


def detect_candidate_cuts(
    tracking: pd.DataFrame,
    displacement_threshold: float = 0.05,
    min_support_fraction: float = 0.5,
    count_change_fraction: float = 0.4,
) -> pd.DataFrame:
    """Return frame-level evidence for candidate broadcast discontinuities.

    A boundary is supported by the fraction of tracks with adjacent-frame
    displacements above ``displacement_threshold``.  A visible-count change
    is an additional signal.  The first frame has no predecessor and is not a
    candidate.  Isolated player anomalies therefore do not qualify when the
    support threshold is set above one track.
    """
    _required_columns(tracking)
    frames = sorted(int(value) for value in tracking["frame"].unique())
    rows = []
    by_frame = {frame: group for frame, group in tracking.groupby("frame", sort=False)}
    for frame in frames:
        previous_frame = frame - 1
        current = by_frame.get(frame, tracking.iloc[0:0])
        previous = by_frame.get(previous_frame, tracking.iloc[0:0])
        current_visible = current[current["visible"].astype(bool)]
        previous_visible = previous[previous["visible"].astype(bool)]
        prev_count = len(previous_visible)
        curr_count = len(current_visible)
        count_change = abs(curr_count - prev_count) / max(prev_count, 1)
        joined = previous_visible[["team", "player_id", "x", "y"]].merge(
            current_visible[["team", "player_id", "x", "y"]],
            on=["team", "player_id"], suffixes=("_previous", "_current"),
        )
        if joined.empty:
            high_motion_fraction = 0.0
            mean_displacement = 0.0
        else:
            displacement = np.hypot(
                joined["x_current"] - joined["x_previous"],
                joined["y_current"] - joined["y_previous"],
            )
            high_motion_fraction = float((displacement > displacement_threshold).mean())
            mean_displacement = float(displacement.mean())
        support = high_motion_fraction >= min_support_fraction
        count_support = count_change >= count_change_fraction
        cut_score = float(0.65 * high_motion_fraction + 0.35 * min(count_change, 1.0))
        rows.append({
            "frame": frame,
            "frame_gap": int(frame - previous_frame) if previous_frame in by_frame else None,
            "candidate_cut": bool(support or (count_support and high_motion_fraction >= min_support_fraction / 2)),
            "cut_score": cut_score,
            "visible_count_previous": prev_count,
            "visible_count_current": curr_count,
            "visible_count_change_fraction": count_change,
            "matched_track_count": int(len(joined)),
            "high_motion_fraction": high_motion_fraction,
            "mean_displacement": mean_displacement,
        })
    return pd.DataFrame(rows)


def _bounded_gap_metrics(tracking: pd.DataFrame) -> Dict[str, Any]:
    gaps = []
    for _, group in tracking.groupby(["team", "player_id"], sort=True):
        visible = group.sort_values("frame")["visible"].astype(bool).to_numpy()
        index = 0
        while index < len(visible):
            if visible[index]:
                index += 1
                continue
            start = index
            while index < len(visible) and not visible[index]:
                index += 1
            gaps.append(index - start)
    return {
        "unresolved_gap_count": int(len(gaps)),
        "mean_gap_length": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap_length": int(max(gaps)) if gaps else 0,
    }


def step80_tracking_metrics(tracking: pd.DataFrame, cut_signal: pd.DataFrame, expected_players: Optional[int] = None) -> Dict[str, Any]:
    """Calculate reproducible tracking metrics used by the Step 80 table."""
    metrics = tracking_quality_metrics(tracking, expected_players=expected_players, large_motion_threshold=0.05)
    metrics.update(_bounded_gap_metrics(tracking))
    velocities = []
    accelerations = []
    continuity_steps = 0
    for _, group in tracking.groupby(["team", "player_id"], sort=True):
        group = group.sort_values("frame")
        valid = group[group["visible"].astype(bool)].copy()
        valid = valid[valid["x"].notna() & valid["y"].notna()]
        if len(valid) < 2:
            continue
        frame_delta = valid["frame"].diff().to_numpy(dtype=float)
        displacement = np.hypot(valid["x"].diff(), valid["y"].diff()).to_numpy(dtype=float)
        adjacent = frame_delta == 1
        speed = displacement[adjacent] / 0.04
        velocities.extend(speed.tolist())
        continuity_steps += int(adjacent.sum())
        if len(speed) > 1:
            accelerations.extend(np.diff(speed).tolist())
    metrics.update({
        "trajectory_continuity": float(continuity_steps / max(len(tracking) - tracking["frame"].nunique(), 1)),
        "mean_displacement": float(np.mean(velocities) * 0.04) if velocities else 0.0,
        "p95_displacement": float(np.percentile(np.asarray(velocities) * 0.04, 95)) if velocities else 0.0,
        "velocity_stability": float(np.std(accelerations)) if accelerations else 0.0,
        "acceleration_stability": float(np.std(np.diff(accelerations))) if len(accelerations) > 1 else 0.0,
        "candidate_cut_count": int(cut_signal["candidate_cut"].sum()) if not cut_signal.empty else 0,
    })
    return metrics


def causal_kalman_filter(tracking: pd.DataFrame, process_noise: float = 0.001, measurement_noise: float = 0.01) -> pd.DataFrame:
    """Apply a causal alpha-beta constant-velocity filter to observed points.

    State is reset after missing observations, so this filter never performs
    unbounded extrapolation or fills gaps.  It uses only current and prior
    observations and is intentionally a small auditable Kalman-style filter.
    """
    _required_columns(tracking)
    if process_noise <= 0 or measurement_noise <= 0:
        raise ValueError("process_noise and measurement_noise must be positive")
    out = tracking.copy()
    for _, indices in out.groupby(["team", "player_id"], sort=True).groups.items():
        group = out.loc[indices].sort_values("frame")
        state = None
        velocity = np.zeros(2)
        uncertainty = float(measurement_noise)
        previous_frame = None
        for index, row in group.iterrows():
            if not bool(row["visible"]) or pd.isna(row["x"]) or pd.isna(row["y"]):
                state = None
                previous_frame = None
                continue
            measurement = np.array([float(row["x"]), float(row["y"])])
            if state is None or previous_frame is None or int(row["frame"]) != previous_frame + 1:
                state = measurement.copy()
                velocity = np.zeros(2)
            else:
                prediction = state + velocity
                gain = (uncertainty + process_noise) / (uncertainty + process_noise + measurement_noise)
                residual = measurement - prediction
                state = prediction + gain * residual
                velocity = velocity + 0.5 * gain * residual
                uncertainty = (1.0 - gain) * (uncertainty + process_noise)
                out.loc[index, "x"], out.loc[index, "y"] = state
            previous_frame = int(row["frame"])
    return out


def _smooth_without_crossing_cuts(tracking: pd.DataFrame, method: str, cut_frames: Iterable[int], parameters: Mapping[str, Any]) -> pd.DataFrame:
    cut_frames = set(int(frame) for frame in cut_frames)
    if not cut_frames:
        return smooth_trajectory(tracking, method=method, **dict(parameters))
    pieces = []
    for _, group in tracking.groupby(["team", "player_id"], sort=False):
        group = group.sort_values("frame")
        boundaries = [frame for frame in cut_frames if group["frame"].min() < frame <= group["frame"].max()]
        starts = [group["frame"].min()] + boundaries
        ends = boundaries + [group["frame"].max() + 1]
        for start, end in zip(starts, ends):
            piece = group[group["frame"].between(start, end - 1)].copy()
            pieces.append(smooth_trajectory(piece, method=method, **dict(parameters)))
    return pd.concat(pieces).sort_index() if pieces else tracking.copy()


def apply_mitigation(tracking: pd.DataFrame, method: str, config: Mapping[str, Any], cut_frames: Iterable[int] = ()) -> pd.DataFrame:
    """Apply one bounded Step 80 condition using the existing primitives."""
    if method not in METHODS:
        raise ValueError(f"Unknown mitigation method: {method}")
    interpolation_cfg = config.get("interpolation", {})
    smoothing_cfg = config.get("smoothing", {})
    result = tracking.copy()
    if method.startswith("interpolation") or method == "interpolation":
        result = interpolate_trajectory(result, max_gap=int(interpolation_cfg.get("max_gap", 10)), imputed_confidence=float(interpolation_cfg.get("imputed_confidence", 0.5)))
    if method.endswith("moving_average") or method == "moving_average":
        result = _smooth_without_crossing_cuts(result, "moving_average", cut_frames, {"window_size": int(smoothing_cfg.get("moving_average_window", 5))})
    elif method.endswith("savgol") or method == "savgol":
        result = _smooth_without_crossing_cuts(result, "savgol", cut_frames, {"window_length": int(smoothing_cfg.get("savgol_window", 7)), "polyorder": int(smoothing_cfg.get("savgol_polyorder", 2))})
    elif method.endswith("kalman") or method == "kalman":
        result = causal_kalman_filter(result)
    return result


def run_step80(config: Mapping[str, Any], root: Optional[Path] = None) -> pd.DataFrame:
    """Run the configured Step 80 grid and return the machine-readable table."""
    root = root or _repo_root()
    rows = []
    trajectory_example = None
    output_directory = Path(config.get("output", {}).get("directory", "results/step80"))
    if not output_directory.is_absolute():
        output_directory = root / output_directory
    checkpoint_directory = output_directory / "metrics"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / "step80_checkpoint.csv"
    checkpoint_meta_path = checkpoint_directory / "step80_checkpoint_meta.json"
    expected_count = len(config["experiment_windows"]) * len(config["severities"]) * len(config["seeds"]) * len(METHODS)
    config_fingerprint = hashlib.sha256(json.dumps(_json_safe(config), sort_keys=True).encode("utf-8")).hexdigest()
    checkpoint_rows = pd.DataFrame()
    if checkpoint_path.exists() and checkpoint_meta_path.exists():
        metadata = json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
        if metadata.get("config_fingerprint") == config_fingerprint:
            checkpoint_rows = pd.read_csv(checkpoint_path)
            rows.extend(checkpoint_rows.to_dict(orient="records"))

    def checkpoint_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (row["severity"], int(row["seed"]), row["window"], row["mitigation_method"])

    completed_keys = {checkpoint_key(row) for row in rows}

    if len(completed_keys) == expected_count and expected_count > 0:
        result = checkpoint_rows.copy()
        result.attrs["trajectory_example"] = trajectory_example
        return result

    def persist_checkpoint() -> None:
        checkpoint_table = pd.DataFrame(rows)
        temporary_path = checkpoint_path.with_suffix(".tmp.csv")
        checkpoint_table.to_csv(temporary_path, index=False)
        temporary_path.replace(checkpoint_path)
        checkpoint_meta_path.write_text(json.dumps({
            "config_fingerprint": config_fingerprint,
            "expected_condition_count": expected_count,
            "completed_condition_count": len(completed_keys),
            "complete": len(completed_keys) == expected_count,
        }, indent=2, sort_keys=True), encoding="utf-8")

    for window in config["experiment_windows"]:
        window_cfg = copy.deepcopy(dict(config))
        fps = float(config["match"].get("fps", 25.0))
        window_cfg["match"] = dict(config["match"], start_frame=int(round(window[0] * fps)) + 1, end_frame=int(round(window[1] * fps)))
        clean, events = load_metrica_subset(window_cfg, root)
        cut_signal = detect_candidate_cuts(clean, **dict(config.get("camera_cut", {})))
        cut_frames = cut_signal.loc[cut_signal["candidate_cut"], "frame"].tolist()
        expected = int(clean[["team", "player_id"]].drop_duplicates().shape[0])
        clean_output = run_inference_condition(clean, window_cfg)
        for severity in config["severities"]:
            for seed in config["seeds"]:
                degraded, _ = degrade_tracking(clean, severity=severity, seed=int(seed))
                for method in METHODS:
                    condition_window = f"{window[0]}_{window[1]}s"
                    if (severity, int(seed), condition_window, method) in completed_keys:
                        continue
                    mitigated = apply_mitigation(degraded, method, config, cut_frames)
                    if trajectory_example is None and method == "interpolation_savgol":
                        identity = clean[["team", "player_id"]].drop_duplicates().iloc[0].to_dict()
                        example_frames = clean[clean["frame"].between(clean["frame"].min(), min(clean["frame"].min() + 49, clean["frame"].max()))]
                        example_rows = []
                        for label, frame_data in (("clean", clean), ("degraded", degraded), ("mitigated", mitigated)):
                            selected = frame_data.merge(pd.DataFrame([identity]), on=["team", "player_id"])
                            selected = selected[selected["frame"].isin(example_frames["frame"])]
                            selected = selected[["frame", "x"]].copy()
                            selected["series"] = label
                            example_rows.append(selected)
                        trajectory_example = pd.concat(example_rows, ignore_index=True)
                        trajectory_example["severity"] = severity
                        trajectory_example["seed"] = int(seed)
                        trajectory_example["window"] = f"{window[0]}_{window[1]}s"
                        trajectory_example["mitigation_method"] = method
                    output = run_inference_condition(mitigated, window_cfg)
                    quality = step80_tracking_metrics(mitigated, cut_signal, expected)
                    possession = possession_change_metrics(output["candidates"], events, fps, float(config["event_matching"]["tolerance_sec"]))
                    tactical = tactical_stability_metrics(clean_output["tactical"], output["tactical"], int(config["tactical"]["top_k"]))
                    rows.append({
                        "condition": "STEP80", "severity": severity, "seed": int(seed), "window": f"{window[0]}_{window[1]}s",
                        "mitigation_method": method, "missingness": quality["missingness_rate"],
                        "mean_gap_length": quality["mean_gap_length"], "max_gap_length": quality["max_gap_length"],
                        "trajectory_continuity": quality["trajectory_continuity"], "velocity_stability": quality["velocity_stability"],
                        "possession_precision": possession["precision"], "possession_recall": possession["recall"], "possession_f1": possession["f1"],
                        "tas_mae": tactical["mean_absolute_score_error"], "candidate_cut_count": quality["candidate_cut_count"],
                        "unresolved_gap_count": quality["unresolved_gap_count"], "p95_displacement": quality["p95_displacement"],
                    })
                    completed_keys.add(checkpoint_key(rows[-1]))
                    persist_checkpoint()
    result = pd.DataFrame(rows)
    result.attrs["trajectory_example"] = trajectory_example
    return result


def _summary_statistics(table: pd.DataFrame) -> Dict[str, Any]:
    table = table.copy()
    table.attrs = {}
    summary: Dict[str, Any] = {}
    for method, group in table.groupby("mitigation_method"):
        summary[method] = {}
        for metric in ("possession_f1", "tas_mae"):
            values = group[metric].astype(float).to_numpy()
            summary[method][metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }

    paired: Dict[str, Any] = {}
    baseline = table[table["mitigation_method"] == "none"].set_index(["severity", "seed", "window"])
    for method in METHODS:
        if method == "none":
            continue
        current = table[table["mitigation_method"] == method].set_index(["severity", "seed", "window"])
        joined = current.join(baseline, lsuffix="_method", rsuffix="_none", how="inner")
        paired[method] = {}
        for metric in ("possession_f1", "tas_mae"):
            differences = (joined[f"{metric}_method"] - joined[f"{metric}_none"]).dropna().to_numpy()
            mean_difference = float(np.mean(differences)) if len(differences) else 0.0
            standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else 0.0
            paired[method][metric] = {
                "n": int(len(differences)),
                "mean_difference_vs_none": mean_difference,
                "median_difference_vs_none": float(np.median(differences)) if len(differences) else 0.0,
                "std_difference_vs_none": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
                "ci95_approx": [mean_difference - 1.96 * standard_error, mean_difference + 1.96 * standard_error],
            }
    return {"by_method": summary, "paired_vs_none": paired}


def save_step80_outputs(
    table: pd.DataFrame,
    cut_signal: pd.DataFrame,
    config: Mapping[str, Any],
    root: Optional[Path] = None,
    trajectory_example: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    root = root or _repo_root()
    output = config.get("output", {})
    directory = Path(output.get("directory", "results/step80"))
    if not directory.is_absolute():
        directory = root / directory
    metrics = directory / "metrics"
    figures = directory / "figures"
    metrics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table_path = metrics / "step80_ablation.csv"
    summary_path = metrics / "step80_summary.json"
    cut_path = metrics / "step80_candidate_cuts.csv"
    table.to_csv(table_path, index=False)
    cut_signal.to_csv(cut_path, index=False)
    trajectory_example = trajectory_example if trajectory_example is not None else table.attrs.get("trajectory_example")
    summary = {
        "rows": int(len(table)), "methods": list(METHODS), "config": _json_safe(config),
        "statistics": _json_safe(_summary_statistics(table)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        table.boxplot(column="trajectory_continuity", by="mitigation_method", ax=axes[0], rot=35)
        table.boxplot(column="velocity_stability", by="mitigation_method", ax=axes[1], rot=35)
        axes[0].set_title("Trajectory continuity"); axes[0].set_xlabel("Mitigation method")
        axes[1].set_title("Velocity stability"); axes[1].set_xlabel("Mitigation method")
        fig.suptitle("Step 80 tracking quality: Metrica Sample Game 1, 0-60s")
        fig.tight_layout(); fig.savefig(figures / "tracking_quality_by_method.png", dpi=150); plt.close(fig)
        for value, ylabel, filename in (("possession_f1", "Possession F1", "possession_f1_by_method.png"), ("tas_mae", "TAS MAE relative to CLEAN", "tas_mae_by_method.png")):
            fig, axis = plt.subplots(figsize=(10, 5))
            table.boxplot(column=value, by="mitigation_method", ax=axis, rot=35)
            axis.set_title(f"Step 80: {ylabel}"); axis.set_xlabel("Mitigation method"); axis.set_ylabel(ylabel)
            fig.suptitle(""); fig.tight_layout(); fig.savefig(figures / filename, dpi=150); plt.close(fig)
        fig, axis = plt.subplots(figsize=(10, 4)); axis.plot(cut_signal["frame"], cut_signal["cut_score"]); axis.scatter(cut_signal.loc[cut_signal["candidate_cut"], "frame"], cut_signal.loc[cut_signal["candidate_cut"], "cut_score"], color="red"); axis.set(title="Step 80 candidate camera-cut/discontinuity signal", xlabel="Frame", ylabel="Cut score"); fig.tight_layout(); fig.savefig(figures / "candidate_cut_timeline.png", dpi=150); plt.close(fig)
        if trajectory_example is not None and not trajectory_example.empty:
            fig, axis = plt.subplots(figsize=(10, 5))
            for label, style in (("clean", "-"), ("degraded", "--"), ("mitigated", ":")):
                subset = trajectory_example[trajectory_example["series"] == label]
                axis.plot(subset["frame"], subset["x"], style, label=label.upper())
            example = trajectory_example.iloc[0]
            axis.set(title=f"Step 80 trajectory example: {example['window']}, {example['severity']}, seed {example['seed']}, {example['mitigation_method']}", xlabel="Frame", ylabel="Normalized x")
            axis.legend(); fig.tight_layout(); fig.savefig(figures / "trajectory_example.png", dpi=150); plt.close(fig)
    except ImportError:
        pass
    return {"table": table_path, "summary": summary_path, "candidate_cuts": cut_path}


def _make_trajectory_example(config: Mapping[str, Any], root: Path) -> pd.DataFrame:
    window = config["experiment_windows"][0]
    window_cfg = copy.deepcopy(dict(config))
    fps = float(config["match"].get("fps", 25.0))
    window_cfg["match"] = dict(config["match"], start_frame=int(round(window[0] * fps)) + 1, end_frame=int(round(window[1] * fps)))
    clean, _ = load_metrica_subset(window_cfg, root)
    degraded, _ = degrade_tracking(clean, severity=config["severities"][0], seed=int(config["seeds"][0]))
    cut_signal = detect_candidate_cuts(clean, **dict(config.get("camera_cut", {})))
    mitigated = apply_mitigation(degraded, "interpolation_savgol", config, cut_signal.loc[cut_signal["candidate_cut"], "frame"].tolist())
    identity = clean[["team", "player_id"]].drop_duplicates().iloc[0].to_dict()
    example_frames = clean[clean["frame"].between(clean["frame"].min(), min(clean["frame"].min() + 49, clean["frame"].max()))]
    example_rows = []
    for label, frame_data in (("clean", clean), ("degraded", degraded), ("mitigated", mitigated)):
        selected = frame_data.merge(pd.DataFrame([identity]), on=["team", "player_id"])
        selected = selected[selected["frame"].isin(example_frames["frame"])][["frame", "x"]].copy()
        selected["series"] = label
        example_rows.append(selected)
    example = pd.concat(example_rows, ignore_index=True)
    example["severity"] = config["severities"][0]
    example["seed"] = int(config["seeds"][0])
    example["window"] = f"{window[0]}_{window[1]}s"
    example["mitigation_method"] = "interpolation_savgol"
    return example


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated Step 80 broadcast-noise analysis.")
    parser.add_argument("--config", default="configs/step80_broadcast_noise.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    table = run_step80(config)
    tracking, _ = load_metrica_subset(config)
    cuts = detect_candidate_cuts(tracking, **dict(config.get("camera_cut", {})))
    example = table.attrs.get("trajectory_example")
    if example is None:
        example = _make_trajectory_example(config, _repo_root())
    paths = save_step80_outputs(table, cuts, config, trajectory_example=example)
    print(table.groupby("mitigation_method")[["possession_f1", "tas_mae"]].mean().to_string())
    print(paths)


if __name__ == "__main__":
    main()