"""Task-driven robustness benchmark for ball-free downstream analysis.

The module deliberately separates inference from evaluation.  Tracking-only
condition processing feeds the existing spatial graph, possession, and TAS
baselines; Metrica events are loaded only after those outputs are final and
are used solely as post-hoc references.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.metrica_event_parser import load_metrica_events
from src.data.metrica_parser import load_metrica_match
from src.noise.degradation import degrade_tracking
from src.noise.interpolation import interpolate_trajectory
from src.noise.smoothing import smooth_trajectory
from src.possession.possession_baseline import infer_events, predict_possession
from src.possession.spatial_graph import build_frame_graph
from src.tactics.tactical_features import extract_tactical_features
from src.tactics.tactical_score import calculate_tactical_score


CONDITIONS = ("CLEAN", "SYNTHETIC-DEGRADED", "NOISE-MITIGATED")
_TRACKING_COLUMNS = [
    "match_id", "frame", "timestamp", "player_id", "team", "x", "y",
    "confidence", "visible",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas values to stable JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def load_config(config: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    """Load YAML (or JSON, a YAML subset) with no hidden defaults."""
    if isinstance(config, Mapping):
        return dict(config)
    config_path = Path(config)
    try:
        import yaml
    except ImportError:
        # The shipped configuration is JSON, which is valid YAML. This keeps
        # the benchmark runnable in the project venv without a parser dependency.
        parsed = json.loads(config_path.read_text(encoding="utf-8-sig"))
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    if not isinstance(parsed, dict):
        raise ValueError("Benchmark configuration must be a YAML mapping.")
    return parsed


def _absolute_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def load_metrica_subset(config: Mapping[str, Any], root: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the configured canonical Metrica subset and evaluation references.

    The event frame filter happens here for efficiency, but events are returned
    separately and are never passed to the condition inference functions.
    """
    root = root or _repo_root()
    match = config["match"]
    tracking, _ = load_metrica_match(
        str(_absolute_path(match["home_tracking_path"], root)),
        str(_absolute_path(match["away_tracking_path"], root)),
        match["match_id"],
    )
    start_frame = int(match["start_frame"])
    end_frame = int(match["end_frame"])
    if end_frame < start_frame:
        raise ValueError("match.end_frame must be >= match.start_frame")
    tracking = tracking[tracking["frame"].between(start_frame, end_frame)].copy()
    tracking = tracking.sort_values(["frame", "team", "player_id"]).reset_index(drop=True)
    if tracking.empty:
        raise ValueError("Configured tracking subset is empty.")
    events = load_metrica_events(
        str(_absolute_path(match["events_path"], root)), match["match_id"]
    )
    events = events[events["start_frame"].between(start_frame, end_frame)].copy()
    return tracking, events


def build_conditions(clean_tracking: pd.DataFrame, config: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    """Create the three benchmark conditions without mutating clean tracking."""
    degradation = config["degradation"]
    degraded, _ = degrade_tracking(
        clean_tracking,
        severity=degradation.get("severity", "moderate"),
        seed=int(degradation["seed"]),
        enabled_degradations=degradation.get("enabled_degradations"),
        custom_config=degradation.get("custom_config"),
    )
    interpolation = config["interpolation"]
    mitigated = interpolate_trajectory(
        degraded,
        max_gap=int(interpolation["max_gap"]),
        confidence_decay=bool(interpolation.get("confidence_decay", False)),
        imputed_confidence=float(interpolation.get("imputed_confidence", 0.5)),
    )
    smoothing = config["smoothing"]
    mitigated = smooth_trajectory(
        mitigated,
        method=smoothing["method"],
        **dict(smoothing.get("parameters", {})),
    )
    return {
        "CLEAN": clean_tracking.copy(),
        "SYNTHETIC-DEGRADED": degraded,
        "NOISE-MITIGATED": mitigated,
    }


def tracking_quality_metrics(
    tracking: pd.DataFrame,
    expected_players: Optional[int] = None,
    large_motion_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Compute condition quality metrics using only canonical tracking fields."""
    if tracking.empty:
        return {
            "missingness_rate": 0.0, "visible_player_completeness": 0.0,
            "coordinate_validity_rate": 0.0, "interpolation_fraction": 0.0,
            "large_motion_anomaly_count": 0, "large_motion_anomaly_rate": 0.0,
            "n_consecutive_visible_steps": 0,
        }
    visible = tracking["visible"].astype(bool)
    visible_rows = tracking.loc[visible]
    finite = np.isfinite(visible_rows["x"].to_numpy(dtype=float)) & np.isfinite(visible_rows["y"].to_numpy(dtype=float))
    in_bounds = (
        visible_rows["x"].between(0.0, 1.0).to_numpy()
        & visible_rows["y"].between(0.0, 1.0).to_numpy()
    )
    expected = expected_players or int(tracking[["team", "player_id"]].drop_duplicates().shape[0])
    visible_per_frame = tracking.groupby("frame", sort=False)["visible"].sum()
    imputed = tracking.get("imputed", pd.Series(False, index=tracking.index)).astype(bool)

    anomaly_count = 0
    step_count = 0
    for _, player in tracking.groupby(["team", "player_id"], sort=True):
        player = player.sort_values("frame")
        current = player[["frame", "x", "y", "visible"]]
        previous = current.shift(1)
        valid_step = (
            current["visible"].astype(bool)
            & previous["visible"].fillna(False).astype(bool)
            & ((current["frame"] - previous["frame"]) == 1)
        )
        if valid_step.any():
            displacement = np.sqrt(
                (current["x"] - previous["x"]) ** 2 + (current["y"] - previous["y"]) ** 2
            )
            step_count += int(valid_step.sum())
            anomaly_count += int((valid_step & (displacement > large_motion_threshold)).sum())
    return {
        "missingness_rate": float(1.0 - visible.mean()),
        "visible_player_completeness": float((visible_per_frame / expected).mean()),
        "coordinate_validity_rate": float((finite & in_bounds).mean()) if len(visible_rows) else 0.0,
        "interpolation_fraction": float(imputed.mean()),
        "large_motion_anomaly_count": anomaly_count,
        "large_motion_anomaly_rate": float(anomaly_count / step_count) if step_count else 0.0,
        "n_consecutive_visible_steps": step_count,
    }


def _tracking_only(tracking: pd.DataFrame) -> pd.DataFrame:
    """Apply the explicit anti-leakage whitelist before downstream inference."""
    missing = set(_TRACKING_COLUMNS) - set(tracking.columns)
    if missing:
        raise ValueError(f"Tracking is missing canonical columns: {sorted(missing)}")
    return tracking[_TRACKING_COLUMNS].copy()


def tactical_scores(
    tracking: pd.DataFrame,
    possession: pd.DataFrame,
    home_attacks_x1: bool,
    density_radius: float,
    knn_k: Optional[int],
) -> pd.DataFrame:
    """Extract per-frame TAS from tracking plus inferred possession only."""
    tracking = _tracking_only(tracking)
    by_frame = {int(frame): group for frame, group in tracking.groupby("frame", sort=False)}
    output = []
    for row in possession.sort_values("frame").itertuples(index=False):
        frame_data = by_frame[int(row.frame)]
        graph = build_frame_graph(
            frame_data, int(row.frame), float(row.timestamp),
            density_radius=density_radius, knn_k=knn_k,
        )
        team = row.team if row.team in {"home", "away"} else None
        features = extract_tactical_features(graph, row.possessor_id, home_attacks_x1=home_attacks_x1)
        score = calculate_tactical_score(features, team) if team else 0.0
        output.append({
            "frame": int(row.frame), "timestamp": float(row.timestamp),
            "possessor_id": row.possessor_id, "team": team, "tactical_score": score,
        })
    return pd.DataFrame(output)


def run_inference_condition(tracking: pd.DataFrame, config: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    """Run graph â†’ ball-free possession â†’ tactical TAS without labels or ball data."""
    tracking = _tracking_only(tracking)
    possession_parameters = dict(config["possession"])
    possession = predict_possession(tracking, **possession_parameters)
    candidates = infer_events(possession)
    tactical = tactical_scores(
        tracking, possession,
        home_attacks_x1=bool(config["orientation"]["home_attacks_x1"]),
        density_radius=float(possession_parameters.get("density_radius", 0.05)),
        knn_k=possession_parameters.get("knn_k", 5),
    )
    return {"possession": possession, "candidates": candidates, "tactical": tactical}


def _greedy_match(predicted_frames: Iterable[int], reference_frames: Iterable[int], tolerance_frames: int) -> Tuple[int, list[Tuple[int, int]]]:
    predicted = list(map(int, predicted_frames))
    reference = list(map(int, reference_frames))
    candidates = sorted(
        (abs(pred - ref), pred_i, ref_i)
        for pred_i, pred in enumerate(predicted)
        for ref_i, ref in enumerate(reference)
        if abs(pred - ref) <= tolerance_frames
    )
    used_pred, used_ref, matches = set(), set(), []
    for _, pred_i, ref_i in candidates:
        if pred_i not in used_pred and ref_i not in used_ref:
            used_pred.add(pred_i)
            used_ref.add(ref_i)
            matches.append((pred_i, ref_i))
    return len(matches), matches


def possession_change_metrics(candidates: pd.DataFrame, reference_events: pd.DataFrame, fps: float, tolerance_sec: float) -> Dict[str, Any]:
    """Evaluate unique possession-change candidates against Metrica annotations."""
    predicted_frames = np.array([], dtype=int)
    if not candidates.empty:
        predicted_frames = candidates[candidates["event_type"].isin(["pass_candidate", "turnover_candidate", "possession_change"])]["frame"].drop_duplicates().to_numpy(dtype=int)
    reference_frames = reference_events[
        reference_events["event_type"].isin(["PASS", "BALL LOST", "BALL OUT", "RECOVERY"])
    ]["start_frame"].drop_duplicates().to_numpy(dtype=int)
    tolerance_frames = int(round(tolerance_sec * fps))
    matched, pairs = _greedy_match(predicted_frames, reference_frames, tolerance_frames)
    precision = matched / len(predicted_frames) if len(predicted_frames) else 0.0
    recall = matched / len(reference_frames) if len(reference_frames) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    diffs = [abs(int(predicted_frames[p]) - int(reference_frames[r])) for p, r in pairs]
    return {
        "tolerance_sec": tolerance_sec, "tolerance_frames": tolerance_frames,
        "n_predicted_candidates": int(len(predicted_frames)), "n_reference_events": int(len(reference_frames)),
        "n_matched": matched, "precision": precision, "recall": recall, "f1": f1,
        "mean_matched_temporal_error_frames": float(np.mean(diffs)) if diffs else None,
        "median_matched_temporal_error_frames": float(np.median(diffs)) if diffs else None,
    }


def _possession_segment_mae(merged: pd.DataFrame) -> Optional[float]:
    clean = merged[["possessor_id_clean", "absolute_tactical_score_difference"]].copy()
    clean["segment"] = clean["possessor_id_clean"].ne(clean["possessor_id_clean"].shift()).cumsum()
    grouped = clean[clean["possessor_id_clean"].notna()].groupby("segment")["absolute_tactical_score_difference"].mean()
    return float(grouped.mean()) if len(grouped) else None


def tactical_stability_metrics(clean_tactical: pd.DataFrame, condition_tactical: pd.DataFrame, top_k: int) -> Dict[str, Any]:
    """Measure TAS stability relative to CLEAN, not tactical ground truth."""
    merged = clean_tactical.merge(condition_tactical, on="frame", suffixes=("_clean", "_condition"))
    if merged.empty:
        return {"n_compared_frames": 0, "mean_absolute_score_error": None, "median_absolute_score_error": None,
                "correlation_with_clean": None, "top_k": 0, "top_k_agreement": None, "possession_level_score_stability": None}
    differences = (merged["tactical_score_condition"] - merged["tactical_score_clean"]).abs()
    merged["absolute_tactical_score_difference"] = differences
    if len(merged) >= 2 and merged["tactical_score_clean"].nunique() > 1 and merged["tactical_score_condition"].nunique() > 1:
        correlation: Optional[float] = float(merged["tactical_score_clean"].corr(merged["tactical_score_condition"]))
    else:
        correlation = None
    k = min(int(top_k), len(merged))
    clean_top = set(merged.sort_values(["tactical_score_clean", "frame"], ascending=[False, True], kind="mergesort").head(k)["frame"])
    condition_top = set(merged.sort_values(["tactical_score_condition", "frame"], ascending=[False, True], kind="mergesort").head(k)["frame"])
    return {
        "n_compared_frames": int(len(merged)),
        "mean_absolute_score_error": float(differences.mean()),
        "median_absolute_score_error": float(differences.median()),
        "correlation_with_clean": correlation,
        "top_k": k,
        "top_k_agreement": float(len(clean_top & condition_top) / k) if k else None,
        "possession_level_score_stability": _possession_segment_mae(merged),
    }


def _summary_row(condition: str, quality: Mapping[str, Any], possession: Mapping[str, Any], tactical: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"condition": condition}
    row.update({f"tracking_{key}": value for key, value in quality.items()})
    row.update({f"possession_{key}": value for key, value in possession.items()})
    row.update({f"tactical_{key}": value for key, value in tactical.items()})
    return row


def _write_tactical_svg(tactical_scores_by_condition: Mapping[str, pd.DataFrame], figure_path: Path) -> None:
    """Write a dependency-free SVG TAS comparison when matplotlib is absent."""
    width, height, margin = 1000, 420, 55
    series = [scores for scores in tactical_scores_by_condition.values() if not scores.empty]
    frames = np.concatenate([scores["frame"].to_numpy(dtype=float) for scores in series])
    minimum, maximum = float(frames.min()), float(frames.max())
    span = max(maximum - minimum, 1.0)
    colors = {"CLEAN": "#1f77b4", "SYNTHETIC-DEGRADED": "#d62728", "NOISE-MITIGATED": "#2ca02c"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="55" y="24" font-family="sans-serif" font-size="16">Tactical Advantage Score by tracking condition</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
    ]
    for condition, scores in tactical_scores_by_condition.items():
        if scores.empty:
            continue
        points = " ".join(
            f'{margin + (frame - minimum) / span * (width - 2 * margin):.2f},{height - margin - score * (height - 2 * margin):.2f}'
            for frame, score in zip(scores["frame"], scores["tactical_score"])
        )
        color = colors.get(condition, "#555555")
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1"/>')
        y = 45 + 18 * list(tactical_scores_by_condition).index(condition)
        lines.append(f'<text x="{width - 250}" y="{y}" font-family="sans-serif" font-size="12" fill="{color}">{condition}</text>')
    lines.extend([
        f'<text x="{width / 2 - 25}" y="{height - 12}" font-family="sans-serif" font-size="12">Frame</text>',
        f'<text x="8" y="{height / 2}" font-family="sans-serif" font-size="12">TAS (0–1)</text>',
        '</svg>',
    ])
    figure_path.write_text("\n".join(lines), encoding="utf-8")
def save_outputs(result: Mapping[str, Any], config: Mapping[str, Any], root: Optional[Path] = None) -> Dict[str, Path]:
    """Persist compact benchmark artifacts; never write raw/intermediate tracks."""
    root = root or _repo_root()
    output = config.get("output", {})
    metrics_dir = _absolute_path(output.get("metrics_dir", "results/metrics"), root)
    runs_dir = _absolute_path(output.get("runs_dir", "results/runs"), root)
    figures_dir = _absolute_path(output.get("figures_dir", "results/figures"), root)
    for directory in (metrics_dir, runs_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    run_id = output.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"robustness_benchmark_{run_id}"
    summary_csv = metrics_dir / f"{prefix}_summary.csv"
    summary_json = metrics_dir / f"{prefix}_summary.json"
    config_json = runs_dir / f"{prefix}_config.json"
    tactical_csv = runs_dir / f"{prefix}_tactical_scores.csv"
    figure_path = figures_dir / f"{prefix}_tas_comparison.png"
    result["summary_table"].to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(_json_safe(result["condition_summaries"]), indent=2, sort_keys=True), encoding="utf-8")
    config_json.write_text(json.dumps(_json_safe(config), indent=2, sort_keys=True), encoding="utf-8")
    pd.concat(
        [scores.assign(condition=condition) for condition, scores in result["tactical_scores"].items()],
        ignore_index=True,
    ).to_csv(tactical_csv, index=False)
    try:
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(10, 4))
        for condition, scores in result["tactical_scores"].items():
            axis.plot(scores["frame"], scores["tactical_score"], label=condition, linewidth=0.8)
        axis.set(title="Tactical Advantage Score by tracking condition", xlabel="Frame", ylabel="TAS")
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
    except ImportError:
        figure_path = figures_dir / f"{prefix}_tas_comparison.svg"
        _write_tactical_svg(result["tactical_scores"], figure_path)
    return {"summary_csv": summary_csv, "summary_json": summary_json, "config_json": config_json, "tactical_csv": tactical_csv, "figure": figure_path}


def run_benchmark(config: Union[str, Path, Mapping[str, Any]], root: Optional[Path] = None, save: bool = True) -> Dict[str, Any]:
    """Run all conditions then perform event evaluation strictly post-hoc."""
    benchmark_config = load_config(config)
    root = root or _repo_root()
    clean_tracking, reference_events = load_metrica_subset(benchmark_config, root)
    conditions = build_conditions(clean_tracking, benchmark_config)
    expected_players = int(clean_tracking[["team", "player_id"]].drop_duplicates().shape[0])
    outputs = {condition: run_inference_condition(tracking, benchmark_config) for condition, tracking in conditions.items()}
    clean_tactical = outputs["CLEAN"]["tactical"]
    quality_config = benchmark_config["tracking_quality"]
    fps = float(benchmark_config["match"]["fps"])
    tolerance = float(benchmark_config["event_matching"]["tolerance_sec"])
    summaries: Dict[str, Dict[str, Any]] = {}
    for condition in CONDITIONS:
        quality = tracking_quality_metrics(conditions[condition], expected_players, float(quality_config["large_motion_threshold"]))
        possession = possession_change_metrics(outputs[condition]["candidates"], reference_events, fps, tolerance)
        tactical = tactical_stability_metrics(clean_tactical, outputs[condition]["tactical"], int(benchmark_config["tactical"]["top_k"]))
        summaries[condition] = {"condition": condition, "tracking_quality_metrics": quality, "possession_metrics": possession, "tactical_stability_metrics": tactical}
    summary_table = pd.DataFrame([
        _summary_row(condition, summaries[condition]["tracking_quality_metrics"], summaries[condition]["possession_metrics"], summaries[condition]["tactical_stability_metrics"])
        for condition in CONDITIONS
    ])
    result: Dict[str, Any] = {
        "condition_summaries": summaries,
        "summary_table": summary_table,
        "tactical_scores": {condition: outputs[condition]["tactical"] for condition in CONDITIONS},
        "subset": {"start_frame": int(benchmark_config["match"]["start_frame"]), "end_frame": int(benchmark_config["match"]["end_frame"]), "fps": fps},
        "seed": int(benchmark_config["degradation"]["seed"]),
        "orientation": {"home_attacks_x1": bool(benchmark_config["orientation"]["home_attacks_x1"])},
        "gsr_included": False,
    }
    if save:
        result["files"] = save_outputs(result, benchmark_config, root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ball-free robustness benchmark.")
    parser.add_argument("--config", default="configs/robustness_benchmark.yaml", help="YAML configuration path")
    args = parser.parse_args()
    result = run_benchmark(args.config)
    print(result["summary_table"].to_string(index=False))
    print("Saved:")
    for label, path in result.get("files", {}).items():
        if str(path):
            print(f"  {label}: {path}")


if __name__ == "__main__":
    main()



