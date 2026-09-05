"""Step 81 controlled ball-free event evaluation.

This module evaluates candidate transitions generated from player tracking only.
Metrica event annotations are loaded separately and consumed only after inference.
It is controlled annotated-tracking validation, not SoccerNet-GSR validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.possession.possession_baseline import infer_events
from src.validation.robustness_benchmark import (
    _repo_root,
    _tracking_only,
    load_config,
    load_metrica_subset,
    run_inference_condition,
)

DEFAULT_TOLERANCES_SEC = (0.20, 0.50, 1.00)
EVENT_TYPE_MAPPING = {
    "pass_candidate": ("PASS",),
    "turnover_candidate": ("BALL LOST", "BALL OUT"),
    "recovery_candidate": ("RECOVERY",),
}


def extract_candidate_events(tracking: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Infer candidate transitions from tracking only; no labels are accepted."""
    inference = run_inference_condition(_tracking_only(tracking), config)
    candidates = inference["candidates"].copy()
    if candidates.empty:
        return candidates
    allowed = set(EVENT_TYPE_MAPPING)
    candidates = candidates[candidates["event_type"].isin(allowed)].copy()
    return candidates.drop_duplicates(subset=["event_type", "frame"], keep="first").reset_index(drop=True)


def _greedy_match(predicted_frames: Sequence[int], reference_frames: Sequence[int], tolerance_frames: int) -> list[Tuple[int, int]]:
    pairs = sorted(
        (abs(int(predicted) - int(reference)), pred_index, reference_index)
        for pred_index, predicted in enumerate(predicted_frames)
        for reference_index, reference in enumerate(reference_frames)
        if abs(int(predicted) - int(reference)) <= tolerance_frames
    )
    used_predicted: set[int] = set()
    used_reference: set[int] = set()
    matches = []
    for _, pred_index, reference_index in pairs:
        if pred_index not in used_predicted and reference_index not in used_reference:
            used_predicted.add(pred_index)
            used_reference.add(reference_index)
            matches.append((pred_index, reference_index))
    return matches


def _event_rows(candidates: pd.DataFrame, reference_events: pd.DataFrame, prediction_type: str, reference_types: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    predicted = candidates.loc[candidates["event_type"] == prediction_type, "frame"].drop_duplicates().astype(int).to_numpy() if not candidates.empty else np.array([], dtype=int)
    reference = reference_events.loc[reference_events["event_type"].isin(reference_types), "start_frame"].drop_duplicates().astype(int).to_numpy() if not reference_events.empty else np.array([], dtype=int)
    return predicted, reference


def evaluate_event_candidates(
    candidates: pd.DataFrame,
    reference_events: pd.DataFrame,
    fps: float,
    tolerances_sec: Sequence[float] = DEFAULT_TOLERANCES_SEC,
) -> pd.DataFrame:
    """Return per-type and aggregate one-to-one event metrics by tolerance."""
    rows = []
    for tolerance_sec in tolerances_sec:
        tolerance_frames = int(round(float(tolerance_sec) * float(fps)))
        aggregate_predicted = 0
        aggregate_reference = 0
        aggregate_matches = 0
        aggregate_errors = []
        for prediction_type, reference_types in EVENT_TYPE_MAPPING.items():
            predicted, reference = _event_rows(candidates, reference_events, prediction_type, reference_types)
            matches = _greedy_match(predicted, reference, tolerance_frames)
            errors = [abs(int(predicted[p]) - int(reference[r])) / float(fps) for p, r in matches]
            matched = len(matches)
            precision = matched / len(predicted) if len(predicted) else 0.0
            recall = matched / len(reference) if len(reference) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append({
                "scope": "event_type", "event_type": prediction_type,
                "reference_event_types": "|".join(reference_types),
                "tolerance_sec": float(tolerance_sec), "tolerance_frames": tolerance_frames,
                "n_predicted": int(len(predicted)), "n_reference": int(len(reference)),
                "true_positive": matched, "false_positive": int(len(predicted) - matched),
                "false_negative": int(len(reference) - matched), "precision": precision,
                "recall": recall, "f1": f1,
                "mean_timing_error_sec": float(np.mean(errors)) if errors else 0.0,
                "median_timing_error_sec": float(np.median(errors)) if errors else 0.0,
            })
            aggregate_predicted += len(predicted)
            aggregate_reference += len(reference)
            aggregate_matches += matched
            aggregate_errors.extend(errors)
        precision = aggregate_matches / aggregate_predicted if aggregate_predicted else 0.0
        recall = aggregate_matches / aggregate_reference if aggregate_reference else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "scope": "aggregate", "event_type": "all_supported",
            "reference_event_types": "PASS|BALL LOST|BALL OUT|RECOVERY",
            "tolerance_sec": float(tolerance_sec), "tolerance_frames": tolerance_frames,
            "n_predicted": aggregate_predicted, "n_reference": aggregate_reference,
            "true_positive": aggregate_matches, "false_positive": aggregate_predicted - aggregate_matches,
            "false_negative": aggregate_reference - aggregate_matches, "precision": precision,
            "recall": recall, "f1": f1,
            "mean_timing_error_sec": float(np.mean(aggregate_errors)) if aggregate_errors else 0.0,
            "median_timing_error_sec": float(np.median(aggregate_errors)) if aggregate_errors else 0.0,
        })
    return pd.DataFrame(rows)


def run_step81(config: Mapping[str, Any], root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run fixed-threshold clean tracking evaluation on the configured window."""
    root = root or _repo_root()
    window = config["experiment_window"]
    fps = float(config["match"].get("fps", 25.0))
    run_config = dict(config)
    run_config["match"] = dict(config["match"], start_frame=int(round(window[0] * fps)) + 1, end_frame=int(round(window[1] * fps)))
    tracking, reference_events = load_metrica_subset(run_config, root)
    candidates = extract_candidate_events(tracking, run_config)
    metrics = evaluate_event_candidates(candidates, reference_events, fps, config.get("tolerances_sec", DEFAULT_TOLERANCES_SEC))
    return candidates, metrics, reference_events


def save_step81_outputs(candidates: pd.DataFrame, metrics: pd.DataFrame, reference_events: pd.DataFrame, config: Mapping[str, Any], root: Path | None = None) -> Dict[str, Path]:
    root = root or _repo_root()
    directory = Path(config.get("output_directory", "results/step81"))
    if not directory.is_absolute():
        directory = root / directory
    metric_dir = directory / "metrics"
    figure_dir = directory / "figures"
    log_dir = directory / "logs"
    for path in (metric_dir, figure_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    metrics_path = metric_dir / "step81_event_metrics.csv"
    tolerance_path = metric_dir / "step81_tolerance_analysis.csv"
    summary_path = metric_dir / "step81_summary.json"
    metrics.to_csv(metrics_path, index=False)
    metrics[metrics["scope"] == "aggregate"].to_csv(tolerance_path, index=False)
    aggregate = metrics[metrics["scope"] == "aggregate"].sort_values("tolerance_sec")
    summary = {
        "description": "Controlled ball-free event validation using annotated Metrica tracking/events; not GSR validation.",
        "config": config,
        "n_ground_truth_supported_events": int(aggregate.iloc[0]["n_reference"]) if not aggregate.empty else 0,
        "n_predicted_events": int(aggregate.iloc[0]["n_predicted"]) if not aggregate.empty else 0,
        "best_f1_tolerance_sec": float(aggregate.loc[aggregate["f1"].idxmax(), "tolerance_sec"]) if not aggregate.empty else None,
        "aggregate_metrics": aggregate.to_dict(orient="records"),
        "unsupported_source_event_types": sorted(set(reference_events["event_type"]) - {v for values in EVENT_TYPE_MAPPING.values() for v in values}),
        "inference_uses_event_labels": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        aggregate.plot(x="tolerance_sec", y="f1", marker="o", ax=plt.subplots(figsize=(7, 4))[1], legend=False)
        plt.xlabel("Matching tolerance (s)"); plt.ylabel("Aggregate F1"); plt.title("Step 81 F1 versus temporal tolerance"); plt.tight_layout(); plt.savefig(figure_dir / "tolerance_f1_curve.png", dpi=150); plt.close("all")
        typed = metrics[metrics["scope"] == "event_type"]
        pivot = typed.pivot(index="event_type", columns="tolerance_sec", values="f1")
        pivot.plot(kind="bar", figsize=(9, 5)); plt.ylabel("F1"); plt.title("Step 81 event-type F1 by tolerance"); plt.tight_layout(); plt.savefig(figure_dir / "event_pr_curve.png", dpi=150); plt.close("all")
        plt.figure(figsize=(7, 4));
        timing_errors = []
        widest_tolerance = max(config.get("tolerances_sec", DEFAULT_TOLERANCES_SEC))
        fps = float(config["match"].get("fps", 25.0))
        tolerance_frames = int(round(widest_tolerance * fps))
        for prediction_type, reference_types in EVENT_TYPE_MAPPING.items():
            predicted, reference = _event_rows(candidates, reference_events, prediction_type, reference_types)
            timing_errors.extend(abs(int(predicted[p]) - int(reference[r])) / fps for p, r in _greedy_match(predicted, reference, tolerance_frames))
        if timing_errors:
            plt.hist(timing_errors, bins=min(10, max(1, len(timing_errors))))
        plt.xlabel("Timing error (s)"); plt.ylabel("Count"); plt.title("Step 81 candidate timing differences"); plt.tight_layout(); plt.savefig(figure_dir / "event_timing_error.png", dpi=150); plt.close("all")
    except ImportError:
        pass
    return {"metrics": metrics_path, "tolerance": tolerance_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled Step 81 ball-free event evaluation.")
    parser.add_argument("--config", default="configs/step81_event_evaluation.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    candidates, metrics, reference_events = run_step81(config)
    paths = save_step81_outputs(candidates, metrics, reference_events, config)
    print(metrics[metrics["scope"] == "aggregate"].to_string(index=False))
    print(paths)


if __name__ == "__main__":
    main()
