"""Run the deterministic Step 82 player-only tactical pass experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tactics.pass_candidates import FEATURE_NAMES, generate_pass_candidates
from src.tactics.pass_probability import PassProbabilityModel, fit_pass_model
from src.tactics.pass_ranking import evaluate_rankings, rank_counterfactual_passes
from src.validation.robustness_benchmark import load_config, load_metrica_subset


def _finite(value):
    return float(value) if np.isfinite(value) else 0.0


def _tracking_player_id(canonical_id: str) -> str:
    """Map parser IDs (HOME_19/AWAY_19) to Metrica jersey IDs."""
    return str(canonical_id).split("_", 1)[-1]


def _plot_outputs(output: Path, rankings: dict[str, list[pd.DataFrame]], actual: list[str], model: PassProbabilityModel) -> None:
    import matplotlib.pyplot as plt
    figure = output / "figures"
    figure.mkdir(parents=True, exist_ok=True)
    def message(name: str, text: str) -> None:
        plt.figure(figsize=(7, 4)); plt.text(0.5, 0.5, text, ha="center", va="center"); plt.axis("off"); plt.tight_layout(); plt.savefig(figure / name, dpi=150); plt.close()
    message("calibration_curve.png", "Calibration unavailable: Metrica has no explicit pass-completion labels")
    message("probability_distribution.png", "Probability distribution unavailable: no defensible binary outcomes")
    metrics = {name: evaluate_rankings(rows, actual) for name, rows in rankings.items()}
    plt.figure(figsize=(8, 4)); names = list(metrics); values = [metrics[name].iloc[0]["top_1_agreement"] for name in names]
    plt.bar(names, values); plt.ylabel("Top-1 agreement"); plt.ylim(0, 1); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(figure / "ranking_quality.png", dpi=150); plt.close()
    weights = [0.20, 0.35, 0.05, 0.20, 0.25, 0.25, 0.08, -0.20, 0.05, 0.05]
    plt.figure(figsize=(9, 4)); plt.bar(FEATURE_NAMES, weights); plt.xticks(rotation=45, ha="right"); plt.ylabel("Fixed heuristic contribution"); plt.tight_layout(); plt.savefig(figure / "feature_importance.png", dpi=150); plt.close()
    example = rankings["pass_model"][0] if rankings["pass_model"] else pd.DataFrame()
    if example.empty:
        message("counterfactual_example.png", "No identifiable PASS state available")
    else:
        plt.figure(figsize=(8, 4)); plt.bar(example["target_player_id"], example["decision_score"]); plt.ylabel("Decision score"); plt.title("Counterfactual options at first identifiable PASS state"); plt.tight_layout(); plt.savefig(figure / "counterfactual_example.png", dpi=150); plt.close()


def run(config_path: str = "configs/step82_tactical_pass_model.yaml") -> dict[str, Path]:
    config = load_config(ROOT / config_path)
    tracking, events = load_metrica_subset(config, ROOT)
    output = ROOT / config["output_directory"]
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pass_events = events[(events["event_type"] == "PASS") & events["from_player_id"].notna() & events["to_player_id"].notna()].copy()
    states, candidates, actual_targets = [], [], []
    for _, event in pass_events.sort_values("start_frame").iterrows():
        state = tracking[tracking["frame"] == int(event["start_frame"])].copy()
        source_id = _tracking_player_id(str(event["from_player_id"]))
        target_id = _tracking_player_id(str(event["to_player_id"]))
        if state.empty or source_id not in set(state["player_id"]):
            continue
        generated = generate_pass_candidates(state, source_id, {**config["candidates"], **config["orientation"]})
        if target_id not in set(generated["target_player_id"]):
            continue
        states.append(state); candidates.append(generated); actual_targets.append(target_id)
    all_candidates = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    labels = np.array([], dtype=int)
    model = fit_pass_model(pd.DataFrame(columns=FEATURE_NAMES), labels, config["model"])
    ranked = {"pass_model": [], "nearest_teammate": [], "most_forward_teammate": [], "tas_baseline": []}
    for state, generated in zip(states, candidates):
        ranked["pass_model"].append(rank_counterfactual_passes(state, model, str(generated.iloc[0]["source_player_id"]), {**config["candidates"], **config["orientation"], "ranking_weights": config["ranking_weights"]}))
        ranked["nearest_teammate"].append(generated.sort_values(["pass_distance", "target_player_id"]).reset_index(drop=True))
        ranked["most_forward_teammate"].append(generated.sort_values(["forward_progress", "target_player_id"], ascending=[False, True]).reset_index(drop=True))
        ranked["tas_baseline"].append(generated.assign(decision_score=generated["forward_progress"] + generated["target_space"] - generated["target_pressure"]).sort_values(["decision_score", "target_player_id"], ascending=[False, True]).reset_index(drop=True))
    rankings = []
    for name, rows in ranked.items():
        result = evaluate_rankings(rows, actual_targets, tuple(config["top_k"]))
        result.insert(0, "baseline", name)
        rankings.append(result)
    ranking_metrics = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    ranking_metrics.to_csv(metrics_dir / "ranking_metrics.csv", index=False)
    if not all_candidates.empty:
        all_candidates.to_csv(metrics_dir / "candidate_passes.csv", index=False)
    model_metrics = pd.DataFrame([{"model": model.model_type, "n_positive": 0, "n_negative": 0, "n_reliable_completion_labels": 0, "train_count": 0, "evaluation_count": 0, "status": "fallback_due_to_missing_outcome_labels"}])
    model_metrics.to_csv(metrics_dir / "pass_model_metrics.csv", index=False)
    probability_metrics = pd.DataFrame([{"metric": metric, "value": 0.0, "valid": False, "reason": "canonical Metrica events do not encode pass completion"} for metric in ("roc_auc", "pr_auc", "brier_score", "log_loss", "calibration_error")])
    probability_metrics.to_csv(metrics_dir / "pass_probability_metrics.csv", index=False)
    ablation = pd.DataFrame([{"ablation": name, "n_labels": 0, "status": "unavailable_no_completion_labels", "roc_auc": 0.0, "brier_score": 0.0} for name in ("geometry_only", "geometry_pressure", "geometry_pressure_support_space", "full_model")])
    ablation.to_csv(metrics_dir / "ablation_results.csv", index=False)
    summary = {"step": "82", "dataset": config["dataset"], "frames": [1, 1500], "n_states": len(states), "n_reliable_labeled_passes": 0, "n_identifiable_pass_events": int(len(pass_events)), "n_candidate_passes": int(len(all_candidates)), "model": model.to_dict(), "probability_metrics_status": "not_estimable", "ranking_metrics": ranking_metrics.to_dict(orient="records"), "ranking_ground_truth": "annotated PASS target only; no completion claim", "limitations": ["Metrica PASS annotations identify source/receiver but not completion outcome", "single match and small ranking sample", "ranking does not claim the actual player chose the top option"], "inference_uses_ball_coordinates": False, "inference_uses_event_labels": False}
    (metrics_dir / "step82_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _plot_outputs(output, ranked, actual_targets, model)
    return {"summary": metrics_dir / "step82_summary.json", "ranking": metrics_dir / "ranking_metrics.csv"}


if __name__ == "__main__":
    print(run())