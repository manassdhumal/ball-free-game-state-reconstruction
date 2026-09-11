"""Run the bounded Step 83 counterfactual decision analysis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tactics.counterfactual_analysis import (
    compare_rankings,
    perturb_snapshot,
    rank_counterfactual_actions,
    ranking_metrics,
    summarize_rankings,
)
from src.tactics.pass_candidates import FEATURE_NAMES, generate_pass_candidates
from src.tactics.pass_probability import fit_pass_model
from src.tactics.pass_ranking import rank_counterfactual_passes
from src.validation.robustness_benchmark import load_config, load_metrica_subset


def _tracking_id(canonical_id: str) -> str:
    return str(canonical_id).split("_", 1)[-1]


def _state_records(tracking: pd.DataFrame, events: pd.DataFrame, config: dict) -> list[dict]:
    records = []
    pass_events = events[(events["event_type"] == "PASS") & events["from_player_id"].notna() & events["to_player_id"].notna()].sort_values("start_frame")
    state_config = {**config["candidates"], **config["orientation"]}
    for _, event in pass_events.iterrows():
        state = tracking[tracking["frame"] == int(event["start_frame"])].copy()
        source = _tracking_id(str(event["from_player_id"]))
        target = _tracking_id(str(event["to_player_id"]))
        if state.empty or source not in set(state["player_id"]):
            continue
        candidates = generate_pass_candidates(state, source, state_config)
        if target not in set(candidates["target_player_id"]):
            continue
        records.append({"state_id": len(records), "state": state, "source": source, "target": target, "candidates": candidates})
    return records


def _ranking(state: pd.DataFrame, source: str, model, config: dict, weights: dict, method: str) -> pd.DataFrame:
    rank_config = {**config["candidates"], **config["orientation"], "ranking_weights": weights}
    if method in {"step82_fallback", "step83_balanced", "probability_heavy", "tactical_heavy"}:
        return rank_counterfactual_actions(state, model, source, rank_config)
    candidates = generate_pass_candidates(state, source, {**config["candidates"], **config["orientation"]})
    if method == "nearest_teammate":
        return candidates.sort_values(["pass_distance", "target_player_id"]).reset_index(drop=True).assign(rank=lambda frame: np.arange(1, len(frame) + 1))
    if method == "most_forward_teammate":
        return candidates.sort_values(["forward_progress", "target_player_id"], ascending=[False, True]).reset_index(drop=True).assign(rank=lambda frame: np.arange(1, len(frame) + 1))
    if method == "tas_baseline":
        candidates = candidates.assign(decision_score=candidates["forward_progress"] + candidates["target_space"] - candidates["target_pressure"])
        return candidates.sort_values(["decision_score", "target_player_id"], ascending=[False, True]).reset_index(drop=True).assign(rank=lambda frame: np.arange(1, len(frame) + 1))
    raise ValueError(f"Unknown ranking method: {method}")


def _actual_table(records: list[dict], rankings: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for method, method_rankings in rankings.items():
        for record, ranking in zip(records, method_rankings):
            metric = ranking_metrics(ranking, record["target"])
            rows.append({"state_id": record["state_id"], "frame": int(record["state"]["frame"].iloc[0]), "source_player_id": record["source"], "actual_target": record["target"], "method": method, **metric})
    return pd.DataFrame(rows)


def _scenario_rows(records: list[dict], rankings: list[pd.DataFrame]) -> pd.DataFrame:
    scored = []
    for record, ranking in zip(records, rankings):
        candidates = record["candidates"]
        scored.append({"record": record, "ranking": ranking, "risk": float(candidates.iloc[0]["source_pressure"]), "space": float(candidates["target_space"].max()), "forward": float(candidates["forward_progress"].max()), "crowded": float(candidates["line_obstruction"].mean())})
    choices = {}
    for label, key, ascending in (("high_pressure", "risk", True), ("open_space", "space", False), ("forward_opportunity", "forward", False), ("crowded", "crowded", False)):
        ordered = sorted(scored, key=lambda item: (item[key], item["record"]["state_id"]), reverse=not ascending)
        for item in ordered:
            state_id = item["record"]["state_id"]
            if state_id not in choices:
                choices[state_id] = (label, item)
                break
    rows = []
    for label, item in choices.values():
        ranking = item["ranking"].head(3)
        for position, (_, candidate) in enumerate(ranking.iterrows(), 1):
            rows.append({"scenario": label, "state_id": item["record"]["state_id"], "frame": int(item["record"]["state"]["frame"].iloc[0]), "source_player_id": item["record"]["source"], "rank": position, "target_player_id": candidate["target_player_id"], "probability_proxy": float(candidate.get("pass_completion_probability", np.nan)), "tactical_value": float(candidate.get("tactical_value", np.nan)), "decision_score": float(candidate.get("decision_score", np.nan)), "interpretation": "higher score reflects the configured probability/tactical tradeoff; this is not an optimality claim"})
    return pd.DataFrame(rows)


def _plot(output: Path, stability: pd.DataFrame, flips: pd.DataFrame, baseline: pd.DataFrame, sensitivity: pd.DataFrame, scenarios: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    figure = output / "figures"
    figure.mkdir(parents=True, exist_ok=True)
    stability.groupby("severity", sort=False)["top_1_retention"].mean().plot(marker="o", figsize=(7, 4), ylim=(0, 1), ylabel="Top-1 retention", title="Counterfactual ranking stability")
    plt.tight_layout(); plt.savefig(figure / "ranking_stability.png", dpi=150); plt.close()
    flips.groupby("perturbation", sort=False)["decision_flip_rate"].mean().plot(kind="bar", figsize=(7, 4), ylim=(0, 1), ylabel="Decision flip rate", title="Decision flips under perturbation")
    plt.tight_layout(); plt.savefig(figure / "decision_flip_rate.png", dpi=150); plt.close()
    baseline.set_index("method")[["mrr", "top_1", "top_3"]].plot(kind="bar", figsize=(9, 4), ylim=(0, 1), ylabel="Metric", title="Ranking comparison")
    plt.tight_layout(); plt.savefig(figure / "baseline_comparison.png", dpi=150); plt.close()
    sensitivity.set_index("weight_profile")[["mrr", "top_1", "top_3"]].plot(kind="bar", figsize=(8, 4), ylim=(0, 1), ylabel="Metric", title="Probability/tactical weight sensitivity")
    plt.tight_layout(); plt.savefig(figure / "weight_sensitivity.png", dpi=150); plt.close()
    if not scenarios.empty:
        scenarios.pivot_table(index="scenario", columns="target_player_id", values="decision_score", aggfunc="mean").plot(kind="bar", figsize=(10, 4), ylabel="Decision score", title="Observed counterfactual scenarios")
    else:
        plt.figure(figsize=(7, 4)); plt.text(0.5, 0.5, "No scenario states", ha="center", va="center"); plt.axis("off")
    plt.tight_layout(); plt.savefig(figure / "counterfactual_scenarios.png", dpi=150); plt.close()


def run(config_path: str = "configs/step83_counterfactual_decision.yaml") -> dict[str, Path]:
    config = load_config(ROOT / config_path)
    tracking, events = load_metrica_subset(config, ROOT)
    records = _state_records(tracking, events, config)
    model = fit_pass_model(pd.DataFrame(columns=FEATURE_NAMES), [], {"minimum_examples_per_class": 10})
    methods = ["nearest_teammate", "most_forward_teammate", "tas_baseline", "step82_fallback", "step83_balanced"]
    weights = config["weights"]
    clean_rankings = {method: [_ranking(record["state"], record["source"], model, config, weights.get(method, weights["step83_balanced"]), method) for record in records] for method in methods}
    output = ROOT / config["output_directory"]
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    final_candidates = []
    for record, ranking in zip(records, clean_rankings["step83_balanced"]):
        frame = ranking.copy(); frame.insert(0, "state_id", record["state_id"]); frame["actual_target"] = record["target"]; final_candidates.append(frame)
    pd.concat(final_candidates, ignore_index=True).to_csv(metrics_dir / "counterfactual_candidates.csv", index=False)
    actual = _actual_table(records, clean_rankings)
    actual.to_csv(metrics_dir / "actual_target_ranking.csv", index=False)
    baseline_rows = [summarize_rankings(rankings, [record["target"] for record in records], method) for method, rankings in clean_rankings.items()]
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(metrics_dir / "baseline_comparison.csv", index=False)
    stability_rows, flip_rows = [], []
    seeds = config["random_seeds"]
    for method in methods:
        for perturbation in config["perturbations"]:
            for severity in config["severities"]:
                comparisons = []
                for index, record in enumerate(records):
                    perturbed = perturb_snapshot(record["state"], perturbation, severity, int(seeds[index % len(seeds)]))
                    ranking = _ranking(perturbed, record["source"], model, config, weights.get(method, weights["step83_balanced"]), method)
                    comparisons.append(compare_rankings(clean_rankings[method][index], ranking, top_k=3))
                valid = [row for row in comparisons if row["common_candidates"]]
                score_values = [row["score_variance"] for row in valid if row["score_variance"] is not None]
                stability_rows.append({"method": method, "perturbation": perturbation, "severity": severity, "n_states": len(valid), "top_1_retention": float(np.mean([row["top_1_retention"] for row in valid])) if valid else 0.0, "top_3_retention": float(np.mean([row["top_k_retention"] for row in valid])) if valid else 0.0, "spearman": float(np.mean([row["spearman"] for row in valid])) if valid else 0.0, "kendall": float(np.mean([row["kendall"] for row in valid])) if valid else 0.0, "average_rank_displacement": float(np.mean([row["mean_rank_displacement"] for row in valid])) if valid else 0.0, "score_variance": float(np.mean(score_values)) if score_values else 0.0})
                flip_rows.append({"method": method, "perturbation": perturbation, "severity": severity, "n_states": len(comparisons), "decision_flip_rate": float(np.mean([row["decision_flip"] for row in comparisons])) if comparisons else 0.0, "changed_top_3_rate": float(1.0 - np.mean([row["top_k_retention"] for row in comparisons])) if comparisons else 0.0})
    stability = pd.DataFrame(stability_rows); flips = pd.DataFrame(flip_rows)
    stability.to_csv(metrics_dir / "ranking_stability.csv", index=False); flips.to_csv(metrics_dir / "decision_flip_rates.csv", index=False)
    sensitivity_rows = []
    for profile in ("probability_heavy", "step83_balanced", "tactical_heavy"):
        ranked = [_ranking(record["state"], record["source"], model, config, weights[profile], profile) for record in records]
        result = summarize_rankings(ranked, [record["target"] for record in records], profile)
        result["weight_profile"] = profile
        reference = clean_rankings["step83_balanced"]
        comparisons = [compare_rankings(reference[index], ranked[index]) for index in range(len(ranked))]
        result.update({"top_1_retention_vs_balanced": float(np.mean([item["top_1_retention"] for item in comparisons])), "spearman_vs_balanced": float(np.mean([item["spearman"] for item in comparisons]))})
        sensitivity_rows.append(result)
    sensitivity = pd.DataFrame(sensitivity_rows); sensitivity.to_csv(metrics_dir / "weight_sensitivity.csv", index=False)
    scenarios = _scenario_rows(records, clean_rankings["step83_balanced"]); scenarios.to_csv(metrics_dir / "scenario_analysis.csv", index=False)
    overall_flips = {method: float(flips.loc[flips["method"] == method, "decision_flip_rate"].mean()) for method in methods}
    summary = {"step": "83", "dataset": config["dataset"], "frames": [config["match"]["start_frame"], config["match"]["end_frame"]], "n_evaluated_states": len(records), "n_identifiable_actual_targets": len(records), "n_candidates": int(sum(len(record["candidates"]) for record in records)), "candidates_per_state": float(np.mean([len(record["candidates"]) for record in records])) if records else 0.0, "perturbation_count": len(config["perturbations"]) * len(config["severities"]), "random_seeds": config["random_seeds"], "methods": methods, "weights": config["weights"], "probability_model": "Step 82 heuristic fallback; no reliable completion labels", "ranking_metrics": baseline.to_dict(orient="records"), "stability": stability.to_dict(orient="records"), "decision_flips": flips.to_dict(orient="records"), "overall_decision_flip_rate": overall_flips, "weight_sensitivity": sensitivity.to_dict(orient="records"), "scenario_count": int(scenarios["scenario"].nunique()) if not scenarios.empty else 0, "anti_leakage": {"ball_coordinates": False, "event_labels_in_scoring": False, "future_frames": False, "outcomes_in_scoring": False}, "limitations": ["15 identifiable actual targets from one 60-second Metrica window", "counterfactual actions are hypothetical and not simulated futures", "no reliable pass-completion labels", "controlled Metrica tracking is not real-broadcast/GSR validation"]}
    (metrics_dir / "step83_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _plot(output, stability, flips, baseline, sensitivity, scenarios)
    return {"summary": metrics_dir / "step83_summary.json", "candidates": metrics_dir / "counterfactual_candidates.csv"}


if __name__ == "__main__":
    print(run())