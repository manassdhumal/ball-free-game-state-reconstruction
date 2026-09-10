"""Counterfactual ranking analysis built on the Step 82 scorer."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.tactics.pass_probability import PassProbabilityModel
from src.tactics.pass_ranking import rank_counterfactual_passes


def rank_with_positions(ranking: pd.DataFrame) -> pd.DataFrame:
    """Return a ranking with deterministic one-based positions."""
    result = ranking.copy().reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=int))
    return result


def rank_counterfactual_actions(
    state: pd.DataFrame,
    model: PassProbabilityModel,
    source_id: str,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Score and rank hypothetical actions from the current snapshot only."""
    return rank_with_positions(rank_counterfactual_passes(state, model, source_id, config))


def actual_target_rank(ranking: pd.DataFrame, target_id: str) -> dict[str, float | str | None]:
    """Compare an observed receiver with a ranking without calling it optimal."""
    ids = ranking.get("target_player_id", pd.Series(dtype=str)).tolist()
    if target_id not in ids:
        return {"actual_target": target_id, "rank": None, "percentile": None, "included": False}
    rank = ids.index(target_id) + 1
    percentile = 1.0 - (rank - 1) / max(len(ids), 1)
    return {"actual_target": target_id, "rank": int(rank), "percentile": float(percentile), "included": True}


def ranking_metrics(ranking: pd.DataFrame, target_id: str, top_k: Sequence[int] = (1, 3)) -> dict[str, float | int | None]:
    comparison = actual_target_rank(ranking, target_id)
    rank = comparison["rank"]
    return {
        "rank": rank,
        "percentile": comparison["percentile"],
        "top_1": float(rank == 1) if rank is not None else 0.0,
        "top_3": float(rank <= 3) if rank is not None else 0.0,
        "mrr": float(1.0 / rank) if rank is not None else 0.0,
        "n_candidates": int(len(ranking)),
    }


def _rank_map(ranking: pd.DataFrame) -> dict[str, int]:
    return {str(target): index + 1 for index, target in enumerate(ranking["target_player_id"].tolist())}


def compare_rankings(clean: pd.DataFrame, perturbed: pd.DataFrame, top_k: int = 3) -> dict[str, float | int | None]:
    """Compare two rankings over their common candidate targets."""
    clean_map, perturbed_map = _rank_map(clean), _rank_map(perturbed)
    common = sorted(set(clean_map) & set(perturbed_map))
    if not common:
        return {"common_candidates": 0, "top_1_retention": 0.0, "top_k_retention": 0.0, "spearman": None, "kendall": None, "mean_rank_displacement": None, "score_variance": None, "decision_flip": 1.0}
    clean_top = clean["target_player_id"].iloc[0] if len(clean) else None
    perturbed_top = perturbed["target_player_id"].iloc[0] if len(perturbed) else None
    clean_top_k = set(clean["target_player_id"].head(top_k))
    perturbed_top_k = set(perturbed["target_player_id"].head(top_k))
    x = np.asarray([clean_map[target] for target in common], dtype=float)
    y = np.asarray([perturbed_map[target] for target in common], dtype=float)
    if len(common) > 1:
        x_rank = pd.Series(x).rank(method="average").to_numpy()
        y_rank = pd.Series(y).rank(method="average").to_numpy()
        denominator = float(np.sqrt(np.sum((x_rank - x_rank.mean()) ** 2) * np.sum((y_rank - y_rank.mean()) ** 2)))
        spearman = float(np.sum((x_rank - x_rank.mean()) * (y_rank - y_rank.mean())) / denominator) if denominator else 1.0
        concordant = discordant = 0
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                delta_x, delta_y = x[i] - x[j], y[i] - y[j]
                if delta_x * delta_y > 0:
                    concordant += 1
                elif delta_x * delta_y < 0:
                    discordant += 1
        total_pairs = concordant + discordant
        kendall = float((concordant - discordant) / total_pairs) if total_pairs else 1.0
    else:
        spearman = kendall = 1.0
    score_variance = None
    if "decision_score" in perturbed.columns and common:
        score_variance = float(np.var(perturbed.set_index("target_player_id").loc[common, "decision_score"].to_numpy(float)))
    return {
        "common_candidates": int(len(common)),
        "top_1_retention": float(clean_top == perturbed_top),
        "top_k_retention": float(bool(clean_top_k & perturbed_top_k)),
        "spearman": spearman,
        "kendall": kendall,
        "mean_rank_displacement": float(np.mean(np.abs(x - y))),
        "score_variance": score_variance,
        "decision_flip": float(clean_top != perturbed_top),
    }


def perturb_snapshot(state: pd.DataFrame, perturbation: str, severity: str, seed: int) -> pd.DataFrame:
    """Apply bounded current-frame perturbations; future rows are never read."""
    scales = {"clean": 0.0, "mild": 0.001, "moderate": 0.003, "severe": 0.008}
    dropout = {"clean": 0.0, "mild": 0.02, "moderate": 0.08, "severe": 0.20}
    if severity not in scales:
        raise ValueError(f"Unknown severity: {severity}")
    result = state.copy(deep=True)
    rng = np.random.default_rng(seed)
    if perturbation == "coordinate_jitter":
        noise = rng.normal(0.0, scales[severity], size=(len(result), 2))
        visible = result["visible"].astype(bool).to_numpy()
        result.loc[visible, ["x", "y"]] = np.clip(result.loc[visible, ["x", "y"]].to_numpy(float) + noise[visible], 0.0, 1.0)
    elif perturbation == "player_dropout":
        candidates = result.index[result["visible"].astype(bool)].to_numpy()
        count = int(round(len(candidates) * dropout[severity]))
        if count:
            dropped = rng.choice(candidates, size=count, replace=False)
            result.loc[dropped, "visible"] = False
    elif perturbation == "availability_jitter":
        result = perturb_snapshot(result, "coordinate_jitter", severity, seed)
        candidates = result.index[result["visible"].astype(bool)].to_numpy()
        count = int(round(len(candidates) * dropout[severity] / 2.0))
        if count:
            dropped = rng.choice(candidates, size=count, replace=False)
            result.loc[dropped, "visible"] = False
    elif perturbation != "clean":
        raise ValueError(f"Unknown perturbation: {perturbation}")
    return result


def summarize_rankings(rankings: list[pd.DataFrame], targets: list[str], method: str) -> dict[str, float | int | str]:
    metrics = [ranking_metrics(ranking, target) for ranking, target in zip(rankings, targets)]
    valid = [metric for metric in metrics if metric["rank"] is not None]
    if not valid:
        return {"method": method, "n_states": 0, "mrr": 0.0, "top_1": 0.0, "top_3": 0.0, "mean_rank": 0.0}
    return {"method": method, "n_states": len(valid), "mrr": float(np.mean([m["mrr"] for m in valid])), "top_1": float(np.mean([m["top_1"] for m in valid])), "top_3": float(np.mean([m["top_3"] for m in valid])), "mean_rank": float(np.mean([m["rank"] for m in valid]))}