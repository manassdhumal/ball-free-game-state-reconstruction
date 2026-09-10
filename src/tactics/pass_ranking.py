"""Counterfactual pass ranking and label-safe ranking metrics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.tactics.pass_candidates import generate_pass_candidates
from src.tactics.pass_probability import PassProbabilityModel, predict_pass_probability


DEFAULT_WEIGHTS = {"success": 0.55, "progress": 0.20, "space": 0.15, "support": 0.10, "pressure": 0.10, "distance": 0.05}


def rank_counterfactual_passes(state: pd.DataFrame, model: PassProbabilityModel,
                                source_id: str, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    candidates = generate_pass_candidates(state, source_id, config)
    if candidates.empty:
        candidates["pass_completion_probability"] = pd.Series(dtype=float)
        candidates["tactical_value"] = pd.Series(dtype=float)
        candidates["decision_score"] = pd.Series(dtype=float)
        return candidates
    candidates = candidates.copy()
    candidates["pass_completion_probability"] = predict_pass_probability(model, candidates)
    weights = {**DEFAULT_WEIGHTS, **dict((config or {}).get("ranking_weights", {}))}
    candidates["tactical_value"] = (
        weights["progress"] * np.clip(candidates["forward_progress"] / 0.35, -1, 1)
        + weights["space"] * np.clip(candidates["target_space"] / 0.30, 0, 1)
        + weights["support"] * np.clip(candidates["support_count"] / 4, 0, 1)
        - weights["pressure"] * np.clip(1 - candidates["target_pressure"] / 0.30, 0, 1)
        - weights["distance"] * np.clip(candidates["pass_distance"] / 0.70, 0, 1)
    )
    candidates["decision_score"] = weights["success"] * candidates["pass_completion_probability"] + candidates["tactical_value"]
    return candidates.sort_values(["decision_score", "target_player_id"], ascending=[False, True]).reset_index(drop=True)


def evaluate_rankings(ranked_rows: list[pd.DataFrame], actual_targets: list[str], top_k: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    rows = []
    reciprocal, ndcg = [], []
    for ranking, target in zip(ranked_rows, actual_targets):
        ids = ranking["target_player_id"].tolist()
        if target not in ids:
            continue
        rank = ids.index(target) + 1
        reciprocal.append(1.0 / rank)
        ndcg.append(1.0 / np.log2(rank + 1))
        for k in top_k:
            rows.append({"k": k, "top_k_inclusion": float(rank <= k), "rank": rank})
    if not reciprocal:
        return pd.DataFrame([{"n_ranked_states": 0, "mrr": 0.0, "ndcg": 0.0, "top_1_agreement": 0.0, "top_3_inclusion": 0.0}])
    return pd.DataFrame([{
        "n_ranked_states": len(reciprocal), "mrr": float(np.mean(reciprocal)), "ndcg": float(np.mean(ndcg)),
        "top_1_agreement": float(np.mean([r["top_k_inclusion"] for r in rows if r["k"] == 1])),
        "top_3_inclusion": float(np.mean([r["top_k_inclusion"] for r in rows if r["k"] == 3])),
    }])