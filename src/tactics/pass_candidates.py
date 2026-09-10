"""Player-only hypothetical pass candidates and interpretable features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "pass_distance", "forward_progress", "pass_angle", "source_pressure",
    "target_pressure", "target_space", "support_count", "line_obstruction",
    "team_width", "team_length",
)


@dataclass(frozen=True)
class CandidateConfig:
    max_distance: float = 0.70
    min_distance: float = 0.02
    support_radius: float = 0.15
    obstruction_radius: float = 0.035
    max_obstruction_fraction: float = 0.65


def _cfg(config: Mapping[str, Any] | None) -> CandidateConfig:
    values = dict(config or {})
    return CandidateConfig(**{k: values[k] for k in CandidateConfig.__annotations__ if k in values})


def _team_rows(state: pd.DataFrame, team: str) -> pd.DataFrame:
    return state[(state["team"] == team) & state["visible"].astype(bool)].copy()


def _pressure(row: pd.Series, opponents: pd.DataFrame) -> float:
    if opponents.empty:
        return 1.0
    return float(np.sqrt(((opponents[["x", "y"]].to_numpy(float) - row[["x", "y"]].to_numpy(float)) ** 2).sum(axis=1)).min())


def build_pass_features(state: pd.DataFrame, source_id: str, target_id: str,
                        config: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Build a feature vector from one current player-only snapshot."""
    required = {"player_id", "team", "x", "y", "visible"}
    missing = required - set(state.columns)
    if missing:
        raise ValueError(f"state is missing required columns: {sorted(missing)}")
    visible = state[state["visible"].astype(bool)].copy()
    source = visible[visible["player_id"] == source_id]
    target = visible[visible["player_id"] == target_id]
    if source.empty or target.empty:
        raise ValueError("source_id and target_id must be visible in the snapshot")
    source_row, target_row = source.iloc[0], target.iloc[0]
    if source_row["team"] != target_row["team"] or source_id == target_id:
        raise ValueError("source and target must be distinct teammates")
    teammates = visible[(visible["team"] == source_row["team"]) & (visible["player_id"] != source_id)]
    opponents = visible[visible["team"] != source_row["team"]]
    vector = target_row[["x", "y"]].to_numpy(float) - source_row[["x", "y"]].to_numpy(float)
    distance = float(np.linalg.norm(vector))
    attacks_x1 = bool(config.get("home_attacks_x1", True)) if config else True
    direction = 1.0 if (source_row["team"] == "home") == attacks_x1 else -1.0
    forward = float(direction * vector[0])
    angle = float(np.arctan2(vector[1], direction * vector[0]))
    segment = target_row[["x", "y"]].to_numpy(float) - source_row[["x", "y"]].to_numpy(float)
    denom = float(np.dot(segment, segment)) or 1.0
    opp_xy = opponents[["x", "y"]].to_numpy(float)
    source_xy = source_row[["x", "y"]].to_numpy(float)
    projection = np.clip(((opp_xy - source_xy) @ segment) / denom, 0.0, 1.0)
    nearest = source_xy + projection[:, None] * segment if len(opp_xy) else np.empty((0, 2))
    line_dist = np.linalg.norm(opp_xy - nearest, axis=1) if len(opp_xy) else np.array([])
    obstruction = float(np.sum((line_dist <= _cfg(config).obstruction_radius) & (projection > 0.05) & (projection < 0.95)))
    support_dist = np.linalg.norm(teammates[["x", "y"]].to_numpy(float) - target_row[["x", "y"]].to_numpy(float), axis=1)
    return {
        "pass_distance": distance,
        "forward_progress": forward,
        "pass_angle": angle,
        "source_pressure": _pressure(source_row, opponents),
        "target_pressure": _pressure(target_row, opponents),
        "target_space": _pressure(target_row, opponents),
        "support_count": float(np.sum((support_dist <= _cfg(config).support_radius) & (support_dist > 0))),
        "line_obstruction": obstruction,
        "team_width": float(teammates["y"].max() - teammates["y"].min()) if len(teammates) else 0.0,
        "team_length": float(teammates["x"].max() - teammates["x"].min()) if len(teammates) else 0.0,
    }


def generate_pass_candidates(state: pd.DataFrame, source_id: str | None = None,
                             config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Generate hypothetical same-team pass options from one snapshot."""
    cfg = _cfg(config)
    visible = state[state["visible"].astype(bool)].copy()
    sources = [source_id] if source_id is not None else visible["player_id"].tolist()
    rows = []
    for source in sources:
        source_rows = visible[visible["player_id"] == source]
        if source_rows.empty:
            continue
        team = source_rows.iloc[0]["team"]
        for target in visible[(visible["team"] == team) & (visible["player_id"] != source)]["player_id"]:
            features = build_pass_features(state, source, target, config)
            if cfg.min_distance <= features["pass_distance"] <= cfg.max_distance:
                row = {"source_player_id": source, "target_player_id": target, **features}
                if "frame" in state.columns:
                    row["frame"] = int(state["frame"].iloc[0])
                if "timestamp" in state.columns:
                    row["timestamp"] = float(state["timestamp"].iloc[0])
                for player_id in (source, target):
                    player = visible[visible["player_id"] == player_id].iloc[0]
                    row[f"{ 'source' if player_id == source else 'target'}_x"] = float(player["x"])
                    row[f"{ 'source' if player_id == source else 'target'}_y"] = float(player["y"])
                rows.append(row)
    columns = ["frame", "timestamp", "source_player_id", "target_player_id", "source_x", "source_y", "target_x", "target_y", *FEATURE_NAMES]
    return pd.DataFrame(rows).reindex(columns=columns)