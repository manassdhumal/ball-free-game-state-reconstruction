from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.metrica_event_parser import load_metrica_events
from src.data.metrica_parser import load_metrica_match
from src.possession.possession_baseline import score_frame
from src.possession.spatial_graph import build_frame_graph
from src.tactics.pass_candidates import generate_pass_candidates
from src.tactics.pass_probability import PassProbabilityModel
from src.tactics.pass_ranking import rank_counterfactual_passes

METRICA = ROOT / "data" / "raw" / "metrica" / "data"
RESULTS = ROOT / "results"
FPS = 25.0


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if pd.isna(value) if not isinstance(value, (tuple, set)) else False:
        return None
    return value


def _paths(match_id: str) -> tuple[Path, Path, Path]:
    if not match_id.startswith("metrica_game"):
        raise KeyError(match_id)
    number = match_id.removeprefix("metrica_game")
    directory = METRICA / f"Sample_Game_{number}"
    return (
        directory / f"Sample_Game_{number}_RawTrackingData_Home_Team.csv",
        directory / f"Sample_Game_{number}_RawTrackingData_Away_Team.csv",
        directory / f"Sample_Game_{number}_RawEventsData.csv",
    )


@lru_cache(maxsize=2)
def tracking(match_id: str) -> pd.DataFrame:
    home, away, _ = _paths(match_id)
    players, _ball = load_metrica_match(str(home), str(away), match_id)
    return players


@lru_cache(maxsize=2)
def events(match_id: str) -> pd.DataFrame:
    _home, _away, event_path = _paths(match_id)
    return load_metrica_events(str(event_path), match_id)


def matches() -> list[dict[str, Any]]:
    result = []
    for match_id in ("metrica_game1", "metrica_game2"):
        try:
            frame_count = int(tracking(match_id)["frame"].max())
        except (FileNotFoundError, KeyError):
            continue
        result.append({
            "match_id": match_id,
            "label": match_id.replace("_", " ").title(),
            "fps": FPS,
            "frame_count": frame_count,
            "events_available": _paths(match_id)[2].exists(),
            "tracking_available": True,
        })
    return result


def frame(match_id: str, frame_number: int) -> dict[str, Any]:
    df = tracking(match_id)
    rows = df[df["frame"] == frame_number]
    if rows.empty:
        raise IndexError(frame_number)
    players = []
    for row in rows.itertuples(index=False):
        players.append({
            "player_id": f"{str(row.team).upper()}_{row.player_id}",
            "team": row.team,
            "x": None if pd.isna(row.x) else float(row.x),
            "y": None if pd.isna(row.y) else float(row.y),
            "confidence": float(row.confidence),
            "visible": bool(row.visible),
        })
    timestamp = float(rows["timestamp"].iloc[0])
    return {"match_id": match_id, "frame": frame_number, "timestamp": timestamp,
            "fps": FPS, "players": players,
            "provenance": {"dataset": "Metrica Sample Game 1" if match_id.endswith("1") else "Metrica Sample Game 2",
                           "sequence": match_id, "frame_range": str(frame_number), "fps": FPS}}


def player_trajectory(match_id: str, player_id: str, limit: int = 250) -> list[dict[str, Any]]:
    df = tracking(match_id)
    raw_id = player_id.split("_", 1)[-1]
    rows = df[df["player_id"].astype(str) == raw_id]
    if "_" in player_id:
        team = player_id.split("_", 1)[0].lower()
        rows = rows[rows["team"] == team]
    rows = rows.head(max(1, min(limit, 1000)))
    return [{"frame": int(r.frame), "timestamp": float(r.timestamp), "x": None if pd.isna(r.x) else float(r.x),
             "y": None if pd.isna(r.y) else float(r.y), "visible": bool(r.visible)} for r in rows.itertuples()]


def possession(match_id: str, frame_number: int) -> dict[str, Any]:
    state = tracking(match_id)
    rows = state[state["frame"] == frame_number]
    if rows.empty:
        raise IndexError(frame_number)
    graph = build_frame_graph(rows, frame_number, float(rows["timestamp"].iloc[0]))
    scores = score_frame(graph)
    if scores.empty:
        return {"status": "unavailable", "message": "No visible players at this frame."}
    top = scores.iloc[0]
    return {"status": "available", "possessor_id": f"{str(top.team).upper()}_{top.player_id}",
            "team": str(top.team), "score": float(top.raw_score), "state": "inferred",
            "provenance": {"dataset": match_id, "frame_range": str(frame_number), "method": "ball-free heuristic"}}


def tactical(match_id: str, frame_number: int) -> dict[str, Any]:
    from src.tactics.tactical_features import extract_tactical_features
    state = tracking(match_id)
    rows = state[state["frame"] == frame_number]
    if rows.empty:
        raise IndexError(frame_number)
    graph = build_frame_graph(rows, frame_number, float(rows["timestamp"].iloc[0]))
    return {"status": "available", "features": _clean(extract_tactical_features(graph)),
            "provenance": {"dataset": match_id, "frame_range": str(frame_number), "method": "player-only spatial graph"}}


def pass_candidates(match_id: str, frame_number: int, player_id: str) -> list[dict[str, Any]]:
    state = tracking(match_id)
    rows = state[state["frame"] == frame_number].copy()
    raw_id = player_id.split("_", 1)[-1]
    candidates = generate_pass_candidates(rows, raw_id)
    if candidates.empty:
        return []
    return _clean(candidates.to_dict(orient="records"))


def pass_ranking(match_id: str, frame_number: int, player_id: str) -> list[dict[str, Any]]:
    state = tracking(match_id)
    rows = state[state["frame"] == frame_number].copy()
    raw_id = player_id.split("_", 1)[-1]
    ranking = rank_counterfactual_passes(rows, PassProbabilityModel(), raw_id)
    if ranking.empty:
        return []
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return _clean(ranking.to_dict(orient="records"))


def result_rows(step: str, filename: str) -> list[dict[str, Any]]:
    path = RESULTS / step / "metrics" / filename
    if not path.exists():
        return []
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return _clean(value if isinstance(value, list) else [value])
    return _clean(pd.read_csv(path).to_dict(orient="records"))
