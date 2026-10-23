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
    interim_csv = ROOT / "data" / "interim" / f"{match_id}_canonical_tracking.csv"
    if interim_csv.exists() or match_id.startswith("uploaded_") or match_id.startswith("video_"):
        return interim_csv, interim_csv, interim_csv
    if not match_id.startswith("metrica_game"):
        raise KeyError(match_id)
    number = match_id.removeprefix("metrica_game")
    directory = METRICA / f"Sample_Game_{number}"
    return (
        directory / f"Sample_Game_{number}_RawTrackingData_Home_Team.csv",
        directory / f"Sample_Game_{number}_RawTrackingData_Away_Team.csv",
        directory / f"Sample_Game_{number}_RawEventsData.csv",
    )


def _get_frame_rows(df: pd.DataFrame, frame_number: int) -> tuple[pd.DataFrame, int]:
    if df.empty:
        raise IndexError(frame_number)
    rows = df[df["frame"] == frame_number]
    if not rows.empty:
        return rows, frame_number
    min_f = int(df["frame"].min())
    max_f = int(df["frame"].max())
    if frame_number < min_f - 25 or frame_number > max_f + 25:
        raise IndexError(frame_number)
    import numpy as np
    avail = df["frame"].unique()
    closest_f = int(avail[np.abs(avail - frame_number).argmin()])
    return df[df["frame"] == closest_f], closest_f


def _infer_video_events(match_id: str) -> pd.DataFrame:
    try:
        df = tracking(match_id)
        if df.empty:
            return pd.DataFrame(columns=["match_id", "frame", "timestamp", "type", "team", "player_id"])
        unique_frames = sorted(df["frame"].unique())
        fps = FPS
        events_list = []
        prev_possessor = None
        prev_team = None

        for f in unique_frames:
            f_rows = df[df["frame"] == f]
            if len(f_rows) < 4:
                continue
            f_time = float(f / fps)
            graph = build_frame_graph(f_rows, int(f), f_time)
            scores = score_frame(graph)
            if not scores.empty:
                top = scores.iloc[0]
                cur_possessor = f"{str(top.team).upper()}_{top.player_id}"
                cur_team = str(top.team)
                if prev_possessor is None:
                    prev_possessor = cur_possessor
                    prev_team = cur_team
                    events_list.append({
                        "match_id": match_id,
                        "event_id": len(events_list) + 1,
                        "frame": int(f),
                        "start_frame": int(f),
                        "end_frame": int(f),
                        "timestamp": f_time,
                        "start_timestamp": f_time,
                        "end_timestamp": f_time,
                        "type": "RECOVERY",
                        "event_type": "RECOVERY",
                        "team": cur_team,
                        "player_id": cur_possessor,
                    })
                elif cur_team != prev_team:
                    events_list.append({
                        "match_id": match_id,
                        "event_id": len(events_list) + 1,
                        "frame": int(f),
                        "start_frame": int(f),
                        "end_frame": int(f),
                        "timestamp": f_time,
                        "start_timestamp": f_time,
                        "end_timestamp": f_time,
                        "type": "BALL LOST",
                        "event_type": "BALL LOST",
                        "team": prev_team,
                        "player_id": prev_possessor,
                    })
                    events_list.append({
                        "match_id": match_id,
                        "event_id": len(events_list) + 1,
                        "frame": int(f),
                        "start_frame": int(f),
                        "end_frame": int(f),
                        "timestamp": f_time,
                        "start_timestamp": f_time,
                        "end_timestamp": f_time,
                        "type": "RECOVERY",
                        "event_type": "RECOVERY",
                        "team": cur_team,
                        "player_id": cur_possessor,
                    })
                    prev_possessor = cur_possessor
                    prev_team = cur_team
                elif cur_possessor != prev_possessor:
                    events_list.append({
                        "match_id": match_id,
                        "event_id": len(events_list) + 1,
                        "frame": int(f),
                        "start_frame": int(f),
                        "end_frame": int(f),
                        "timestamp": f_time,
                        "start_timestamp": f_time,
                        "end_timestamp": f_time,
                        "type": "PASS",
                        "event_type": "PASS",
                        "team": cur_team,
                        "player_id": cur_possessor,
                    })
                    prev_possessor = cur_possessor
        if events_list:
            return pd.DataFrame(events_list)
    except Exception:
        pass
    return pd.DataFrame(columns=["match_id", "event_id", "frame", "start_frame", "timestamp", "start_timestamp", "type", "event_type", "team", "player_id"])


@lru_cache(maxsize=16)
def tracking(match_id: str) -> pd.DataFrame:
    interim_csv = ROOT / "data" / "interim" / f"{match_id}_canonical_tracking.csv"
    if interim_csv.exists() or match_id.startswith("uploaded_") or match_id.startswith("video_"):
        if not interim_csv.exists():
            raise FileNotFoundError(f"Uploaded tracking file not found: {interim_csv}")
        return pd.read_csv(interim_csv)
    home, away, _ = _paths(match_id)
    players, _ball = load_metrica_match(str(home), str(away), match_id)
    return players


@lru_cache(maxsize=16)
def events(match_id: str) -> pd.DataFrame:
    if match_id.startswith("uploaded_") or match_id.startswith("video_"):
        return _infer_video_events(match_id)
    _home, _away, event_path = _paths(match_id)
    return load_metrica_events(str(event_path), match_id)


METRICA_FRAME_COUNTS = {
    "metrica_game1": 145006,
    "metrica_game2": 141156,
}


def matches() -> list[dict[str, Any]]:
    import json
    result = []
    seen_ids = set()
    for match_id in ("metrica_game1", "metrica_game2"):
        try:
            home_path, _, event_path = _paths(match_id)
            if not home_path.exists():
                continue
            frame_count = METRICA_FRAME_COUNTS.get(match_id, 145000)
            seen_ids.add(match_id)
            result.append({
                "match_id": match_id,
                "label": match_id.replace("_", " ").title(),
                "fps": FPS,
                "frame_count": frame_count,
                "events_available": event_path.exists(),
                "tracking_available": True,
            })
        except (FileNotFoundError, KeyError):
            continue

    registry_file = ROOT / "data" / "interim" / "uploads" / "video_registry.json"
    registry: dict[str, Any] = {}
    if registry_file.exists():
        try:
            registry = json.loads(registry_file.read_text(encoding="utf-8"))
        except Exception:
            registry = {}

    # Dynamically include uploaded video gameplay matches
    interim_dir = ROOT / "data" / "interim"
    if interim_dir.exists():
        for csv_file in sorted(interim_dir.glob("*_canonical_tracking.csv"), key=lambda f: f.stat().st_mtime, reverse=True):
            m_id = csv_file.name.replace("_canonical_tracking.csv", "")
            if m_id in seen_ids:
                continue
            seen_ids.add(m_id)
            try:
                v_id = m_id.replace("uploaded_", "")
                summary_file = interim_dir / f"{m_id}_summary.json"
                frame_count = 120
                if summary_file.exists():
                    try:
                        s_data = json.loads(summary_file.read_text(encoding="utf-8"))
                        frame_count = int(s_data.get("frame_count") or s_data.get("max_frame") or 120)
                    except Exception:
                        frame_count = 120
                elif registry.get(v_id, {}).get("total_frames"):
                    frame_count = int(registry[v_id]["total_frames"])
                
                friendly_name = registry.get(v_id, {}).get("filename") if registry else None
                label_text = f"Video: {friendly_name}" if friendly_name else f"Video: {v_id.replace('_', ' ').title()}"
                result.append({
                    "match_id": m_id,
                    "label": label_text,
                    "fps": FPS,
                    "frame_count": frame_count,
                    "events_available": True,
                    "tracking_available": True,
                    "is_video": True,
                })
            except Exception:
                continue
    return result


def frame(match_id: str, frame_number: int) -> dict[str, Any]:
    df = tracking(match_id)
    rows, resolved_f = _get_frame_rows(df, frame_number)
    players = []
    for row in rows.itertuples(index=False):
        pid = str(row.player_id)
        if not pid.startswith(f"{str(row.team).upper()}_"):
            pid = f"{str(row.team).upper()}_{pid}"
        conf = getattr(row, "confidence", 1.0)
        conf_val = 1.0 if pd.isna(conf) else float(conf)
        jersey_color = getattr(row, "jersey_color", None)
        players.append({
            "player_id": pid,
            "team": row.team,
            "x": None if pd.isna(row.x) else float(row.x),
            "y": None if pd.isna(row.y) else float(row.y),
            "confidence": conf_val,
            "visible": bool(getattr(row, "visible", True)),
            "jersey_color": None if pd.isna(jersey_color) else str(jersey_color),
        })
    timestamp = float(rows["timestamp"].iloc[0]) if "timestamp" in rows else float(resolved_f / FPS)
    is_video = match_id.startswith("uploaded_") or match_id.startswith("video_")
    return {"match_id": match_id, "frame": resolved_f, "timestamp": timestamp,
            "fps": FPS, "players": players,
            "provenance": {"dataset": "Uploaded Video Perception" if is_video else ("Metrica Sample Game 1" if match_id.endswith("1") else "Metrica Sample Game 2"),
                           "sequence": match_id, "frame_range": str(resolved_f), "fps": FPS}}


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
    rows, resolved_f = _get_frame_rows(state, frame_number)
    timestamp = float(rows["timestamp"].iloc[0]) if "timestamp" in rows else float(resolved_f / FPS)
    graph = build_frame_graph(rows, resolved_f, timestamp)
    scores = score_frame(graph)
    if scores.empty:
        return {"status": "unavailable", "message": "No visible players at this frame."}
    top = scores.iloc[0]
    return {"status": "available", "possessor_id": f"{str(top.team).upper()}_{top.player_id}",
            "team": str(top.team), "score": float(top.raw_score), "state": "inferred",
            "provenance": {"dataset": match_id, "frame_range": str(resolved_f), "method": "ball-free heuristic"}}


def tactical(match_id: str, frame_number: int) -> dict[str, Any]:
    from src.tactics.tactical_features import extract_tactical_features
    state = tracking(match_id)
    rows, resolved_f = _get_frame_rows(state, frame_number)
    timestamp = float(rows["timestamp"].iloc[0]) if "timestamp" in rows else float(resolved_f / FPS)
    graph = build_frame_graph(rows, resolved_f, timestamp)
    return {"status": "available", "features": _clean(extract_tactical_features(graph)),
            "provenance": {"dataset": match_id, "frame_range": str(resolved_f), "method": "player-only spatial graph"}}


def pass_candidates(match_id: str, frame_number: int, player_id: str) -> list[dict[str, Any]]:
    state = tracking(match_id)
    rows, _ = _get_frame_rows(state, frame_number)
    rows = rows.copy()
    raw_id = player_id.split("_", 1)[-1]
    candidates = generate_pass_candidates(rows, raw_id)
    if candidates.empty:
        return []
    return _clean(candidates.to_dict(orient="records"))


def pass_ranking(match_id: str, frame_number: int, player_id: str) -> list[dict[str, Any]]:
    state = tracking(match_id)
    rows, _ = _get_frame_rows(state, frame_number)
    rows = rows.copy()
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
