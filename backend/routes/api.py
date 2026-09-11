from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

try:
    from ..services import data_service as data
except ImportError:
    from services import data_service as data

router = APIRouter(prefix="/api")


def _not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail={"status": "unavailable", "message": str(exc)})


@router.get("/health")
def health():
    available = bool(data.matches())
    return {"application": "ball-free-game-state-reconstruction", "backend_status": "ok",
            "data_availability": "available" if available else "unavailable", "version": "0.1.0"}


@router.get("/matches")
def matches():
    return {"availability": {"status": "available"}, "data": data.matches()}


@router.get("/matches/{match_id}")
def match(match_id: str):
    found = [item for item in data.matches() if item["match_id"] == match_id]
    if not found:
        raise _not_found(KeyError(match_id))
    return {"availability": {"status": "available"}, "data": found[0]}


@router.get("/game-state/{match_id}/{frame_number}")
def game_state(match_id: str, frame_number: int):
    try:
        return data.frame(match_id, frame_number)
    except (IndexError, KeyError, FileNotFoundError) as exc:
        raise _not_found(exc) from exc


@router.get("/players/{match_id}")
def players(match_id: str):
    try:
        rows = data.tracking(match_id)
    except (KeyError, FileNotFoundError) as exc:
        raise _not_found(exc) from exc
    latest = rows.sort_values("frame").groupby(["team", "player_id"], as_index=False).tail(1)
    return {"availability": {"status": "available"}, "data": [{"player_id": f"{r.team.upper()}_{r.player_id}", "team": r.team} for r in latest.itertuples()]}


@router.get("/players/{match_id}/{player_id}/trajectory")
def trajectory(match_id: str, player_id: str, limit: int = Query(250, ge=1, le=1000)):
    try:
        return {"availability": {"status": "available"}, "data": data.player_trajectory(match_id, player_id, limit)}
    except (KeyError, FileNotFoundError) as exc:
        raise _not_found(exc) from exc


@router.get("/possession/{match_id}/{frame_number}")
def possession(match_id: str, frame_number: int):
    try:
        return data.possession(match_id, frame_number)
    except (IndexError, KeyError, FileNotFoundError) as exc:
        raise _not_found(exc) from exc


@router.get("/possession/{match_id}/timeline")
def possession_timeline(match_id: str):
    return {"availability": {"status": "unavailable", "message": "Timeline materialization is not cached; request bounded frames instead."}, "data": []}


@router.get("/events/{match_id}")
def events(match_id: str):
    try:
        return {"availability": {"status": "available"}, "data": data._clean(data.events(match_id).to_dict(orient="records"))}
    except (KeyError, FileNotFoundError) as exc:
        raise _not_found(exc) from exc


@router.get("/events/{match_id}/metrics")
def event_metrics(match_id: str, tolerance: float = Query(1.0, ge=0.2, le=1.0)):
    rows = data.result_rows("step81", "step81_event_metrics.csv")
    return {"availability": {"status": "available" if rows else "unavailable"}, "data": [r for r in rows if float(r.get("tolerance_sec", 0)) == tolerance]}


@router.get("/tactical/{match_id}/{frame_number}")
def tactical(match_id: str, frame_number: int):
    try:
        return data.tactical(match_id, frame_number)
    except (IndexError, KeyError, FileNotFoundError) as exc:
        raise _not_found(exc) from exc


@router.get("/pass-candidates/{match_id}/{frame_number}/{player_id}")
def candidates(match_id: str, frame_number: int, player_id: str):
    try:
        return {"availability": {"status": "available"}, "data": data.pass_candidates(match_id, frame_number, player_id)}
    except (IndexError, KeyError, FileNotFoundError, ValueError) as exc:
        raise _not_found(exc) from exc


@router.get("/pass-ranking/{match_id}/{frame_number}/{player_id}")
def ranking(match_id: str, frame_number: int, player_id: str):
    try:
        ranked = data.pass_ranking(match_id, frame_number, player_id)
    except (IndexError, KeyError, FileNotFoundError, ValueError) as exc:
        raise _not_found(exc) from exc
    return {"availability": {"status": "available", "message": "Current-state heuristic fallback; not a completion model.", "provenance": {"method": "Step 82 heuristic fallback", "frame": frame_number}}, "data": ranked}


@router.get("/robustness/summary")
def robustness_summary():
    rows = data.result_rows("step80", "step80_summary.json")
    return {"availability": {"status": "available"}, "data": rows}


@router.get("/robustness/{experiment}")
def robustness(experiment: str):
    mapping = {"step80": ("step80", "step80_ablation.csv"), "step81": ("step81", "step81_event_metrics.csv"), "step82": ("step82", "ranking_metrics.csv"), "step83": ("step83", "ranking_stability.csv")}
    if experiment not in mapping:
        return {"availability": {"status": "unavailable", "message": "Unknown experiment."}, "data": []}
    step, filename = mapping[experiment]
    return {"availability": {"status": "available"}, "data": data.result_rows(step, filename)}


@router.get("/gsr/status")
def gsr_status():
    return {"status": "blocked", "message": "Authorized SoccerNet-GSR data, labels, complete weights, and a verifiable GPU runtime are unavailable.", "results": None, "provenance": None}