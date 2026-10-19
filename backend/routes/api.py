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
    return {
        "status": "ready",
        "message": "AI Video Game State Reconstruction perception engine is ready. Upload a gameplay video to extract canonical player tracking and tactical insights.",
        "results": {"supported_formats": [".mp4", ".mov", ".avi"], "model": "YOLOv8n + ByteTrack"},
        "provenance": {"detection": "YOLOv8n", "tracking": "ByteTrack", "homography": "Perspective 2D"}
    }


# ---------------------------------------------------------------------------
# Video Upload & Processing Endpoints
# ---------------------------------------------------------------------------
import os
import shutil
import uuid
from pathlib import Path
from pydantic import BaseModel
from fastapi import File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from src.vision.pipeline import VideoAnalysisPipeline

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "interim" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory progress tracking
VIDEO_PROGRESS: dict[str, dict[str, Any]] = {}


class ProcessVideoRequest(BaseModel):
    video_id: str
    preset: str = "main_broadcast"
    max_frames: int = 150
    frame_stride: int = 1
    apply_mitigation: bool = True
    custom_corners: Optional[list[list[float]]] = None


@router.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    
    video_id = f"video_{uuid.uuid4().hex[:8]}"
    file_ext = Path(file.filename).suffix or ".mp4"
    dest_path = UPLOAD_DIR / f"{video_id}{file_ext}"

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    size = os.path.getsize(dest_path)
    VIDEO_PROGRESS[video_id] = {"progress": 0.0, "status": "Uploaded, ready for analysis"}

    return {
        "status": "uploaded",
        "video_id": video_id,
        "filename": file.filename,
        "size_bytes": size,
        "match_id": f"uploaded_{video_id}",
    }


@router.post("/video/sample-demo")
def sample_demo():
    """Immediately processes the bundled sample football clip."""
    sample_file = Path(__file__).resolve().parents[2] / "data" / "samples" / "sample_match_clip.mp4"
    if not sample_file.exists():
        raise HTTPException(status_code=404, detail="Sample match video not found on disk.")

    demo_id = "sample_demo_clip"
    dest_video = UPLOAD_DIR / f"{demo_id}.mp4"
    if not dest_video.exists():
        shutil.copyfile(sample_file, dest_video)

    match_id = f"uploaded_{demo_id}"
    VIDEO_PROGRESS[demo_id] = {"progress": 0.1, "status": "Processing bundled match clip"}

    def progress_callback(ratio: float, msg: str):
        VIDEO_PROGRESS[demo_id] = {"progress": round(ratio, 2), "status": msg}

    pipeline = VideoAnalysisPipeline(homography_preset="main_broadcast")
    result = pipeline.analyze_video(
        video_path=str(dest_video),
        match_id=match_id,
        max_frames=120,
        frame_stride=1,
        apply_mitigation=True,
        progress_callback=progress_callback,
    )

    data.tracking.cache_clear()
    return {"status": "success", "video_id": demo_id, "match_id": match_id, "data": result}


@router.post("/video/process")
def process_video(req: ProcessVideoRequest):
    # Locate video file
    matching = list(UPLOAD_DIR.glob(f"{req.video_id}.*"))
    if not matching:
        raise HTTPException(status_code=404, detail=f"Video ID not found: {req.video_id}")
    video_path = matching[0]

    match_id = f"uploaded_{req.video_id}"
    VIDEO_PROGRESS[req.video_id] = {"progress": 0.05, "status": "Initializing tracking"}

    def progress_callback(ratio: float, msg: str):
        VIDEO_PROGRESS[req.video_id] = {"progress": round(ratio, 2), "status": msg}

    from src.vision.homography import PitchHomography
    pipeline = VideoAnalysisPipeline(homography_preset=req.preset)
    if req.custom_corners and len(req.custom_corners) == 4:
        pipeline.homography = PitchHomography.from_four_points(req.custom_corners)

    result = pipeline.analyze_video(
        video_path=str(video_path),
        match_id=match_id,
        max_frames=req.max_frames,
        frame_stride=req.frame_stride,
        apply_mitigation=req.apply_mitigation,
        progress_callback=progress_callback,
    )

    # Invalidate tracking cache in data service
    data.tracking.cache_clear()

    return {"status": "success", "match_id": match_id, "data": result}


@router.get("/video/status/{video_id}")
def video_status(video_id: str):
    if video_id not in VIDEO_PROGRESS:
        return {"progress": 0.0, "status": "idle"}
    return VIDEO_PROGRESS[video_id]


@router.get("/video/stream/{video_id}")
def stream_video(video_id: str):
    matching = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not matching:
        # Check bundled samples
        sample_path = Path(__file__).resolve().parents[2] / "data" / "samples" / f"{video_id}.mp4"
        if sample_path.exists():
            return FileResponse(str(sample_path), media_type="video/mp4")
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = matching[0]
    return FileResponse(str(video_path), media_type="video/mp4")