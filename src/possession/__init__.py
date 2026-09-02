"""
Ball-free possession inference package.

Provides spatial graph construction and heuristic possession baseline
that operate exclusively on player tracking data without ball coordinates.
"""

from src.possession.spatial_graph import (
    FrameGraph,
    build_frame_graph,
    build_spatial_features,
)
from src.possession.possession_baseline import (
    DEFAULT_WEIGHTS,
    score_frame,
    predict_possession,
    infer_events,
    evaluate_predictions,
    possession_summary,
)

__all__ = [
    # Spatial graph API
    "FrameGraph",
    "build_frame_graph",
    "build_spatial_features",
    # Possession baseline API
    "DEFAULT_WEIGHTS",
    "score_frame",
    "predict_possession",
    "infer_events",
    "evaluate_predictions",
    "possession_summary",
]
