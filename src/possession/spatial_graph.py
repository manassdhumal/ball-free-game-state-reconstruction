"""
Spatial Graph Construction for Ball-Free Possession Inference

Builds a per-frame spatial graph from canonical player tracking data.
Each visible player is a node with computed spatial features:
  - pairwise distances to all other visible players
  - nearest teammate distance
  - nearest opponent distance
  - local player density (count within a configurable radius)
  - optional k-nearest-neighbour edge list

ANTI-LEAKAGE: This module never accesses Ball_X, Ball_Y, or any
ball-derived column.  It operates exclusively on player tracking rows.

Design: returns a lightweight FrameGraph dataclass (not NetworkX)
to keep dependencies minimal and representations explainable.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameGraph:
    """Lightweight spatial graph for a single tracking frame.

    Attributes
    ----------
    frame : int
        Frame number from the canonical tracking data.
    timestamp : float
        Timestamp in seconds.
    node_df : pd.DataFrame
        One row per visible player with columns:
            player_id, team, x, y,
            nearest_teammate_id, nearest_teammate_dist,
            nearest_opponent_id, nearest_opponent_dist,
            local_density
    distance_matrix : np.ndarray
        N×N symmetric matrix of pairwise Euclidean distances
        (diagonal = inf).  Row/column order matches node_df index.
    player_ids : list[str]
        Ordered player IDs corresponding to distance_matrix axes.
    edges : list[tuple[str, str, float]]
        Optional KNN edge list as (player_a, player_b, distance).
        Empty list when KNN is disabled.
    n_visible : int
        Number of visible players in this frame.
    """
    frame: int
    timestamp: float
    node_df: pd.DataFrame
    distance_matrix: np.ndarray
    player_ids: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str, float]] = field(default_factory=list)
    n_visible: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        X/Y coordinates of N players.

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric distance matrix with inf on the diagonal (no self-edges).
    """
    # Vectorised pairwise distance via broadcasting
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
    np.fill_diagonal(dist, np.inf)
    return dist


def _nearest_by_mask(
    dist_matrix: np.ndarray,
    player_ids: List[str],
    mask: np.ndarray,
    row_idx: int,
) -> Tuple[Optional[str], float]:
    """Find nearest player within a boolean mask for a given row.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Pairwise distance matrix.
    player_ids : list[str]
        Ordered player IDs.
    mask : np.ndarray, shape (N,)
        Boolean mask indicating candidate columns.
    row_idx : int
        Index of the query player.

    Returns
    -------
    (player_id, distance) or (None, inf) if no candidates exist.
    """
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return None, np.inf
    dists = dist_matrix[row_idx, candidates]
    best = candidates[np.argmin(dists)]
    return player_ids[best], dist_matrix[row_idx, best]


def _compute_local_density(
    dist_matrix: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Count neighbouring players within *radius* for each player.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Pairwise distance matrix (diagonal = inf).
    radius : float
        Radius threshold in coordinate units.

    Returns
    -------
    np.ndarray, shape (N,)
        Integer counts of neighbours within radius.
    """
    return np.sum(dist_matrix < radius, axis=1)


def _build_knn_edges(
    dist_matrix: np.ndarray,
    player_ids: List[str],
    k: int,
) -> List[Tuple[str, str, float]]:
    """Build a k-nearest-neighbour edge list.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Pairwise distance matrix.
    player_ids : list[str]
        Ordered player IDs.
    k : int
        Number of neighbours per node.

    Returns
    -------
    list of (player_a, player_b, distance) tuples.
    Edges are directed (A→B does not imply B→A in the list,
    but both will typically appear since each node emits k edges).
    """
    n = len(player_ids)
    k_actual = min(k, n - 1)
    if k_actual <= 0:
        return []

    edges = []
    for i in range(n):
        neighbours = np.argsort(dist_matrix[i])[:k_actual]
        for j in neighbours:
            edges.append((player_ids[i], player_ids[j], dist_matrix[i, j]))
    return edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Columns that this module is allowed to read from canonical tracking data
_REQUIRED_COLUMNS = {"player_id", "team", "x", "y", "visible"}


def build_frame_graph(
    frame_df: pd.DataFrame,
    frame: int,
    timestamp: float,
    density_radius: float = 0.05,
    knn_k: Optional[int] = 5,
) -> FrameGraph:
    """Construct a spatial graph for a single frame.

    Parameters
    ----------
    frame_df : pd.DataFrame
        Subset of canonical tracking data for a single frame.
        Must contain at least: player_id, team, x, y, visible.
        May contain additional columns (including ball columns) —
        they are **ignored**.
    frame : int
        Frame number.
    timestamp : float
        Frame timestamp in seconds.
    density_radius : float, default 0.05
        Radius (in normalised pitch coordinates) used to compute
        local player density.  0.05 ≈ 5 m on a 105 m pitch.
    knn_k : int or None, default 5
        Number of nearest neighbours for KNN edge construction.
        Set to ``None`` to disable edge construction entirely.

    Returns
    -------
    FrameGraph
        Lightweight spatial graph for this frame.

    Raises
    ------
    ValueError
        If required columns are missing from *frame_df*.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    missing = _REQUIRED_COLUMNS - set(frame_df.columns)
    if missing:
        raise ValueError(
            f"frame_df is missing required columns: {sorted(missing)}"
        )

    # ------------------------------------------------------------------
    # Filter to visible players only
    # ------------------------------------------------------------------
    visible = frame_df[frame_df["visible"] == True].copy()  # noqa: E712
    n = len(visible)

    if n == 0:
        return FrameGraph(
            frame=frame,
            timestamp=timestamp,
            node_df=pd.DataFrame(
                columns=[
                    "player_id", "team", "x", "y",
                    "nearest_teammate_id", "nearest_teammate_dist",
                    "nearest_opponent_id", "nearest_opponent_dist",
                    "local_density",
                ]
            ),
            distance_matrix=np.empty((0, 0)),
            player_ids=[],
            edges=[],
            n_visible=0,
        )

    # ------------------------------------------------------------------
    # Extract coordinates and IDs
    # ------------------------------------------------------------------
    player_ids = visible["player_id"].tolist()
    teams = visible["team"].values
    coords = visible[["x", "y"]].values.astype(float)

    # ------------------------------------------------------------------
    # Pairwise distances
    # ------------------------------------------------------------------
    dist_matrix = _compute_pairwise_distances(coords)

    # ------------------------------------------------------------------
    # Nearest teammate / opponent
    # ------------------------------------------------------------------
    nearest_tm_ids = []
    nearest_tm_dists = []
    nearest_opp_ids = []
    nearest_opp_dists = []

    for i in range(n):
        same_team = teams == teams[i]
        same_team[i] = False  # exclude self
        diff_team = teams != teams[i]

        tm_id, tm_dist = _nearest_by_mask(dist_matrix, player_ids, same_team, i)
        opp_id, opp_dist = _nearest_by_mask(dist_matrix, player_ids, diff_team, i)

        nearest_tm_ids.append(tm_id)
        nearest_tm_dists.append(tm_dist)
        nearest_opp_ids.append(opp_id)
        nearest_opp_dists.append(opp_dist)

    # ------------------------------------------------------------------
    # Local density
    # ------------------------------------------------------------------
    density = _compute_local_density(dist_matrix, density_radius)

    # ------------------------------------------------------------------
    # Build node DataFrame
    # ------------------------------------------------------------------
    node_df = pd.DataFrame({
        "player_id": player_ids,
        "team": teams,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "nearest_teammate_id": nearest_tm_ids,
        "nearest_teammate_dist": nearest_tm_dists,
        "nearest_opponent_id": nearest_opp_ids,
        "nearest_opponent_dist": nearest_opp_dists,
        "local_density": density.astype(int),
    })

    # ------------------------------------------------------------------
    # KNN edges
    # ------------------------------------------------------------------
    edges = []
    if knn_k is not None and knn_k > 0 and n > 1:
        edges = _build_knn_edges(dist_matrix, player_ids, knn_k)

    return FrameGraph(
        frame=frame,
        timestamp=timestamp,
        node_df=node_df,
        distance_matrix=dist_matrix,
        player_ids=player_ids,
        edges=edges,
        n_visible=n,
    )


def build_spatial_features(
    tracking_df: pd.DataFrame,
    density_radius: float = 0.05,
    knn_k: Optional[int] = 5,
    frame_col: str = "frame",
    timestamp_col: str = "timestamp",
) -> List[FrameGraph]:
    """Build spatial graphs for all frames in a canonical tracking DataFrame.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Full canonical tracking DataFrame (multi-frame, multi-player).
    density_radius : float
        Passed to :func:`build_frame_graph`.
    knn_k : int or None
        Passed to :func:`build_frame_graph`.
    frame_col : str
        Column name for frame numbers.
    timestamp_col : str
        Column name for timestamps.

    Returns
    -------
    list[FrameGraph]
        One FrameGraph per unique frame, in frame order.
    """
    missing = _REQUIRED_COLUMNS - set(tracking_df.columns)
    if missing:
        raise ValueError(
            f"tracking_df is missing required columns: {sorted(missing)}"
        )

    graphs = []
    for (frame_num, ts), grp in tracking_df.groupby(
        [frame_col, timestamp_col], sort=True
    ):
        g = build_frame_graph(
            grp,
            frame=int(frame_num),
            timestamp=float(ts),
            density_radius=density_radius,
            knn_k=knn_k,
        )
        graphs.append(g)
    return graphs
