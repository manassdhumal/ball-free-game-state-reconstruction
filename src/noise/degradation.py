"""
Synthetic Tracking Degradation + Robustness Framework

Generates controlled, reproducible degraded versions of clean canonical
tracking data for robustness benchmarking and validation.

Conceptual Hierarchy:
  A. Observation-Level Degradations:
     - Random missing detections
     - Contiguous track gaps (geometric distribution duration model)
     - Coordinate jitter (configurable distribution API)
     - Isolated coordinate jumps (speed-anomaly calibrated)
  B. Identity-Level Degradations:
     - Track fragmentation (deterministic '<original_id>_frag_<seg>' syntax)
     - Identity-switch simulation (same-team pairwise swaps)

Design Guarantees:
  - Strict input immutability (returns new DataFrames, never mutates in-place).
  - Full determinism via explicit numpy.random.Generator instances.
  - Comprehensive experiment metadata recording degradation classes and parameters.
  - Zero modification to raw datasets.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Degradation Classification Taxonomy
# ---------------------------------------------------------------------------

DEGRADATION_HIERARCHY: Dict[str, List[str]] = {
    "observation_level": [
        "random_missing",
        "contiguous_gaps",
        "coordinate_jitter",
        "isolated_jumps",
    ],
    "identity_level": [
        "track_fragmentation",
        "identity_switch",
    ],
}

ALL_DEGRADATION_NAMES: List[str] = [
    name
    for sublist in DEGRADATION_HIERARCHY.values()
    for name in sublist
]

# ---------------------------------------------------------------------------
# Severity Configuration System
# ---------------------------------------------------------------------------
#
# Classification Key:
#   [ED]  = Empirically Derived   - Directly measured from Metrica Game 1 distributions.
#   [EM]  = Empirically Motivated - Informed by dataset properties (25 FPS, duration, motion)
#                                   and optical tracking domain dynamics.
#   [TBD] = To Be Determined      - Research parameterization for unobserved failure modes.

SEVERITY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "clean": {
        # --- Observation-Level: Random Missing ---
        "missing_prob": 0.0,

        # --- Observation-Level: Contiguous Gaps ---
        "gap_start_prob": 0.0,
        "gap_p_continue": 0.0,
        "gap_min_length": 0,
        "gap_max_length": 0,

        # --- Observation-Level: Coordinate Jitter ---
        "jitter_scale": 0.0,
        "jitter_distribution": "gaussian",

        # --- Observation-Level: Isolated Jumps ---
        "jump_prob": 0.0,
        "jump_magnitude_min": 0.0,
        "jump_magnitude_max": 0.0,

        # --- Identity-Level: Track Fragmentation ---
        "frag_prob": 0.0,
        "frag_num_splits": 0,

        # --- Identity-Level: Identity Switches ---
        "switch_prob": 0.0,
        "switch_duration_min": 0,
        "switch_duration_max": 0,
    },

    "mild": {
        # [EM] Metrica Game 1 has 100% completeness for active on-pitch players;
        #      mild degradation simulates minor optical detector dropouts (2%).
        "missing_prob": 0.02,

        # [EM] Low gap initiation frequency for short occlusion bursts.
        "gap_start_prob": 0.001,
        # [EM] Geometric p=0.08 yields expected unclipped length of 1/0.08 = 12.5 frames (0.5 s).
        "gap_p_continue": 0.08,
        # [EM] 5 frames = 0.20 s at 25 FPS.
        "gap_min_length": 5,
        # [EM] 25 frames = 1.00 s at 25 FPS.
        "gap_max_length": 25,

        # [EM] Empirical P99 per-frame displacement is 0.00288 norm units.
        #      Mild jitter standard deviation (0.001) is chosen below median per-frame noise.
        "jitter_scale": 0.001,
        "jitter_distribution": "gaussian",

        # [EM] Non-systemic isolated tracking glitches are sparse in broadcast feeds.
        "jump_prob": 0.0002,
        # [ED] Displacements 0.05-0.15 norm units exceed physical sprinting threshold (0.010/frame).
        "jump_magnitude_min": 0.05,
        "jump_magnitude_max": 0.15,

        # [TBD] Identity fragmentation disabled at mild severity.
        "frag_prob": 0.0,
        "frag_num_splits": 0,

        # [TBD] Identity switches disabled at mild severity.
        "switch_prob": 0.0,
        "switch_duration_min": 0,
        "switch_duration_max": 0,
    },

    "moderate": {
        # [EM] Moderate optical tracking failure rate (8% dropout).
        "missing_prob": 0.08,

        # [EM] Moderate occlusion burst frequency.
        "gap_start_prob": 0.005,
        # [EM] Geometric p=0.03 yields expected unclipped length of 1/0.03 = 33.3 frames (~1.33 s).
        "gap_p_continue": 0.03,
        # [EM] 10 frames = 0.40 s at 25 FPS.
        "gap_min_length": 10,
        # [EM] 75 frames = 3.00 s at 25 FPS.
        "gap_max_length": 75,

        # [ED] Jitter scale (0.003) matches empirical 99.0th percentile step displacement (0.00288).
        "jitter_scale": 0.003,
        "jitter_distribution": "gaussian",

        # [EM] 0.1% jump rate simulates intermittent re-acquisition coordinate errors.
        "jump_prob": 0.001,
        # [ED] Upper bound (0.30) encompasses maximum non-systemic jump observed in Game 1 (0.2328 at frame 71281).
        "jump_magnitude_min": 0.08,
        "jump_magnitude_max": 0.30,

        # [TBD] 5% player fragmentation rate with 2 splits per affected track.
        "frag_prob": 0.05,
        "frag_num_splits": 2,

        # [TBD] 2% pairwise switch probability within team.
        "switch_prob": 0.02,
        # [EM] Switch duration 25-125 frames (1.0 s - 5.0 s at 25 FPS).
        "switch_duration_min": 25,
        "switch_duration_max": 125,
    },

    "severe": {
        # [EM] Heavy tracking degradation / low-quality broadcast feed (20% missing).
        "missing_prob": 0.20,

        # [EM] Controlled tracking dropouts tuned down from 0.015 to prevent missingness collapse.
        "gap_start_prob": 0.010,
        # [EM] Geometric p=0.01 yields expected unclipped length of 1/0.01 = 100 frames (4.0 s).
        "gap_p_continue": 0.01,
        # [EM] 25 frames = 1.0 s minimum duration.
        "gap_min_length": 25,
        # [EM] 250 frames = 10.0 s maximum duration.
        "gap_max_length": 250,

        # [EM] Heavy jitter (0.008) substantially exceeds the 99.9th percentile displacement (0.00384).
        "jitter_scale": 0.008,
        "jitter_distribution": "gaussian",

        # [EM] 0.5% jump rate models highly volatile tracker re-identification.
        "jump_prob": 0.005,
        # [ED] Severe jumps up to 0.50 normalized pitch units (half the pitch length).
        "jump_magnitude_min": 0.12,
        "jump_magnitude_max": 0.50,

        # [TBD] 15% player fragmentation rate with 4 splits per affected track.
        "frag_prob": 0.15,
        "frag_num_splits": 4,

        # [TBD] 8% pairwise switch probability within team.
        "switch_prob": 0.08,
        # [EM] Extended identity switches up to 500 frames (20.0 s at 25 FPS).
        "switch_duration_min": 25,
        "switch_duration_max": 500,
    },
}


def _make_rng(seed: int) -> np.random.Generator:
    """Instantiate an isolated NumPy random Generator."""
    return np.random.default_rng(seed)


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate that the configuration contains all mandatory degradation keys."""
    required_keys = {
        "missing_prob",
        "gap_start_prob",
        "gap_p_continue",
        "gap_min_length",
        "gap_max_length",
        "jitter_scale",
        "jitter_distribution",
        "jump_prob",
        "jump_magnitude_min",
        "jump_magnitude_max",
        "frag_prob",
        "frag_num_splits",
        "switch_prob",
        "switch_duration_min",
        "switch_duration_max",
    }
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Configuration is missing required parameter keys: {missing}")


# ---------------------------------------------------------------------------
# A. Observation-Level Degradations
# ---------------------------------------------------------------------------

def apply_random_missing(
    df: pd.DataFrame,
    missing_prob: float = 0.05,
    seed: int = 42,
    affected_teams: Optional[Sequence[str]] = None,
    affected_players: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Randomly hide individual player observations.

    Sets visible=False, x=NaN, y=NaN, confidence=0.0 on selected visible rows.
    Row count and ordering are strictly preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical tracking DataFrame (not modified in-place).
    missing_prob : float
        Probability of dropping each visible observation.
    seed : int
        Random seed.
    affected_teams : sequence of str, optional
        Filter candidate rows to specific teams.
    affected_players : sequence of str, optional
        Filter candidate rows to specific player_ids.

    Returns
    -------
    pd.DataFrame
        A new degraded DataFrame.
    """
    if missing_prob <= 0.0:
        return df.copy()

    rng = _make_rng(seed)
    out = df.copy()

    mask = out["visible"].values.copy()
    if affected_teams is not None:
        mask &= out["team"].isin(affected_teams).values
    if affected_players is not None:
        mask &= out["player_id"].isin(affected_players).values

    candidate_indices = np.where(mask)[0]
    if len(candidate_indices) == 0:
        return out

    drop_flags = rng.random(len(candidate_indices)) < missing_prob
    drop_indices = candidate_indices[drop_flags]

    target_idx = out.index[drop_indices]
    out.loc[target_idx, "visible"] = False
    out.loc[target_idx, "x"] = np.nan
    out.loc[target_idx, "y"] = np.nan
    out.loc[target_idx, "confidence"] = 0.0

    return out


def apply_contiguous_gaps(
    df: pd.DataFrame,
    gap_start_prob: float = 0.005,
    gap_p_continue: float = 0.03,
    gap_min_length: int = 10,
    gap_max_length: int = 75,
    seed: int = 42,
) -> pd.DataFrame:
    """Create contiguous missing segments using a geometric distribution duration model.

    For each active player track, gaps are triggered with probability `gap_start_prob`.
    Gap duration is drawn from Geometric(gap_p_continue) and clipped to
    [gap_min_length, gap_max_length].

    Parameters
    ----------
    df : pd.DataFrame
        Canonical tracking DataFrame (not modified in-place).
    gap_start_prob : float
        Per-frame initiation probability of a contiguous gap.
    gap_p_continue : float
        Success probability parameter p for the geometric distribution.
        Mean gap length is 1/p before clipping.
    gap_min_length : int
        Minimum gap duration in frames.
    gap_max_length : int
        Maximum gap duration in frames.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        A new degraded DataFrame.
    """
    if gap_start_prob <= 0.0 or gap_max_length <= 0:
        return df.copy()

    rng = _make_rng(seed)
    out = df.copy()

    p_geom = max(min(gap_p_continue, 1.0), 1e-4) if gap_p_continue > 0 else 0.05

    for (_, _), grp in out.groupby(["team", "player_id"]):
        grp_sorted = grp.sort_values("frame")
        vis_indices = np.where(grp_sorted["visible"].values)[0]

        if len(vis_indices) == 0:
            continue

        starts = rng.random(len(vis_indices)) < gap_start_prob
        start_positions = vis_indices[starts]

        for spos in start_positions:
            raw_length = int(rng.geometric(p_geom))
            gap_len = int(np.clip(raw_length, gap_min_length, gap_max_length))

            end_pos = min(spos + gap_len, len(grp_sorted))
            gap_iloc = grp_sorted.index[spos:end_pos]

            out.loc[gap_iloc, "visible"] = False
            out.loc[gap_iloc, "x"] = np.nan
            out.loc[gap_iloc, "y"] = np.nan
            out.loc[gap_iloc, "confidence"] = 0.0

    return out


def _sample_noise(
    distribution: str,
    size: int,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample coordinate noise from a configurable statistical distribution."""
    dist_clean = distribution.strip().lower()
    if dist_clean == "gaussian":
        return rng.normal(loc=0.0, scale=scale, size=size)
    elif dist_clean == "uniform":
        # Zero-mean uniform with standard deviation matching scale: a = scale * sqrt(3)
        half_width = scale * np.sqrt(3.0)
        return rng.uniform(low=-half_width, high=half_width, size=size)
    elif dist_clean == "laplace":
        # Scale parameter b = scale / sqrt(2)
        b = scale / np.sqrt(2.0)
        return rng.laplace(loc=0.0, scale=b, size=size)
    else:
        raise ValueError(
            f"Unsupported jitter distribution '{distribution}'. Supported: 'gaussian', 'uniform', 'laplace'."
        )


def apply_coordinate_jitter(
    df: pd.DataFrame,
    scale: float = 0.003,
    distribution: str = "gaussian",
    seed: int = 42,
    scale_x: Optional[float] = None,
    scale_y: Optional[float] = None,
) -> pd.DataFrame:
    """Perturb visible coordinates with noise from a configurable distribution.

    Preserves missing coordinates as NaN without altering row visibility.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical tracking DataFrame (not modified in-place).
    scale : float
        Base noise scale (standard deviation in normalized pitch coordinates).
    distribution : str
        Noise distribution: "gaussian" (default), "uniform", or "laplace".
    seed : int
        Random seed.
    scale_x, scale_y : float, optional
        Per-axis scale overrides. If None, `scale` is applied to both axes.

    Returns
    -------
    pd.DataFrame
        A new degraded DataFrame.
    """
    sx = scale_x if scale_x is not None else scale
    sy = scale_y if scale_y is not None else scale

    if sx <= 0.0 and sy <= 0.0:
        return df.copy()

    rng = _make_rng(seed)
    out = df.copy()

    vis = out["visible"].values
    n_vis = int(vis.sum())
    if n_vis == 0:
        return out

    noise_x = _sample_noise(distribution, n_vis, sx, rng)
    noise_y = _sample_noise(distribution, n_vis, sy, rng)

    x_vals = out["x"].values.copy().astype(float)
    y_vals = out["y"].values.copy().astype(float)

    x_vals[vis] += noise_x
    y_vals[vis] += noise_y

    out["x"] = x_vals
    out["y"] = y_vals

    return out


def apply_isolated_jumps(
    df: pd.DataFrame,
    jump_prob: float = 0.001,
    magnitude_min: float = 0.08,
    magnitude_max: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Introduce sparse isolated coordinate perturbations.

    Magnitudes are drawn uniformly in [magnitude_min, magnitude_max] in an
    isotropic 2D direction. Not modeled on the systemic half-time reset.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical tracking DataFrame (not modified in-place).
    jump_prob : float
        Per-visible-observation probability of an isolated jump.
    magnitude_min, magnitude_max : float
        Magnitude bounds of spatial displacement in normalized pitch coordinates.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        A new degraded DataFrame.
    """
    if jump_prob <= 0.0 or magnitude_max <= 0.0:
        return df.copy()

    rng = _make_rng(seed)
    out = df.copy()

    vis_idx = np.where(out["visible"].values)[0]
    if len(vis_idx) == 0:
        return out

    jump_flags = rng.random(len(vis_idx)) < jump_prob
    jump_positions = vis_idx[jump_flags]
    n_jumps = len(jump_positions)
    if n_jumps == 0:
        return out

    magnitudes = rng.uniform(magnitude_min, magnitude_max, size=n_jumps)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_jumps)

    dx = magnitudes * np.cos(angles)
    dy = magnitudes * np.sin(angles)

    x_vals = out["x"].values.copy().astype(float)
    y_vals = out["y"].values.copy().astype(float)

    x_vals[jump_positions] += dx
    y_vals[jump_positions] += dy

    out["x"] = x_vals
    out["y"] = y_vals

    return out


# ---------------------------------------------------------------------------
# B. Identity-Level Degradations
# ---------------------------------------------------------------------------

def apply_track_fragmentation(
    df: pd.DataFrame,
    frag_prob: float = 0.05,
    num_splits: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate track identity fragmentation into sequential segments.

    Generates deterministic IDs formatted as `<original_player_id>_frag_<segment_number>`.
    These represent synthetic track identities, not new physical players.
    Temporal ordering and key uniqueness are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical tracking DataFrame (not modified in-place).
    frag_prob : float
        Probability of selecting a given player for fragmentation.
    num_splits : int
        Number of cut-points (yielding num_splits + 1 fragments).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        A new degraded DataFrame.
    """
    if frag_prob <= 0.0 or num_splits <= 0:
        return df.copy()

    rng = _make_rng(seed)
    out = df.copy()

    player_groups = out.groupby(["team", "player_id"]).groups

    # Sort keys for deterministic iteration order
    sorted_player_keys = sorted(player_groups.keys())

    for (team, pid) in sorted_player_keys:
        if rng.random() >= frag_prob:
            continue

        idx = player_groups[(team, pid)]
        grp = out.loc[idx].sort_values("frame")
        n_rows = len(grp)

        if n_rows <= num_splits:
            continue

        cut_points = np.sort(
            rng.choice(np.arange(1, n_rows), size=min(num_splits, n_rows - 1), replace=False)
        )

        segments = np.split(grp.index.values, cut_points)
        for seg_idx, seg_rows in enumerate(segments):
            frag_id = f"{pid}_frag_{seg_idx}"
            out.loc[seg_rows, "player_id"] = frag_id

    return out


def apply_identity_switch(
    df: pd.DataFrame,
    switch_prob: float = 0.02,
    duration_min: int = 25,
    duration_max: int = 125,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate pairwise identity switches between same-team players.

    Swaps player_id between two teammates over a contiguous temporal window.
    Strictly preserves team identity and validates that no duplicate
    (match_id, frame, team, player_id) keys are created.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical tracking DataFrame (not modified in-place).
    switch_prob : float
        Probability of an identity switch occurring for each candidate pair.
    duration_min, duration_max : int
        Bounds on the switch window duration in frames.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        A new degraded DataFrame.
    """
    if switch_prob <= 0.0 or duration_max <= 0:
        return df.copy()

    rng = _make_rng(seed)
    out = df.copy()

    for team_name in sorted(out["team"].unique()):
        team_players = sorted(out[out["team"] == team_name]["player_id"].unique())
        if len(team_players) < 2:
            continue

        pairs = [
            (team_players[i], team_players[j])
            for i in range(len(team_players))
            for j in range(i + 1, len(team_players))
        ]

        for p1, p2 in pairs:
            if rng.random() >= switch_prob:
                continue

            frames_p1 = set(out[(out["team"] == team_name) & (out["player_id"] == p1)]["frame"].values)
            frames_p2 = set(out[(out["team"] == team_name) & (out["player_id"] == p2)]["frame"].values)
            common_frames = sorted(frames_p1 & frames_p2)

            if len(common_frames) < duration_min:
                continue

            dur = int(rng.integers(duration_min, duration_max + 1))
            dur = min(dur, len(common_frames))

            start_idx = int(rng.integers(0, len(common_frames) - dur + 1))
            switch_frames = set(common_frames[start_idx:start_idx + dur])

            mask_p1 = (
                (out["team"] == team_name)
                & (out["player_id"] == p1)
                & (out["frame"].isin(switch_frames))
            )
            mask_p2 = (
                (out["team"] == team_name)
                & (out["player_id"] == p2)
                & (out["frame"].isin(switch_frames))
            )

            out.loc[mask_p1, "player_id"] = p2
            out.loc[mask_p2, "player_id"] = p1

    dups = out.duplicated(subset=["match_id", "frame", "team", "player_id"])
    if dups.any():
        raise RuntimeError(
            f"Identity switch resulted in {dups.sum()} duplicate player-frame keys."
        )

    return out


# ---------------------------------------------------------------------------
# Composition API & Metadata
# ---------------------------------------------------------------------------

def degrade_tracking(
    df: pd.DataFrame,
    severity: str = "moderate",
    seed: int = 42,
    enabled_degradations: Optional[List[str]] = None,
    custom_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compose multiple degradation methods into a reproducible degraded dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Clean canonical tracking DataFrame.
    severity : str
        Preset name: "clean", "mild", "moderate", "severe", or "custom".
    seed : int
        Master random seed from which deterministic sub-seeds are derived.
    enabled_degradations : list of str, optional
        Subset of degradation names to apply. Defaults to all available methods.
    custom_config : dict, optional
        Required when severity="custom".

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        A tuple of (degraded_dataframe, experiment_metadata_dict).
    """
    if severity == "custom":
        if custom_config is None:
            raise ValueError("custom_config dict is required when severity='custom'.")
        config = copy.deepcopy(custom_config)
    elif severity in SEVERITY_CONFIGS:
        config = copy.deepcopy(SEVERITY_CONFIGS[severity])
    else:
        valid = list(SEVERITY_CONFIGS.keys()) + ["custom"]
        raise ValueError(f"Unknown severity '{severity}'. Valid options: {valid}")

    _validate_config(config)

    if enabled_degradations is None:
        enabled_degradations = list(ALL_DEGRADATION_NAMES)
    else:
        unknown = set(enabled_degradations) - set(ALL_DEGRADATION_NAMES)
        if unknown:
            raise ValueError(f"Unknown degradation names requested: {unknown}")

    # Derive deterministic sub-seeds
    master_rng = _make_rng(seed)
    sub_seeds = {
        name: int(master_rng.integers(0, 2**31))
        for name in ALL_DEGRADATION_NAMES
    }

    match_ids = df["match_id"].unique()
    source_match_id = match_ids[0] if len(match_ids) == 1 else str(list(match_ids))

    out = df.copy()

    # 1. Observation-Level: Random Missing
    if "random_missing" in enabled_degradations:
        out = apply_random_missing(
            out,
            missing_prob=config["missing_prob"],
            seed=sub_seeds["random_missing"],
        )

    # 2. Observation-Level: Contiguous Gaps
    if "contiguous_gaps" in enabled_degradations:
        out = apply_contiguous_gaps(
            out,
            gap_start_prob=config["gap_start_prob"],
            gap_p_continue=config["gap_p_continue"],
            gap_min_length=config["gap_min_length"],
            gap_max_length=config["gap_max_length"],
            seed=sub_seeds["contiguous_gaps"],
        )

    # 3. Observation-Level: Coordinate Jitter
    if "coordinate_jitter" in enabled_degradations:
        out = apply_coordinate_jitter(
            out,
            scale=config["jitter_scale"],
            distribution=config.get("jitter_distribution", "gaussian"),
            seed=sub_seeds["coordinate_jitter"],
        )

    # 4. Observation-Level: Isolated Coordinate Jumps
    if "isolated_jumps" in enabled_degradations:
        out = apply_isolated_jumps(
            out,
            jump_prob=config["jump_prob"],
            magnitude_min=config["jump_magnitude_min"],
            magnitude_max=config["jump_magnitude_max"],
            seed=sub_seeds["isolated_jumps"],
        )

    # 5. Identity-Level: Track Fragmentation
    if "track_fragmentation" in enabled_degradations:
        out = apply_track_fragmentation(
            out,
            frag_prob=config["frag_prob"],
            num_splits=config["frag_num_splits"],
            seed=sub_seeds["track_fragmentation"],
        )

    # 6. Identity-Level: Identity Switches
    if "identity_switch" in enabled_degradations:
        out = apply_identity_switch(
            out,
            switch_prob=config["switch_prob"],
            duration_min=config["switch_duration_min"],
            duration_max=config["switch_duration_max"],
            seed=sub_seeds["identity_switch"],
        )

    # Structured metadata output
    metadata: Dict[str, Any] = {
        "source_match_id": source_match_id,
        "severity": severity,
        "random_seed": seed,
        "enabled_degradations": enabled_degradations,
        "degradation_classes": {
            cls_name: [d for d in methods if d in enabled_degradations]
            for cls_name, methods in DEGRADATION_HIERARCHY.items()
        },
        "actual_parameters": config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return out, metadata
