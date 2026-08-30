"""
Noise analysis, synthetic tracking degradation, and mitigation package.
"""

from src.noise.degradation import (
    ALL_DEGRADATION_NAMES,
    SEVERITY_CONFIGS,
    apply_contiguous_gaps,
    apply_coordinate_jitter,
    apply_identity_switch,
    apply_isolated_jumps,
    apply_random_missing,
    apply_track_fragmentation,
    degrade_tracking,
)
from src.noise.interpolation import interpolate_trajectory
from src.noise.smoothing import (
    smooth_moving_average,
    smooth_savitzky_golay,
    smooth_trajectory,
)

__all__ = [
    # Degradation API
    "SEVERITY_CONFIGS",
    "ALL_DEGRADATION_NAMES",
    "apply_random_missing",
    "apply_contiguous_gaps",
    "apply_coordinate_jitter",
    "apply_isolated_jumps",
    "apply_track_fragmentation",
    "apply_identity_switch",
    "degrade_tracking",
    # Mitigation API
    "interpolate_trajectory",
    "smooth_moving_average",
    "smooth_savitzky_golay",
    "smooth_trajectory",
]
