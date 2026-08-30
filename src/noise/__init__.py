"""
Noise analysis and synthetic tracking degradation package.
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

__all__ = [
    "SEVERITY_CONFIGS",
    "ALL_DEGRADATION_NAMES",
    "apply_random_missing",
    "apply_contiguous_gaps",
    "apply_coordinate_jitter",
    "apply_isolated_jumps",
    "apply_track_fragmentation",
    "apply_identity_switch",
    "degrade_tracking",
]
