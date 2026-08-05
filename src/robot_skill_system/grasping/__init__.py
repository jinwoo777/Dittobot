"""Local grasp-geometry helpers that never contact robot hardware."""

from robot_skill_system.grasping.rg2_depth import (
    RG2_TOTAL_STROKE_M,
    GripWidthStatistics,
    Rg2DepthDiagnostic,
    build_rg2_depth_diagnostic,
    circular_sagitta_m,
    summarize_grip_widths,
)

__all__ = [
    "RG2_TOTAL_STROKE_M",
    "GripWidthStatistics",
    "Rg2DepthDiagnostic",
    "build_rg2_depth_diagnostic",
    "circular_sagitta_m",
    "summarize_grip_widths",
]
