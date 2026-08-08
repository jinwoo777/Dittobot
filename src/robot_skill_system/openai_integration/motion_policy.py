"""Shared semantic guidance for locally owned motion simplification.

The model may label or recommend only catalogued primitives.  Metric fitting and
simplification remain deterministic local responsibilities.
"""

MAXIMUM_ACTION_MOTION_BLOCKS = 3
MAXIMUM_END_MOTION_BLOCKS = 3

# These are the canonical path-producing motion operations exposed by the
# block editor and emitted by the RGB-D trajectory materializer.  MoveJ is not
# included because recorded task geometry is persisted relative to an anchor,
# never as robot-base joint targets.  ``motion.wait`` is not a path block.
RECORDING_BLOCK_MOTION_OPERATIONS = frozenset(
    {
        "motion.move_l",
        "motion.move_c",
        "motion.move_spline",
        "motion.move_periodic",
    }
)

MOTION_SIMPLIFICATION_INSTRUCTIONS = """Apply this motion-simplification policy to every
semantic recommendation:
- Preserve intended contact phases and every gripper state-transition boundary.
- Avoid unnecessary interval splits; represent each continuous intent with the fewest justified
  motion segments.
- Prefer motion.move_l for a locally verified straight interval.
- Prefer motion.move_c only for an arc interval already verified by local geometry.
- Use motion.move_spline only as a fallback when no verified line, arc, or periodic primitive fits.
- Organize the learned sequence as Grip -> Action -> End. Recommend no more than three path-motion
  blocks for Action and no more than three path-motion blocks for End. Those path blocks may only
  be motion.move_l, motion.move_c, motion.move_spline, or motion.move_periodic.
- For a fixed-workspace pickup, use this semantic shape when the evidence supports it:
  gripper.open -> one approach motion -> gripper.close -> one carry/retract motion ->
  gripper.open.  This is an example of phase structure only, never a request for coordinates.
  Do not add repeated approach/retract motions, alternate between motion types without evidence,
  or create a motion segment for every video frame.  Let the local fitter collapse each continuous
  path before it is materialized.
- Never generate, infer, or modify coordinates, poses, transforms, speed, acceleration, force,
  safety limits, classification thresholds, or other numeric execution parameters.
Propose the minimum primitive sequence consistent with the preserved semantic boundaries. The
local motion fitter is authoritative for line, arc, periodic, and spline classification. Local
deterministic code owns all geometry, thresholds, profiles, simplification, and execution."""


__all__ = [
    "MAXIMUM_ACTION_MOTION_BLOCKS",
    "MAXIMUM_END_MOTION_BLOCKS",
    "MOTION_SIMPLIFICATION_INSTRUCTIONS",
    "RECORDING_BLOCK_MOTION_OPERATIONS",
]
