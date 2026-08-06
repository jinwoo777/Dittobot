"""Shared semantic guidance for locally owned motion simplification.

The model may label or recommend only catalogued primitives.  Metric fitting and
simplification remain deterministic local responsibilities.
"""

MOTION_SIMPLIFICATION_INSTRUCTIONS = """Apply this motion-simplification policy to every
semantic recommendation:
- Preserve intended contact phases and every gripper state-transition boundary.
- Avoid unnecessary interval splits; represent each continuous intent with the fewest justified
  motion segments.
- Prefer motion.move_l for a locally verified straight interval.
- Prefer motion.move_c only for an arc interval already verified by local geometry.
- Use motion.move_spline only as a fallback when no verified line, arc, or periodic primitive fits.
- Never generate, infer, or modify coordinates, poses, transforms, speed, acceleration, force,
  safety limits, classification thresholds, or other numeric execution parameters.
Propose the minimum primitive sequence consistent with the preserved semantic boundaries. The
local motion fitter is authoritative for line, arc, periodic, and spline classification. Local
deterministic code owns all geometry, thresholds, profiles, simplification, and execution."""


__all__ = ["MOTION_SIMPLIFICATION_INSTRUCTIONS"]
