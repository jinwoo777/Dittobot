"""Deterministic synthetic demonstrations for tests and mock teaching flows."""

from __future__ import annotations

import math

import numpy as np

from .models import DemonstrationTrajectory, PoseSample


def _pose_series(
    positions: np.ndarray,  # type: ignore[type-arg]
    *,
    duration_s: float,
    confidence: float,
    noise_std_m: float,
    seed: int,
    surface_distances_m: np.ndarray | None = None,  # type: ignore[type-arg]
    expected_contact: bool | None = None,
) -> DemonstrationTrajectory:
    if len(positions) < 2:
        raise ValueError("synthetic trajectories require at least two samples")
    rng = np.random.default_rng(seed)
    noisy = positions.astype(np.float64, copy=True)
    if noise_std_m > 0.0:
        noisy += rng.normal(0.0, noise_std_m, size=noisy.shape)
    timestamps_ns = np.rint(np.linspace(0.0, duration_s * 1.0e9, len(noisy))).astype(
        np.int64
    )
    samples = tuple(
        PoseSample(
            timestamp_ns=int(timestamps_ns[index]),
            position_m=(
                float(noisy[index, 0]),
                float(noisy[index, 1]),
                float(noisy[index, 2]),
            ),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="fixture_anchor",
            source="synthetic",
            confidence=confidence,
            surface_distance_m=(
                float(surface_distances_m[index])
                if surface_distances_m is not None
                else None
            ),
            expected_contact=expected_contact,
        )
        for index in range(len(noisy))
    )
    return DemonstrationTrajectory(samples=samples, successful=True)


def generate_line_trajectory(
    *,
    sample_count: int = 41,
    start_m: tuple[float, float, float] = (0.0, 0.0, 0.05),
    end_m: tuple[float, float, float] = (0.24, 0.06, 0.05),
    duration_s: float = 2.0,
    noise_std_m: float = 0.0,
    confidence: float = 0.98,
    seed: int = 7,
) -> DemonstrationTrajectory:
    """Generate a timestamped straight Cartesian stroke."""

    fraction = np.linspace(0.0, 1.0, sample_count)[:, None]
    start = np.asarray(start_m, dtype=np.float64)
    end = np.asarray(end_m, dtype=np.float64)
    positions = start + fraction * (end - start)
    return _pose_series(
        positions,
        duration_s=duration_s,
        confidence=confidence,
        noise_std_m=noise_std_m,
        seed=seed,
    )


def generate_arc_trajectory(
    *,
    sample_count: int = 49,
    center_m: tuple[float, float, float] = (0.1, -0.05, 0.04),
    radius_m: float = 0.12,
    start_angle_rad: float = -0.3,
    sweep_rad: float = math.pi * 0.75,
    duration_s: float = 2.4,
    noise_std_m: float = 0.0,
    confidence: float = 0.98,
    seed: int = 11,
) -> DemonstrationTrajectory:
    """Generate a planar circular arc."""

    angles = np.linspace(start_angle_rad, start_angle_rad + sweep_rad, sample_count)
    center = np.asarray(center_m, dtype=np.float64)
    positions = np.column_stack(
        (
            center[0] + radius_m * np.cos(angles),
            center[1] + radius_m * np.sin(angles),
            np.full(sample_count, center[2]),
        )
    )
    return _pose_series(
        positions,
        duration_s=duration_s,
        confidence=confidence,
        noise_std_m=noise_std_m,
        seed=seed,
    )


def generate_periodic_trajectory(
    *,
    sample_count: int = 121,
    cycles: float = 3.0,
    amplitude_m: float = 0.045,
    duration_s: float = 4.0,
    drift_m: float = 0.025,
    noise_std_m: float = 0.0,
    confidence: float = 0.98,
    seed: int = 17,
) -> DemonstrationTrajectory:
    """Generate repeated surface wiping with a small transverse drift."""

    phase = np.linspace(0.0, cycles * 2.0 * np.pi, sample_count)
    fraction = np.linspace(0.0, 1.0, sample_count)
    positions = np.column_stack(
        (
            0.2 + amplitude_m * np.sin(phase),
            -0.1 + drift_m * fraction,
            np.full(sample_count, 0.002),
        )
    )
    distances = np.full(sample_count, 0.002)
    return _pose_series(
        positions,
        duration_s=duration_s,
        confidence=confidence,
        noise_std_m=noise_std_m,
        seed=seed,
        surface_distances_m=distances,
        expected_contact=True,
    )


def generate_novice_wipe_trajectory() -> DemonstrationTrajectory:
    """A slower, noisier novice fixture with modest observation confidence."""

    trajectory = generate_periodic_trajectory(
        sample_count=101,
        cycles=2.25,
        amplitude_m=0.040,
        duration_s=6.0,
        drift_m=0.04,
        noise_std_m=0.0025,
        confidence=0.82,
        seed=101,
    )
    return trajectory.model_copy(update={"operator_role": "novice", "session_id": "novice_wipe"})


def generate_expert_wipe_trajectory() -> DemonstrationTrajectory:
    """A smooth, repeatable expert fixture suitable for candidate comparison."""

    trajectory = generate_periodic_trajectory(
        sample_count=151,
        cycles=3.5,
        amplitude_m=0.048,
        duration_s=3.8,
        drift_m=0.025,
        noise_std_m=0.0005,
        confidence=0.98,
        seed=202,
    )
    return trajectory.model_copy(update={"operator_role": "expert", "session_id": "expert_wipe"})


def generate_wipe_trajectory(*, expert: bool = False) -> DemonstrationTrajectory:
    """Return the named novice or expert synthetic wipe fixture."""

    return generate_expert_wipe_trajectory() if expert else generate_novice_wipe_trajectory()


__all__ = [
    "generate_arc_trajectory",
    "generate_expert_wipe_trajectory",
    "generate_line_trajectory",
    "generate_novice_wipe_trajectory",
    "generate_periodic_trajectory",
    "generate_wipe_trajectory",
]
