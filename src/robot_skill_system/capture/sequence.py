"""RGB-D sequence metadata, inference-rate sampling, and keyframe selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .interfaces import SynchronizedRGBDFrame


@dataclass(frozen=True)
class SequenceMetadata:
    """Recording-rate facts retained alongside derived frame selections."""

    raw_capture_fps: float = 30.0
    timestamps_preserved: bool = True
    source: str = "rgbd_sequence"
    color_format: str = "rgb8"
    depth_unit: str = "metre"

    def __post_init__(self) -> None:
        if self.raw_capture_fps <= 0.0:
            raise ValueError("raw_capture_fps must be positive")
        if not self.timestamps_preserved:
            raise ValueError("RGB-D source timestamps must be preserved")
        if not self.source.strip():
            raise ValueError("sequence source must be non-empty")


@dataclass(frozen=True)
class RGBDSequence:
    """A timestamp-preserving, ordered RGB-D recording."""

    frames: tuple[SynchronizedRGBDFrame, ...]
    metadata: SequenceMetadata = SequenceMetadata()

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("RGB-D sequence must contain at least one frame")
        timestamps = [frame.timestamp_ns for frame in self.frames]
        timestamp_pairs = zip(timestamps[:-1], timestamps[1:], strict=True)
        if any(second <= first for first, second in timestamp_pairs):
            raise ValueError("RGB-D sequence timestamps must be strictly increasing")
        shape = self.frames[0].depth_image_m.shape
        reference_frame = self.frames[0].reference_frame
        if any(frame.depth_image_m.shape != shape for frame in self.frames):
            raise ValueError("all RGB-D sequence frames must have the same dimensions")
        if any(frame.reference_frame != reference_frame for frame in self.frames):
            raise ValueError("all RGB-D sequence frames must share a reference frame")

    @property
    def duration_s(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return (self.frames[-1].timestamp_ns - self.frames[0].timestamp_ns) / 1.0e9


@dataclass(frozen=True)
class InferenceSamplingConfig:
    """Supported local motion-inference and semantic keyframe rates."""

    inference_fps: int = 10
    keyframe_count: int = 8

    def __post_init__(self) -> None:
        if self.inference_fps not in {10, 15}:
            raise ValueError("inference_fps must be either 10 or 15")
        if not 6 <= self.keyframe_count <= 12:
            raise ValueError("keyframe_count must be in [6, 12]")


def select_inference_frames(
    sequence: RGBDSequence, config: InferenceSamplingConfig | None = None
) -> tuple[SynchronizedRGBDFrame, ...]:
    """Subsample by original timestamps without synthesizing new timestamps."""

    effective_config = config or InferenceSamplingConfig()
    target_period_ns = int(round(1.0e9 / effective_config.inference_fps))
    selected: list[SynchronizedRGBDFrame] = [sequence.frames[0]]
    next_target_ns = sequence.frames[0].timestamp_ns + target_period_ns
    for frame in sequence.frames[1:]:
        if frame.timestamp_ns >= next_target_ns:
            selected.append(frame)
            skipped_periods = max(
                1, (frame.timestamp_ns - next_target_ns) // target_period_ns + 1
            )
            next_target_ns += skipped_periods * target_period_ns
    if sequence.frames[-1] is not selected[-1]:
        selected.append(sequence.frames[-1])
    return tuple(selected)


def _change_score(
    previous: SynchronizedRGBDFrame, current: SynchronizedRGBDFrame
) -> float:
    # Downsampling by slicing bounds CPU cost for full-resolution recordings.
    color_previous = previous.color_image_rgb[::8, ::8].astype(np.float32)
    color_current = current.color_image_rgb[::8, ::8].astype(np.float32)
    color_score = float(np.mean(np.abs(color_current - color_previous))) / 255.0
    depth_previous = previous.depth_image_m[::8, ::8]
    depth_current = current.depth_image_m[::8, ::8]
    valid = (depth_previous > 0.0) & (depth_current > 0.0)
    depth_score = (
        float(np.mean(np.abs(depth_current[valid] - depth_previous[valid])))
        if np.any(valid)
        else 0.0
    )
    return color_score + min(1.0, depth_score / 0.05)


def select_keyframes(
    sequence: RGBDSequence,
    config: InferenceSamplingConfig | None = None,
) -> tuple[SynchronizedRGBDFrame, ...]:
    """Select 6–12 representative original frames using time and local change."""

    effective_config = config or InferenceSamplingConfig()
    inference_frames = select_inference_frames(sequence, effective_config)
    target_count = min(effective_config.keyframe_count, len(inference_frames))
    if target_count == len(inference_frames):
        return inference_frames
    selected_indices = {0, len(inference_frames) - 1}
    # Uniform anchors ensure full temporal coverage; high-change frames fill any
    # remaining slots. Every returned frame remains an original captured frame.
    uniform = np.rint(np.linspace(0, len(inference_frames) - 1, target_count)).astype(int)
    selected_indices.update(int(index) for index in uniform)
    scored = sorted(
        (
            (_change_score(inference_frames[index - 1], inference_frames[index]), index)
            for index in range(1, len(inference_frames) - 1)
        ),
        reverse=True,
    )
    for _, index in scored:
        if len(selected_indices) >= target_count:
            break
        selected_indices.add(index)
    # Rounding uniform anchors can rarely collide; fill deterministically.
    for index in range(1, len(inference_frames) - 1):
        if len(selected_indices) >= target_count:
            break
        selected_indices.add(index)
    return tuple(inference_frames[index] for index in sorted(selected_indices))


def sequence_from_frames(
    frames: Sequence[SynchronizedRGBDFrame],
    *,
    raw_capture_fps: float = 30.0,
    source: str = "frames",
) -> RGBDSequence:
    return RGBDSequence(
        frames=tuple(frames),
        metadata=SequenceMetadata(raw_capture_fps=raw_capture_fps, source=source),
    )


__all__ = [
    "InferenceSamplingConfig",
    "RGBDSequence",
    "SequenceMetadata",
    "select_inference_frames",
    "select_keyframes",
    "sequence_from_frames",
]
