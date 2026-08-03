"""Hardware-neutral RGB-D capture contracts and synchronized frame models."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from robot_skill_system.exceptions import NotConfiguredError

ColorImage = NDArray[np.uint8]
DepthImage = NDArray[np.float32]


class CaptureError(RuntimeError):
    """A configured capture backend failed to return valid synchronized data."""


class CaptureMode(str, Enum):
    """User-facing capture trigger modes."""

    SINGLE = "single"
    BURST = "burst"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for an image with pixel-unit focal lengths."""

    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    distortion_model: str = "none"
    distortion_coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("camera image dimensions must be positive")
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True)
class CameraExtrinsics:
    """Rigid transform from a source optical frame to a target frame."""

    source_frame: str
    target_frame: str
    translation_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.source_frame.strip() or not self.target_frame.strip():
            raise ValueError("extrinsic frame identifiers must be non-empty")
        if len(self.translation_m) != 3 or not all(
            math.isfinite(value) for value in self.translation_m
        ):
            raise ValueError("extrinsic translation_m must contain three finite values")
        if len(self.orientation_xyzw) != 4 or not all(
            math.isfinite(value) for value in self.orientation_xyzw
        ):
            raise ValueError("extrinsic orientation_xyzw must contain four finite values")
        norm = math.sqrt(sum(value * value for value in self.orientation_xyzw))
        if not math.isclose(norm, 1.0, rel_tol=1.0e-4, abs_tol=1.0e-4):
            raise ValueError("extrinsic orientation_xyzw must be normalized")


@dataclass(frozen=True)
class SynchronizedRGBDFrame:
    """One depth-to-colour-aligned RGB-D pair in metres.

    ``color_timestamp_ns`` and ``depth_timestamp_ns`` are the timestamps used by
    downstream consumers.  Capture adapters that translate from another clock
    retain the source values and domains in the ``raw_*`` fields so recordings
    remain auditable without mixing device-relative time with Unix time.
    """

    color_image_rgb: ColorImage
    depth_image_m: DepthImage
    color_timestamp_ns: int
    depth_timestamp_ns: int
    color_intrinsics: CameraIntrinsics
    frame_number: int
    reference_frame: str = "camera_color_optical_frame"
    depth_scale_m: float = 1.0
    aligned_depth_to_color: bool = True
    maximum_timestamp_skew_ns: int = 20_000_000
    depth_intrinsics: CameraIntrinsics | None = None
    depth_to_color_extrinsics: CameraExtrinsics | None = None
    timestamp_clock_domain: str = "source_clock"
    raw_color_timestamp_ns: int | None = None
    raw_depth_timestamp_ns: int | None = None
    raw_color_timestamp_clock_domain: str | None = None
    raw_depth_timestamp_clock_domain: str | None = None

    def __post_init__(self) -> None:
        if self.color_image_rgb.dtype != np.uint8:
            raise ValueError("color_image_rgb must have dtype uint8")
        if self.color_image_rgb.ndim != 3 or self.color_image_rgb.shape[2] != 3:
            raise ValueError("color_image_rgb must have shape (height, width, 3)")
        if self.depth_image_m.dtype != np.float32:
            raise ValueError("depth_image_m must have dtype float32")
        if self.depth_image_m.ndim != 2:
            raise ValueError("depth_image_m must have shape (height, width)")
        if self.depth_image_m.shape != self.color_image_rgb.shape[:2]:
            raise ValueError("aligned RGB and depth dimensions must match")
        if self.color_image_rgb.shape[:2] != (
            self.color_intrinsics.height_px,
            self.color_intrinsics.width_px,
        ):
            raise ValueError("frame dimensions must match color intrinsics")
        if self.color_timestamp_ns < 0 or self.depth_timestamp_ns < 0:
            raise ValueError("frame timestamps must be non-negative")
        if self.frame_number < 0:
            raise ValueError("frame_number must be non-negative")
        if self.depth_scale_m <= 0.0:
            raise ValueError("depth_scale_m must be positive")
        if not self.reference_frame.strip():
            raise ValueError("reference_frame must be non-empty")
        if not self.timestamp_clock_domain.strip():
            raise ValueError("timestamp_clock_domain must be non-empty")
        raw_timestamp_metadata = (
            (self.raw_color_timestamp_ns, self.raw_color_timestamp_clock_domain),
            (self.raw_depth_timestamp_ns, self.raw_depth_timestamp_clock_domain),
        )
        for raw_timestamp_ns, raw_clock_domain in raw_timestamp_metadata:
            if (raw_timestamp_ns is None) != (raw_clock_domain is None):
                raise ValueError("raw timestamps and their clock domains must be provided together")
            if raw_timestamp_ns is not None and raw_timestamp_ns < 0:
                raise ValueError("raw frame timestamps must be non-negative")
            if raw_clock_domain is not None and not raw_clock_domain.strip():
                raise ValueError("raw timestamp clock domains must be non-empty")
        if (
            abs(self.color_timestamp_ns - self.depth_timestamp_ns)
            > self.maximum_timestamp_skew_ns
        ):
            raise ValueError("RGB and depth timestamps exceed allowed synchronization skew")

    @property
    def timestamp_ns(self) -> int:
        """Midpoint timestamp for downstream observations."""

        return (self.color_timestamp_ns + self.depth_timestamp_ns) // 2


# Short compatibility name used by callers that already imply synchronization.
RGBDFrame = SynchronizedRGBDFrame


@dataclass(frozen=True)
class CaptureRequest:
    """Parameters for one user-visible capture trigger."""

    mode: CaptureMode = CaptureMode.BURST
    frame_count: int = 5
    strict_single_frame: bool = False

    def __post_init__(self) -> None:
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        if self.frame_count > 1000:
            raise ValueError("frame_count is unreasonably large for one trigger")

    @property
    def effective_frame_count(self) -> int:
        if self.strict_single_frame or self.mode == CaptureMode.SINGLE:
            return 1
        return self.frame_count


def depth_median_consensus(frames: tuple[SynchronizedRGBDFrame, ...]) -> DepthImage:
    """Median valid depth across a short burst, leaving unknown pixels as zero."""

    if not frames:
        raise ValueError("at least one RGB-D frame is required")
    depth_stack = np.stack([frame.depth_image_m for frame in frames]).astype(
        np.float32, copy=False
    )
    valid = np.isfinite(depth_stack) & (depth_stack > 0.0)
    safe = np.where(valid, depth_stack, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        consensus = np.nanmedian(safe, axis=0)
    consensus = np.where(np.isfinite(consensus), consensus, 0.0)
    return np.asarray(consensus, dtype=np.float32)


@dataclass(frozen=True)
class CaptureResult:
    """Frames and local consensus produced by one capture trigger."""

    request: CaptureRequest
    frames: tuple[SynchronizedRGBDFrame, ...]
    consensus_depth_m: DepthImage

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("capture result must contain at least one frame")
        if len(self.frames) != self.request.effective_frame_count:
            raise ValueError("capture frame count does not match the request")
        expected_shape = self.frames[0].depth_image_m.shape
        if self.consensus_depth_m.dtype != np.float32:
            raise ValueError("consensus_depth_m must have dtype float32")
        if self.consensus_depth_m.shape != expected_shape:
            raise ValueError("consensus depth shape must match captured frames")

    @property
    def representative_frame(self) -> SynchronizedRGBDFrame:
        return self.frames[len(self.frames) // 2]

    @property
    def timestamp_ns(self) -> int:
        return self.representative_frame.timestamp_ns


@runtime_checkable
class RGBDCapture(Protocol):
    """Contract shared by mock and optional RealSense backends."""

    def start(self) -> None:
        """Prepare the backend for capture."""

    def stop(self) -> None:
        """Release backend resources."""

    def capture(self, request: CaptureRequest | None = None) -> CaptureResult:
        """Perform one user-visible trigger."""

    def stream(self) -> Iterator[SynchronizedRGBDFrame]:
        """Yield continuous frames until the consumer stops iteration."""


__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "CaptureError",
    "CaptureMode",
    "CaptureRequest",
    "CaptureResult",
    "ColorImage",
    "DepthImage",
    "NotConfiguredError",
    "RGBDCapture",
    "RGBDFrame",
    "SynchronizedRGBDFrame",
    "depth_median_consensus",
]
