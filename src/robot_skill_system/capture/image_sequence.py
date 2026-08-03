"""Adapters for synchronized RGB/depth image sequences."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .interfaces import CameraExtrinsics, CameraIntrinsics, SynchronizedRGBDFrame
from .sequence import RGBDSequence, SequenceMetadata


@runtime_checkable
class RGBDImageSequenceLoader(Protocol):
    """Contract for local recording formats (PNG, bag export, NumPy, and others)."""

    def load(self) -> RGBDSequence:
        """Load synchronized frames while preserving source timestamps."""


@dataclass(frozen=True)
class ArrayImageSequenceAdapter:
    """Build an RGB-D recording from already decoded arrays."""

    color_images_rgb: Sequence[NDArray[np.uint8]]
    depth_images_m: Sequence[NDArray[np.float32]]
    color_timestamps_ns: Sequence[int]
    depth_timestamps_ns: Sequence[int]
    color_intrinsics: CameraIntrinsics
    depth_intrinsics: CameraIntrinsics | None = None
    depth_to_color_extrinsics: CameraExtrinsics | None = None
    maximum_timestamp_skew_ns: int = 20_000_000
    raw_capture_fps: float = 30.0
    reference_frame: str = "camera_color_optical_frame"

    def load(self) -> RGBDSequence:
        lengths = {
            len(self.color_images_rgb),
            len(self.depth_images_m),
            len(self.color_timestamps_ns),
            len(self.depth_timestamps_ns),
        }
        if len(lengths) != 1 or not self.color_images_rgb:
            raise ValueError("RGB, depth, and timestamp sequences must have equal non-zero length")
        frames = tuple(
            SynchronizedRGBDFrame(
                color_image_rgb=np.asarray(color_image, dtype=np.uint8),
                depth_image_m=np.asarray(depth_image, dtype=np.float32),
                color_timestamp_ns=int(self.color_timestamps_ns[index]),
                depth_timestamp_ns=int(self.depth_timestamps_ns[index]),
                color_intrinsics=self.color_intrinsics,
                depth_intrinsics=self.depth_intrinsics,
                depth_to_color_extrinsics=self.depth_to_color_extrinsics,
                frame_number=index,
                reference_frame=self.reference_frame,
                maximum_timestamp_skew_ns=self.maximum_timestamp_skew_ns,
            )
            for index, (color_image, depth_image) in enumerate(
                zip(self.color_images_rgb, self.depth_images_m, strict=True)
            )
        )
        return RGBDSequence(
            frames=frames,
            metadata=SequenceMetadata(
                raw_capture_fps=self.raw_capture_fps,
                source="array_image_sequence",
            ),
        )


@dataclass(frozen=True)
class NumpyDirectorySequenceLoader:
    """Minimal dependency-free fixture loader for ``.npy`` RGB/depth pairs."""

    directory: Path
    timestamps_ns: Sequence[int]
    intrinsics: CameraIntrinsics
    raw_capture_fps: float = 30.0
    maximum_timestamp_skew_ns: int = 20_000_000

    def load(self) -> RGBDSequence:
        color_paths = sorted(self.directory.glob("rgb/*.npy"))
        depth_paths = sorted(self.directory.glob("depth/*.npy"))
        if len(color_paths) != len(depth_paths) or len(color_paths) != len(self.timestamps_ns):
            raise ValueError("NumPy RGB/depth files and timestamps must have equal length")
        colors = [
            np.asarray(np.load(path, allow_pickle=False), dtype=np.uint8)
            for path in color_paths
        ]
        depths = [
            np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
            for path in depth_paths
        ]
        return ArrayImageSequenceAdapter(
            color_images_rgb=colors,
            depth_images_m=depths,
            color_timestamps_ns=self.timestamps_ns,
            depth_timestamps_ns=self.timestamps_ns,
            color_intrinsics=self.intrinsics,
            maximum_timestamp_skew_ns=self.maximum_timestamp_skew_ns,
            raw_capture_fps=self.raw_capture_fps,
        ).load()


# Descriptive alias for integrations that call this an image-sequence capture.
ImageSequenceCapture = ArrayImageSequenceAdapter

__all__ = [
    "ArrayImageSequenceAdapter",
    "ImageSequenceCapture",
    "NumpyDirectorySequenceLoader",
    "RGBDImageSequenceLoader",
]
