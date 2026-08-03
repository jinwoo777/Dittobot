"""Deterministic RGB-D capture for tests and hardware-free vertical slices."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .interfaces import (
    CameraExtrinsics,
    CameraIntrinsics,
    CaptureMode,
    CaptureRequest,
    CaptureResult,
    SynchronizedRGBDFrame,
    depth_median_consensus,
)


@dataclass(frozen=True)
class MockCaptureConfig:
    """Small images keep mock flows fast while preserving RGB-D semantics."""

    width_px: int = 64
    height_px: int = 48
    frames_per_second: float = 30.0
    nominal_depth_m: float = 0.75
    default_mode: CaptureMode = CaptureMode.BURST
    default_burst_frame_count: int = 5
    start_timestamp_ns: int = 1_000_000_000

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("mock image dimensions must be positive")
        if self.frames_per_second <= 0.0:
            raise ValueError("frames_per_second must be positive")
        if self.nominal_depth_m <= 0.0:
            raise ValueError("nominal_depth_m must be positive")
        if self.default_burst_frame_count < 1:
            raise ValueError("default_burst_frame_count must be positive")


@dataclass(frozen=True)
class MockSceneFrame:
    """One mock scene trigger, represented by a single or burst capture."""

    capture: CaptureResult
    scene_hint: str = "mock_tabletop"

    @property
    def mode(self) -> CaptureMode:
        return self.capture.request.mode

    @property
    def frame_count(self) -> int:
        return len(self.capture.frames)

    @property
    def representative_frame(self) -> SynchronizedRGBDFrame:
        return self.capture.representative_frame


class MockCapture:
    """A network-free capture backend with optional pre-recorded frames."""

    def __init__(
        self,
        config: MockCaptureConfig | None = None,
        frames: Sequence[SynchronizedRGBDFrame] | None = None,
    ) -> None:
        self.config = config or MockCaptureConfig()
        self._fixture_frames = tuple(frames or ())
        self._fixture_index = 0
        self._next_frame_number = 0
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def __enter__(self) -> MockCapture:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()

    def _generated_frame(self) -> SynchronizedRGBDFrame:
        config = self.config
        frame_number = self._next_frame_number
        self._next_frame_number += 1
        period_ns = int(round(1.0e9 / config.frames_per_second))
        timestamp_ns = config.start_timestamp_ns + frame_number * period_ns
        y_grid, x_grid = np.mgrid[0 : config.height_px, 0 : config.width_px]
        color = np.empty((config.height_px, config.width_px, 3), dtype=np.uint8)
        color[..., 0] = (x_grid + frame_number) % 256
        color[..., 1] = (2 * y_grid + frame_number) % 256
        color[..., 2] = 96
        depth = np.full(
            (config.height_px, config.width_px),
            config.nominal_depth_m + frame_number * 0.0001,
            dtype=np.float32,
        )
        # A stable invalid pixel exercises consensus handling without affecting
        # the rest of the synthetic plane.
        depth[0, 0] = 0.0
        intrinsics = CameraIntrinsics(
            width_px=config.width_px,
            height_px=config.height_px,
            fx_px=float(config.width_px),
            fy_px=float(config.width_px),
            cx_px=(config.width_px - 1) / 2.0,
            cy_px=(config.height_px - 1) / 2.0,
        )
        return SynchronizedRGBDFrame(
            color_image_rgb=color,
            depth_image_m=depth,
            color_timestamp_ns=timestamp_ns,
            depth_timestamp_ns=timestamp_ns + 100_000,
            color_intrinsics=intrinsics,
            frame_number=frame_number,
            depth_scale_m=1.0,
            depth_intrinsics=intrinsics,
            depth_to_color_extrinsics=CameraExtrinsics(
                source_frame="camera_depth_optical_frame",
                target_frame="camera_color_optical_frame",
                translation_m=(0.0, 0.0, 0.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        )

    def _next_frame(self) -> SynchronizedRGBDFrame:
        if self._fixture_frames:
            frame = self._fixture_frames[self._fixture_index % len(self._fixture_frames)]
            self._fixture_index += 1
            return frame
        return self._generated_frame()

    def capture(self, request: CaptureRequest | None = None) -> CaptureResult:
        if not self._started:
            self.start()
        effective_request = request or CaptureRequest(
            mode=self.config.default_mode,
            frame_count=self.config.default_burst_frame_count,
        )
        frames = tuple(
            self._next_frame() for _ in range(effective_request.effective_frame_count)
        )
        return CaptureResult(
            request=effective_request,
            frames=frames,
            consensus_depth_m=depth_median_consensus(frames),
        )

    def capture_scene_frame(
        self, request: CaptureRequest | None = None, *, scene_hint: str = "mock_tabletop"
    ) -> MockSceneFrame:
        """Convenience wrapper used by mock scene builders."""

        return MockSceneFrame(capture=self.capture(request), scene_hint=scene_hint)

    def stream(self) -> Iterator[SynchronizedRGBDFrame]:
        if not self._started:
            self.start()
        while self._started:
            yield self._next_frame()


__all__ = ["MockCapture", "MockCaptureConfig", "MockSceneFrame"]
