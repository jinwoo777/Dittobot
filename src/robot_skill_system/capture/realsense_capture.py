"""Optional, lazily imported Intel RealSense capture adapter.

Importing this module never imports ``pyrealsense2``.  Core and mock workflows
therefore remain usable on machines without RealSense software or hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

import numpy as np

from .interfaces import (
    CameraIntrinsics,
    CaptureError,
    CaptureMode,
    CaptureRequest,
    CaptureResult,
    NotConfiguredError,
    SynchronizedRGBDFrame,
    depth_median_consensus,
)

HOST_UNIX_EPOCH_CLOCK_DOMAIN = "host_unix_epoch"


def _normalise_timestamp_clock_domain(frame: Any) -> str:
    """Return a stable name for a pyrealsense2 frame timestamp domain."""

    getter = getattr(frame, "get_frame_timestamp_domain", None)
    if not callable(getter):
        return "unknown"
    domain = getter()
    name = getattr(domain, "name", None)
    text = str(name if isinstance(name, str) else domain)
    # pybind enum strings are commonly ``timestamp_domain.hardware_clock``.
    return text.rsplit(".", maxsplit=1)[-1].strip().lower() or "unknown"


class RealSenseTimestampMapper:
    """Map stable RealSense clock domains onto a host Unix-epoch anchor.

    One offset is established per device clock domain.  A shared color/depth
    domain is anchored using the pair midpoint, which preserves both the raw
    inter-frame interval and the color/depth skew exactly.
    """

    def __init__(self) -> None:
        self._anchors: dict[str, tuple[int, int]] = {}

    def reset(self) -> None:
        """Discard mappings when the RealSense pipeline/device clock restarts."""

        self._anchors.clear()

    def map_pair(
        self,
        *,
        raw_color_timestamp_ns: int,
        raw_depth_timestamp_ns: int,
        color_clock_domain: str,
        depth_clock_domain: str,
        observed_host_unix_epoch_ns: int,
    ) -> tuple[int, int]:
        """Return color/depth timestamps in host Unix-epoch nanoseconds."""

        if raw_color_timestamp_ns < 0 or raw_depth_timestamp_ns < 0:
            raise ValueError("raw RealSense timestamps must be non-negative")
        if observed_host_unix_epoch_ns < 0:
            raise ValueError("observed host Unix timestamp must be non-negative")
        if not color_clock_domain.strip() or not depth_clock_domain.strip():
            raise ValueError("RealSense timestamp clock domains must be non-empty")

        if color_clock_domain == depth_clock_domain:
            raw_midpoint_ns = (raw_color_timestamp_ns + raw_depth_timestamp_ns) // 2
            self._set_anchor_if_missing(
                color_clock_domain,
                raw_midpoint_ns,
                observed_host_unix_epoch_ns,
            )
        else:
            self._set_anchor_if_missing(
                color_clock_domain,
                raw_color_timestamp_ns,
                observed_host_unix_epoch_ns,
            )
            self._set_anchor_if_missing(
                depth_clock_domain,
                raw_depth_timestamp_ns,
                observed_host_unix_epoch_ns,
            )

        return (
            self._map(raw_color_timestamp_ns, color_clock_domain),
            self._map(raw_depth_timestamp_ns, depth_clock_domain),
        )

    def _set_anchor_if_missing(
        self, clock_domain: str, raw_timestamp_ns: int, host_unix_epoch_ns: int
    ) -> None:
        if clock_domain not in self._anchors:
            self._anchors[clock_domain] = (raw_timestamp_ns, host_unix_epoch_ns)

    def _map(self, raw_timestamp_ns: int, clock_domain: str) -> int:
        raw_anchor_ns, host_anchor_ns = self._anchors[clock_domain]
        mapped_timestamp_ns = host_anchor_ns + raw_timestamp_ns - raw_anchor_ns
        if mapped_timestamp_ns < 0:
            raise ValueError("mapped RealSense timestamp precedes the Unix epoch")
        return mapped_timestamp_ns


def _load_pyrealsense2() -> ModuleType:
    try:
        import pyrealsense2
    except ImportError as exc:
        raise NotConfiguredError(
            "RealSense capture requires the optional pyrealsense2 package"
        ) from exc
    return cast(ModuleType, pyrealsense2)


@dataclass(frozen=True)
class RealSenseCaptureConfig:
    width_px: int = 640
    height_px: int = 480
    frames_per_second: int = 30
    device_serial: str | None = None
    default_mode: CaptureMode = CaptureMode.BURST
    default_burst_frame_count: int = 5
    wait_timeout_ms: int = 5000
    enable_spatial_filter: bool = True
    enable_temporal_filter: bool = True
    enable_hole_filling: bool = False

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0 or self.frames_per_second <= 0:
            raise ValueError("RealSense stream dimensions and rate must be positive")
        if self.default_burst_frame_count < 1 or self.wait_timeout_ms <= 0:
            raise ValueError("RealSense counts and timeouts must be positive")


class RealSenseCapture:
    """Direct RealSense adapter with RGB-depth alignment and optional filtering."""

    def __init__(
        self,
        config: RealSenseCaptureConfig | None = None,
        *,
        epoch_clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.config = config or RealSenseCaptureConfig()
        self._epoch_clock_ns = epoch_clock_ns
        self._timestamp_mapper = RealSenseTimestampMapper()
        self._rs: ModuleType | None = None
        self._pipeline: Any | None = None
        self._align: Any | None = None
        self._depth_scale_m = 0.0
        self._filters: list[Any] = []

    def start(self) -> None:
        if self._pipeline is not None:
            return
        rs = _load_pyrealsense2()
        pipeline = rs.pipeline()
        stream_config = rs.config()
        if self.config.device_serial:
            stream_config.enable_device(self.config.device_serial)
        stream_config.enable_stream(
            rs.stream.color,
            self.config.width_px,
            self.config.height_px,
            rs.format.rgb8,
            self.config.frames_per_second,
        )
        stream_config.enable_stream(
            rs.stream.depth,
            self.config.width_px,
            self.config.height_px,
            rs.format.z16,
            self.config.frames_per_second,
        )
        try:
            profile = pipeline.start(stream_config)
        except Exception as exc:
            raise NotConfiguredError(
                "RealSense pipeline could not start; verify device and stream configuration"
            ) from exc
        try:
            self._timestamp_mapper.reset()
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale_m = float(depth_sensor.get_depth_scale())
            self._align = rs.align(rs.stream.color)
            filters: list[Any] = []
            if self.config.enable_spatial_filter:
                filters.append(rs.spatial_filter())
            if self.config.enable_temporal_filter:
                filters.append(rs.temporal_filter())
            if self.config.enable_hole_filling:
                filters.append(rs.hole_filling_filter())
            self._filters = filters
            self._rs = rs
            self._pipeline = pipeline
        except Exception:
            pipeline.stop()
            raise

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None
        self._filters = []
        self._timestamp_mapper.reset()

    def __enter__(self) -> RealSenseCapture:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()

    def _capture_frame(self) -> SynchronizedRGBDFrame:
        if self._pipeline is None or self._align is None:
            raise NotConfiguredError("RealSense capture has not been started")
        try:
            frames = self._pipeline.wait_for_frames(self.config.wait_timeout_ms)
            observed_host_unix_epoch_ns = self._epoch_clock_ns()
            aligned = self._align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise CaptureError("RealSense returned an incomplete RGB-D pair")
            for filter_object in self._filters:
                depth_frame = filter_object.process(depth_frame)
            color = np.asanyarray(color_frame.get_data()).astype(np.uint8, copy=False)
            raw_depth = np.asanyarray(depth_frame.get_data())
            depth_m = np.asarray(raw_depth, dtype=np.float32) * self._depth_scale_m
            video_profile = color_frame.profile.as_video_stream_profile()
            intrinsic = video_profile.intrinsics
            intrinsics = CameraIntrinsics(
                width_px=int(intrinsic.width),
                height_px=int(intrinsic.height),
                fx_px=float(intrinsic.fx),
                fy_px=float(intrinsic.fy),
                cx_px=float(intrinsic.ppx),
                cy_px=float(intrinsic.ppy),
                distortion_model=str(intrinsic.model),
                distortion_coefficients=tuple(float(value) for value in intrinsic.coeffs),
            )
            raw_color_timestamp_ns = int(
                round(float(color_frame.get_timestamp()) * 1.0e6)
            )
            raw_depth_timestamp_ns = int(
                round(float(depth_frame.get_timestamp()) * 1.0e6)
            )
            raw_color_clock_domain = _normalise_timestamp_clock_domain(color_frame)
            raw_depth_clock_domain = _normalise_timestamp_clock_domain(depth_frame)
            color_timestamp_ns, depth_timestamp_ns = self._timestamp_mapper.map_pair(
                raw_color_timestamp_ns=raw_color_timestamp_ns,
                raw_depth_timestamp_ns=raw_depth_timestamp_ns,
                color_clock_domain=raw_color_clock_domain,
                depth_clock_domain=raw_depth_clock_domain,
                observed_host_unix_epoch_ns=observed_host_unix_epoch_ns,
            )
            return SynchronizedRGBDFrame(
                color_image_rgb=color,
                depth_image_m=depth_m,
                color_timestamp_ns=color_timestamp_ns,
                depth_timestamp_ns=depth_timestamp_ns,
                color_intrinsics=intrinsics,
                frame_number=int(color_frame.get_frame_number()),
                depth_scale_m=self._depth_scale_m,
                timestamp_clock_domain=HOST_UNIX_EPOCH_CLOCK_DOMAIN,
                raw_color_timestamp_ns=raw_color_timestamp_ns,
                raw_depth_timestamp_ns=raw_depth_timestamp_ns,
                raw_color_timestamp_clock_domain=raw_color_clock_domain,
                raw_depth_timestamp_clock_domain=raw_depth_clock_domain,
            )
        except (CaptureError, ValueError):
            raise
        except Exception as exc:
            raise CaptureError("RealSense frame capture failed") from exc

    def capture(self, request: CaptureRequest | None = None) -> CaptureResult:
        if self._pipeline is None:
            self.start()
        effective_request = request or CaptureRequest(
            mode=self.config.default_mode,
            frame_count=self.config.default_burst_frame_count,
        )
        frames = tuple(
            self._capture_frame() for _ in range(effective_request.effective_frame_count)
        )
        return CaptureResult(
            request=effective_request,
            frames=frames,
            consensus_depth_m=depth_median_consensus(frames),
        )

    def stream(self) -> Iterator[SynchronizedRGBDFrame]:
        if self._pipeline is None:
            self.start()
        while self._pipeline is not None:
            yield self._capture_frame()


__all__ = [
    "HOST_UNIX_EPOCH_CLOCK_DOMAIN",
    "NotConfiguredError",
    "RealSenseCapture",
    "RealSenseCaptureConfig",
    "RealSenseTimestampMapper",
]
