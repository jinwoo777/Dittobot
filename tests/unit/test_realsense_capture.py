from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_skill_system.capture import (
    HOST_UNIX_EPOCH_CLOCK_DOMAIN,
    CaptureMode,
    CaptureRequest,
    RealSenseCapture,
    RealSenseCaptureConfig,
    RealSenseTimestampMapper,
)


class _FakeVideoProfile:
    def __init__(self) -> None:
        self.intrinsics = SimpleNamespace(
            width=2,
            height=2,
            fx=2.0,
            fy=2.0,
            ppx=0.5,
            ppy=0.5,
            model="none",
            coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
        )

    def as_video_stream_profile(self) -> _FakeVideoProfile:
        return self


class _FakeFrame:
    def __init__(
        self,
        data: np.ndarray,
        *,
        timestamp_ms: float,
        frame_number: int,
        clock_domain: str = "hardware_clock",
    ) -> None:
        self._data = data
        self._timestamp_ms = timestamp_ms
        self._frame_number = frame_number
        self._clock_domain = SimpleNamespace(name=clock_domain)
        self.profile = _FakeVideoProfile()

    def get_data(self) -> np.ndarray:
        return self._data

    def get_timestamp(self) -> float:
        return self._timestamp_ms

    def get_frame_timestamp_domain(self) -> SimpleNamespace:
        return self._clock_domain

    def get_frame_number(self) -> int:
        return self._frame_number


class _FakeFrameSet:
    def __init__(self, color_frame: _FakeFrame, depth_frame: _FakeFrame) -> None:
        self._color_frame = color_frame
        self._depth_frame = depth_frame

    def get_color_frame(self) -> _FakeFrame:
        return self._color_frame

    def get_depth_frame(self) -> _FakeFrame:
        return self._depth_frame


class _FakePipeline:
    def __init__(self, frame_sets: list[_FakeFrameSet]) -> None:
        self._frame_sets = frame_sets

    def wait_for_frames(self, timeout_ms: int) -> _FakeFrameSet:
        assert timeout_ms > 0
        return self._frame_sets.pop(0)

    def stop(self) -> None:
        return None


class _FakeAlign:
    def process(self, frames: _FakeFrameSet) -> _FakeFrameSet:
        return frames


def _fake_frame_set(timestamp_ms: float, frame_number: int) -> _FakeFrameSet:
    color = np.full((2, 2, 3), 7, dtype=np.uint8)
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    return _FakeFrameSet(
        _FakeFrame(
            color,
            timestamp_ms=timestamp_ms,
            frame_number=frame_number,
        ),
        _FakeFrame(
            depth,
            timestamp_ms=timestamp_ms + 0.4,
            frame_number=frame_number,
        ),
    )


def test_timestamp_mapper_anchors_pair_and_preserves_device_intervals() -> None:
    mapper = RealSenseTimestampMapper()
    host_epoch_ns = 1_700_000_000_000_000_000
    first = mapper.map_pair(
        raw_color_timestamp_ns=10_000_000_000,
        raw_depth_timestamp_ns=10_000_400_000,
        color_clock_domain="hardware_clock",
        depth_clock_domain="hardware_clock",
        observed_host_unix_epoch_ns=host_epoch_ns,
    )
    second = mapper.map_pair(
        raw_color_timestamp_ns=10_033_000_000,
        raw_depth_timestamp_ns=10_033_400_000,
        color_clock_domain="hardware_clock",
        depth_clock_domain="hardware_clock",
        # Once anchored, host callback jitter cannot alter device intervals.
        observed_host_unix_epoch_ns=host_epoch_ns + 900_000_000,
    )

    assert (first[0] + first[1]) // 2 == host_epoch_ns
    assert first[1] - first[0] == 400_000
    assert second[0] - first[0] == 33_000_000
    assert second[1] - first[1] == 33_000_000

    mapper.reset()
    reset_mapping = mapper.map_pair(
        raw_color_timestamp_ns=10_000_000_000,
        raw_depth_timestamp_ns=10_000_400_000,
        color_clock_domain="hardware_clock",
        depth_clock_domain="hardware_clock",
        observed_host_unix_epoch_ns=host_epoch_ns + 5_000_000_000,
    )
    assert (reset_mapping[0] + reset_mapping[1]) // 2 == host_epoch_ns + 5_000_000_000


def test_fake_realsense_capture_maps_to_epoch_and_retains_raw_clock_metadata() -> None:
    host_epoch_ns = 1_700_000_000_000_000_000
    observed_times = iter((host_epoch_ns, host_epoch_ns + 900_000_000))
    capture = RealSenseCapture(
        RealSenseCaptureConfig(
            width_px=2,
            height_px=2,
            enable_spatial_filter=False,
            enable_temporal_filter=False,
        ),
        epoch_clock_ns=lambda: next(observed_times),
    )
    capture._pipeline = _FakePipeline(  # noqa: SLF001 - injected optional hardware boundary
        [_fake_frame_set(12_345.0, 11), _fake_frame_set(12_378.0, 12)]
    )
    capture._align = _FakeAlign()  # noqa: SLF001 - injected optional hardware boundary
    capture._depth_scale_m = 0.001  # noqa: SLF001 - injected optional hardware boundary

    request = CaptureRequest(mode=CaptureMode.SINGLE)
    first = capture.capture(request).representative_frame
    second = capture.capture(request).representative_frame

    assert first.timestamp_clock_domain == HOST_UNIX_EPOCH_CLOCK_DOMAIN
    assert first.timestamp_ns == host_epoch_ns
    assert first.color_timestamp_ns > 1_600_000_000_000_000_000
    assert first.raw_color_timestamp_ns == 12_345_000_000
    assert first.raw_depth_timestamp_ns == 12_345_400_000
    assert first.raw_color_timestamp_clock_domain == "hardware_clock"
    assert first.raw_depth_timestamp_clock_domain == "hardware_clock"
    assert second.timestamp_ns - first.timestamp_ns == 33_000_000
    assert np.allclose(first.depth_image_m, 1.0)


def test_realsense_capture_skips_unsynchronized_startup_pairs() -> None:
    color = np.full((2, 2, 3), 7, dtype=np.uint8)
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    unsynchronized = _FakeFrameSet(
        _FakeFrame(color, timestamp_ms=1_000.0, frame_number=1),
        _FakeFrame(depth, timestamp_ms=1_250.0, frame_number=1),
    )
    synchronized = _fake_frame_set(1_300.0, 2)
    capture = RealSenseCapture(
        RealSenseCaptureConfig(
            width_px=2,
            height_px=2,
            enable_spatial_filter=False,
            enable_temporal_filter=False,
            synchronization_retry_count=2,
        ),
        epoch_clock_ns=lambda: 1_700_000_000_000_000_000,
    )
    capture._pipeline = _FakePipeline(  # noqa: SLF001 - injected hardware boundary
        [unsynchronized, synchronized]
    )
    capture._align = _FakeAlign()  # noqa: SLF001 - injected hardware boundary
    capture._depth_scale_m = 0.001  # noqa: SLF001 - injected hardware boundary

    frame = capture.capture(CaptureRequest(mode=CaptureMode.SINGLE)).representative_frame

    assert frame.frame_number == 2
    assert abs(frame.color_timestamp_ns - frame.depth_timestamp_ns) == 400_000
