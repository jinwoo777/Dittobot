from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from robot_skill_system.capture.interfaces import NotConfiguredError, SynchronizedRGBDFrame
from robot_skill_system.demonstrations import rgbd_dataset
from robot_skill_system.demonstrations.rgbd_dataset import (
    RGBDDatasetValidationError,
    load_rgbd_dataset,
    segment_rgbd_dataset,
)
from robot_skill_system.perception.hand_pose import (
    FingerGripperState,
    FingerObservation,
    FingerTrackingResult,
    classify_gripper_state,
)
from robot_skill_system.scene.models import Vector3


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_dataset(root: Path, *, frame_count: int = 12) -> dict[str, object]:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    frames: list[dict[str, object]] = []
    for index in range(frame_count):
        rgb_payload = b"\xff\xd8" + f"test-frame-{index}".encode() + b"\xff\xd9"
        depth_buffer = io.BytesIO()
        np.savez_compressed(
            depth_buffer,
            depth_m=np.ones((2, 3), dtype=np.float32),
        )
        depth_payload = depth_buffer.getvalue()
        rgb_name = f"{index:06d}.jpg"
        depth_name = f"{index:06d}.npz"
        (root / "rgb" / rgb_name).write_bytes(rgb_payload)
        (root / "depth" / depth_name).write_bytes(depth_payload)
        timestamp_ns = index * 100_000_000
        frames.append(
            {
                "index": index,
                "frame_number": 100 + index,
                "timestamp_ns": timestamp_ns,
                "color_timestamp_ns": timestamp_ns,
                "depth_timestamp_ns": timestamp_ns,
                "timestamp_clock_domain": "host_unix_epoch",
                "raw_color_timestamp_ns": timestamp_ns,
                "raw_depth_timestamp_ns": timestamp_ns,
                "raw_color_timestamp_clock_domain": "hardware_clock",
                "raw_depth_timestamp_clock_domain": "hardware_clock",
                "reference_frame": "camera_color_optical_frame",
                "depth_scale_m": 0.001,
                # These intentionally point at the pre-relocation directory.
                "rgb_uri": f"demonstrations/rgbd_original/rgb/{rgb_name}",
                "depth_uri": f"demonstrations/rgbd_original/depth/{depth_name}",
                "rgb_checksum_sha256": _sha256(rgb_payload),
                "depth_checksum_sha256": _sha256(depth_payload),
                "color_intrinsics": {
                    "width_px": 3,
                    "height_px": 2,
                    "fx_px": 2.0,
                    "fy_px": 2.0,
                    "cx_px": 1.0,
                    "cy_px": 0.5,
                    "distortion_model": "none",
                    "distortion_coefficients": [],
                },
            }
        )
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "recording_id": "rgbd_original",
        "status": "finished",
        "error": None,
        "frame_count": frame_count,
        "recording_fps": 10.0,
        "timestamps_preserved": True,
        "rgb_format": "jpeg",
        "depth_format": "numpy_npz_float32_metres",
        "depth_aligned_to_color": True,
        "frames": frames,
    }
    (root / "rgbd_manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )
    return manifest


def _test_decoder(_payload: bytes) -> np.ndarray:
    return np.zeros((2, 3, 3), dtype=np.uint8)


def test_relocated_dataset_uses_local_index_and_rechecks_bytes(tmp_path: Path) -> None:
    root = tmp_path / "hammer1"
    _write_dataset(root, frame_count=2)

    dataset = load_rgbd_dataset(root, rgb_decoder=_test_decoder)

    assert len(dataset) == 2
    assert dataset.manifest.recording_id == "rgbd_original"
    assert dataset.frames[0].recorded_rgb_uri.startswith("demonstrations/rgbd_original/")
    assert dataset.frames[0].rgb_path == (root / "rgb" / "000000.jpg").resolve()
    frame = dataset.load_frame(0)
    assert frame.color_image_rgb.shape == (2, 3, 3)
    assert frame.depth_image_m.dtype == np.float32
    assert np.all(frame.depth_image_m == 1.0)

    # Eager validation is not treated as a permanent trust decision.
    (root / "rgb" / "000000.jpg").write_bytes(b"\xff\xd8tampered\xff\xd9")
    with pytest.raises(RGBDDatasetValidationError, match="checksum"):
        dataset.load_frame(0)


def test_dataset_rejects_checksum_mismatch_and_unsafe_recorded_uri(tmp_path: Path) -> None:
    mismatched = tmp_path / "mismatched"
    mismatch_manifest = _write_dataset(mismatched, frame_count=1)
    mismatch_frames = mismatch_manifest["frames"]
    assert isinstance(mismatch_frames, list)
    assert isinstance(mismatch_frames[0], dict)
    mismatch_frames[0]["depth_checksum_sha256"] = "0" * 64
    (mismatched / "rgbd_manifest.json").write_text(
        json.dumps(mismatch_manifest), encoding="utf-8"
    )
    with pytest.raises(RGBDDatasetValidationError, match="checksum"):
        load_rgbd_dataset(mismatched, rgb_decoder=_test_decoder)

    unsafe = tmp_path / "unsafe"
    unsafe_manifest = _write_dataset(unsafe, frame_count=1)
    unsafe_frames = unsafe_manifest["frames"]
    assert isinstance(unsafe_frames, list)
    assert isinstance(unsafe_frames[0], dict)
    unsafe_frames[0]["rgb_uri"] = "../../rgb/000000.jpg"
    (unsafe / "rgbd_manifest.json").write_text(
        json.dumps(unsafe_manifest), encoding="utf-8"
    )
    with pytest.raises(RGBDDatasetValidationError, match="unsafe path"):
        load_rgbd_dataset(unsafe, rgb_decoder=_test_decoder)


class _SequenceTracker:
    def __init__(self) -> None:
        self.seen: list[int] = []

    def track(self, frame: SynchronizedRGBDFrame) -> FingerTrackingResult:
        self.seen.append(frame.frame_number)
        frame_index = frame.frame_number - 100
        distance_m = 0.05 if frame_index < 3 or frame_index >= 6 else 0.02
        half_distance = distance_m / 2.0
        observation = FingerObservation(
            frame_number=frame.frame_number,
            timestamp_ns=frame.timestamp_ns,
            reference_frame=frame.reference_frame,
            status="valid",
            thumb_normalized_xy=(0.25, 0.5),
            index_normalized_xy=(0.75, 0.5),
            thumb_pixel_xy=(0, 1),
            index_pixel_xy=(2, 1),
            thumb_depth_m=1.0,
            index_depth_m=1.0,
            thumb_point_camera_m=Vector3(x=-half_distance, y=0.0, z=1.0),
            index_point_camera_m=Vector3(x=half_distance, y=0.0, z=1.0),
            midpoint_camera_m=Vector3(x=0.0, y=0.0, z=1.0),
            distance_m=distance_m,
            candidate_state=classify_gripper_state(distance_m),
            stabilization_progress_frames=1,
            required_stable_frames=3,
            stable_state=(
                FingerGripperState.OPEN if frame_index >= 8 else None
            ),
            confidence=1.0,
        )
        return FingerTrackingResult(observation=observation)


def test_dataset_tracks_every_frame_then_segments_grip_action_end(tmp_path: Path) -> None:
    root = tmp_path / "hammer1"
    _write_dataset(root)
    dataset = load_rgbd_dataset(root, rgb_decoder=_test_decoder)
    tracker = _SequenceTracker()

    result = segment_rgbd_dataset(dataset, estimator=tracker)

    assert tracker.seen == list(range(100, 112))
    assert len(result.observations) == 12
    assert result.segmentation.grip.end_frame_index == 5
    assert result.segmentation.action.start_frame_index == 5
    assert result.segmentation.action.end_frame_index == 6
    assert result.segmentation.end_motion.start_frame_index == 6
    assert result.segmentation.end_motion.end_frame_index == 11


def test_default_segmentation_reports_optional_mediapipe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "hammer1"
    _write_dataset(root, frame_count=1)
    dataset = load_rgbd_dataset(root, rgb_decoder=_test_decoder)

    class _UnavailableMediaPipe:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def track(self, _frame: SynchronizedRGBDFrame) -> FingerTrackingResult:
            raise NotConfiguredError("optional mediapipe package is absent")

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        rgbd_dataset,
        "MediaPipeHandPoseEstimator",
        _UnavailableMediaPipe,
    )

    with pytest.raises(NotConfiguredError, match="dataset checksum"):
        segment_rgbd_dataset(dataset)
