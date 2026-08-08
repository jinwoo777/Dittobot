from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from robot_skill_system.capture import (
    CameraStateError,
    MockCapture,
    MockCaptureConfig,
    RGBDCameraController,
    RGBDRecordingWriter,
)
from robot_skill_system.storage.artifact_store import LocalArtifactStore


def _controller(tmp_path: Path) -> RGBDCameraController:
    store = LocalArtifactStore(tmp_path)
    return RGBDCameraController(
        lambda: MockCapture(
            MockCaptureConfig(width_px=32, height_px=24, frames_per_second=30.0)
        ),
        store,
        frames_per_second=30.0,
        recording_frames_per_second=10.0,
        maximum_recording_duration_s=2.0,
        startup_timeout_s=2.0,
    )


def test_preview_and_recording_persist_synchronized_rgbd_artifacts(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    try:
        status = controller.start_preview()
        assert status["state"] == "streaming"
        assert status["preview_ready"] is True
        rgb_chunk = next(controller.iter_mjpeg("rgb"))
        depth_chunk = next(controller.iter_mjpeg("depth"))
        assert b"Content-Type: image/jpeg" in rgb_chunk
        assert b"Content-Type: image/jpeg" in depth_chunk

        recording = controller.start_recording(maximum_duration_s=1.0)
        time.sleep(0.12)
        summary = controller.stop_recording(recording["recording_id"])

        assert summary["status"] == "finished"
        assert summary["frame_count"] >= 1
        assert summary["timestamps_preserved"] is True
        assert summary["depth_aligned_to_color"] is True
        assert summary["raw_capture_fps"] == 30.0
        assert summary["recording_fps"] == 10.0
        assert summary["manifest_uri"]
        manifest_path = controller._store.path_for(  # noqa: SLF001 - artifact assertion
            summary["manifest_uri"]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(manifest["frames"]) == summary["frame_count"]
        first = manifest["frames"][0]
        assert first["rgb_uri"].endswith(".jpg")
        assert first["depth_uri"].endswith(".npz")
        assert controller._store.path_for(first["rgb_uri"]).is_file()  # noqa: SLF001
        assert controller._store.path_for(first["depth_uri"]).is_file()  # noqa: SLF001
    finally:
        assert controller.stop_preview()["state"] == "stopped"


def test_recording_requires_preview_and_enforces_duration_bound(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    try:
        with pytest.raises(CameraStateError, match="preview"):
            controller.start_recording(maximum_duration_s=1.0)
        controller.start_preview()
        with pytest.raises(ValueError, match="must be in"):
            controller.start_recording(maximum_duration_s=3.0)
    finally:
        controller.close()


def test_recording_writer_time_samples_raw_frames_at_ten_fps(tmp_path: Path) -> None:
    now_ns = [0]
    capture = MockCapture(
        MockCaptureConfig(width_px=8, height_px=6, frames_per_second=30.0)
    )
    capture.start()
    frame = next(capture.stream())
    writer = RGBDRecordingWriter(
        LocalArtifactStore(tmp_path),
        recording_id="rgbd_rate_test",
        frames_per_second=30.0,
        recording_frames_per_second=10.0,
        maximum_duration_s=1.0,
        monotonic_clock_ns=lambda: now_ns[0],
        epoch_clock_ns=lambda: 1_000_000_000 + now_ns[0],
    )

    for timestamp_ms in (0, 33, 66, 99, 100, 133, 166, 199, 200, 233, 266, 299):
        now_ns[0] = timestamp_ms * 1_000_000
        assert writer.submit(frame) is True

    summary = writer.stop()

    assert summary["frame_count"] == 3
    assert summary["raw_capture_fps"] == 30.0
    assert summary["recording_fps"] == 10.0
    assert summary["dropped_frame_count"] == 0


def test_finalized_recording_is_discovered_and_previewed_after_restart(
    tmp_path: Path,
) -> None:
    first = _controller(tmp_path)
    try:
        first.start_preview()
        recording = first.start_recording(maximum_duration_s=1.0)
        time.sleep(0.12)
        finished = first.stop_recording(recording["recording_id"])
    finally:
        first.close()

    restarted = _controller(tmp_path)
    try:
        catalog = restarted.list_recordings()
        assert catalog["invalid_recording_count"] == 0
        assert [item["recording_id"] for item in catalog["recordings"]] == [
            finished["recording_id"]
        ]
        persisted = restarted.get_recording(finished["recording_id"])
        assert persisted["frame_count"] == finished["frame_count"]
        assert persisted["recording_fps"] == 10.0
        assert restarted.get_recording_frame_jpeg(
            finished["recording_id"], 0, "rgb"
        ).startswith(b"\xff\xd8")
        assert restarted.get_recording_frame_jpeg(
            finished["recording_id"], 0, "depth"
        ).startswith(b"\xff\xd8")
        selected = restarted.select_recording_keyframes(
            finished["recording_id"], maximum_count=8
        )
        assert selected
        assert all(path.is_file() for _index, path in selected)
        selected_rgbd = restarted.select_recording_rgbd_keyframes(
            finished["recording_id"], maximum_count=8
        )
        assert [index for index, _rgb, _depth in selected_rgbd] == [
            index for index, _rgb in selected
        ]
        assert all(
            rgb_path.read_bytes().startswith(b"\xff\xd8")
            and depth_path.read_bytes().startswith(b"\xff\xd8")
            for _index, rgb_path, depth_path in selected_rgbd
        )
    finally:
        restarted.close()


def test_imported_recording_directory_is_discovered_and_previewed(
    tmp_path: Path,
) -> None:
    first = _controller(tmp_path)
    try:
        first.start_preview()
        recording = first.start_recording(maximum_duration_s=1.0)
        time.sleep(0.12)
        finished = first.stop_recording(recording["recording_id"])
    finally:
        first.close()

    canonical_directory = tmp_path / "demonstrations" / finished["recording_id"]
    imported_directory = tmp_path / "demonstrations" / "hammer1"
    canonical_directory.rename(imported_directory)

    restarted = _controller(tmp_path)
    try:
        catalog = restarted.list_recordings()
        assert catalog["invalid_recording_count"] == 0
        assert len(catalog["recordings"]) == 1
        imported = catalog["recordings"][0]
        assert imported["recording_id"] == finished["recording_id"]
        assert imported["source_label"] == "hammer1"
        assert imported["manifest_uri"] == "demonstrations/hammer1/rgbd_manifest.json"

        assert restarted.get_recording_frame_jpeg(
            finished["recording_id"], 0, "rgb"
        ).startswith(b"\xff\xd8")
        assert restarted.get_recording_frame_jpeg(
            finished["recording_id"], 0, "depth"
        ).startswith(b"\xff\xd8")
        selected = restarted.select_recording_keyframes(
            finished["recording_id"], maximum_count=8
        )
        assert selected
        assert all(path.parent == imported_directory / "rgb" for _index, path in selected)
        assert all(path.is_file() for _index, path in selected)
    finally:
        restarted.close()
