from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import robot_skill_system.application as application_module
from robot_skill_system.api.contracts import RecordingSkillDraftRequest
from robot_skill_system.application import MVPApplication
from robot_skill_system.capture.interfaces import CameraIntrinsics, SynchronizedRGBDFrame
from robot_skill_system.exceptions import NotConfiguredError
from robot_skill_system.perception.hand_pose import (
    FingerStateStabilizer,
    MediaPipeHandPoseEstimator,
)
from robot_skill_system.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        {"ARTIFACT_ROOT": str(tmp_path / "artifacts")},
        root=Path(__file__).resolve().parents[2],
    )


def _frame(frame_index: int) -> SynchronizedRGBDFrame:
    timestamp_ns = 1_000_000_000 + frame_index * 100_000_000
    return SynchronizedRGBDFrame(
        color_image_rgb=np.zeros((101, 101, 3), dtype=np.uint8),
        depth_image_m=np.ones((101, 101), dtype=np.float32),
        color_timestamp_ns=timestamp_ns,
        depth_timestamp_ns=timestamp_ns + 100_000,
        color_intrinsics=CameraIntrinsics(
            width_px=101,
            height_px=101,
            fx_px=100.0,
            fy_px=100.0,
            cx_px=50.0,
            cy_px=50.0,
        ),
        frame_number=frame_index,
    )


def _manifest(frame_count: int) -> dict[str, Any]:
    return {
        "frame_count": frame_count,
        "depth_aligned_to_color": True,
        "frames": [
            {
                "index": index,
                "frame_number": index,
                "color_timestamp_ns": 1_000_000_000 + index * 100_000_000,
                "depth_timestamp_ns": 1_000_100_000 + index * 100_000_000,
                "reference_frame": "camera_color_optical_frame",
            }
            for index in range(frame_count)
        ],
    }


class _FrameController:
    def __init__(self, *, corrupt_index: int | None = None) -> None:
        self.corrupt_index = corrupt_index
        self.loaded_indices: list[int] = []

    def load_recording_rgbd_frame(
        self, _recording_id: str, frame_index: int
    ) -> SynchronizedRGBDFrame:
        self.loaded_indices.append(frame_index)
        if frame_index == self.corrupt_index:
            raise ValueError("corrupted frame")
        return _frame(frame_index)


def _service(settings: Settings, controller: _FrameController) -> MVPApplication:
    service = object.__new__(MVPApplication)
    service.settings = settings
    service.camera_controller = controller  # type: ignore[assignment]
    return service


def test_full_manifest_tracking_resets_on_frame_error_and_orders_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ScriptedTracker(MediaPipeHandPoseEstimator):
        def track(self, frame: SynchronizedRGBDFrame):  # type: ignore[no-untyped-def]
            landmarks = (
                {4: (0.2, 0.5), 8: (0.8, 0.5)}
                if frame.frame_number <= 5
                else {4: (0.49, 0.5), 8: (0.51, 0.5)}
            )
            return self.track_normalized_landmarks(frame, landmarks, confidence=0.9)

    monkeypatch.setattr(application_module, "MediaPipeHandPoseEstimator", ScriptedTracker)
    controller = _FrameController(corrupt_index=2)
    tracking = _service(_settings(tmp_path), controller)._track_recording_fingertips(
        "rgbd_test", _manifest(9)
    )

    assert controller.loaded_indices == list(range(9))
    assert tracking["processed_frame_count"] == 9
    assert tracking["valid_frame_count"] == 8
    assert [item["frame_index"] for item in tracking["finger_observations"]] == list(
        range(9)
    )
    assert tracking["finger_observations"][2]["status"] == "uncertain"
    assert tracking["finger_observations"][2]["stabilization_progress_frames"] == 0
    assert tracking["invalid_frames"] == [
        {
            "frame_index": 2,
            "timestamp_ns": 1_200_050_000,
            "reason": "ValueError: corrupted frame",
        }
    ]
    assert [
        (item["frame_index"], item["previous_state"], item["state"])
        for item in tracking["state_transitions"]
    ] == [(5, None, "open"), (8, "open", "closed")]
    assert len(tracking["compact_fingertip_trace"]) == 9
    serialized_trace = json.dumps(tracking["compact_fingertip_trace"])
    assert "distance_m" not in serialized_trace
    assert "depth_m" not in serialized_trace
    assert "candidate_state" not in serialized_trace


def test_missing_optional_mediapipe_yields_uncertain_evidence_for_every_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MissingTracker:
        def __init__(self, *, stable_frames: int, **_kwargs: object) -> None:
            self.state_stabilizer = FingerStateStabilizer(stable_frames=stable_frames)
            self.track_calls = 0

        def track(self, _frame: SynchronizedRGBDFrame) -> object:
            self.track_calls += 1
            raise NotConfiguredError("mediapipe package unavailable")

        def close(self) -> None:
            pass

    monkeypatch.setattr(application_module, "MediaPipeHandPoseEstimator", MissingTracker)
    controller = _FrameController()
    tracking = _service(_settings(tmp_path), controller)._track_recording_fingertips(
        "rgbd_test", _manifest(4)
    )

    assert controller.loaded_indices == [0, 1, 2, 3]
    assert tracking["processed_frame_count"] == 4
    assert tracking["valid_frame_count"] == 0
    assert tracking["state_transitions"] == []
    assert tracking["optional_dependency_error"] == "mediapipe package unavailable"
    assert [item["reason"] for item in tracking["invalid_frames"]] == [
        "mediapipe_not_configured"
    ] * 4
    assert all(
        item["stabilization_progress_frames"] == 0
        for item in tracking["finger_observations"]
    )


def test_full_manifest_tracking_rejects_unaligned_depth_before_processing(
    tmp_path: Path,
) -> None:
    controller = _FrameController()
    manifest = _manifest(2)
    manifest["depth_aligned_to_color"] = False

    with pytest.raises(ValueError, match="requires aligned depth"):
        _service(_settings(tmp_path), controller)._track_recording_fingertips(
            "rgbd_test", manifest
        )
    assert controller.loaded_indices == []


def test_recording_draft_capabilities_expose_selectable_rgb_contract(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    service = _service(settings, _FrameController())

    capabilities = service.get_recording_skill_draft_capabilities()
    assert capabilities["maximum_keyframes"] == settings.openai_max_keyframes
    assert capabilities["uploaded_rgb_frame_count"] == (
        "operator_selected_1_to_maximum_keyframes"
    )
    assert capabilities["maximum_compact_trace_frames"] == 6_000
    assert capabilities["uploads_rgb_and_aligned_depth_pairs"] is False
    assert capabilities["depth_stays_local"] is True
    assert capabilities["local_finger_state_authoritative"] is True
    assert capabilities["request_keyframe_count_is_deprecated_and_ignored"] is False

    request = RecordingSkillDraftRequest(
        recording_id="rgbd_0123456789abcdef0123456789abcdef",
        operator_instruction="두 손가락으로 작업한다",
    )
    assert request.model_dump()["keyframe_count"] == 8
    assert "deprecated" not in RecordingSkillDraftRequest.model_json_schema()[
        "properties"
    ]["keyframe_count"]


def test_complete_rgbd_promotion_evidence_requires_full_trace_coverage() -> None:
    payload = {
        "full_recording_frame_count": 4,
        "transport": {"trace_frame_count": 4},
        "local_fingertip_tracking": {"tracking_id": "track_01"},
    }

    assert MVPApplication._recording_has_complete_rgbd_evidence(payload) is True
    assert (
        MVPApplication._recording_has_complete_rgbd_evidence(
            {**payload, "transport": {"trace_frame_count": 3}}
        )
        is False
    )
    assert (
        MVPApplication._recording_has_complete_rgbd_evidence(
            {**payload, "local_fingertip_tracking": None}
        )
        is False
    )


def test_recording_start_rejects_duration_that_cannot_keep_every_trace_frame(
    tmp_path: Path,
) -> None:
    settings = Settings.from_env(
        {
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "REALSENSE_FRAMES_PER_SECOND": "90",
            "REALSENSE_RECORDING_FRAMES_PER_SECOND": "90",
            "REALSENSE_MAXIMUM_RECORDING_DURATION_S": "600",
        },
        root=Path(__file__).resolve().parents[2],
    )
    service = _service(settings, _FrameController())

    with pytest.raises(ValueError, match="full-frame fingertip trace limit"):
        service.start_camera_recording({"maximum_duration_s": 100.0})
