from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import robot_skill_system.application as application_module
from robot_skill_system.application import MVPApplication
from robot_skill_system.capture import MockCapture, MockCaptureConfig, RGBDCameraController
from robot_skill_system.capture.interfaces import SynchronizedRGBDFrame
from robot_skill_system.perception.hand_pose import (
    FingerTrackingResult,
    MediaPipeHandPoseEstimator,
)
from robot_skill_system.settings import Settings
from robot_skill_system.storage.artifact_store import LocalArtifactStore


def test_selected_rgb_full_trace_task_plane_and_candidate_direct_pipeline(
    tmp_path: Path,
) -> None:
    settings = Settings.from_env(
        {
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
            "OPENAI_MODE": "mock",
            "ROBOT_EXECUTION_MODE": "mock",
            "DRY_RUN": "true",
        },
        root=Path(__file__).resolve().parents[2],
    )
    camera = RGBDCameraController(
        lambda: MockCapture(
            MockCaptureConfig(width_px=64, height_px=48, frames_per_second=30.0)
        ),
        LocalArtifactStore(settings.artifact_root),
        frames_per_second=30.0,
        recording_frames_per_second=10.0,
        maximum_recording_duration_s=2.0,
        startup_timeout_s=2.0,
    )
    service = MVPApplication(settings, camera_controller=camera)
    try:
        service.start_camera_preview()
        recording = service.start_camera_recording({"maximum_duration_s": 1.0})
        time.sleep(0.45)
        stopped = service.stop_camera_recording(str(recording["recording_id"]))
        assert stopped["frame_count"] >= 4
        second_recording = service.start_camera_recording({"maximum_duration_s": 1.0})
        time.sleep(0.35)
        second_stopped = service.stop_camera_recording(
            str(second_recording["recording_id"])
        )
        assert second_stopped["frame_count"] >= 3

        draft = service.create_recording_skill_draft(
            {
                "recording_id": recording["recording_id"],
                "recording_ids": [
                    recording["recording_id"],
                    second_recording["recording_id"],
                ],
                "name_hint": "direct_rgbd_wipe",
                "operator_instruction": "걸레로 테이블 표면을 닦는다",
                "keyframe_count": 4,
            }
        )
        transport = draft["transport"]
        assert transport["mode"] == "rgb_keyframes_plus_compact_fingertip_trace"
        assert transport["first_frame_index"] == 0
        assert transport["image_count"] == 4
        assert transport["demonstration_count"] == 2
        assert [
            item["recording_id"] for item in transport["demonstration_cases"]
        ] == [recording["recording_id"], second_recording["recording_id"]]
        assert [
            len(item["local_keyframe_indices"])
            for item in transport["demonstration_cases"]
        ] == [2, 2]
        assert len(transport["keyframe_indices"]) == 4
        assert transport["keyframe_indices"][0] == 0
        assert transport["keyframe_indices"][-1] == (
            stopped["frame_count"] + second_stopped["frame_count"] - 1
        )
        assert transport["depth_image_count"] == 0
        assert transport["trace_frame_count"] == (
            stopped["frame_count"] + second_stopped["frame_count"]
        )
        assert transport["requested_keyframe_count"] == 4
        assert transport["fallback_used"] is False
        tracking_metadata = draft["local_fingertip_tracking"]
        tracking = json.loads(
            service.store.read_bytes(
                tracking_metadata["artifact_uri"],
                expected_checksum_sha256=tracking_metadata[
                    "artifact_checksum_sha256"
                ],
            )
        )
        assert tracking["processed_frame_count"] == stopped["frame_count"]
        assert len(tracking["compact_fingertip_trace"]) == stopped["frame_count"]
        assert all(
            "depth" not in json.dumps(item)
            and "distance" not in json.dumps(item)
            and "candidate_state" not in json.dumps(item)
            for item in tracking["compact_fingertip_trace"]
        )

        calibration = service.calibrate_recording_draft_surface(
            draft["draft_id"],
            {
                "frame_index": 0,
                "surface_anchor_id": "teaching_surface",
                "origin_px": {"x_px": 10, "y_px": 10},
                "positive_x_px": {"x_px": 20, "y_px": 10},
                "positive_y_px": {"x_px": 10, "y_px": 20},
                "operator_confirmed": True,
            },
        )
        assert calibration["transform_convention"] == "T_camera_task_plane"
        assert calibration["base_chain"]["verified"] is False
        assert service.list_task_planes()["count"] == 1

        last_frame = stopped["frame_count"] - 1
        trajectory = service.create_recording_draft_tcp_trajectory(
            draft["draft_id"],
            {
                "method": "manual_two_fingertip",
                "annotations": [
                    {
                        "frame_index": 0,
                        "jaw_tip_a_px": {"x_px": 20, "y_px": 20},
                        "jaw_tip_b_px": {"x_px": 25, "y_px": 20},
                    },
                    {
                        "frame_index": last_frame,
                        "jaw_tip_a_px": {"x_px": 30, "y_px": 20},
                        "jaw_tip_b_px": {"x_px": 35, "y_px": 20},
                    },
                ],
                "operator_confirmed": True,
            },
        )
        assert trajectory["quality"]["sample_count"] == 2
        readiness = service.get_recording_skill_draft(draft["draft_id"])[
            "promotion_readiness"
        ]
        assert readiness["status"] == "ready_for_candidate"
        assert readiness["can_register_candidate"] is True
        assert readiness["warnings"]
        pending_mock_check = next(
            item for item in readiness["checks"] if item["id"] == "mock_validation"
        )
        assert pending_mock_check["passed"] is False
        assert pending_mock_check["pending"] is True
        assert pending_mock_check["required"] is False
        assert pending_mock_check["blocking"] is False

        candidate = service.register_recording_draft_candidate(
            draft["draft_id"], {"acknowledge_mock_only": True}
        )
        assert candidate["mock_validation_passed"] is True
        registered_readiness = service.get_recording_skill_draft(draft["draft_id"])[
            "promotion_readiness"
        ]
        final_mock_check = next(
            item
            for item in registered_readiness["checks"]
            if item["id"] == "mock_validation"
        )
        assert final_mock_check["passed"] is True
        assert final_mock_check["pending"] is False
        assert final_mock_check["required"] is True
        assert final_mock_check["blocking"] is True
        graph = service.get_skill(candidate["skill_id"], candidate["version"])[
            "skill_graph"
        ]
        assert graph["uncertainty"]["promotion_warnings"]
        assert graph["uncertainty"]["initial_scene_anchor_inputs"]
        assert graph["uncertainty"]["hardware_validated"] is False
    finally:
        service.close()


def test_mediapipe_full_recording_generates_chronological_gripper_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeterministicMediaPipeTracker:
        def __init__(self, **kwargs: object) -> None:
            self._delegate = MediaPipeHandPoseEstimator(**kwargs)  # type: ignore[arg-type]
            self._index = 0

        def track(self, frame: SynchronizedRGBDFrame) -> FingerTrackingResult:
            width = frame.color_intrinsics.width_px
            height = frame.color_intrinsics.height_px
            midpoint_x = 15 + self._index * 2
            separation_px = 2 if 3 <= self._index < 6 else 6
            half = separation_px / 2.0
            landmarks = {
                4: ((midpoint_x - half) / (width - 1), 24 / (height - 1)),
                8: ((midpoint_x + half) / (width - 1), 24 / (height - 1)),
            }
            self._index += 1
            return self._delegate.track_normalized_landmarks(
                frame, landmarks, confidence=0.95
            )

        def close(self) -> None:
            self._delegate.close()

    monkeypatch.setattr(
        application_module,
        "MediaPipeHandPoseEstimator",
        DeterministicMediaPipeTracker,
    )
    settings = Settings.from_env(
        {
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
            "OPENAI_MODE": "mock",
            "ROBOT_EXECUTION_MODE": "mock",
            "DRY_RUN": "true",
        },
        root=Path(__file__).resolve().parents[2],
    )
    camera = RGBDCameraController(
        lambda: MockCapture(
            MockCaptureConfig(width_px=64, height_px=48, frames_per_second=30.0)
        ),
        LocalArtifactStore(settings.artifact_root),
        frames_per_second=30.0,
        recording_frames_per_second=10.0,
        maximum_recording_duration_s=2.0,
        startup_timeout_s=2.0,
    )
    service = MVPApplication(settings, camera_controller=camera)
    try:
        service.start_camera_preview()
        recording = service.start_camera_recording({"maximum_duration_s": 2.0})
        time.sleep(1.05)
        stopped = service.stop_camera_recording(str(recording["recording_id"]))
        assert stopped["frame_count"] >= 9
        draft = service.create_recording_skill_draft(
            {
                "recording_id": recording["recording_id"],
                "name_hint": "mediapipe_sequence",
                "operator_instruction": "물체 위에서 손을 직선으로 움직인다",
                "keyframe_count": 1,
            }
        )
        assert draft["local_fingertip_tracking"]["transition_count"] == 3
        service.calibrate_recording_draft_surface(
            draft["draft_id"],
            {
                "frame_index": 0,
                "surface_anchor_id": "teaching_surface",
                "origin_px": {"x_px": 10, "y_px": 10},
                "positive_x_px": {"x_px": 20, "y_px": 10},
                "positive_y_px": {"x_px": 10, "y_px": 20},
                "operator_confirmed": True,
            },
        )
        trajectory = service.create_recording_draft_tcp_trajectory(
            draft["draft_id"],
            {
                "method": "mediapipe_rgbd",
                "annotations": [],
                "operator_confirmed": True,
            },
        )
        assert [item["state"] for item in trajectory["state_transitions"]] == [
            "open",
            "closed",
            "open",
        ]
        assert trajectory["finger_tracking_settings"]["close_threshold_m"] == 0.03
        candidate = service.register_recording_draft_candidate(
            draft["draft_id"], {"acknowledge_mock_only": True}
        )
        assert candidate["mock_validation_passed"] is True
        assert candidate["validation"]["runtime"]["gripper_commands"] == [
            "connect",
            "open",
            "close",
            "open",
        ]
    finally:
        service.close()
