from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import robot_skill_system.application as application_module
from robot_skill_system.api.app import create_app
from robot_skill_system.api.contracts import SkillInduceRequest
from robot_skill_system.api.routes import catalog as catalog_routes
from robot_skill_system.api.routes import skills as skill_routes
from robot_skill_system.application import MVPApplication
from robot_skill_system.demonstrations.rgbd_dataset import RGBDDatasetSegmentation
from robot_skill_system.demonstrations.stage_segmentation import segment_grip_action_end
from robot_skill_system.perception.hand_pose import (
    DEFAULT_FINGER_STATE_STABLE_FRAMES,
    FingerGripperState,
    FingerObservation,
)
from robot_skill_system.scene.models import Vector3
from robot_skill_system.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        {
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
            "OPENAI_MODE": "mock",
            "ROBOT_EXECUTION_MODE": "mock",
            "DRY_RUN": "true",
        },
        root=Path(__file__).resolve().parents[2],
    )


def _raw_dataset(settings: Settings, *, folder_name: str = "semantic-looking-hammer") -> Path:
    root = settings.artifact_root / "demonstrations" / folder_name
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    rgb_payload = b"fixture-jpeg-bytes-validated-by-checksum-before-mocked-decoder"
    depth_payload = b"fixture-depth-bytes-validated-by-checksum-before-mocked-decoder"
    (root / "rgb/000000.jpg").write_bytes(rgb_payload)
    (root / "depth/000000.npz").write_bytes(depth_payload)
    color_timestamp_ns = 1_000_000_000
    depth_timestamp_ns = 1_000_000_002
    recording_id = "rgbd_induction_fixture"
    manifest = {
        "schema_version": "1.0",
        "recording_id": recording_id,
        "status": "finished",
        "error": None,
        "depth_aligned_to_color": True,
        "timestamps_preserved": True,
        "rgb_format": "jpeg",
        "depth_format": "numpy_npz_float32_metres",
        "recording_fps": 10,
        "frame_count": 1,
        "frames": [
            {
                "index": 0,
                "frame_number": 1,
                "timestamp_ns": (color_timestamp_ns + depth_timestamp_ns) // 2,
                "color_timestamp_ns": color_timestamp_ns,
                "depth_timestamp_ns": depth_timestamp_ns,
                "timestamp_clock_domain": "host_monotonic",
                "reference_frame": "camera_color_optical_frame",
                "depth_scale_m": 0.001,
                "color_intrinsics": {
                    "width_px": 2,
                    "height_px": 2,
                    "fx_px": 1.0,
                    "fy_px": 1.0,
                    "cx_px": 0.5,
                    "cy_px": 0.5,
                    "distortion_model": "none",
                    "distortion_coefficients": [],
                },
                "rgb_uri": f"demonstrations/{recording_id}/rgb/000000.jpg",
                "rgb_checksum_sha256": hashlib.sha256(rgb_payload).hexdigest(),
                "depth_uri": f"demonstrations/{recording_id}/depth/000000.npz",
                "depth_checksum_sha256": hashlib.sha256(depth_payload).hexdigest(),
            }
        ],
    }
    (root / "rgbd_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root


def _observation(frame_index: int, state: FingerGripperState) -> FingerObservation:
    distance_m = 0.02 if state is FingerGripperState.CLOSED else 0.05
    midpoint_x = frame_index * 0.001
    return FingerObservation(
        frame_number=frame_index,
        timestamp_ns=1_000_000_000 + frame_index * 250_000_000,
        reference_frame="camera_color_optical_frame",
        status="valid",
        thumb_normalized_xy=(0.4, 0.5),
        index_normalized_xy=(0.6, 0.5),
        thumb_pixel_xy=(0, 1),
        index_pixel_xy=(1, 1),
        thumb_depth_m=1.0,
        index_depth_m=1.0,
        thumb_point_camera_m=Vector3(
            x=midpoint_x - distance_m / 2.0, y=0.0, z=1.0
        ),
        index_point_camera_m=Vector3(
            x=midpoint_x + distance_m / 2.0, y=0.0, z=1.0
        ),
        midpoint_camera_m=Vector3(x=midpoint_x, y=0.0, z=1.0),
        distance_m=distance_m,
        candidate_state=state,
        stabilization_progress_frames=1,
        required_stable_frames=DEFAULT_FINGER_STATE_STABLE_FRAMES,
        confidence=0.9,
    )


def _successful_segmentation(dataset: Any, **_kwargs: Any) -> RGBDDatasetSegmentation:
    observations = tuple(
        _observation(index, state)
        for index, state in enumerate(
            [FingerGripperState.CLOSED] * 3 + [FingerGripperState.OPEN] * 6
        )
    )
    speeds = [0.02] * len(observations)
    speeds[6:9] = [0.003, 0.003, 0.003]
    result = segment_grip_action_end(observations, speed_mps=speeds)
    return RGBDDatasetSegmentation(
        manifest_checksum_sha256=dataset.manifest.manifest_checksum_sha256,
        observations=observations,
        transitions=(),
        segmentation=result,
    )


def _missing_close(dataset: Any, **_kwargs: Any) -> RGBDDatasetSegmentation:
    observations = tuple(
        _observation(index, FingerGripperState.OPEN) for index in range(6)
    )
    # The application must persist this typed structural failure, not replace it
    # with the legacy fixed wipe graph.
    result = segment_grip_action_end(observations, speed_mps=[0.01] * len(observations))
    return RGBDDatasetSegmentation(
        manifest_checksum_sha256=dataset.manifest.manifest_checksum_sha256,
        observations=observations,
        transitions=(),
        segmentation=result,
    )


def test_rgbd_induce_registers_only_non_executable_component_evidence(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = _settings(tmp_path)
    source = _raw_dataset(settings)
    monkeypatch.setattr(
        application_module, "segment_rgbd_dataset", _successful_segmentation
    )
    service = MVPApplication(settings)
    try:
        assert "/skills/induce" in create_app(service).openapi()["paths"]
        result = skill_routes.induce_skill(
            SkillInduceRequest(
                demo_path=str(source),
                transcript_text=(
                    "망치를 가져와. object_class_id=hammer action_id=bring "
                    "required_roles=$destination"
                ),
            ),
            service,
        )
        assert result["status"] == "candidate"
        assert result["skill_id"] is None
        assert result["active"] is False
        assert result["executable"] is False
        assert result["hardware_compatible"] is False
        assert result["dataset"]["integrity_verified"] is True
        assert result["segmentation"]["status"] == "succeeded"
        assert result["semantic_classification"]["result"]["object_class_id"] == "hammer"
        assert result["semantic_classification"]["result"]["action_id"] == "bring"

        assert result["catalog"]["object"]["status"] == "draft"
        assert result["catalog"]["action"]["status"] == "draft"
        assert result["selected_components"]["grip"]["status"] == "candidate"
        assert result["selected_components"]["grip"]["hardware_compatible"] is False
        assert result["selected_components"]["action"]["definition_created"] is False
        assert result["selected_components"]["end_motion"]["mapping_created"] is False
        assert skill_routes.list_skills(service)["skills"] == []
        assert catalog_routes.list_objects(service, status="draft")["objects"][0][
            "canonical_id"
        ] == "hammer"
        assert catalog_routes.get_action("bring", service)["metadata"][
            "required_roles"
        ] == ["$destination"]

        report_bytes = service.store.read_bytes(
            result["segmentation_report_uri"],
            expected_checksum_sha256=result[
                "segmentation_report_checksum_sha256"
            ],
        )
        assert json.loads(report_bytes)["status"] == "succeeded"
    finally:
        service.close()


def test_rgbd_induce_missing_semantics_is_structured_and_does_not_guess_folder(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = _settings(tmp_path)
    source = _raw_dataset(settings, folder_name="hammer-bring-wipe")
    monkeypatch.setattr(
        application_module, "segment_rgbd_dataset", _successful_segmentation
    )
    service = MVPApplication(settings)
    try:
        result = service.induce_skill({"demo_path": str(source)})
        assert result["status"] == "structural_failure"
        assert result["semantic_classification"]["status"] == "semantics_missing"
        assert result["selected_components"]["grip"]["component_status"] == "not_created"
        assert service.list_catalog_objects()["objects"] == []
        assert service.list_catalog_actions()["actions"] == []
        assert service.list_skills()["skills"] == []
        service.store.read_bytes(
            result["segmentation_report_uri"],
            expected_checksum_sha256=result[
                "segmentation_report_checksum_sha256"
            ],
        )
    finally:
        service.close()


def test_rgbd_induce_unresolved_natural_text_creates_no_guessed_drafts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = _settings(tmp_path)
    source = _raw_dataset(settings)
    monkeypatch.setattr(
        application_module, "segment_rgbd_dataset", _successful_segmentation
    )
    service = MVPApplication(settings)
    try:
        result = service.induce_skill(
            {"demo_path": str(source), "transcript_text": "이 물건을 저기로 가져와"}
        )
        assert result["status"] == "structural_failure"
        assert result["semantic_classification"]["status"] == "semantics_unresolved"
        assert service.list_catalog_objects()["objects"] == []
        assert service.list_catalog_actions()["actions"] == []
    finally:
        service.close()


def test_rgbd_induce_accepts_only_checksum_pinned_finalized_transcript_artifact(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = _settings(tmp_path)
    source = _raw_dataset(settings)
    monkeypatch.setattr(
        application_module, "segment_rgbd_dataset", _successful_segmentation
    )
    service = MVPApplication(settings)
    try:
        teaching = service.create_teaching_session(
            {"operator_id": "trainer", "operator_role": "operator"}
        )
        service.finish_teaching_session(
            teaching["session_id"],
            {
                "success": True,
                "transcript_text": "object_class_id=hammer action_id=bring",
            },
        )
        finalized = service.repository.get_teaching_session(teaching["session_id"])
        assert finalized is not None
        assert finalized.artifact_uri is not None
        assert finalized.artifact_checksum_sha256 is not None

        result = service.induce_skill(
            {
                "demo_path": str(source),
                "transcript_artifact_uri": finalized.artifact_uri,
                "transcript_artifact_checksum_sha256": (
                    finalized.artifact_checksum_sha256
                ),
            }
        )
        assert result["status"] == "candidate"
        assert result["semantic_classification"]["source"]["kind"] == (
            "finalized_teaching_artifact"
        )
        assert result["semantic_classification"]["source"]["approved"] is True
    finally:
        service.close()


def test_rgbd_induce_missing_close_persists_typed_failure_without_any_skill(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = _settings(tmp_path)
    source = _raw_dataset(settings)
    monkeypatch.setattr(application_module, "segment_rgbd_dataset", _missing_close)
    service = MVPApplication(settings)
    try:
        result = service.induce_skill(
            {
                "demo_path": str(source),
                "transcript_text": "object_class_id=hammer action_id=bring",
            }
        )
        assert result["status"] == "structural_failure"
        assert result["segmentation"]["status"] == "structural_failure"
        assert result["segmentation"]["failure"]["code"] == "stable_close_missing"
        assert result["selected_components"]["grip"]["component_status"] == "not_created"
        assert service.list_skills()["skills"] == []
        report = json.loads(
            service.store.read_bytes(
                result["segmentation_report_uri"],
                expected_checksum_sha256=result[
                    "segmentation_report_checksum_sha256"
                ],
            )
        )
        assert report["failure"]["code"] == "stable_close_missing"
    finally:
        service.close()
