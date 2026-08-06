from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import robot_skill_system.application as application_module
from robot_skill_system.api.app import create_app
from robot_skill_system.application import MVPApplication
from robot_skill_system.capture import (
    MockCapture,
    MockCaptureConfig,
    RGBDCameraController,
)
from robot_skill_system.openai_integration.mock_client import MockOpenAIClient
from robot_skill_system.openai_integration.recording_skill_analyzer import (
    ImageInputRejectedError,
)
from robot_skill_system.openai_integration.schemas import RecordingSkillDraftInput
from robot_skill_system.settings import Settings
from robot_skill_system.skills.models import SkillGraph
from robot_skill_system.storage.artifact_store import LocalArtifactStore
from robot_skill_system.storage.orm import (
    SkillEmbeddingRecord,
    SkillRecord,
    SkillVersionRecord,
    ValidationRunRecord,
)
from robot_skill_system.vertical_slice import run_offline_demo


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


@pytest.fixture
def service(tmp_path: Path) -> MVPApplication:
    application = MVPApplication(_settings(tmp_path))
    try:
        yield application
    finally:
        application.close()


def test_openapi_exposes_every_required_original_and_supplemental_route(
    service: MVPApplication,
) -> None:
    paths = set(create_app(service).openapi()["paths"])
    required = {
        "/health",
        "/camera/status",
        "/camera/preview/start",
        "/camera/preview/stop",
        "/camera/streams/{kind}.mjpg",
        "/camera/recordings",
        "/camera/recordings/{recording_id}/stop",
        "/camera/recordings/{recording_id}/frames/{frame_index}/{kind}.jpg",
        "/calibration/hand-eye/status",
        "/calibration/hand-eye/start",
        "/calibration/hand-eye/abort",
        "/calibration/hand-eye/import-legacy-npy",
        "/calibration/task-planes",
        "/teaching/sessions",
        "/teaching/sessions/{session_id}/capture",
        "/teaching/sessions/{session_id}/frames",
        "/teaching/sessions/{session_id}/finish",
        "/teaching/sessions/{session_id}/finalize",
        "/scenes/capture",
        "/perception/scene/capture",
        "/skills",
        "/skills/induce",
        "/skills/draft-from-recording",
        "/skills/draft-from-recording/capabilities",
        "/skills/drafts",
        "/skills/drafts/{draft_id}",
        "/skills/drafts/{draft_id}/surface-calibration",
        "/skills/drafts/{draft_id}/surface-calibration/auto",
        "/skills/drafts/{draft_id}/tcp-trajectory",
        "/skills/drafts/{draft_id}/candidate",
        "/skills/editor/catalog",
        "/skills/editor/preview",
        "/skills/editor/candidates",
        "/skills/search",
        "/skills/{skill_id}/versions/{version}/parameter-candidates",
        "/skills/{skill_id}/validate",
        "/skills/{skill_id}/versions/{version}/promote",
        "/runtime/resolve",
        "/runtime/intent",
        "/runtime/bind",
        "/runtime/preflight",
        "/runtime/validate",
        "/runtime/execute",
        "/runtime/abort",
    }
    assert required <= paths


def test_handeye_api_exposes_plan_but_refuses_closed_hardware_gates(
    service: MVPApplication,
) -> None:
    client = TestClient(create_app(service))

    status = client.get("/calibration/hand-eye/status")
    assert status.status_code == 200
    capabilities = status.json()["capabilities"]
    assert capabilities["board"] == {
        "internal_corners": [10, 7],
        "square_size_m": 0.025,
    }
    assert capabilities["locked_joint_indices"] == [1, 2]
    assert capabilities["pose_count"] == 21
    assert capabilities["hardware_authorized"] is False

    started = client.post(
        "/calibration/hand-eye/start",
        json={
            "operator_id": "integration_operator",
            "operator_confirmed": True,
            "board_secured": True,
            "workspace_cleared": True,
            "estop_ready": True,
        },
    )
    assert started.status_code == 403
    assert "authorization is closed" in started.json()["detail"]

    imported = client.post(
        "/calibration/hand-eye/import-legacy-npy",
        json={
            "operator_id": "integration_operator",
            "operator_confirmed": True,
            "acknowledge_candidate_only": True,
        },
    )
    assert imported.status_code == 403
    assert "requires robot frame authorization" in imported.json()["detail"]


def test_camera_api_records_local_rgbd_with_injected_capture(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    controller = RGBDCameraController(
        lambda: MockCapture(
            MockCaptureConfig(width_px=32, height_px=24, frames_per_second=30.0)
        ),
        LocalArtifactStore(settings.artifact_root),
        frames_per_second=30.0,
        maximum_recording_duration_s=2.0,
        startup_timeout_s=2.0,
    )
    application = MVPApplication(settings, camera_controller=controller)
    try:
        client = TestClient(create_app(application))
        assert client.get("/camera/status").json()["state"] == "stopped"
        assert client.post("/camera/preview/start").json()["state"] == "streaming"
        recording = client.post(
            "/camera/recordings", json={"maximum_duration_s": 1.0}
        )
        assert recording.status_code == 200
        recording_id = recording.json()["recording_id"]
        time.sleep(0.12)
        stopped = client.post(f"/camera/recordings/{recording_id}/stop")
        assert stopped.status_code == 200
        assert stopped.json()["frame_count"] >= 1
        assert client.get(f"/camera/recordings/{recording_id}").json()[
            "manifest_uri"
        ]
        catalog = client.get("/camera/recordings")
        assert catalog.status_code == 200
        assert [recording_id] == [
            item["recording_id"] for item in catalog.json()["recordings"]
        ]
        rgb_frame = client.get(
            f"/camera/recordings/{recording_id}/frames/0/rgb.jpg"
        )
        depth_frame = client.get(
            f"/camera/recordings/{recording_id}/frames/0/depth.jpg"
        )
        assert rgb_frame.status_code == 200
        assert depth_frame.status_code == 200
        assert rgb_frame.headers["content-type"] == "image/jpeg"
        assert depth_frame.headers["content-type"] == "image/jpeg"

        capabilities = client.get("/skills/draft-from-recording/capabilities").json()
        assert capabilities["openai_mode"] == "mock"
        assert capabilities["maximum_keyframes"] == 1
        assert capabilities["uploads_rgb_and_aligned_depth_pairs"] is False
        assert capabilities["uploaded_rgb_frame_count"] == 1
        assert capabilities["depth_stays_local"] is True
        assert capabilities["provider_video_input_supported"] is False
        assert capabilities["fallback_analysis_transport"] == "first_rgb_input_file"
        assert capabilities["creates_zip_archive_on_fallback"] is False
        assert capabilities["creates_executable_skill"] is False
        draft_response = client.post(
            "/skills/draft-from-recording",
            json={
                "recording_id": recording_id,
                "name_hint": "table_wipe_recorded",
                "operator_instruction": "걸레로 테이블 표면을 닦는다",
                "keyframe_count": 8,
            },
        )
        assert draft_response.status_code == 200
        draft = draft_response.json()
        assert draft["status"] == "semantic_draft"
        assert draft["source_recording_id"] == recording_id
        assert draft["openai"]["mode"] == "mock"
        assert draft["draft"]["suggested_skill_id"] == "table_wipe_recorded"
        assert draft["draft"]["tcp_proxy_observation"]["detected"] is True
        assert draft["transport"]["mode"] == "first_rgb_plus_compact_fingertip_trace"
        assert draft["transport"]["image_count"] == 1
        assert draft["transport"]["depth_image_count"] == 0
        assert draft["transport"]["trace_frame_count"] == stopped.json()["frame_count"]
        assert draft["local_fingertip_tracking"]["valid_frame_count"] == 0
        assert draft["executable"] is False
        assert draft["requires_pose_trajectory"] is True
        assert application.store.path_for(draft["artifact_uri"]).is_file()
        draft_catalog = client.get("/skills/drafts")
        assert draft_catalog.status_code == 200
        [listed_draft] = draft_catalog.json()["drafts"]
        assert listed_draft["draft_id"] == draft["draft_id"]
        assert listed_draft["promotion_readiness"]["status"] == "needs_calibration"
        assert listed_draft["promotion_readiness"]["can_register_candidate"] is False
        draft_detail = client.get(f"/skills/drafts/{draft['draft_id']}")
        assert draft_detail.status_code == 200
        assert draft_detail.json()["source_recording_id"] == recording_id
        assert client.post("/camera/preview/stop").json()["state"] == "stopped"
    finally:
        application.close()


def test_manual_task_plane_and_metric_tcp_path_register_candidate_without_valid_npy(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    controller = RGBDCameraController(
        lambda: MockCapture(
            MockCaptureConfig(width_px=64, height_px=48, frames_per_second=30.0)
        ),
        LocalArtifactStore(settings.artifact_root),
        frames_per_second=30.0,
        recording_frames_per_second=10.0,
        maximum_recording_duration_s=2.0,
        startup_timeout_s=2.0,
    )
    application = MVPApplication(settings, camera_controller=controller)
    try:
        client = TestClient(create_app(application))
        client.post("/camera/preview/start").raise_for_status()
        recording = client.post(
            "/camera/recordings", json={"maximum_duration_s": 1.0}
        ).json()
        time.sleep(0.45)
        finished = client.post(
            f"/camera/recordings/{recording['recording_id']}/stop"
        ).json()
        assert finished["frame_count"] >= 4
        draft = client.post(
            "/skills/draft-from-recording",
            json={
                "recording_id": recording["recording_id"],
                "name_hint": "rgbd_surface_wipe",
                "operator_instruction": "두 손가락으로 테이블 표면을 닦는다",
                "keyframe_count": 8,
            },
        ).json()

        calibration = client.post(
            f"/skills/drafts/{draft['draft_id']}/surface-calibration",
            json={
                "frame_index": 0,
                "surface_anchor_id": "teaching_surface",
                "origin_px": {"x_px": 10, "y_px": 10},
                "positive_x_px": {"x_px": 20, "y_px": 10},
                "positive_y_px": {"x_px": 10, "y_px": 20},
                "operator_confirmed": True,
            },
        )
        assert calibration.status_code == 200, calibration.text
        assert calibration.json()["transform_convention"] == "T_camera_task_plane"
        assert calibration.json()["method"] == "operator_three_point_aligned_depth"
        last_frame_index = finished["frame_count"] - 1
        trajectory = client.post(
            f"/skills/drafts/{draft['draft_id']}/tcp-trajectory",
            json={
                "method": "manual_two_fingertip",
                "annotations": [
                    {
                        "frame_index": 0,
                        "jaw_tip_a_px": {"x_px": 20, "y_px": 20},
                        "jaw_tip_b_px": {"x_px": 25, "y_px": 20},
                    },
                    {
                        "frame_index": last_frame_index,
                        "jaw_tip_a_px": {"x_px": 30, "y_px": 20},
                        "jaw_tip_b_px": {"x_px": 35, "y_px": 20},
                    },
                ],
                "operator_confirmed": True,
            },
        )
        assert trajectory.status_code == 200, trajectory.text
        assert trajectory.json()["quality"]["sample_count"] == 2
        assert trajectory.json()["method"] == "manual_two_fingertip"
        legacy_result = application.store.put_json(
            "calibrations/legacy_import_integration/result.json",
            {
                "schema_version": "1.0",
                "evidence_type": "legacy_handeye_transform_candidate",
                "import_id": "legacy_import_integration",
                "passed": False,
                "candidate_only": True,
                "hardware_validated": False,
                "runtime_authorized": False,
            },
        )
        ready = client.get(f"/skills/drafts/{draft['draft_id']}").json()
        assert ready["promotion_readiness"]["status"] == "ready_for_candidate"
        assert ready["promotion_readiness"]["can_register_candidate"] is True
        handeye_check = next(
            item
            for item in ready["promotion_readiness"]["checks"]
            if item["id"] == "handeye_transform_candidate"
        )
        assert handeye_check["passed"] is False
        assert handeye_check["required"] is False
        assert handeye_check["blocking"] is False

        candidate = client.post(
            f"/skills/drafts/{draft['draft_id']}/candidate",
            json={"acknowledge_mock_only": True},
        )
        assert candidate.status_code == 200, candidate.text
        assert candidate.json()["mock_validation_passed"] is True, candidate.json()[
            "validation"
        ]
        assert candidate.json()["handeye_transform_candidate"] == {
            "import_id": "legacy_import_integration",
            "passed": False,
            "hardware_validated": False,
            "runtime_authorized": False,
            "artifact_uri": legacy_result.uri,
            "attached_to_candidate": False,
        }
        skill = client.get(
            "/skills/rgbd_surface_wipe",
            params={"version": candidate.json()["version"]},
        ).json()
        assert legacy_result.uri not in skill["skill_graph"]["source_demonstrations"]
        promoted = client.get(f"/skills/drafts/{draft['draft_id']}").json()
        assert promoted["promotion_readiness"]["status"] == "candidate_registered"
        assert promoted["promotion_readiness"]["can_register_candidate"] is False
        registered = client.get("/skills").json()["skills"]
        assert any(
            item["skill_id"] == "rgbd_surface_wipe"
            and item["validation_status"] == "passed"
            for item in registered
        )
    finally:
        application.close()


def test_recording_draft_falls_back_to_first_rgb_input_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    controller = RGBDCameraController(
        lambda: MockCapture(
            MockCaptureConfig(width_px=32, height_px=24, frames_per_second=30.0)
        ),
        LocalArtifactStore(settings.artifact_root),
        frames_per_second=30.0,
        maximum_recording_duration_s=2.0,
        startup_timeout_s=2.0,
    )

    class ForcedFallbackAnalyzer:
        def __init__(self, _settings: Settings) -> None:
            self.mock = MockOpenAIClient()

        def analyze_first_frame_trace(
            self,
            request: RecordingSkillDraftInput,
            *,
            first_rgb_path: Path,
            as_file_fallback: bool = False,
        ) -> object:
            assert first_rgb_path.read_bytes().startswith(b"\xff\xd8")
            if not as_file_fallback:
                raise ImageInputRejectedError("forced image rejection")
            return self.mock.analyze_recording_skill_draft(
                request, "trace-first-rgb-file-fallback"
            )

    monkeypatch.setattr(
        application_module,
        "RecordingSkillDraftAnalyzer",
        ForcedFallbackAnalyzer,
    )
    application = MVPApplication(settings, camera_controller=controller)
    try:
        application.start_camera_preview()
        recording = application.start_camera_recording({"maximum_duration_s": 1.0})
        time.sleep(0.12)
        stopped = application.stop_camera_recording(str(recording["recording_id"]))
        result = application.create_recording_skill_draft(
            {
                "recording_id": recording["recording_id"],
                "name_hint": "table_wipe_recorded",
                "operator_instruction": "걸레로 테이블 표면을 닦는다",
                "keyframe_count": 300,
            }
        )

        assert result["transport"]["mode"] == (
            "first_rgb_input_file_plus_compact_fingertip_trace"
        )
        assert result["transport"]["fallback_used"] is True
        assert result["transport"]["image_count"] == 1
        assert result["transport"]["depth_image_count"] == 0
        assert result["transport"]["trace_frame_count"] == stopped["frame_count"]
        assert result["local_fingertip_tracking"]["valid_frame_count"] == 0
        assert "zip_archive" not in result["transport"]
        assert "analysis_pdf" not in result["transport"]
        tracking_path = application.store.path_for(
            result["local_fingertip_tracking"]["artifact_uri"]
        )
        tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
        assert tracking["processed_frame_count"] == stopped["frame_count"]
        assert len(tracking["finger_observations"]) == stopped["frame_count"]
        assert len(tracking["compact_fingertip_trace"]) == stopped["frame_count"]
        assert len(tracking["invalid_frames"]) == stopped["frame_count"]
    finally:
        application.close()


def test_ui_registry_api_and_static_console_are_connected(
    service: MVPApplication,
) -> None:
    app = create_app(service)
    client = TestClient(app)

    assert client.get("/skills").json() == {"skills": []}
    induced = service.induce_skill(
        {"demo_path": "tests/fixtures/demonstrations/novice_wipe.json"}
    )

    registry = client.get("/skills")
    assert registry.status_code == 200
    [row] = registry.json()["skills"]
    assert row["skill_id"] == "wipe_surface"
    assert row["version"] == induced["version"]
    assert row["status"] == "active"
    assert row["validation_status"] == "passed"
    assert row["node_count"] == len(row["skill_graph"]["nodes"])

    detail = client.get(
        "/skills/wipe_surface", params={"version": induced["version"]}
    )
    assert detail.status_code == 200
    assert detail.json()["skill_graph"]["skill_id"] == "wipe_surface"

    ui_redirect = client.get("/ui", follow_redirects=False)
    assert ui_redirect.status_code in {302, 307}
    assert ui_redirect.headers["location"] == "/ui/"
    assert "Dittobot Operator Console" in client.get("/ui/").text
    assert "분석 초안" in client.get("/ui/").text
    assert "두 손가락" in client.get("/ui/").text
    assert "DittobotApiClient" in client.get("/ui/api-client.js").text


def test_persisted_teaching_runtime_update_promotion_and_rollback(
    service: MVPApplication,
) -> None:
    teaching = service.create_teaching_session(
        {"operator_id": "novice_01", "operator_role": "novice"}
    )
    captured = service.capture_teaching_session(teaching["session_id"], {"mode": "mock"})
    finished = service.finish_teaching_session(
        teaching["session_id"], {"success": True, "transcript_text": "테이블을 닦는다"}
    )
    assert captured["scene_ids"]
    assert finished["status"] == "finished"

    scene_id = captured["scene_ids"][-1]
    induced = service.induce_skill(
        {"demo_path": "tests/fixtures/demonstrations/novice_wipe.json"}
    )
    assert induced["observed_primitive"] == "motion.move_periodic"
    assert induced["teaching_quality"] < 0.9
    tool_dispatcher = service._function_dispatcher(service._scene(scene_id))
    tool_search = tool_dispatcher.dispatch(
        "search_skill_registry",
        {"query": "wipe", "lifecycle": "active", "limit": 5},
    )
    assert [match.skill_id for match in tool_search.matches] == ["wipe_surface"]
    tool_scene = tool_dispatcher.dispatch(
        "get_scene_summary", {"scene_id": scene_id}
    )
    assert tool_scene.entity_counts.tools == 1
    assert service.search_skills({"query": "wipe_surface"})["mode"] == "embedding"
    assert service.bind_runtime(
        {"skill_id": "wipe_surface", "scene_id": scene_id}
    )["bindings"] == {
        "$surface": "table_surface_01",
        "$tool": "wiper_01",
    }
    assert service.preflight_runtime(
        {"skill_id": "wipe_surface", "scene_id": scene_id, "mode": "dry_run"}
    )["passed"]
    execution = service.execute_runtime(
        {
            "run_id": "known_dry_run_01",
            "skill_id": "wipe_surface",
            "scene_id": scene_id,
            "mode": "dry_run",
            "text": "테이블을 닦아줘",
        }
    )
    assert execution["status"] == "succeeded"
    assert execution["robot_commands"][-3:] == [
        "release_force",
        "release_compliance",
        "safe_retract",
    ]
    assert service.get_runtime_run("known_dry_run_01")["events"]

    candidate = service.update_skill(
        "wipe_surface",
        {
            "demo_path": "tests/fixtures/demonstrations/expert_wipe.json",
            "operator_role": "expert",
        },
    )
    assert candidate["status"] == "candidate"
    assert service.get_skill("wipe_surface")["version"] == "1.0.0"
    validation = service.validate_skill(
        "wipe_surface", {"version": candidate["candidate_version"], "mode": "mock"}
    )
    assert validation["passed"]
    assert validation["runtime"]["robot_commands"]
    promoted = service.activate_skill(
        "wipe_surface", {"version": candidate["candidate_version"]}
    )
    assert promoted["version"] == "1.1.0"
    assert service.get_skill("wipe_surface")["version"] == "1.1.0"
    rollback = service.rollback_skill("wipe_surface", {"version": "1.0.0"})
    assert rollback["status"] == "active"
    assert service.get_skill("wipe_surface")["version"] == "1.0.0"

    with service.database.session() as session:
        embeddings = list(session.scalars(select(SkillEmbeddingRecord)))
        validation_runs = list(session.scalars(select(ValidationRunRecord)))
    assert len(embeddings) == 3
    assert len(validation_runs) == 3


def test_failed_candidate_keeps_previous_active(service: MVPApplication, monkeypatch: Any) -> None:
    service.induce_skill(
        {"demo_path": "tests/fixtures/demonstrations/novice_wipe.json"}
    )
    candidate = service.update_skill(
        "wipe_surface",
        {
            "demo_path": "tests/fixtures/demonstrations/expert_wipe.json",
            "operator_role": "expert",
        },
    )
    monkeypatch.setattr(
        service,
        "_mock_regression_validate",
        lambda _row, _graph: {"passed": False, "robot_commands": [], "events": []},
    )

    result = service.validate_skill(
        "wipe_surface", {"version": candidate["candidate_version"]}
    )

    assert result["passed"] is False
    assert service.get_skill("wipe_surface")["version"] == "1.0.0"
    versions = service.get_skill_versions("wipe_surface")["versions"]
    rejected = next(item for item in versions if item["version"].endswith("-candidate"))
    assert rejected["status"] == "rejected"


def test_material_route_update_persists_measured_path_as_separate_variant(
    service: MVPApplication, monkeypatch: Any
) -> None:
    service.induce_skill(
        {"demo_path": "tests/fixtures/demonstrations/novice_wipe.json"}
    )
    measured_route = [
        (0.00, 0.20, 0.0),
        (0.10, 0.20, 0.0),
        (0.20, 0.20, 0.0),
        (0.30, 0.20, 0.0),
    ]
    monkeypatch.setattr(service, "_normalized_path", lambda _trajectory: measured_route)
    monkeypatch.setattr(
        service,
        "_parent_normalized_path",
        lambda _graph: [(0.00, 0.0, 0.0), (0.30, 0.0, 0.0)],
    )

    candidate = service.update_skill(
        "wipe_surface",
        {
            "demo_path": "tests/fixtures/demonstrations/expert_wipe.json",
            "operator_role": "expert",
        },
    )

    assert candidate["variant"] == "expert_precise"
    row = service._find_version("wipe_surface", candidate["candidate_version"])
    graph = SkillGraph.model_validate(row.graph_json)
    target = graph.node_by_id("stroke_forward").arguments["target"]
    assert target["position_m"]["y"] == pytest.approx(0.20)

    assert service.validate_skill(
        "wipe_surface", {"version": candidate["candidate_version"], "mode": "mock"}
    )["passed"]
    promoted = service.activate_skill(
        "wipe_surface", {"version": candidate["candidate_version"]}
    )
    assert promoted["version"] == "1.1.0"
    with service.database.session() as session:
        skills = list(session.scalars(select(SkillRecord)))
    assert {skill.variant for skill in skills} == {"safe", "expert_precise"}
    assert all(skill.active_version_id is not None for skill in skills)


def test_registry_graph_tampering_is_rejected_before_execution(
    service: MVPApplication,
) -> None:
    scene = service.capture_scene({"mode": "mock"})
    service.induce_skill(
        {"demo_path": "tests/fixtures/demonstrations/novice_wipe.json"}
    )
    with service.database.session() as session:
        row = session.scalar(select(SkillVersionRecord))
        assert row is not None
        changed = dict(row.graph_json)
        changed["description"] = "tampered graph"
        row.graph_json = changed

    with pytest.raises(Exception, match="checksum mismatch"):
        service.execute_runtime(
            {
                "skill_id": "wipe_surface",
                "scene_id": scene["scene_id"],
                "mode": "dry_run",
            }
        )


def test_abort_stops_known_active_run_and_failure_events_are_persisted(
    service: MVPApplication, monkeypatch: Any
) -> None:
    scene = service.capture_scene({"mode": "mock"})
    service.induce_skill(
        {"demo_path": "tests/fixtures/demonstrations/novice_wipe.json"}
    )

    async def slow_compiled_run(runtime: Any) -> bool:
        await runtime.execute_primitive("motion.wait", {"duration_s": 0.15}, 1.0, None)
        await runtime.execute_primitive("motion.wait", {"duration_s": 0.01}, 1.0, None)
        return True

    monkeypatch.setattr(
        service, "_load_verified_run", lambda _row, _graph: slow_compiled_run
    )
    errors: list[BaseException] = []

    def execute() -> None:
        try:
            service.execute_runtime(
                {
                    "run_id": "abortable_run_01",
                    "skill_id": "wipe_surface",
                    "scene_id": scene["scene_id"],
                    "mode": "dry_run",
                }
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with service._active_execution_lock:
            if "abortable_run_01" in service._active_executions:
                break
        time.sleep(0.005)
    aborted = service.abort_runtime(
        {"run_id": "abortable_run_01", "reason": "integration_test"}
    )
    worker.join(timeout=2.0)

    assert aborted["abort_requested"] is True
    assert errors
    persisted = service.get_runtime_run("abortable_run_01")
    assert persisted["status"] == "failed"
    assert persisted["events"]


def test_offline_vertical_slice_has_exact_compiled_motion_order(tmp_path: Path) -> None:
    result = run_offline_demo(_settings(tmp_path))
    motions = [
        operation
        for operation in result.robot_commands
        if operation in {"move_l", "move_c", "move_periodic"}
    ]
    assert motions == ["move_l", "move_l", "move_c", "move_l"]
