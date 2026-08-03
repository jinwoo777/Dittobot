from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from robot_skill_system.api.app import create_app
from robot_skill_system.application import MVPApplication
from robot_skill_system.settings import Settings
from robot_skill_system.skills.models import SkillGraph
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
        "/teaching/sessions",
        "/teaching/sessions/{session_id}/capture",
        "/teaching/sessions/{session_id}/frames",
        "/teaching/sessions/{session_id}/finish",
        "/teaching/sessions/{session_id}/finalize",
        "/scenes/capture",
        "/perception/scene/capture",
        "/skills/induce",
        "/skills/search",
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
