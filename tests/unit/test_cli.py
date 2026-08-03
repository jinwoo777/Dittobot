from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_skill_system.cli import (
    EXIT_OK,
    EXIT_OPERATION_FAILED,
    EXIT_SAFETY_REJECTED,
    EXIT_USAGE,
    main,
)
from robot_skill_system.settings import Settings

ROOT = Path(__file__).resolve().parents[2]


def configured_settings(tmp_path: Path, **overrides: str) -> Settings:
    environment = {
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'registry.sqlite3').as_posix()}",
        "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        **overrides,
    }
    return Settings.from_env(environment, root=ROOT)


def read_json(text: str) -> dict[str, object]:
    value = json.loads(text)
    assert isinstance(value, dict)
    return value


def test_inspect_is_json_and_never_exposes_api_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = configured_settings(
        tmp_path,
        OPENAI_API_KEY="sk-test-must-not-appear",
        OPENAI_MODE="live",
    )

    exit_code = main(["inspect"], settings=settings)

    captured = capsys.readouterr()
    document = read_json(captured.out)
    assert exit_code == EXIT_OK
    assert document["ok"] is True
    assert "sk-test-must-not-appear" not in captured.out
    result = document["result"]
    assert isinstance(result, dict)
    assert result["hardware_enabled"] is False
    assert result["primitive_count"] == 29


def test_offline_demo_then_cli_execute_is_complete_and_mock_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Even an accidental live setting is ignored by the explicitly offline demo command.
    settings = configured_settings(
        tmp_path,
        OPENAI_API_KEY="sk-not-contacted",
        OPENAI_MODE="live",
    )
    assert main(["demo-e2e"], settings=settings) == EXIT_OK
    demo = read_json(capsys.readouterr().out)
    demo_result = demo["result"]
    assert isinstance(demo_result, dict)
    assert demo_result["success"] is True
    assert demo_result["preflight_mock"] is True

    exit_code = main(
        [
            "execute",
            "--text",
            "파란 걸레로 오른쪽 테이블을 닦아줘",
            "--mode",
            "mock",
        ],
        settings=configured_settings(tmp_path),
    )

    execution_document = read_json(capsys.readouterr().out)
    result = execution_document["result"]
    assert isinstance(result, dict)
    execution = result["execution"]
    assert isinstance(execution, dict)
    assert exit_code == EXIT_OK
    assert execution["status"] == "succeeded"
    assert execution["mode"] == "mock"
    assert "set_desired_force" in execution["robot_commands"]
    preflight = execution["preflight"]
    assert isinstance(preflight, dict)
    assert preflight["passed"] is True


def test_induce_update_list_and_rollback_lifecycle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = configured_settings(tmp_path)
    novice = "tests/fixtures/demonstrations/novice_wipe.json"
    expert = "tests/fixtures/demonstrations/expert_wipe.json"

    assert main(["induce-skill", "--demo", novice], settings=settings) == EXIT_OK
    induced = read_json(capsys.readouterr().out)
    induced_result = induced["result"]
    assert isinstance(induced_result, dict)
    assert induced_result["version"] == "1.0.0"

    assert (
        main(
            ["update-skill", "--skill-id", "wipe_surface", "--demo", expert],
            settings=settings,
        )
        == EXIT_OK
    )
    update = read_json(capsys.readouterr().out)
    update_result = update["result"]
    assert isinstance(update_result, dict)
    assert update_result["status"] == "candidate"
    assert update_result["force_profile_preserved"] is True

    assert main(["list-skills"], settings=settings) == EXIT_OK
    listing = read_json(capsys.readouterr().out)
    listing_result = listing["result"]
    assert isinstance(listing_result, dict)
    skills = listing_result["skills"]
    assert isinstance(skills, list)
    versions = skills[0]["versions"]
    assert {item["version"] for item in versions} == {"1.0.0", "1.1.0-candidate"}

    assert (
        main(
            ["rollback", "--skill-id", "wipe_surface", "--version", "1.0.0"],
            settings=settings,
        )
        == EXIT_OK
    )
    rollback = read_json(capsys.readouterr().out)
    rollback_result = rollback["result"]
    assert isinstance(rollback_result, dict)
    assert rollback_result["rolled_back"] is True


def test_hardware_mode_is_rejected_before_application_is_created(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = configured_settings(
        tmp_path,
        ROBOT_EXECUTION_MODE="hardware",
        ENABLE_HARDWARE_EXECUTION="true",
        ROBOT_BACKEND="doosan",
        ENABLE_REAL_ROBOT="true",
        DRY_RUN="false",
    )

    exit_code = main(
        ["execute", "--text", "wipe the table", "--mode", "hardware"],
        settings=settings,
    )

    captured = capsys.readouterr()
    error = read_json(captured.err)
    assert exit_code == EXIT_SAFETY_REJECTED
    assert error["ok"] is False
    assert not (tmp_path / "registry.sqlite3").exists()


def test_operation_and_usage_failures_have_stable_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = configured_settings(tmp_path)
    exit_code = main(
        ["rollback", "--skill-id", "missing", "--version", "1.0.0"],
        settings=settings,
    )
    operation_error = read_json(capsys.readouterr().err)
    assert exit_code == EXIT_OPERATION_FAILED
    assert operation_error["ok"] is False

    with pytest.raises(SystemExit) as caught:
        main(["execute"], settings=settings)
    usage_error = read_json(capsys.readouterr().err)
    assert caught.value.code == EXIT_USAGE
    assert usage_error["error"]["type"] == "usage_error"


def test_transcribe_uses_mock_sidecar_without_constructing_network_client(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "wipe_command.wav"
    audio_path.write_bytes(b"mock audio bytes")
    audio_path.with_suffix(".txt").write_text(
        "파란 걸레로 오른쪽 테이블을 닦아줘", encoding="utf-8"
    )

    def fail_create(_self: object) -> object:
        raise AssertionError("mock transcription must never construct a live client")

    monkeypatch.setattr(
        "robot_skill_system.openai_integration.client.OpenAIClientFactory.create",
        fail_create,
    )
    exit_code = main(
        ["transcribe", str(audio_path)], settings=configured_settings(tmp_path)
    )

    document = read_json(capsys.readouterr().out)
    result = document["result"]
    assert isinstance(result, dict)
    assert exit_code == EXIT_OK
    assert result["text"] == "파란 걸레로 오른쪽 테이블을 닦아줘"
    assert result["language"] == "ko"
    assert result["confidence"] == 1.0
    assert result["source_audio_uri"] == "wipe_command.wav"


def test_supplemental_offline_aliases_share_existing_safety_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = configured_settings(tmp_path)
    with pytest.raises(SystemExit) as missing_gate:
        main(["execute-skill", "--skill", "wipe_surface"], settings=settings)
    gate_error = read_json(capsys.readouterr().err)
    assert missing_gate.value.code == EXIT_USAGE
    assert "--dry-run" in gate_error["error"]["message"]

    assert main(["demo-offline"], settings=settings) == EXIT_OK
    demo = read_json(capsys.readouterr().out)
    assert demo["command"] == "demo-offline"

    assert (
        main(
            ["capture-scene", "--backend", "mock", "--mode", "burst"],
            settings=settings,
        )
        == EXIT_OK
    )
    scene = read_json(capsys.readouterr().out)
    scene_result = scene["result"]
    assert isinstance(scene_result, dict)
    assert scene_result["backend"] == "mock"
    assert scene_result["camera_metadata"]["capture_mode"] == "burst"

    exit_code = main(
        ["execute-skill", "--skill", "wipe_surface", "--dry-run"],
        settings=settings,
    )
    execution_document = read_json(capsys.readouterr().out)
    result = execution_document["result"]
    assert isinstance(result, dict)
    execution = result["execution"]
    assert isinstance(execution, dict)
    assert exit_code == EXIT_OK
    assert execution["mode"] == "dry_run"
    assert execution["status"] == "succeeded"
