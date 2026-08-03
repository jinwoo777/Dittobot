from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError
from pydantic import ValidationError

from robot_skill_system.exceptions import SemanticCatalogViolationError
from robot_skill_system.openai_integration._live import parse_structured_response
from robot_skill_system.openai_integration.client import (
    OpenAIClientFactory,
    RetryExecutor,
)
from robot_skill_system.openai_integration.demonstration_analyzer import DemonstrationAnalyzer
from robot_skill_system.openai_integration.embeddings import (
    SkillEmbeddingService,
    SkillSearchDocument,
    cosine_similarity,
)
from robot_skill_system.openai_integration.intent_resolver import RuntimeIntentResolver
from robot_skill_system.openai_integration.mock_client import MockOpenAIClient
from robot_skill_system.openai_integration.schemas import (
    APICallMetadata,
    DemonstrationAnalysis,
    DemonstrationAnalysisInput,
    RuntimeIntent,
    SkillGraphProposal,
)
from robot_skill_system.openai_integration.skill_composer import SkillGraphComposer
from robot_skill_system.openai_integration.transcription import TranscriptionService
from robot_skill_system.settings import Settings


def settings(tmp_path: Path, **overrides: str) -> Settings:
    environment = {"ARTIFACT_ROOT": str(tmp_path), **overrides}
    return Settings.from_env(environment, root=tmp_path)


def analysis_input(confidence: float = 0.9) -> DemonstrationAnalysisInput:
    return DemonstrationAnalysisInput(
        transcript_text="오른쪽 테이블을 닦는다",
        scene_summary={"tool_id": "wiper_01", "target_ids": ["surface_01"]},
        pose_summary={"sample_count": 10},
        entity_catalog=["wiper_01", "surface_01"],
        primitive_catalog=["motion.move_l"],
        approved_motion_profiles=["linear_normal"],
        approved_force_profiles=["wipe_standard"],
        motion_fitting_candidates=[{"operation": "motion.move_l"}],
        confidence_summary={"pose": confidence},
    )


def test_mock_mode_never_constructs_live_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_create(_self: object) -> object:
        raise AssertionError("network client must not be constructed")

    monkeypatch.setattr(OpenAIClientFactory, "create", fail_create)
    configured = settings(tmp_path)
    analysis, metadata = DemonstrationAnalyzer(configured).analyze(analysis_input())
    intent, _ = RuntimeIntentResolver(configured).resolve(
        "파란 걸레로 오른쪽 테이블을 숙련자 방식으로 닦아줘",
        entity_ids=["wiper_01", "surface_01"],
        approved_motion_profiles=["linear_normal"],
        approved_force_profiles=["wipe_standard"],
    )
    assert analysis.task_name == "wipe_surface"
    assert intent.intent == "wipe_surface"
    assert metadata.trace_id


def test_offline_test_suite_blocks_network_connections() -> None:
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.create_connection(("example.invalid", 443))


def test_low_confidence_mock_analysis_requires_reteach(tmp_path: Path) -> None:
    analysis, _ = DemonstrationAnalyzer(settings(tmp_path)).analyze(analysis_input(0.2))
    assert analysis.reteach_required is True
    assert analysis.unresolved_ambiguities


def test_strict_output_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RuntimeIntent.model_validate(
            {
                "intent": "wipe_surface",
                "style": "normal",
                "repetitions": 1,
                "candidate_entity_ids": [],
                "confidence": 0.9,
                "unresolved_ambiguities": [],
                "execute_robot_directly": True,
            }
        )


def test_responses_parse_requires_validated_output(tmp_path: Path) -> None:
    expected = DemonstrationAnalyzer(settings(tmp_path)).analyze(analysis_input())[0]
    response = SimpleNamespace(
        output_parsed=expected,
        id="resp_mock",
        usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
    )
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(parse=lambda **_kwargs: response)
    )
    result, metadata = parse_structured_response(
        client=fake_client,
        retry=RetryExecutor(0, sleep=lambda _seconds: None),
        model="test-model",
        instructions="strict",
        payload=analysis_input(),
        output_type=DemonstrationAnalysis,
        trace_id="trace-test",
    )
    assert result == expected
    assert metadata.response_id == "resp_mock"
    assert metadata.total_tokens == 30


def test_responses_parse_rejects_wrong_schema() -> None:
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **_kwargs: SimpleNamespace(output_parsed={"not": "validated"})
        )
    )
    with pytest.raises(ValueError, match="did not return"):
        parse_structured_response(
            client=fake_client,
            retry=RetryExecutor(0, sleep=lambda _seconds: None),
            model="test-model",
            instructions="strict",
            payload={},
            output_type=RuntimeIntent,
            trace_id="trace-test",
        )


def test_timeout_is_retried_with_injected_backoff() -> None:
    attempts = 0
    delays: list[float] = []

    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
        return "ok"

    result, count = RetryExecutor(
        2, sleep=delays.append, random_value=lambda: 0.0
    ).call(flaky, "trace")
    assert result == "ok"
    assert count == 3
    assert delays == [0.25, 0.5]


def test_embedding_is_deterministic_and_searchable(tmp_path: Path) -> None:
    service = SkillEmbeddingService(settings(tmp_path))
    document = SkillSearchDocument(
        name="wipe surface",
        description="wipe a table",
        target_objects=("table",),
        tools=("wiper",),
    )
    first = service.embed_document(document)
    second = service.embed_document(document)
    assert first.vector == second.vector
    assert cosine_similarity(first.vector, second.vector) == pytest.approx(1.0)


def test_mock_transcription_uses_audio_sidecar(tmp_path: Path) -> None:
    audio = tmp_path / "command.wav"
    audio.write_bytes(b"RIFF-mock")
    audio.with_suffix(".txt").write_text("테이블을 닦아줘", encoding="utf-8")
    result = TranscriptionService(settings(tmp_path)).transcribe(audio)
    assert result.text == "테이블을 닦아줘"
    assert result.language == "ko"


def test_api_key_not_present_in_safe_summary(tmp_path: Path) -> None:
    configured = settings(tmp_path, OPENAI_API_KEY="super-secret")
    rendered = repr(configured.safe_summary())
    assert "super-secret" not in rendered
    assert "openai_api_key" not in rendered


def test_live_mode_requires_api_key(tmp_path: Path) -> None:
    configured = settings(tmp_path, OPENAI_MODE="live")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClientFactory(configured).create()


class _InvalidAnalysisMock(MockOpenAIClient):
    def analyze_demonstration(
        self, request: DemonstrationAnalysisInput, trace_id: str
    ) -> tuple[DemonstrationAnalysis, APICallMetadata]:
        analysis, metadata = super().analyze_demonstration(request, trace_id)
        recommendation = analysis.primitive_recommendations[0].model_copy(
            update={
                "operation": "motion.teleport",
                "motion_profile_id": "unapproved_motion",
                "force_profile_id": "unapproved_force",
            }
        )
        return analysis.model_copy(
            update={
                "involved_entity_ids": [*analysis.involved_entity_ids, "invented_entity"],
                "primitive_recommendations": [recommendation],
            }
        ), metadata


class _InvalidIntentMock(MockOpenAIClient):
    def resolve_intent(
        self,
        command: str,
        entity_ids: list[str],
        approved_motion_profiles: list[str],
        approved_force_profiles: list[str],
        trace_id: str,
    ) -> tuple[RuntimeIntent, APICallMetadata]:
        intent, metadata = super().resolve_intent(
            command,
            entity_ids,
            approved_motion_profiles,
            approved_force_profiles,
            trace_id,
        )
        return intent.model_copy(
            update={
                "candidate_entity_ids": [*entity_ids, "invented_entity"],
                "speed_profile_request": "unapproved_motion",
                "force_profile_request": "unapproved_force",
            }
        ), metadata


class _InvalidComposerMock(MockOpenAIClient):
    def compose_skill_graph(
        self,
        analysis: DemonstrationAnalysis,
        primitive_catalog: list[str],
        binding_ref_catalog: list[str],
        approved_motion_profiles: list[str],
        approved_force_profiles: list[str],
        trace_id: str,
    ) -> tuple[SkillGraphProposal, APICallMetadata]:
        proposal, metadata = super().compose_skill_graph(
            analysis,
            primitive_catalog,
            binding_ref_catalog,
            approved_motion_profiles,
            approved_force_profiles,
            trace_id,
        )
        node = proposal.nodes[0].model_copy(
            update={
                "operation": "motion.teleport",
                "binding_refs": ["$invented"],
                "motion_profile_id": "unapproved_motion",
                "force_profile_id": "unapproved_force",
            }
        )
        return proposal.model_copy(
            update={"required_entity_roles": {"$invented": "target"}, "nodes": [node]}
        ), metadata


def test_analyzer_rejects_identifiers_outside_authoritative_catalogs(tmp_path: Path) -> None:
    analyzer = DemonstrationAnalyzer(settings(tmp_path), mock=_InvalidAnalysisMock())
    with pytest.raises(SemanticCatalogViolationError) as raised:
        analyzer.analyze(analysis_input())
    message = str(raised.value)
    assert "invented_entity" in message
    assert "motion.teleport" in message
    assert "unapproved_motion" in message
    assert "unapproved_force" in message


def test_intent_resolver_rejects_entities_and_profiles_outside_catalogs(
    tmp_path: Path,
) -> None:
    resolver = RuntimeIntentResolver(settings(tmp_path), mock=_InvalidIntentMock())
    with pytest.raises(SemanticCatalogViolationError) as raised:
        resolver.resolve(
            "wipe the table",
            entity_ids=["wiper_01", "surface_01"],
            approved_motion_profiles=["linear_normal"],
            approved_force_profiles=["wipe_standard"],
        )
    message = str(raised.value)
    assert "invented_entity" in message
    assert "unapproved_motion" in message
    assert "unapproved_force" in message


def test_composer_rejects_operations_profiles_and_bindings_outside_catalogs(
    tmp_path: Path,
) -> None:
    analysis, _ = DemonstrationAnalyzer(settings(tmp_path)).analyze(analysis_input())
    composer = SkillGraphComposer(settings(tmp_path), mock=_InvalidComposerMock())
    with pytest.raises(SemanticCatalogViolationError) as raised:
        composer.compose(
            analysis,
            ["motion.move_l"],
            binding_ref_catalog=["$surface"],
            approved_motion_profiles=["linear_normal"],
            approved_force_profiles=["wipe_standard"],
        )
    message = str(raised.value)
    assert "motion.teleport" in message
    assert "$invented" in message
    assert "unapproved_motion" in message
    assert "unapproved_force" in message
