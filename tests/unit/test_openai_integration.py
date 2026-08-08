from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, BadRequestError
from pydantic import ValidationError

from robot_skill_system.exceptions import SemanticCatalogViolationError
from robot_skill_system.openai_integration._live import parse_structured_response
from robot_skill_system.openai_integration.client import (
    OpenAIClientFactory,
    RetryExecutor,
)
from robot_skill_system.openai_integration.demonstration_analyzer import (
    ANALYZER_INSTRUCTIONS,
    DemonstrationAnalyzer,
)
from robot_skill_system.openai_integration.embeddings import (
    SkillEmbeddingService,
    SkillSearchDocument,
    cosine_similarity,
)
from robot_skill_system.openai_integration.intent_resolver import RuntimeIntentResolver
from robot_skill_system.openai_integration.mock_client import MockOpenAIClient
from robot_skill_system.openai_integration.recording_skill_analyzer import (
    FIRST_FRAME_TRACE_INSTRUCTIONS,
    RECORDING_PDF_FALLBACK_INSTRUCTIONS,
    RECORDING_SKILL_INSTRUCTIONS,
    ImageInputRejectedError,
    RecordingSkillDraftAnalyzer,
)
from robot_skill_system.openai_integration.schemas import (
    APICallMetadata,
    DemonstrationAnalysis,
    DemonstrationAnalysisInput,
    RecordingSkillDraft,
    RecordingSkillDraftInput,
    RuntimeIntent,
    SkillGraphProposal,
)
from robot_skill_system.openai_integration.skill_composer import (
    COMPOSER_INSTRUCTIONS,
    SkillGraphComposer,
)
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


def test_all_semantic_motion_prompts_share_local_simplification_policy() -> None:
    prompts = (
        RECORDING_SKILL_INSTRUCTIONS,
        RECORDING_PDF_FALLBACK_INSTRUCTIONS,
        FIRST_FRAME_TRACE_INSTRUCTIONS,
        ANALYZER_INSTRUCTIONS,
        COMPOSER_INSTRUCTIONS,
    )

    for prompt in prompts:
        assert "Preserve intended contact phases" in prompt
        assert "gripper state-transition boundary" in prompt
        assert "motion.move_l" in prompt
        assert "motion.move_c only" in prompt
        assert "motion.move_spline only as a fallback" in prompt
        assert "classification thresholds" in prompt
        assert "minimum primitive sequence" in prompt
        assert "local motion fitter is authoritative" in prompt
        assert "fixed-workspace pickup" in prompt
        assert "gripper.open -> one approach motion -> gripper.close" in prompt
        assert "Do not add repeated approach/retract motions" in prompt

    for prompt in (
        RECORDING_SKILL_INSTRUCTIONS,
        RECORDING_PDF_FALLBACK_INSTRUCTIONS,
        FIRST_FRAME_TRACE_INSTRUCTIONS,
    ):
        assert "semantic object" in prompt
        assert "compact thumb/index trace" in prompt
        assert "complete recording" in prompt
        assert "coordinates or motion targets" in prompt

    assert "exactly two inputs" in FIRST_FRAME_TRACE_INSTRUCTIONS
    assert "one RGB image from the first manifest frame" in FIRST_FRAME_TRACE_INSTRUCTIONS
    assert "covering every stored video frame" in FIRST_FRAME_TRACE_INSTRUCTIONS
    assert "excludes depth, metric distance, gripper classification" in (
        FIRST_FRAME_TRACE_INSTRUCTIONS
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


def test_live_recording_draft_sends_bounded_rgb_depth_pairs_with_store_disabled(
    tmp_path: Path,
) -> None:
    rgb_image = tmp_path / "rgb_keyframe.jpg"
    depth_image = tmp_path / "depth_keyframe.jpg"
    rgb_image.write_bytes(b"\xff\xd8mock-rgb-jpeg\xff\xd9")
    depth_image.write_bytes(b"\xff\xd8mock-depth-jpeg\xff\xd9")
    request = RecordingSkillDraftInput(
        recording_id="rgbd_0123456789abcdef0123456789abcdef",
        name_hint="recorded_wipe",
        operator_instruction="걸레로 표면을 닦는다",
        recording_summary={"frame_count": 10, "recording_fps": 10},
        primitive_catalog=["motion.move_l"],
        entity_role_catalog=["tool", "target_surface"],
        keyframe_indices=[5],
        limitations=["No trusted robot-base pose trajectory."],
    )
    expected = MockOpenAIClient().analyze_recording_skill_draft(request, "trace")[0]
    captured: dict[str, object] = {}

    def parse(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(output_parsed=expected, id="resp_recording", usage=None)

    fake_client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    analyzer = RecordingSkillDraftAnalyzer(
        settings(
            tmp_path,
            OPENAI_MODE="live",
            OPENAI_API_KEY="test-secret",
            OPENAI_MAX_KEYFRAMES="1",
        ),
        client=fake_client,
        retry=RetryExecutor(0, sleep=lambda _seconds: None),
    )

    result, metadata = analyzer.analyze(
        request,
        rgb_paths=[rgb_image],
        depth_paths=[depth_image],
    )

    assert result == expected
    assert metadata.response_id == "resp_recording"
    assert captured["store"] is False
    [message] = captured["input"]  # type: ignore[misc]
    assert [item["type"] for item in message["content"]] == [
        "input_text",
        "input_image",
        "input_image",
    ]
    assert message["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
    assert message["content"][2]["image_url"].startswith("data:image/jpeg;base64,")
    assert "midpoint_between_fingertips" in message["content"][0]["text"]
    assert "EVERY supplied keyframe" in captured["instructions"]
    assert "scene_observation" in captured["instructions"]
    assert "test-secret" not in repr(captured)


def test_first_frame_trace_transport_sends_one_rgb_and_image_coordinates_only(
    tmp_path: Path,
) -> None:
    first_rgb = tmp_path / "000000.jpg"
    first_rgb.write_bytes(b"\xff\xd8mock-first-rgb\xff\xd9")
    request = RecordingSkillDraftInput(
        recording_id="rgbd_0123456789abcdef0123456789abcdef",
        name_hint="recorded_wipe",
        operator_instruction="두 손가락으로 표면을 닦는다",
        recording_summary={"frame_count": 2, "recording_fps": 10},
        primitive_catalog=["motion.move_l"],
        entity_role_catalog=["tool", "target_surface"],
        keyframe_indices=[0],
        first_frame_index=0,
        fingertip_trace=[
            {
                "frame_index": 0,
                "timestamp_ns": 1_000,
                "thumb_tip": {
                    "landmark_index": 4,
                    "normalized_xy": (0.4, 0.5),
                    "pixel_xy": (40, 50),
                },
                "index_tip": {
                    "landmark_index": 8,
                    "normalized_xy": (0.6, 0.5),
                    "pixel_xy": (60, 50),
                },
                "status": "valid",
            },
            {
                "frame_index": 1,
                "timestamp_ns": 2_000,
                "thumb_tip": {
                    "landmark_index": 4,
                    "normalized_xy": None,
                    "pixel_xy": None,
                },
                "index_tip": {
                    "landmark_index": 8,
                    "normalized_xy": None,
                    "pixel_xy": None,
                },
                "status": "uncertain",
            },
        ],
        visual_input_policy="first_rgb_plus_local_fingertip_trace",
        limitations=["Metric hand state remains local."],
    )
    expected = MockOpenAIClient().analyze_recording_skill_draft(request, "trace")[0]
    captured: dict[str, object] = {}

    def parse(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(output_parsed=expected, id="resp_first_rgb", usage=None)

    analyzer = RecordingSkillDraftAnalyzer(
        settings(tmp_path, OPENAI_MODE="live", OPENAI_API_KEY="test-secret"),
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
        retry=RetryExecutor(0, sleep=lambda _seconds: None),
    )
    result, metadata = analyzer.analyze_first_frame_trace(
        request, first_rgb_path=first_rgb
    )

    assert result == expected
    assert metadata.response_id == "resp_first_rgb"
    assert captured["instructions"] == FIRST_FRAME_TRACE_INSTRUCTIONS
    [message] = captured["input"]  # type: ignore[misc]
    assert [item["type"] for item in message["content"]] == [
        "input_text",
        "input_image",
    ]
    assert message["content"][1]["image_url"].startswith(
        "data:image/jpeg;base64,"
    )
    payload = json.loads(message["content"][0]["text"])
    trace = payload["fingertip_trace"]
    assert len(trace) == 2
    assert set(trace[0]) == {
        "frame_index",
        "timestamp_ns",
        "thumb_tip",
        "index_tip",
        "status",
    }
    assert set(trace[0]["thumb_tip"]) == {
        "landmark_index",
        "normalized_xy",
        "pixel_xy",
    }
    serialized_trace = json.dumps(trace)
    for forbidden in ("depth", "distance", "candidate_state", "stable_state"):
        assert forbidden not in serialized_trace
    assert payload["keyframe_indices"] == [0]
    assert "image_pair_order" not in payload
    assert "depth_visualization" not in payload

    captured.clear()
    analyzer.analyze_first_frame_trace(
        request,
        first_rgb_path=first_rgb,
        as_file_fallback=True,
    )
    [fallback_message] = captured["input"]  # type: ignore[misc]
    assert [item["type"] for item in fallback_message["content"]] == [
        "input_text",
        "input_file",
    ]
    assert fallback_message["content"][1]["filename"] == "000000.jpg"


def test_compact_trace_schema_rejects_metric_or_state_fields() -> None:
    base = {
        "recording_id": "rgbd_0123456789abcdef0123456789abcdef",
        "name_hint": "recorded_wipe",
        "operator_instruction": "두 손가락으로 표면을 닦는다",
        "recording_summary": {"frame_count": 1},
        "primitive_catalog": ["motion.move_l"],
        "entity_role_catalog": ["target_surface"],
        "keyframe_indices": [0],
        "first_frame_index": 0,
        "visual_input_policy": "first_rgb_plus_local_fingertip_trace",
        "limitations": ["Metric state remains local."],
    }
    trace_frame = {
        "frame_index": 0,
        "timestamp_ns": 1,
        "thumb_tip": {
            "landmark_index": 4,
            "normalized_xy": (0.4, 0.5),
            "pixel_xy": (40, 50),
        },
        "index_tip": {
            "landmark_index": 8,
            "normalized_xy": (0.6, 0.5),
            "pixel_xy": (60, 50),
        },
        "status": "valid",
        "distance_m": 0.02,
    }

    with pytest.raises(ValidationError, match="distance_m"):
        RecordingSkillDraftInput.model_validate(
            {**base, "fingertip_trace": [trace_frame]}
        )


def test_responses_parse_sends_pdf_as_base64_input_file(tmp_path: Path) -> None:
    pdf = tmp_path / "rgb_contact_sheet.pdf"
    pdf.write_bytes(b"%PDF-1.4\nmock\n%%EOF")
    request = RecordingSkillDraftInput(
        recording_id="rgbd_0123456789abcdef0123456789abcdef",
        name_hint="recorded_wipe",
        operator_instruction="걸레로 표면을 닦는다",
        recording_summary={"frame_count": 10, "recording_fps": 10},
        primitive_catalog=["motion.move_l"],
        entity_role_catalog=["tool", "target_surface"],
        keyframe_indices=[5],
        limitations=["No trusted robot-base pose trajectory."],
    )
    expected = MockOpenAIClient().analyze_recording_skill_draft(request, "trace")[0]
    captured: dict[str, object] = {}

    def parse(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(output_parsed=expected, id="resp_pdf", usage=None)

    result, _metadata = parse_structured_response(
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
        retry=RetryExecutor(0, sleep=lambda _seconds: None),
        model="test-model",
        instructions="strict",
        payload=request,
        output_type=RecordingSkillDraft,
        trace_id="trace-pdf",
        file_paths=[pdf],
        image_detail="low",
    )

    assert result == expected
    [message] = captured["input"]  # type: ignore[misc]
    assert [item["type"] for item in message["content"]] == [
        "input_text",
        "input_file",
    ]
    file_part = message["content"][1]
    assert file_part["filename"] == "rgb_contact_sheet.pdf"
    assert file_part["detail"] == "low"
    assert file_part["file_data"].startswith("data:application/pdf;base64,")


def test_recording_analyzer_marks_rejected_images_for_file_fallback(
    tmp_path: Path,
) -> None:
    rgb_image = tmp_path / "rgb_keyframe.jpg"
    depth_image = tmp_path / "depth_keyframe.jpg"
    rgb_image.write_bytes(b"\xff\xd8mock-rgb-jpeg\xff\xd9")
    depth_image.write_bytes(b"\xff\xd8mock-depth-jpeg\xff\xd9")
    request = RecordingSkillDraftInput(
        recording_id="rgbd_0123456789abcdef0123456789abcdef",
        name_hint="recorded_wipe",
        operator_instruction="걸레로 표면을 닦는다",
        recording_summary={"frame_count": 10, "recording_fps": 10},
        primitive_catalog=["motion.move_l"],
        entity_role_catalog=["tool", "target_surface"],
        keyframe_indices=[5],
        limitations=["No trusted robot-base pose trajectory."],
    )

    def reject_images(**_kwargs: object) -> object:
        response = httpx.Response(
            400,
            request=httpx.Request("POST", "https://example.invalid/v1/responses"),
        )
        raise BadRequestError(
            "input_image payload is not supported",
            response=response,
            body={"error": "input_image payload is not supported"},
        )

    analyzer = RecordingSkillDraftAnalyzer(
        settings(
            tmp_path,
            OPENAI_MODE="live",
            OPENAI_API_KEY="test-secret",
            OPENAI_MAX_KEYFRAMES="1",
        ),
        client=SimpleNamespace(responses=SimpleNamespace(parse=reject_images)),
        retry=RetryExecutor(0, sleep=lambda _seconds: None),
    )

    with pytest.raises(ImageInputRejectedError, match="RGB-D image payload"):
        analyzer.analyze(
            request,
            rgb_paths=[rgb_image],
            depth_paths=[depth_image],
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
