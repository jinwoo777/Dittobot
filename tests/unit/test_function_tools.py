from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_skill_system.openai_integration._live import parse_structured_response
from robot_skill_system.openai_integration.client import RetryExecutor
from robot_skill_system.openai_integration.function_tools import (
    FORBIDDEN_TOOL_NAMES,
    FunctionScene,
    FunctionSceneEntity,
    FunctionSkillManifest,
    FunctionSkillVersion,
    InvalidToolArgumentsError,
    LocalReadOnlyToolProvider,
    SafeFunctionDispatcher,
    SearchSkillRegistryArguments,
    SearchSkillRegistryOutput,
    ToolNotAllowedError,
    ToolOutputValidationError,
    UnknownToolError,
)
from robot_skill_system.openai_integration.schemas import RuntimeIntent
from robot_skill_system.openai_integration.tool_schemas import SAFE_FUNCTION_TOOLS


def _skill(
    version: str,
    *,
    lifecycle: str,
    validation: str,
    checksum_character: str,
    hardware_compatible: bool = False,
) -> FunctionSkillVersion:
    return FunctionSkillVersion(
        skill_id="wipe_surface",
        name="Wipe surface",
        intent="contact",
        version=version,
        lifecycle_status=lifecycle,
        validation_status=validation,
        variant="default",
        description="Wipe a table with a verified wiper.",
        graph_checksum_sha256=checksum_character * 64,
        generated_code_checksum_sha256=checksum_character * 64,
        hardware_compatible=hardware_compatible,
    )


def _dispatcher() -> SafeFunctionDispatcher:
    scene = FunctionScene(
        scene_id="scene_01",
        timestamp_ns=1_700_000_000_000_000_000,
        reference_frame="base",
        valid_for_ms=1000,
        calibration_id="calibration_01",
        confidence=0.96,
        entities=[
            FunctionSceneEntity(
                entity_id="wiper_01",
                entity_kind="tool",
                class_or_role="blue_wiper",
                confidence=0.98,
                attached=True,
            ),
            FunctionSceneEntity(
                entity_id="surface_01",
                entity_kind="surface",
                class_or_role="contact_target",
                confidence=0.97,
                attached=None,
            ),
        ],
    )
    provider = LocalReadOnlyToolProvider(
        skill_versions=[
            _skill("1.0.0", lifecycle="active", validation="passed", checksum_character="a"),
            _skill(
                "1.1.0",
                lifecycle="validated",
                validation="passed",
                checksum_character="b",
                hardware_compatible=True,
            ),
            _skill(
                "2.0.0-candidate",
                lifecycle="candidate",
                validation="pending",
                checksum_character="c",
            ),
        ],
        manifests=[
            FunctionSkillManifest(
                skill_id="wipe_surface",
                version="1.0.0",
                skill_graph_uri="skills/wipe_surface/1.0.0/skill_graph.json",
                skill_graph_checksum_sha256="a" * 64,
                compiled_skill_uri="skills/wipe_surface/1.0.0/compiled_skill.py",
                compiled_skill_checksum_sha256="d" * 64,
                validation_report_uri="skills/wipe_surface/1.0.0/validation.json",
                validation_report_checksum_sha256="e" * 64,
            )
        ],
        scenes=[scene],
    )
    return SafeFunctionDispatcher(provider)


def test_only_six_read_only_strict_function_schemas_are_exposed() -> None:
    expected_names = {
        "search_skill_registry",
        "get_skill_manifest",
        "get_primitive_catalog",
        "query_scene_entities",
        "get_scene_summary",
        "compare_skill_versions",
    }
    assert {str(item["name"]) for item in SAFE_FUNCTION_TOOLS} == expected_names
    assert expected_names.isdisjoint(FORBIDDEN_TOOL_NAMES)
    for tool in SAFE_FUNCTION_TOOLS:
        parameters = tool["parameters"]
        assert isinstance(parameters, dict)
        assert tool["type"] == "function"
        assert tool["strict"] is True
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])


def test_dispatcher_executes_all_six_read_only_tools() -> None:
    dispatcher = _dispatcher()

    search = dispatcher.dispatch(
        "search_skill_registry",
        {"query": "wipe", "lifecycle": "validated", "limit": 10},
    )
    assert isinstance(search, SearchSkillRegistryOutput)
    assert [item.version for item in search.matches] == ["1.0.0", "1.1.0"]

    manifest = dispatcher.dispatch(
        "get_skill_manifest", {"skill_id": "wipe_surface", "version": "1.0.0"}
    )
    assert manifest.manifest.skill_graph_checksum_sha256 == "a" * 64

    primitives = dispatcher.dispatch(
        "get_primitive_catalog", {"category": "motion", "limit": 100}
    )
    assert primitives.primitives
    assert all(item.operation_name.startswith("motion.") for item in primitives.primitives)

    entities = dispatcher.dispatch(
        "query_scene_entities",
        {
            "scene_id": "scene_01",
            "query": "blue",
            "entity_kind": "tool",
            "limit": 10,
        },
    )
    assert [item.entity_id for item in entities.entities] == ["wiper_01"]

    summary = dispatcher.dispatch("get_scene_summary", {"scene_id": "scene_01"})
    assert summary.entity_counts.tools == 1
    assert summary.entity_counts.surfaces == 1

    comparison = dispatcher.dispatch(
        "compare_skill_versions",
        {
            "skill_id": "wipe_surface",
            "left_version": "1.0.0",
            "right_version": "1.1.0",
        },
    )
    assert comparison.left.graph_checksum_sha256 == "a" * 64
    assert "graph_checksum_sha256" in comparison.differing_fields
    assert "hardware_compatible" in comparison.differing_fields


@pytest.mark.parametrize(
    "name",
    [
        "execute_robot_motion",
        "raw_move_j",
        "raw_force_control",
        "execute_shell",
        "arbitrary_file_write",
        "disable_safety",
    ],
)
def test_dispatcher_explicitly_rejects_execution_file_shell_and_safety_tools(
    name: str,
) -> None:
    with pytest.raises(ToolNotAllowedError):
        _dispatcher().dispatch(name, {})


def test_dispatcher_rejects_unknown_tool() -> None:
    with pytest.raises(UnknownToolError):
        _dispatcher().dispatch("look_up_anything", {})


@pytest.mark.parametrize(
    "arguments",
    [
        "not-json",
        {"category": "motion"},
        {"category": "motion", "limit": 1, "execute": True},
        {"category": "motion", "limit": "1"},
        {"category": "motion", "limit": 101},
        {"category": "raw_motion", "limit": 1},
    ],
)
def test_dispatcher_rejects_malformed_missing_extra_and_out_of_range_arguments(
    arguments: str | dict[str, object],
) -> None:
    with pytest.raises(InvalidToolArgumentsError):
        _dispatcher().dispatch("get_primitive_catalog", arguments)


class _InvalidOutputProvider(LocalReadOnlyToolProvider):
    def search_skill_registry(
        self, arguments: SearchSkillRegistryArguments
    ) -> SearchSkillRegistryOutput:
        del arguments
        return {"matches": [], "unexpected": True}  # type: ignore[return-value]


def test_dispatcher_validates_provider_output() -> None:
    dispatcher = SafeFunctionDispatcher(_InvalidOutputProvider())
    with pytest.raises(ToolOutputValidationError):
        dispatcher.dispatch(
            "search_skill_registry",
            {"query": "wipe", "lifecycle": "active", "limit": 5},
        )


def test_responses_parse_runs_serial_read_only_tool_then_structured_output(
    tmp_path: Path,
) -> None:
    del tmp_path
    expected = RuntimeIntent(
        intent="wipe_surface",
        style="normal",
        repetitions=1,
        candidate_entity_ids=[],
        confidence=0.95,
    )
    calls: list[dict[str, object]] = []
    responses = iter(
        (
            SimpleNamespace(
                output_parsed=None,
                id="resp_tool",
                output=[
                    {
                        "type": "reasoning",
                        "id": "reasoning_01",
                        "summary": [],
                        "encrypted_content": "encrypted-test-content",
                    },
                    SimpleNamespace(
                        type="function_call",
                        name="get_primitive_catalog",
                        arguments='{"category":"motion","limit":1}',
                        call_id="call_01",
                    )
                ],
                usage=SimpleNamespace(input_tokens=3, output_tokens=4, total_tokens=7),
            ),
            SimpleNamespace(
                output_parsed=expected,
                id="resp_final",
                output=[],
                usage=SimpleNamespace(input_tokens=5, output_tokens=6, total_tokens=11),
            ),
        )
    )

    def fake_parse(**kwargs: object) -> object:
        calls.append(kwargs)
        return next(responses)

    result, metadata = parse_structured_response(
        client=SimpleNamespace(responses=SimpleNamespace(parse=fake_parse)),
        retry=RetryExecutor(0, sleep=lambda _seconds: None),
        model="test-model",
        instructions="return bounded intent",
        payload={"command": "wipe"},
        output_type=RuntimeIntent,
        trace_id="trace-function-test",
        tool_dispatcher=_dispatcher(),
    )

    assert result == expected
    assert len(calls) == 2
    assert calls[0]["parallel_tool_calls"] is False
    assert calls[0]["max_tool_calls"] == 1
    assert {item["name"] for item in calls[0]["tools"]} == {
        item["name"] for item in SAFE_FUNCTION_TOOLS
    }
    second_input = calls[1]["input"]
    assert second_input[-1]["type"] == "function_call_output"
    assert second_input[-3]["type"] == "reasoning"
    assert second_input[-3]["encrypted_content"] == "encrypted-test-content"
    tool_output = json.loads(second_input[-1]["output"])
    assert tool_output["primitives"][0]["operation_name"].startswith("motion.")
    assert metadata.response_id == "resp_final"
    assert metadata.attempts == 2
    assert metadata.input_tokens == 8
    assert metadata.output_tokens == 10
    assert metadata.total_tokens == 18


def test_responses_structured_output_default_path_exposes_no_tools() -> None:
    expected = RuntimeIntent(
        intent="inspect_scene",
        style="normal",
        repetitions=1,
        candidate_entity_ids=[],
        confidence=0.9,
    )
    calls: list[dict[str, object]] = []

    def fake_parse(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(output_parsed=expected, id="resp_structured", usage=None)

    result, _ = parse_structured_response(
        client=SimpleNamespace(responses=SimpleNamespace(parse=fake_parse)),
        retry=RetryExecutor(0, sleep=lambda _seconds: None),
        model="test-model",
        instructions="strict structured output",
        payload={"command": "inspect"},
        output_type=RuntimeIntent,
        trace_id="trace-structured-test",
    )

    assert result == expected
    assert len(calls) == 1
    assert "tools" not in calls[0]
    assert "parallel_tool_calls" not in calls[0]
