"""Resolve text/STT commands into non-executable runtime intent."""

from __future__ import annotations

from openai import OpenAI

from robot_skill_system.settings import OpenAIMode, Settings

from ._live import parse_structured_response
from .client import OpenAIClientFactory, RetryExecutor, new_trace_id
from .function_tools import SafeFunctionDispatcher
from .mock_client import MockOpenAIClient
from .schemas import APICallMetadata, RuntimeIntent
from .semantic_validation import validate_runtime_intent

INTENT_INSTRUCTIONS = """Map the command to a strict RuntimeIntent. Select only supplied entity
IDs and approved profile IDs. Do not invent coordinates, forces, speeds, code, or robot calls.
Report ambiguity rather than guessing an unlisted entity."""


class RuntimeIntentResolver:
    """Convert a natural-language command into a bounded search/binding intent."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: OpenAI | None = None,
        mock: MockOpenAIClient | None = None,
        retry: RetryExecutor | None = None,
        tool_dispatcher: SafeFunctionDispatcher | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._mock = mock or MockOpenAIClient()
        self._retry = retry or RetryExecutor(settings.openai_api_max_retries)
        self._tool_dispatcher = tool_dispatcher

    def resolve(
        self,
        command: str,
        *,
        entity_ids: list[str],
        approved_motion_profiles: list[str],
        approved_force_profiles: list[str],
    ) -> tuple[RuntimeIntent, APICallMetadata]:
        """Resolve intent for later local retrieval and validation; never execute it."""

        trace_id = new_trace_id()
        if self.settings.openai_mode is OpenAIMode.MOCK:
            intent, metadata = self._mock.resolve_intent(
                command,
                entity_ids,
                approved_motion_profiles,
                approved_force_profiles,
                trace_id,
            )
        else:
            client = self._client or OpenAIClientFactory(self.settings).create()
            payload = {
                "command": command,
                "entity_ids": entity_ids,
                "approved_motion_profiles": approved_motion_profiles,
                "approved_force_profiles": approved_force_profiles,
            }
            intent, metadata = parse_structured_response(
                client=client,
                retry=self._retry,
                model=self.settings.openai_reasoning_model,
                instructions=INTENT_INSTRUCTIONS,
                payload=payload,
                output_type=RuntimeIntent,
                trace_id=trace_id,
                tool_dispatcher=self._tool_dispatcher,
            )
        validate_runtime_intent(
            intent,
            entity_catalog=entity_ids,
            approved_motion_profiles=approved_motion_profiles,
            approved_force_profiles=approved_force_profiles,
        )
        return intent, metadata
