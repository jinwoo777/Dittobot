"""Compose a semantic graph proposal for deterministic local materialization."""

from __future__ import annotations

from openai import OpenAI

from robot_skill_system.settings import OpenAIMode, Settings

from ._live import parse_structured_response
from .client import OpenAIClientFactory, RetryExecutor, new_trace_id
from .mock_client import MockOpenAIClient
from .motion_policy import MOTION_SIMPLIFICATION_INSTRUCTIONS
from .schemas import APICallMetadata, DemonstrationAnalysis, SkillGraphProposal
from .semantic_validation import validate_skill_graph_proposal

COMPOSER_INSTRUCTIONS = """Propose only supplied primitive operations, binding references, and
approved profile IDs. Do not supply motion coordinates or numeric speed, acceleration, force,
safety, or hardware parameters. Local validation and deterministic compilation are authoritative."""
COMPOSER_INSTRUCTIONS += f"\n\n{MOTION_SIMPLIFICATION_INSTRUCTIONS}"


class SkillGraphComposer:
    """Produce a strict, non-executable SkillGraph proposal."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: OpenAI | None = None,
        mock: MockOpenAIClient | None = None,
        retry: RetryExecutor | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._mock = mock or MockOpenAIClient()
        self._retry = retry or RetryExecutor(settings.openai_api_max_retries)

    def compose(
        self,
        analysis: DemonstrationAnalysis,
        primitive_catalog: list[str],
        *,
        binding_ref_catalog: list[str],
        approved_motion_profiles: list[str],
        approved_force_profiles: list[str],
    ) -> tuple[SkillGraphProposal, APICallMetadata]:
        """Compose semantics; local fitting supplies the final typed arguments."""

        trace_id = new_trace_id()
        if self.settings.openai_mode is OpenAIMode.MOCK:
            proposal, metadata = self._mock.compose_skill_graph(
                analysis,
                primitive_catalog,
                binding_ref_catalog,
                approved_motion_profiles,
                approved_force_profiles,
                trace_id,
            )
        else:
            client = self._client or OpenAIClientFactory(self.settings).create()
            payload = {
                "analysis": analysis.model_dump(mode="json"),
                "primitive_catalog": primitive_catalog,
                "binding_ref_catalog": binding_ref_catalog,
                "approved_motion_profiles": approved_motion_profiles,
                "approved_force_profiles": approved_force_profiles,
            }
            proposal, metadata = parse_structured_response(
                client=client,
                retry=self._retry,
                model=self.settings.openai_reasoning_model,
                instructions=COMPOSER_INSTRUCTIONS,
                payload=payload,
                output_type=SkillGraphProposal,
                trace_id=trace_id,
            )
        validate_skill_graph_proposal(
            proposal,
            primitive_catalog=primitive_catalog,
            binding_ref_catalog=binding_ref_catalog,
            approved_motion_profiles=approved_motion_profiles,
            approved_force_profiles=approved_force_profiles,
        )
        return proposal, metadata
