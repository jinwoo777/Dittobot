"""Semantic demonstration analysis; geometry remains entirely local."""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from robot_skill_system.settings import OpenAIMode, Settings

from ._live import parse_structured_response
from .client import OpenAIClientFactory, RetryExecutor, new_trace_id
from .mock_client import MockOpenAIClient
from .schemas import APICallMetadata, DemonstrationAnalysis, DemonstrationAnalysisInput
from .semantic_validation import validate_demonstration_analysis

ANALYZER_INSTRUCTIONS = """You label the semantics of locally computed robot demonstrations.
Return only the strict schema. Treat entity IDs and local motion-fit intervals as authoritative.
Never invent XYZ poses, orientation, speed, acceleration, force values, code, or robot calls.
Use only IDs from entity_catalog, primitive_catalog, approved_motion_profiles, and
approved_force_profiles. Mark ambiguity explicitly."""


class DemonstrationAnalyzer:
    """Analyze keyframes and local summaries into a strict semantic schema."""

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

    def analyze(
        self, request: DemonstrationAnalysisInput
    ) -> tuple[DemonstrationAnalysis, APICallMetadata]:
        """Return semantic labels without changing local interval boundaries."""

        trace_id = new_trace_id()
        if self.settings.openai_mode is OpenAIMode.MOCK:
            analysis, metadata = self._mock.analyze_demonstration(request, trace_id)
        else:
            client = self._client or OpenAIClientFactory(self.settings).create()
            paths = [
                Path(item)
                for item in request.keyframe_paths[: self.settings.openai_max_keyframes]
            ]
            analysis, metadata = parse_structured_response(
                client=client,
                retry=self._retry,
                model=self.settings.openai_reasoning_model,
                instructions=ANALYZER_INSTRUCTIONS,
                payload=request,
                output_type=DemonstrationAnalysis,
                trace_id=trace_id,
                image_paths=paths,
                image_detail=self.settings.openai_image_detail,
            )
        validate_demonstration_analysis(
            analysis,
            entity_catalog=request.entity_catalog,
            primitive_catalog=request.primitive_catalog,
            approved_motion_profiles=request.approved_motion_profiles,
            approved_force_profiles=request.approved_force_profiles,
        )
        return analysis, metadata
