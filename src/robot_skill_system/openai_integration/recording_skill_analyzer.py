"""Bounded semantic drafting from chronological aligned RGB-D keyframes."""

from __future__ import annotations

from pathlib import Path

from openai import APIStatusError, OpenAI

from robot_skill_system.settings import OpenAIMode, Settings

from ._live import parse_structured_response
from .client import OpenAIClientFactory, RetryExecutor, new_trace_id
from .mock_client import MockOpenAIClient
from .schemas import (
    APICallMetadata,
    RecordingSkillDraft,
    RecordingSkillDraftInput,
)
from .semantic_validation import validate_recording_skill_draft

RECORDING_SKILL_INSTRUCTIONS = """Analyze chronological aligned RGB-D evidence for a robot
teaching recording and return only the strict schema. The JSON metadata is followed by exactly two
images for each keyframe index, always RGB first and its aligned depth colormap second. In the depth
colormap, nearer valid pixels are warmer and invalid pixels are black; use it only as qualitative
foreground/background and contact evidence, never as metric geometry.

The operator may imitate a two-jaw gripper with two intentionally extended fingertips, normally
thumb and index finger. Treat those two visible fingertip ends as the jaw tips. The visual TCP proxy
is their midpoint. Return exactly one observed_states item for EVERY supplied keyframe. When both
jaw tips are visible, return normalized [0,1] image positions for tip A, tip B, and their midpoint.
When either tip is missing or confused with a tool endpoint, set landmarks_detected=false, set all
three positions to null, classify the state as occluded or uncertain, and explain the failure.
Ignore curled or incidental fingers. Audit the whole sequence with trajectory_status,
valid_landmark_frame_count, usable_for_local_depth_path, and an explicit failure_reason whenever
fewer than four frames contain valid landmark pairs.

Also return scene_observation for the person's hand shape, tool shape, and workbench/work-surface
shape. For each item choose a representative supplied frame and a normalized image region when it
is visible. Work-surface output is a semantic ROI and plane-likelihood hint only. Local code will
re-open raw aligned depth and calibrated intrinsics to estimate metric geometry.

Suggest only operations and entity roles present in the supplied catalogs. Never invent
coordinates, poses, transforms, speed, acceleration, force, code, robot calls, or execution
permission. RGB-D images and the two-finger proxy contain no trusted robot-base trajectory or TF,
so executable must remain false, requires_pose_trajectory must remain true,
tcp_proxy_observation.semantic_only must remain true, and robot_tcp_pose_available must remain
false. Normalized image locations are not robot coordinates. Do not output camera coordinates,
metric depth, robot poses, or metric transforms. State important visual ambiguity explicitly."""

RECORDING_PDF_FALLBACK_INSTRUCTIONS = """Analyze the chronological RGB-D contact-sheet PDF as
evidence for a robot teaching recording and return only the strict schema. Each RGB image is paired
with its aligned depth colormap and labeled with the original frame index. Nearer valid depth pixels
are warmer and invalid pixels are black. The operator may imitate a gripper with two intentionally
extended fingertips; treat the fingertip ends as jaw tips and their midpoint as a qualitative TCP
proxy. Return one audited state for EVERY labeled keyframe. When both tips are visible, output their
normalized [0,1] image positions and midpoint; otherwise use null positions and an explicit failure
reason. Return hand, tool, and work-surface shapes with representative frame indices and normalized
regions as semantic hints for local raw-depth processing. Never output camera coordinates, metric
depth, poses, transforms,
speed, acceleration, force, code, robot calls, or execution permission. Suggest only catalog roles
and operations. The PDF contains no trusted robot-base trajectory or TF, so executable must remain
false, requires_pose_trajectory and tcp_proxy_observation.semantic_only must remain true, and
robot_tcp_pose_available must remain false. State important visual ambiguity explicitly."""


class ImageInputRejectedError(RuntimeError):
    """The provider rejected the direct image payload and a file fallback may be attempted."""


class RecordingSkillDraftAnalyzer:
    """Create a semantic draft; local code remains responsible for all executable geometry."""

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
        self,
        request: RecordingSkillDraftInput,
        *,
        rgb_paths: list[Path],
        depth_paths: list[Path],
    ) -> tuple[RecordingSkillDraft, APICallMetadata]:
        """Analyze checksum-verified RGB-depth pairs without client-selected paths."""

        if len(rgb_paths) != len(request.keyframe_indices) or len(depth_paths) != len(
            request.keyframe_indices
        ):
            raise ValueError("RGB, depth, and keyframe indices must have the same length")
        if len(rgb_paths) > self.settings.openai_max_keyframes:
            raise ValueError("keyframe count exceeds the configured OpenAI bound")
        if any(not path.is_file() for path in [*rgb_paths, *depth_paths]):
            raise ValueError("every selected RGB-D keyframe must be a local file")
        image_paths = [
            path
            for rgb_path, depth_path in zip(rgb_paths, depth_paths, strict=True)
            for path in (rgb_path, depth_path)
        ]

        trace_id = new_trace_id()
        if self.settings.openai_mode is OpenAIMode.MOCK:
            draft, metadata = self._mock.analyze_recording_skill_draft(request, trace_id)
        else:
            client = self._client or OpenAIClientFactory(self.settings).create()
            try:
                draft, metadata = parse_structured_response(
                    client=client,
                    retry=self._retry,
                    model=self.settings.openai_reasoning_model,
                    instructions=RECORDING_SKILL_INSTRUCTIONS,
                    payload=request,
                    output_type=RecordingSkillDraft,
                    trace_id=trace_id,
                    image_paths=image_paths,
                    image_detail=self.settings.openai_image_detail,
                )
            except APIStatusError as exc:
                if _is_image_input_rejection(exc):
                    raise ImageInputRejectedError(
                        "OpenAI rejected the direct RGB-D image payload"
                    ) from exc
                raise
        self._validate_draft(draft, request)
        return draft, metadata

    def analyze_pdf(
        self,
        request: RecordingSkillDraftInput,
        *,
        contact_sheet_path: Path,
    ) -> tuple[RecordingSkillDraft, APICallMetadata]:
        """Analyze a locally generated PDF when the provider rejects direct images."""

        if contact_sheet_path.suffix.lower() != ".pdf" or not contact_sheet_path.is_file():
            raise ValueError("contact-sheet fallback must be an existing PDF")
        trace_id = new_trace_id()
        if self.settings.openai_mode is OpenAIMode.MOCK:
            draft, metadata = self._mock.analyze_recording_skill_draft(request, trace_id)
        else:
            client = self._client or OpenAIClientFactory(self.settings).create()
            draft, metadata = parse_structured_response(
                client=client,
                retry=self._retry,
                model=self.settings.openai_reasoning_model,
                instructions=RECORDING_PDF_FALLBACK_INSTRUCTIONS,
                payload=request,
                output_type=RecordingSkillDraft,
                trace_id=trace_id,
                file_paths=[contact_sheet_path],
                image_detail=self.settings.openai_image_detail,
            )
        self._validate_draft(draft, request)
        return draft, metadata

    @staticmethod
    def _validate_draft(
        draft: RecordingSkillDraft,
        request: RecordingSkillDraftInput,
    ) -> None:
        validate_recording_skill_draft(
            draft,
            primitive_catalog=request.primitive_catalog,
            entity_role_catalog=request.entity_role_catalog,
            keyframe_indices=request.keyframe_indices,
        )


def _is_image_input_rejection(exc: APIStatusError) -> bool:
    if exc.status_code in {413, 415, 422}:
        return True
    if exc.status_code != 400:
        return False
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "image",
            "input_image",
            "payload too large",
            "request too large",
            "unsupported media",
        )
    )


__all__ = ["ImageInputRejectedError", "RecordingSkillDraftAnalyzer"]
