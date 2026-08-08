"""Bounded semantic drafting from chronological aligned RGB-D keyframes."""

from __future__ import annotations

from pathlib import Path

from openai import APIStatusError, OpenAI

from robot_skill_system.settings import OpenAIMode, Settings

from ._live import parse_structured_response
from .client import OpenAIClientFactory, RetryExecutor, new_trace_id
from .mock_client import MockOpenAIClient
from .motion_policy import MOTION_SIMPLIFICATION_INSTRUCTIONS
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

Use the first RGB image to identify only semantic object and tool regions.
Use the compact thumb/index trace as chronological hand-motion evidence for the complete recording.
Read it from the JSON metadata. The trace is a semantic sequence, not metric geometry. Do not turn
its pixels, depths, distances, or timestamps into coordinates or motion targets. Attached RGB-D
keyframes are bounded visual audit context and do not override the locally computed trace.

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
RECORDING_SKILL_INSTRUCTIONS += f"\n\n{MOTION_SIMPLIFICATION_INSTRUCTIONS}"

RECORDING_PDF_FALLBACK_INSTRUCTIONS = """Analyze the chronological RGB-D contact-sheet PDF as
evidence for a robot teaching recording and return only the strict schema. Each RGB image is paired
with its aligned depth colormap and labeled with the original frame index. Nearer valid depth pixels
are warmer and invalid pixels are black. Use the first RGB panel to identify only semantic object
and tool regions. Use the compact thumb/index trace supplied in the JSON metadata as the
chronological hand-motion evidence for the complete recording. Do not convert trace pixels,
depths, distances, or timestamps into coordinates or motion targets. The contact sheet is bounded
visual audit context and does not override the locally computed trace. The operator may imitate a
gripper with two intentionally extended fingertips; treat the fingertip ends as jaw tips and their
midpoint as a qualitative TCP proxy. Return one audited state for EVERY labeled keyframe. When both
tips are visible, output their normalized [0,1] image positions and midpoint; otherwise use null
positions and an explicit failure
reason. Return hand, tool, and work-surface shapes with representative frame indices and normalized
regions as semantic hints for local raw-depth processing. Never output camera coordinates, metric
depth, poses, transforms,
speed, acceleration, force, code, robot calls, or execution permission. Suggest only catalog roles
and operations. The PDF contains no trusted robot-base trajectory or TF, so executable must remain
false, requires_pose_trajectory and tcp_proxy_observation.semantic_only must remain true, and
robot_tcp_pose_available must remain false. State important visual ambiguity explicitly."""
RECORDING_PDF_FALLBACK_INSTRUCTIONS += f"\n\n{MOTION_SIMPLIFICATION_INSTRUCTIONS}"

FIRST_FRAME_TRACE_INSTRUCTIONS = """Analyze a robot teaching recording using exactly two inputs:
(1) one RGB image from the first manifest frame and (2) the compact thumb/index trace in the JSON
metadata, covering every stored video frame in the complete recording. The trace contains
image-space locations only. It deliberately excludes depth, metric distance, gripper classification,
and robot poses.

Use the first RGB image only to identify semantic object and tool regions for the required tool,
target object, and work surface. Every representative_frame_index in scene_observation must refer
to that first frame.
Return normalized [0,1] regions, never metric coordinates. A detected target object must use the
target_object field. These regions are hints: local code reopens aligned depth and calibrated
intrinsics, reconstructs a camera-frame anchor, and requires operator confirmation before the
result can affect a skill.

Use the complete thumb/index trace to describe how the hand moves relative to those semantic
regions and propose the smallest chronological sequence of catalogued primitives. The local metric
trajectory fitter is authoritative for MoveL, verified MoveC, periodic, and spline classification.
Do not convert trace pixels or timestamps into coordinates or motion targets.
The tcp_proxy_observation audits only the supplied first RGB frame; it is advisory and must not
override MediaPipe+depth state inference. Never output or infer depth, metric distance, coordinates,
poses, transforms, velocity, acceleration, force, thresholds, code, robot calls, or execution
permission. executable must remain false, requires_pose_trajectory must remain true,
tcp_proxy_observation.semantic_only must remain true, and robot_tcp_pose_available must remain
false. State occlusion and ambiguity explicitly."""
FIRST_FRAME_TRACE_INSTRUCTIONS += f"\n\n{MOTION_SIMPLIFICATION_INSTRUCTIONS}"

RGB_KEYFRAME_TRACE_INSTRUCTIONS = """Analyze a robot teaching recording using two bounded
evidence streams: (1) the supplied chronological RGB keyframes whose global frame indices are
listed in keyframe_indices and (2) the compact thumb/index trace in the JSON metadata covering
every stored frame. No depth image, metric fingertip distance, gripper classification, or robot
pose is sent to you.

When demonstration_cases contains more than one item, treat each item as an independent example
of the same requested skill. Respect its global_frame_offset and keyframe mapping; do not infer a
physical motion transition between the end of one case and the beginning of the next case.

Use the RGB keyframes to identify semantic object, tool, hand, and work-surface regions. Every
representative_frame_index and TCP observed state must refer to one of the supplied keyframe
indices. Return normalized [0,1] image locations only. Use the full compact trace to describe the
chronological hand motion between keyframes, but never convert trace pixels or timestamps into
coordinates or motion targets. Local aligned depth, MediaPipe tracking, and the local metric
trajectory fitter remain authoritative.

Never output or infer depth, metric distance, coordinates, poses, transforms, velocity,
acceleration, force, thresholds, code, robot calls, or execution permission. executable must
remain false, requires_pose_trajectory must remain true, tcp_proxy_observation.semantic_only must
remain true, and robot_tcp_pose_available must remain false. State occlusion and ambiguity
explicitly."""
RGB_KEYFRAME_TRACE_INSTRUCTIONS += f"\n\n{MOTION_SIMPLIFICATION_INSTRUCTIONS}"


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

    def analyze_first_frame_trace(
        self,
        request: RecordingSkillDraftInput,
        *,
        first_rgb_path: Path,
        as_file_fallback: bool = False,
    ) -> tuple[RecordingSkillDraft, APICallMetadata]:
        """Analyze one initial RGB image plus a full local fingertip trace."""

        if request.visual_input_policy != "first_rgb_plus_local_fingertip_trace":
            raise ValueError("first-frame analysis requires the compact trace input policy")
        if not first_rgb_path.is_file():
            raise ValueError("first RGB frame must be a checksum-verified local file")
        trace_id = new_trace_id()
        if self.settings.openai_mode is OpenAIMode.MOCK:
            draft, metadata = self._mock.analyze_recording_skill_draft(request, trace_id)
        else:
            client = self._client or OpenAIClientFactory(self.settings).create()
            semantic_payload = request.model_dump(
                mode="json",
                exclude={"image_pair_order", "depth_visualization"},
            )
            try:
                if as_file_fallback:
                    draft, metadata = parse_structured_response(
                        client=client,
                        retry=self._retry,
                        model=self.settings.openai_reasoning_model,
                        instructions=FIRST_FRAME_TRACE_INSTRUCTIONS,
                        payload=semantic_payload,
                        output_type=RecordingSkillDraft,
                        trace_id=trace_id,
                        file_paths=[first_rgb_path],
                        image_detail=self.settings.openai_image_detail,
                    )
                else:
                    draft, metadata = parse_structured_response(
                        client=client,
                        retry=self._retry,
                        model=self.settings.openai_reasoning_model,
                        instructions=FIRST_FRAME_TRACE_INSTRUCTIONS,
                        payload=semantic_payload,
                        output_type=RecordingSkillDraft,
                        trace_id=trace_id,
                        image_paths=[first_rgb_path],
                        image_detail=self.settings.openai_image_detail,
                    )
            except APIStatusError as exc:
                if not as_file_fallback and _is_image_input_rejection(exc):
                    raise ImageInputRejectedError(
                        "OpenAI rejected the first-frame image payload"
                    ) from exc
                raise
        self._validate_draft(draft, request)
        return draft, metadata

    def analyze_rgb_keyframe_trace(
        self,
        request: RecordingSkillDraftInput,
        *,
        rgb_paths: list[Path],
        as_file_fallback: bool = False,
    ) -> tuple[RecordingSkillDraft, APICallMetadata]:
        """Analyze selected RGB frames plus a complete local-only fingertip trace."""

        if request.visual_input_policy != "rgb_keyframes_plus_local_fingertip_trace":
            raise ValueError("RGB keyframe analysis requires the RGB trace input policy")
        if len(rgb_paths) != len(request.keyframe_indices):
            raise ValueError("RGB paths and keyframe indices must have the same length")
        if len(rgb_paths) > self.settings.openai_max_keyframes:
            raise ValueError("keyframe count exceeds the configured OpenAI bound")
        if any(not path.is_file() for path in rgb_paths):
            raise ValueError("every RGB keyframe must be a checksum-verified local file")
        trace_id = new_trace_id()
        if self.settings.openai_mode is OpenAIMode.MOCK:
            draft, metadata = self._mock.analyze_recording_skill_draft(request, trace_id)
        else:
            client = self._client or OpenAIClientFactory(self.settings).create()
            semantic_payload = request.model_dump(
                mode="json",
                exclude={"image_pair_order", "depth_visualization"},
            )
            try:
                draft, metadata = parse_structured_response(
                    client=client,
                    retry=self._retry,
                    model=self.settings.openai_reasoning_model,
                    instructions=RGB_KEYFRAME_TRACE_INSTRUCTIONS,
                    payload=semantic_payload,
                    output_type=RecordingSkillDraft,
                    trace_id=trace_id,
                    file_paths=rgb_paths if as_file_fallback else None,
                    image_paths=None if as_file_fallback else rgb_paths,
                    image_detail=self.settings.openai_image_detail,
                )
            except APIStatusError as exc:
                if not as_file_fallback and _is_image_input_rejection(exc):
                    raise ImageInputRejectedError(
                        "OpenAI rejected the RGB keyframe image payload"
                    ) from exc
                raise
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
