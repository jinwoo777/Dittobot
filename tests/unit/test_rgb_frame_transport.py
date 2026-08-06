from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from robot_skill_system.api.contracts import (
    DraftTCPPathRequest,
    RecordingSkillDraftRequest,
)
from robot_skill_system.capture.rgb_frame_transport import (
    build_rgb_contact_sheet_pdf,
    build_rgb_keyframe_zip,
    build_rgbd_contact_sheet_pdf,
    build_rgbd_keyframe_zip,
)
from robot_skill_system.settings import Settings


def _jpeg(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (64, 48), color).save(path, format="JPEG")


def test_rgb_transport_builds_deterministic_zip_and_vision_pdf(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _jpeg(first, (255, 0, 0))
    _jpeg(second, (0, 0, 255))
    keyframes = [(3, first), (17, second)]

    archive_bytes = build_rgb_keyframe_zip("rgbd_transport_test", keyframes)
    assert archive_bytes == build_rgb_keyframe_zip("rgbd_transport_test", keyframes)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert archive.namelist() == [
            "manifest.json",
            "rgb/000_frame_000003.jpg",
            "rgb/001_frame_000017.jpg",
        ]
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["frame_count"] == 2
        assert [item["frame_index"] for item in manifest["frames"]] == [3, 17]

    pdf_bytes = build_rgb_contact_sheet_pdf(keyframes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1_000


def test_rgbd_transport_preserves_chronological_image_pairs(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.jpg"
    depth = tmp_path / "depth.jpg"
    _jpeg(rgb, (255, 0, 0))
    _jpeg(depth, (0, 255, 255))

    archive_bytes = build_rgbd_keyframe_zip(
        "rgbd_transport_test",
        [(4, rgb, depth)],
    )
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert archive.namelist() == [
            "manifest.json",
            "rgb/000_frame_000004.jpg",
            "depth/000_frame_000004.jpg",
        ]
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["keyframe_pair_count"] == 1
        assert manifest["pair_order"] == "rgb_then_aligned_depth_per_keyframe"

    pdf_bytes = build_rgbd_contact_sheet_pdf([(4, rgb, depth)])
    assert pdf_bytes.startswith(b"%PDF")


def test_rgb_transport_rejects_more_than_300_frames(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    _jpeg(image, (1, 2, 3))

    with pytest.raises(ValueError, match="300"):
        build_rgb_keyframe_zip(
            "rgbd_transport_test",
            [(index, image) for index in range(301)],
        )


def test_recording_draft_contract_and_settings_allow_300_frames(tmp_path: Path) -> None:
    request = RecordingSkillDraftRequest(
        recording_id="rgbd_transport_test",
        name_hint="recorded_skill",
        operator_instruction="작업을 분석한다",
        keyframe_count=300,
    )
    configured = Settings.from_env({"ARTIFACT_ROOT": str(tmp_path)}, root=tmp_path)

    assert request.keyframe_count == 300
    assert configured.openai_max_keyframes == 300
    with pytest.raises(ValidationError):
        RecordingSkillDraftRequest(
            recording_id="rgbd_transport_test",
            name_hint="recorded_skill",
            operator_instruction="작업을 분석한다",
            keyframe_count=301,
        )


def test_low_confidence_tcp_materialization_requires_mock_only_acknowledgement() -> None:
    with pytest.raises(ValidationError, match="Mock-only acknowledgement"):
        DraftTCPPathRequest(
            method="openai_rgbd_low_confidence_mock",
            operator_confirmed=True,
        )

    request = DraftTCPPathRequest(
        method="openai_rgbd_low_confidence_mock",
        operator_confirmed=True,
        acknowledge_low_confidence_mock_only=True,
    )

    assert request.acknowledge_low_confidence_mock_only is True


def test_operator_confirmed_fingertip_tcp_proxy_requires_acknowledgement() -> None:
    with pytest.raises(ValidationError, match="two-fingertip midpoint"):
        DraftTCPPathRequest(
            method="openai_rgbd_operator_confirmed",
            annotations=[],
            operator_confirmed=True,
        )
    request = DraftTCPPathRequest(
        method="openai_rgbd_operator_confirmed",
        annotations=[],
        operator_confirmed=True,
        acknowledge_two_fingertip_tcp_proxy=True,
    )
    assert request.acknowledge_two_fingertip_tcp_proxy is True
