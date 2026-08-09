from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_skill_system.application import MVPApplication
from robot_skill_system.capture.interfaces import CameraIntrinsics, SynchronizedRGBDFrame
from robot_skill_system.grasping.live_profile_catalog import (
    load_registered_live_grip_profile,
    register_configured_live_grip_profile,
)
from robot_skill_system.perception.interfaces import ObjectDetection2D
from robot_skill_system.perception.live_scene import LiveSceneBuilder, load_learned_grip_point
from robot_skill_system.scene.models import BoundingBox2D
from robot_skill_system.settings import Settings
from robot_skill_system.storage.artifact_store import LocalArtifactStore
from robot_skill_system.storage.database import Database, StorageRepository

ROOT = Path(__file__).resolve().parents[2]


class _HammerDetector:
    def detect(self, frame: SynchronizedRGBDFrame) -> list[ObjectDetection2D]:
        del frame
        return [
            ObjectDetection2D(
                detection_id="hammer_1",
                class_name="hammer",
                bounding_box=BoundingBox2D(
                    x_min_px=4,
                    y_min_px=8,
                    x_max_px=28,
                    y_max_px=24,
                ),
                confidence=0.95,
            )
        ]


def test_configured_live_grip_is_registered_selected_and_checksum_verified(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'registry.db').as_posix()}")
    database.create_schema()
    repository = StorageRepository(database)
    store = LocalArtifactStore(tmp_path / "artifacts")
    repository.create_catalog_entry(
        kind="object",
        canonical_id="hammer",
        display_name="hammer",
    )
    source_path = ROOT / "configs/grasp_profiles/hammer_live_grasp.json"
    configured = load_learned_grip_point(source_path)

    selected = register_configured_live_grip_profile(
        repository,
        store,
        configured,
    )

    assert selected is not None
    assert selected.record.status == "active"
    assert selected.record.validation_status == "passed"
    assert selected.record.metadata_json["runtime_live_grip_profile"] is True
    active = repository.active_grip_profile_version(object_class_id="hammer")
    assert active is not None
    assert active.id == selected.record.id
    loaded = load_registered_live_grip_profile(
        store,
        active,
        source_path=source_path,
    )
    assert loaded.target_gripper_width_m == configured.target_gripper_width_m
    assert loaded.jaw_relative_angle_rad == configured.jaw_relative_angle_rad
    database.close()


def test_application_startup_activates_existing_hammer_grip_catalog(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "registry.db"
    database = Database(f"sqlite:///{database_path.as_posix()}")
    database.create_schema()
    StorageRepository(database).create_catalog_entry(
        kind="object",
        canonical_id="hammer",
        display_name="hammer",
    )
    database.close()
    settings = Settings.from_env(
        {
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "DATABASE_URL": f"sqlite:///{database_path.as_posix()}",
            "OPENAI_MODE": "mock",
            "ROBOT_EXECUTION_MODE": "mock",
            "DRY_RUN": "true",
        },
        root=ROOT,
    )

    service = MVPApplication(settings)
    try:
        active = service.repository.active_grip_profile_version(
            object_class_id="hammer"
        )
        assert active is not None
        assert active.id == service._live_grip_profile_version_id
        assert active.metadata_json["runtime_live_grip_profile"] is True
    finally:
        service.close()


def test_live_grip_accepts_one_aligned_pair_with_large_raw_timestamp_offset() -> None:
    color = np.full((32, 32, 3), 240, dtype=np.uint8)
    color[8:24, 4:28] = (40, 60, 90)
    depth = np.full((32, 32), 0.5, dtype=np.float32)
    frame = SynchronizedRGBDFrame(
        color_image_rgb=color,
        depth_image_m=depth,
        color_timestamp_ns=4_000_000_000,
        depth_timestamp_ns=4_000_000_000,
        color_intrinsics=CameraIntrinsics(
            width_px=32,
            height_px=32,
            fx_px=40.0,
            fy_px=40.0,
            cx_px=16.0,
            cy_px=16.0,
        ),
        frame_number=1,
        raw_color_timestamp_ns=1_000_000_000,
        raw_depth_timestamp_ns=4_000_000_000,
        raw_color_timestamp_clock_domain="hardware_clock",
        raw_depth_timestamp_clock_domain="hardware_clock",
    )
    profile = load_learned_grip_point(
        ROOT / "configs/grasp_profiles/hammer_live_grasp.json"
    )
    builder = LiveSceneBuilder(
        _HammerDetector(),
        profile,
        grip_profile_version_id="grip_v1",
        grip_profile_checksum_sha256="a" * 64,
    )

    anchor = builder.build_object(
        frame,
        base_to_camera=np.eye(4),
        base_to_plane=np.eye(4),
    )

    assert anchor.attributes["rgbd_pairing_policy"] == "aligned_frameset_pair"
    assert anchor.attributes["raw_rgbd_timestamp_skew_ms"] == 3000.0
    assert anchor.attributes["grip_profile_version_id"] == "grip_v1"
