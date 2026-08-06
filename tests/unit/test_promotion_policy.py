from __future__ import annotations

from robot_skill_system.skills.promotion import PromotionPolicy


def _manual_task_plane() -> dict[str, object]:
    return {
        "calibration_id": "cal_manual_01",
        "operator_confirmed": True,
        "transform_convention": "T_camera_task_plane",
        "method": "operator_three_point_aligned_depth",
    }


def _trajectory() -> dict[str, object]:
    return {
        "calibration_id": "cal_manual_01",
        "operator_confirmed": True,
        "quality": {
            "sample_count": 2,
            "path_length_m": 0.01,
            "maximum_step_m": 0.01,
            "mean_confidence": 0.2,
            "coverage_ratio": 0.1,
        },
        "state_transitions": [],
    }


def test_recording_promotion_relaxes_observation_quality_but_not_geometry() -> None:
    decision = PromotionPolicy().evaluate_recording(
        calibration=_manual_task_plane(),
        trajectory=_trajectory(),
        has_rgbd_evidence=False,
        has_semantic_schema=False,
        gpt_fingertips_detected=False,
        handeye_verified=False,
        semantic_confidence=0.2,
    )

    assert decision.eligible is True
    assert decision.blockers == ()
    assert any("gripper behavior unresolved" in warning for warning in decision.warnings)
    assert any("camera-relative Mock" in warning for warning in decision.warnings)
    assert any("semantic confidence=0.200" in warning for warning in decision.warnings)
    blocking_checks = [item for item in decision.checks if item.blocking]
    assert {item.check_id for item in blocking_checks} == {
        "operator_task_plane",
        "metric_trajectory",
        "anchor_geometry",
    }
    assert all(item.passed for item in blocking_checks)


def test_auto_plane_hint_and_revision_mismatch_remain_blocking() -> None:
    auto_hint = {
        **_manual_task_plane(),
        "method": "local_depth_ransac_plane_hint",
        "transform_convention": "T_camera_task_plane_hint",
    }
    auto_decision = PromotionPolicy().evaluate_recording(
        calibration=auto_hint,
        trajectory=_trajectory(),
        has_rgbd_evidence=True,
        has_semantic_schema=True,
        gpt_fingertips_detected=True,
        handeye_verified=True,
    )
    assert auto_decision.eligible is False
    assert any("원점·+X·+Y" in blocker for blocker in auto_decision.blockers)

    mismatch = {**_trajectory(), "calibration_id": "cal_stale"}
    mismatch_decision = PromotionPolicy().evaluate_recording(
        calibration=_manual_task_plane(),
        trajectory=mismatch,
        has_rgbd_evidence=True,
        has_semantic_schema=True,
        gpt_fingertips_detected=True,
        handeye_verified=True,
    )
    assert mismatch_decision.eligible is False
    assert any("동일 revision" in blocker for blocker in mismatch_decision.blockers)


def test_block_candidate_needs_only_explicit_confirmation() -> None:
    policy = PromotionPolicy()
    assert policy.evaluate_block_candidate(operator_confirmed=True).eligible is True
    assert policy.evaluate_block_candidate(operator_confirmed=False).eligible is False
