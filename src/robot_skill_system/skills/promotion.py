"""Shared candidate-readiness policy for recorded and block-authored skills.

Only evidence quality is relaxed here.  Schema, primitive catalog, profiles,
compiler integrity, Mock regression, activation, and hardware preflight remain
owned by their existing fail-closed validators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    check_id: str
    label: str
    passed: bool
    blocking: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "label": self.label,
            "passed": self.passed,
            "required": self.blocking,
            "blocking": self.blocking,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    checks: tuple[PromotionCheck, ...]

    @property
    def eligible(self) -> bool:
        return all(item.passed for item in self.checks if item.blocking)

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(
            item.detail for item in self.checks if item.blocking and not item.passed
        )

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            item.detail for item in self.checks if not item.blocking and not item.passed
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "checks": [item.as_dict() for item in self.checks],
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


class PromotionPolicy:
    """Evaluate one source without weakening downstream execution gates."""

    minimum_metric_pose_count = 2
    final_task_plane_method = "operator_three_point_aligned_depth"

    def evaluate_recording(
        self,
        *,
        calibration: dict[str, Any] | None,
        trajectory: dict[str, Any] | None,
        has_rgbd_evidence: bool,
        has_semantic_schema: bool,
        gpt_fingertips_detected: bool,
        handeye_verified: bool,
        semantic_confidence: float | None = None,
    ) -> PromotionDecision:
        fixed_workspace_reuse = bool(
            calibration
            and calibration.get("fixed_workspace_reuse") is True
            and calibration.get("transform_convention") == "T_camera_task_plane"
        )
        calibration_valid = bool(
            calibration
            and calibration.get("transform_convention") == "T_camera_task_plane"
            and (
                fixed_workspace_reuse
                or (
                    calibration.get("operator_confirmed") is True
                    and calibration.get("method") == self.final_task_plane_method
                )
            )
        )
        quality = trajectory.get("quality") if trajectory else None
        calibration_id = calibration.get("calibration_id") if calibration else None
        metric_pose_count = (
            int(quality.get("sample_count") or 0) if isinstance(quality, dict) else 0
        )
        same_calibration = bool(
            calibration_valid
            and (
                fixed_workspace_reuse
                or (
                    trajectory
                    and trajectory.get("operator_confirmed") is True
                    and trajectory.get("calibration_id") == calibration_id
                )
            )
        )
        geometry_verified = bool(
            fixed_workspace_reuse
            or (
                same_calibration
                and metric_pose_count >= self.minimum_metric_pose_count
                and isinstance(quality, dict)
                and float(quality.get("path_length_m") or 0.0) >= 0.005
                and float(quality.get("maximum_step_m") or 0.0) <= 0.5
            )
        )
        stable_gripper_state = bool(
            trajectory
            and isinstance(trajectory.get("state_transitions"), list)
            and trajectory.get("state_transitions")
        )
        mean_confidence = (
            float(quality.get("mean_confidence") or 0.0)
            if isinstance(quality, dict)
            else 0.0
        )
        coverage_ratio = (
            float(quality.get("coverage_ratio") or 0.0)
            if isinstance(quality, dict)
            else 0.0
        )
        return PromotionDecision(
            checks=(
                PromotionCheck(
                    "operator_task_plane",
                    (
                        "고정 workspace task-plane TF"
                        if fixed_workspace_reuse
                        else "운영자 확인 수동 3점 task-plane TF"
                    ),
                    calibration_valid,
                    True,
                    (
                        "고정한 [0,0,90,0,90,-90] 기준 자세의 NPZ/URDF "
                        "workspace 좌표계를 재사용합니다."
                        if fixed_workspace_reuse
                        else f"{calibration['calibration_id']} · 수동 3점 task plane"
                        if calibration_valid and calibration
                        else "원점·+X·+Y를 지정해 최종 task-plane TF를 확정하세요."
                    ),
                ),
                PromotionCheck(
                    "metric_trajectory",
                    (
                        "고정 workspace 기반 pose trajectory"
                        if fixed_workspace_reuse
                        else "같은 calibration의 metric pose trajectory"
                    ),
                    (
                        fixed_workspace_reuse
                        or (
                            same_calibration
                            and metric_pose_count >= self.minimum_metric_pose_count
                        )
                    ),
                    True,
                    (
                        "Candidate 등록 시 녹화 RGB-D trace를 고정 task-plane으로 "
                        "자동 변환합니다."
                        if fixed_workspace_reuse and trajectory is None
                        else f"동일 calibration에서 유효 pose {metric_pose_count}개"
                        if same_calibration
                        else "최종 task plane과 동일 revision으로 trajectory를 생성하세요."
                    ),
                ),
                PromotionCheck(
                    "anchor_geometry",
                    (
                        "고정 workspace anchor-relative geometry"
                        if fixed_workspace_reuse
                        else "anchor-relative geometry 검증"
                    ),
                    geometry_verified,
                    True,
                    (
                        "고정 task-plane anchor로 생성합니다."
                        if fixed_workspace_reuse and trajectory is None
                        else "로컬 geometry envelope와 연속성 검증 통과"
                        if geometry_verified
                        else "경로 길이·step·anchor-relative geometry 검증이 필요합니다."
                    ),
                ),
                PromotionCheck(
                    "rgbd_evidence",
                    "시간 정렬 RGB-D 증거",
                    has_rgbd_evidence,
                    False,
                    "RGB-D manifest 품질은 경고이며 Candidate 생성을 차단하지 않습니다.",
                ),
                PromotionCheck(
                    "semantic_schema",
                    "구조화 semantic 분석",
                    has_semantic_schema,
                    False,
                    "semantic 분석이 없어도 로컬 geometry Candidate는 허용됩니다.",
                ),
                PromotionCheck(
                    "semantic_confidence",
                    "semantic 분석 신뢰도",
                    bool(
                        semantic_confidence is not None
                        and semantic_confidence >= 0.5
                    ),
                    False,
                    (
                        f"semantic confidence={semantic_confidence:.3f}; "
                        "낮은 신뢰도는 advisory warning으로 보존됩니다."
                        if semantic_confidence is not None
                        else "semantic confidence가 없어 advisory warning으로 보존됩니다."
                    ),
                ),
                PromotionCheck(
                    "gpt_fingertips",
                    "GPT fingertip 관찰",
                    gpt_fingertips_detected,
                    False,
                    "GPT fingertip 결과는 advisory이며 MediaPipe+depth가 최종 권한입니다.",
                ),
                PromotionCheck(
                    "gripper_behavior",
                    "안정된 로컬 gripper 상태",
                    stable_gripper_state,
                    False,
                    "gripper behavior unresolved; motion-only Candidate로 저장됩니다.",
                ),
                PromotionCheck(
                    "trajectory_confidence",
                    "trajectory 관측 품질",
                    mean_confidence >= 0.6 and coverage_ratio >= 0.5,
                    False,
                    (
                        f"mean confidence={mean_confidence:.3f}, coverage={coverage_ratio:.3f}; "
                        "낮은 품질은 비활성 Candidate 경고로 보존됩니다."
                    ),
                ),
                PromotionCheck(
                    "handeye_transform_candidate",
                    "검증된 flange → camera TF",
                    handeye_verified,
                    False,
                    "hand-eye/base chain 부재 시 camera-relative Mock 전용입니다.",
                ),
            )
        )

    def evaluate_block_candidate(self, *, operator_confirmed: bool) -> PromotionDecision:
        """Block skills need no demonstration or OpenAI observation."""

        return PromotionDecision(
            checks=(
                PromotionCheck(
                    "operator_confirmation",
                    "운영자 확인",
                    operator_confirmed,
                    True,
                    "블록 Candidate 저장에는 운영자 확인이 필요합니다.",
                ),
            )
        )


__all__ = ["PromotionCheck", "PromotionDecision", "PromotionPolicy"]
