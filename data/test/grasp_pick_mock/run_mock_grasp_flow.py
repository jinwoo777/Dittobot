#!/usr/bin/env python3
"""Replay the learned grip-point-to-close flow without robot or LLM calls.

The input is the validated-schema output of the offline grip-point experiment.
Object selection stands in for an LLM semantic result, while numeric point,
width, orientation, and Z diagnostics are owned by local code.  The generated
sequence intentionally stops at grasp verification and cannot contact hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from robot_skill_system.grasping.rg2_depth import (
    build_rg2_depth_diagnostic,
    summarize_grip_widths,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "object_relative_grip_point_train_validation":
        raise ValueError("input is not a grip-point train/validation result")
    objects = payload.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ValueError("input contains no learned objects")
    return payload


def _image_axis_angle_deg(axis_xy: list[float]) -> float:
    if len(axis_xy) != 2:
        raise ValueError("predicted image jaw axis must have two components")
    x, y = (float(axis_xy[0]), float(axis_xy[1]))
    if not all(math.isfinite(value) for value in (x, y)) or math.hypot(x, y) <= 1.0e-9:
        raise ValueError("predicted image jaw axis is invalid")
    angle = math.degrees(math.atan2(y, x))
    # A parallel-jaw axis is undirected.  Canonicalize to [-90, 90).
    return (angle + 90.0) % 180.0 - 90.0


def _build_object_flow(object_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    model = payload["learned_grip_model"]
    train_cases = [case for case in payload["train_cases"] if case.get("usable")]
    samples_m = [float(case["observed_fingertip_distance_m"]) for case in train_cases]
    width = summarize_grip_widths(samples_m)
    stored_width = model["grip_width_statistics_m"]
    if not math.isclose(
        width.mean_m,
        float(stored_width["example_gripper_width_m"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{object_name}: stored example width does not match source samples")

    validation = payload["validation"][0]
    prediction = validation["prediction"]
    depth = build_rg2_depth_diagnostic(
        closed_reference_descent_m=0.184,
        table_clearance_m=0.005,
        example_grip_width_m=width.mean_m,
    )
    image_yaw_deg = _image_axis_angle_deg(prediction["predicted_jaw_axis_pixel_xy"])
    grasp_camera_m = prediction["predicted_grasp_point_camera_m"]
    geometry_ready = grasp_camera_m is not None
    blockers = [
        "recording has no task-plane revision or T_camera_task_plane",
        "recording has no synchronized robot pose or verified hand-eye chain",
        "image-plane jaw angle is not a task-plane TCP yaw",
        "RG2 width-to-Z depth compensation is not calibrated for installed fingertips",
        "Doosan and RG2 hardware adapters are not verified/configured",
    ]
    commands = [
        {
            "sequence": 1,
            "operation": "runtime.bind_object_anchor",
            "mode": "mock_semantic_result",
            "object_class": object_name,
            "llm_numeric_geometry_accepted": False,
        },
        {
            "sequence": 2,
            "operation": "local.resolve_rgbd_grasp",
            "camera_point_m": grasp_camera_m,
            "example_width_m": width.mean_m,
            "image_jaw_axis_deg": image_yaw_deg,
            "ready": geometry_ready,
        },
        {
            "sequence": 3,
            "operation": "gripper.open",
            "tool": "$tool",
            "mode": "mock",
        },
        {
            "sequence": 4,
            "operation": "motion.move_j",
            "purpose": "move to anchor-relative pregrasp",
            "target": None,
            "status": "unresolved_without_task_plane_and_local_ik",
        },
        {
            "sequence": 5,
            "operation": "motion.align_tcp_yaw",
            "purpose": "locally resolve wrist orientation before descent",
            "image_plane_yaw_deg_diagnostic": image_yaw_deg,
            "joint6_target_deg": None,
            "status": "unresolved_without_camera_to_task_plane_rotation_and_ik",
        },
        {
            "sequence": 6,
            "operation": "motion.move_l",
            "purpose": "straight descent along task-plane minus Z",
            "closed_reference_baseline_descent_m": depth.baseline_descent_m,
            "executable_target": None,
            "status": "mock_comparison_only",
        },
        {
            "sequence": 7,
            "operation": "gripper.close",
            "tool": "$tool",
            "example_learned_contact_width_m": width.mean_m,
            "mode": "mock",
        },
        {
            "sequence": 8,
            "operation": "gripper.verify_state",
            "expected_state": "holding",
            "mode": "mock_unverified",
        },
    ]
    return {
        "runtime_object_binding": object_name,
        "skill_template": "grasp_at_runtime_target",
        "skill_template_contains_object_name": False,
        "validation_frame_policy": prediction["input_policy"],
        "semantic_perception": {
            "mode": "mock",
            "llm_called": False,
            "accepted_output": {"object_class": object_name},
            "note": "an online LLM may select semantics only; local code owns all geometry",
        },
        "learned_width": {
            **width.to_dict(),
            "source_recordings": [str(case["recording"]) for case in train_cases],
            "example_gripper_width_m": width.mean_m,
            "interpretation": "advisory human fingertip contact span",
        },
        "grasp_prediction": {
            "camera_point_m": grasp_camera_m,
            "image_jaw_axis_deg": image_yaw_deg,
            "task_plane_yaw_deg": None,
            "joint6_target_deg": None,
        },
        "depth_diagnostic": depth.to_dict(),
        "commands": commands,
        "mock_flow_completed_through_gripper_close": True,
        "hardware_preflight": {
            "passed": False,
            "hardware_commands_sent": False,
            "blockers": blockers,
        },
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    rows = []
    sections = []
    for object_name, payload in result["objects"].items():
        width = payload["learned_width"]
        depth = payload["depth_diagnostic"]
        grasp = payload["grasp_prediction"]
        samples = ", ".join(f"{value * 1000.0:.1f}" for value in width["samples_m"])
        rows.append(
            f"| {object_name} | {samples} | {width['mean_m'] * 1000.0:.1f} | "
            f"{grasp['image_jaw_axis_deg']:+.1f} | "
            f"{depth['baseline_descent_m'] * 1000.0:.1f} | "
            f"{depth['mock_descent_with_corrected_hypothesis_m'] * 1000.0:.1f} |"
        )
        flow_text = (
            "runtime binding → RGB-D 파지점 → gripper.open → pregrasp → yaw 정렬 → "
            "MoveL 하강 → gripper.close → holding 확인"
        )
        sections.append(
            f"""## {object_name}

- 동작 순서: `{flow_text}`
- 학습 폭 출처: `{', '.join(width['source_recordings'])}`
- 예시 파지 폭: `{width['mean_m'] * 1000.0:.2f} mm`
- 최종 상태: Mock close까지 완료, hardware preflight 차단
"""
        )
    report = f"""# RG2 파지까지 Mock flow

이 결과는 물체 이름을 스킬 정의에 고정하지 않고 런타임 anchor로만 바인딩합니다. 검증 이미지의
첫 RGB-D 프레임에서 얻은 파지점과 이미지 jaw 축을 사용했으며, LLM·DB·로봇·그리퍼 호출은
하지 않았습니다.

| 물체 | fingertip 표본 mm | 평균 폭 mm | image jaw ° | 184−5 mm | 수정식 적용 mm |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

마지막 열은 `R=110 mm`, `sin(theta)=(평균 폭/2)/R`라고 가정해
`184−5+R(1−cos(theta))`를 계산한 비교값입니다. 공식 사양의 110 mm는 전체 스트로크이므로
이 값은 실제 Z 명령으로 선택되지 않았습니다.

{chr(10).join(sections)}
## 실제 실행 전에 필요한 것

- 동일 active TCP·fingertip·task-plane revision에서 폭별 Z 보정 실측표
- validation 첫 프레임의 camera point/축을 task-plane으로 옮길 TF
- local IK와 collision/preflight 결과로 정한 TCP yaw; image angle을 joint 6에 직접 대입하지 않음
- Doosan/RG2 adapter 및 force/velocity profile의 hardware verification
"""
    path.write_text(report, encoding="utf-8")


def run(input_path: Path, output_root: Path) -> dict[str, Any]:
    source = _load_payload(input_path.resolve())
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    objects = {
        object_name: _build_object_flow(object_name, payload)
        for object_name, payload in source["objects"].items()
    }
    result = {
        "schema_version": "1.0",
        "experiment": "rg2_grasp_until_close_mock",
        "execution_mode": "mock_diagnostic_only",
        "llm_called": False,
        "robot_called": False,
        "gripper_called": False,
        "database_modified": False,
        "closed_reference_joint_deg": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0],
        "closed_reference_descent_m": 0.184,
        "table_clearance_m": 0.005,
        "objects": objects,
    }
    _write_json(output_root / "result.json", result)
    _write_report(output_root / "report.md", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grip-result",
        type=Path,
        default=Path("data/test/grip_point/result.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/test/grasp_pick_mock"),
    )
    args = parser.parse_args()
    try:
        result = run(args.grip_result, args.output)
    except Exception as exc:
        print(f"mock grasp flow failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    summary = {
        name: {
            "example_width_mm": payload["learned_width"]["mean_m"] * 1000.0,
            "mock_flow_completed": payload["mock_flow_completed_through_gripper_close"],
            "hardware_preflight_passed": payload["hardware_preflight"]["passed"],
        }
        for name, payload in result["objects"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"artifacts: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
