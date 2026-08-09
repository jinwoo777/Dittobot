"""Deterministic conversion from a live grasp anchor to an object-relative pick.

This module deliberately creates an *ephemeral* graph for a current scene.  It
does not overwrite the taught skill or persist robot-base targets: every pose
remains relative to the current ``$object`` contact anchor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from aruco.object_width_workspace import RuntimeWorkspace, check_tcp_point_plane
from robot_skill_system.perception._geometry import rotation_matrix_to_quaternion_xyzw
from robot_skill_system.scene.models import ObjectInstance, Quaternion, Vector3
from robot_skill_system.scene.transforms import AnchorType, RelativePose
from robot_skill_system.skills.models import (
    BindingSpec,
    EntityKind,
    SkillEdge,
    SkillGraph,
    SkillNode,
)

LIVE_OBJECT_BINDING = "$object"
LIVE_TOOL_BINDING = "$tool"
DEFAULT_PREGRASP_DISTANCE_M = 0.050
DEFAULT_GRASP_DEPTH_OFFSET_M = -0.005
DEFAULT_WORKSPACE_XY_TOLERANCE_M = 0.001


def _finite_attribute(attributes: Mapping[str, Any], name: str) -> float:
    try:
        value = float(attributes[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"live grasp anchor is missing finite {name!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"live grasp anchor has non-finite {name!r}")
    return value


def _target_pose(
    *,
    position_m: tuple[float, float, float], orientation_xyzw: tuple[float, float, float, float]
) -> RelativePose:
    return RelativePose(
        anchor_id=LIVE_OBJECT_BINDING,
        anchor_type=AnchorType.OBJECT,
        position_m=Vector3(x=position_m[0], y=position_m[1], z=position_m[2]),
        orientation_xyzw=Quaternion(
            x=orientation_xyzw[0],
            y=orientation_xyzw[1],
            z=orientation_xyzw[2],
            w=orientation_xyzw[3],
        ),
    )


def _plane_target_orientation(
    reference: Mapping[str, object], *, jaw_yaw_plane_rad: float
) -> tuple[float, float, float, float]:
    """Rotate the verified reference TCP pose around the plane normal only."""

    reference_tcp = np.asarray(reference["T_plane_tcp_reference"], dtype=float)
    if reference_tcp.shape != (4, 4) or not np.isfinite(reference_tcp).all():
        raise ValueError("T_plane_tcp must be a finite 4x4 matrix")
    cosine, sine = math.cos(jaw_yaw_plane_rad), math.sin(jaw_yaw_plane_rad)
    yaw = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return rotation_matrix_to_quaternion_xyzw(yaw @ reference_tcp[:3, :3])


def _symmetric_jaw_yaw_rad(jaw_yaw_plane_rad: float) -> float:
    """Return the shortest equivalent yaw for a 180-degree-symmetric jaw."""

    if not math.isfinite(jaw_yaw_plane_rad):
        raise ValueError("jaw yaw must be finite")
    return (jaw_yaw_plane_rad + math.pi / 2.0) % math.pi - math.pi / 2.0


def build_live_pick_graph(
    graph: SkillGraph,
    *,
    object_anchor: ObjectInstance,
    reference: Mapping[str, object],
    runtime: RuntimeWorkspace,
    pregrasp_distance_m: float = DEFAULT_PREGRASP_DISTANCE_M,
    grasp_depth_offset_m: float = DEFAULT_GRASP_DEPTH_OFFSET_M,
    workspace_xy_tolerance_m: float = DEFAULT_WORKSPACE_XY_TOLERANCE_M,
) -> SkillGraph:
    """Replace replayed hand motion with a current-object grip sequence.

    ``object_anchor.pose`` is created by :class:`LiveSceneBuilder` in the
    current plane basis, at the predicted contact point.  It is not a base
    target stored into the skill graph.
    """

    if not 0.010 <= pregrasp_distance_m <= 0.150:
        raise ValueError("pregrasp distance must be within [10, 150] mm")
    if not math.isfinite(grasp_depth_offset_m) or not -0.010 <= grasp_depth_offset_m <= 0.010:
        raise ValueError("grasp depth offset must be finite and within [-10, 10] mm")
    if (
        not math.isfinite(workspace_xy_tolerance_m)
        or not 0.0 <= workspace_xy_tolerance_m <= 0.003
    ):
        raise ValueError("workspace XY tolerance must be finite and within [0, 3] mm")
    if object_anchor.class_name != "hammer":
        raise ValueError("live pick currently requires a hammer grasp anchor")
    attributes = object_anchor.attributes
    if attributes.get("anchor_semantics") != (
        "predicted_grasp_xy_with_rgbd_mean_grip_depth"
    ):
        raise ValueError("live grasp anchor was not built from RGB-D mean grip depth")
    mean_depth_camera_m = _finite_attribute(
        attributes, "grip_region_mean_depth_camera_m"
    )
    depth_std_m = _finite_attribute(attributes, "grip_region_depth_std_m")
    depth_valid_fraction = _finite_attribute(
        attributes, "grip_region_depth_valid_fraction"
    )
    depth_sample_count = attributes.get("grip_region_depth_sample_count")
    if (
        isinstance(depth_sample_count, bool)
        or not isinstance(depth_sample_count, int)
        or depth_sample_count < 30
    ):
        raise ValueError("RGB-D mean grip depth requires at least 30 samples")
    if not 0.0 < mean_depth_camera_m <= 2.0:
        raise ValueError("RGB-D mean grip depth is outside the camera envelope")
    if not 0.0 <= depth_std_m <= 0.020:
        raise ValueError("RGB-D grip-region depth spread exceeds 20 mm")
    if not 0.60 <= depth_valid_fraction <= 1.0:
        raise ValueError("RGB-D grip-region valid depth fraction is below 0.60")
    jaw_yaw = _symmetric_jaw_yaw_rad(
        _finite_attribute(attributes, "jaw_yaw_plane_rad")
    )
    width_m = _finite_attribute(attributes, "target_gripper_width_m")
    if not 0.0 < width_m <= 0.110:
        raise ValueError("live grasp target width is outside the RG2 envelope")
    contact_plane = attributes.get("grasp_point_plane_m")
    contact = np.asarray(contact_plane, dtype=np.float64)
    if contact.shape != (3,) or not np.isfinite(contact).all():
        raise ValueError("live grasp anchor lacks a finite plane contact point")
    if not math.isclose(float(contact[2]), 0.0, abs_tol=1e-6, rel_tol=0.0):
        raise ValueError("live grasp anchor must be projected to contact-plane Z=0")
    observed_grasp = np.asarray(
        attributes.get("observed_grasp_point_plane_m"), dtype=np.float64
    )
    if observed_grasp.shape != (3,) or not np.isfinite(observed_grasp).all():
        raise ValueError("live grasp anchor lacks a finite RGB-D mean grip point")
    observed_tcp_height_m = float(observed_grasp[2])
    tcp_height_m = observed_tcp_height_m + grasp_depth_offset_m
    if tcp_height_m <= 0.0:
        raise ValueError("offset RGB-D grip height must be above the contact plane")
    grasp_plane = contact + np.asarray((0.0, 0.0, tcp_height_m), dtype=np.float64)
    pregrasp_plane = grasp_plane + np.asarray(
        (0.0, 0.0, pregrasp_distance_m), dtype=np.float64
    )
    for label, point in (("pregrasp", pregrasp_plane), ("grasp", grasp_plane)):
        safe, details = check_tcp_point_plane(
            reference,
            runtime,
            point,
            xy_tolerance_m=workspace_xy_tolerance_m,
        )
        if not safe:
            reasons = "; ".join(str(item) for item in details["rejection_reasons"])
            raise ValueError(f"live {label} TCP target violates safety workspace: {reasons}")
    orientation = _plane_target_orientation(reference, jaw_yaw_plane_rad=jaw_yaw)
    # Anchor origin is the contact point and uses the plane basis, so the
    # target translation is simple, object-relative +Z rather than base XYZ.
    grasp = _target_pose(
        position_m=(0.0, 0.0, tcp_height_m), orientation_xyzw=orientation
    )
    pregrasp = _target_pose(
        position_m=(0.0, 0.0, tcp_height_m + pregrasp_distance_m),
        orientation_xyzw=orientation,
    )
    # The replay graph may bind a taught surface.  A live pick binds only the
    # current grasp anchor and active tool; retaining $surface would disguise a
    # missing live-object dependency.
    bindings: dict[str, BindingSpec] = {}
    bindings[LIVE_OBJECT_BINDING] = BindingSpec(
        variable=LIVE_OBJECT_BINDING,
        entity_kind=EntityKind.OBJECT,
        class_name="hammer",
        minimum_confidence=0.60,
        minimum_visible_fraction=0.60,
    )
    bindings[LIVE_TOOL_BINDING] = BindingSpec(
        variable=LIVE_TOOL_BINDING,
        entity_kind=EntityKind.TOOL,
        instance_id="active_rg2",
        minimum_confidence=0.80,
        minimum_visible_fraction=1.0,
        must_be_attached=True,
        compatible_skill="take_hammer",
    )
    nodes = [
        SkillNode(
            node_id="live_pick_open",
            operation="gripper.open",
            arguments={"tool": LIVE_TOOL_BINDING},
        ),
        SkillNode(
            node_id="live_pick_rotate_j6",
            operation="motion.rotate_joint_6_relative",
            arguments={
                "delta_rad": jaw_yaw,
                "motion_profile_id": "joint_safe",
            },
        ),
        SkillNode(
            node_id="live_pick_validate_pregrasp",
            operation="workspace.validate_target",
            arguments={"target": pregrasp.model_dump(mode="json")},
        ),
        SkillNode(
            node_id="live_pick_move_pregrasp",
            operation="motion.move_l",
            arguments={
                "target": pregrasp.model_dump(mode="json"),
                "motion_profile_id": "linear_slow",
            },
        ),
        SkillNode(
            node_id="live_pick_validate_grasp",
            operation="workspace.validate_target",
            arguments={"target": grasp.model_dump(mode="json")},
        ),
        SkillNode(
            node_id="live_pick_move_grasp",
            operation="motion.move_l",
            arguments={
                "target": grasp.model_dump(mode="json"),
                "motion_profile_id": "linear_slow",
            },
        ),
        SkillNode(
            node_id="live_pick_set_width",
            operation="gripper.move_width",
            arguments={"tool": LIVE_TOOL_BINDING, "width_m": width_m},
        ),
    ]
    edges = [
        SkillEdge(source_node=left.node_id, target_node=right.node_id)
        for left, right in zip(nodes, nodes[1:], strict=False)
    ]
    uncertainty = dict(graph.uncertainty)
    uncertainty["live_pick"] = {
        "enabled": True,
        "source": "fresh_rgbd_grip_region_mean_depth",
        "z_target_source": "rgbd_grip_region_mean_depth_plus_local_offset",
        "object_width_mm": runtime.object_width_mm,
        "opening_offset_m": runtime.opening_offset_m,
        "allowed_down_from_reference_m": runtime.allowed_down_from_reference_m,
        "safety_margin_m": runtime.safety_margin_m,
        "workspace_xy_tolerance_m": workspace_xy_tolerance_m,
        "j6_relative_rotation_rad": jaw_yaw,
        "observed_grasp_tcp_z_plane_m": observed_tcp_height_m,
        "grasp_depth_offset_m": grasp_depth_offset_m,
        "grasp_tcp_z_plane_m": float(grasp_plane[2]),
        "pregrasp_tcp_z_plane_m": float(pregrasp_plane[2]),
        "z_descent_m": float(pregrasp_plane[2] - grasp_plane[2]),
        "grip_region_mean_depth_camera_m": mean_depth_camera_m,
        "grip_region_depth_std_m": depth_std_m,
        "grip_region_depth_sample_count": depth_sample_count,
        "grip_profile_version_id": attributes.get("grip_profile_version_id"),
        "grip_profile_checksum_sha256": attributes.get(
            "grip_profile_checksum_sha256"
        ),
    }
    return graph.model_copy(
        update={
            "description": f"{graph.description} [live object-targeted pick]",
            "required_entity_roles": {LIVE_OBJECT_BINDING: "hammer", LIVE_TOOL_BINDING: "tool"},
            "bindings": bindings,
            "nodes": nodes,
            "edges": edges,
            "start_node": nodes[0].node_id,
            "terminal_nodes": [nodes[-1].node_id],
            "motion_profiles": ["joint_safe", "linear_slow"],
            "uncertainty": uncertainty,
        },
        deep=True,
    )


__all__ = [
    "DEFAULT_GRASP_DEPTH_OFFSET_M",
    "DEFAULT_PREGRASP_DISTANCE_M",
    "DEFAULT_WORKSPACE_XY_TOLERANCE_M",
    "LIVE_OBJECT_BINDING",
    "build_live_pick_graph",
]
