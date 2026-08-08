#!/usr/bin/env python3
"""Build a non-executable base-frame candidate from an expanded URDF and fixed NPZ.

This module performs only local file parsing and deterministic matrix operations.  The
URDF-derived ``T_base_flange`` is kept separate from transforms that conditionally treat
the URDF flange as the legacy reference TCP used by the frozen ArUco artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_JOINT_DEG = (0.0, 0.0, 90.0, 0.0, 90.0, -90.0)
ALLOWED_CHAIN_JOINT_TYPES = frozenset({"fixed", "revolute", "continuous"})
RIGID_ATOL = 1e-7
COMPOSITION_ATOL = 1e-8
JOINT_MATCH_ATOL_DEG = 1e-9

FloatArray = NDArray[np.float64]


class CandidateValidationError(ValueError):
    """Raised when an input cannot support even a conditional candidate."""


@dataclass(frozen=True)
class _RawJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    element: ET.Element


@dataclass(frozen=True)
class ChainJoint:
    """Validated kinematic data for one joint in base-to-flange order."""

    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz_m: FloatArray
    origin_rpy_rad: FloatArray
    axis_xyz: FloatArray | None
    lower_rad: float | None
    upper_rad: float | None


@dataclass(frozen=True)
class ForwardKinematicsResult:
    """URDF-only FK result and the exact actuated-joint order used for it."""

    T_base_flange: FloatArray
    chain: tuple[ChainJoint, ...]
    actuated_joint_names: tuple[str, ...]
    joint_deg: FloatArray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(
    text: str | None,
    *,
    label: str,
    default: tuple[float, float, float] | None = None,
) -> FloatArray:
    if text is None:
        if default is None:
            raise CandidateValidationError(f"{label} is required")
        values = np.asarray(default, dtype=np.float64)
    else:
        tokens = text.split()
        if len(tokens) != 3:
            raise CandidateValidationError(f"{label} must contain exactly three numbers")
        try:
            values = np.asarray([float(token) for token in tokens], dtype=np.float64)
        except ValueError as exc:
            raise CandidateValidationError(f"{label} must contain only numbers") from exc
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise CandidateValidationError(f"{label} must contain finite values")
    return values


def _finite_float(text: str | None, *, label: str) -> float:
    if text is None:
        raise CandidateValidationError(f"{label} is required")
    try:
        value = float(text)
    except ValueError as exc:
        raise CandidateValidationError(f"{label} must be a number") from exc
    if not math.isfinite(value):
        raise CandidateValidationError(f"{label} must be finite")
    return value


def _validate_rigid_matrix(value: object, *, label: str) -> FloatArray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError(f"{label} must be a numeric 4x4 matrix") from exc
    if matrix.shape != (4, 4):
        raise CandidateValidationError(f"{label} must have shape (4, 4)")
    if not np.all(np.isfinite(matrix)):
        raise CandidateValidationError(f"{label} must contain only finite values")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=RIGID_ATOL, rtol=0.0):
        raise CandidateValidationError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=RIGID_ATOL, rtol=0.0):
        raise CandidateValidationError(f"{label} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=RIGID_ATOL, rel_tol=0.0):
        raise CandidateValidationError(f"{label} rotation determinant must be +1")
    return matrix.copy()


def _rotation_rpy(rpy_rad: FloatArray) -> FloatArray:
    roll, pitch, yaw = (float(value) for value in rpy_rad)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)), dtype=np.float64
    )
    rotation_y = np.asarray(
        ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)), dtype=np.float64
    )
    rotation_z = np.asarray(
        ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64
    )
    return rotation_z @ rotation_y @ rotation_x


def _origin_transform(xyz_m: FloatArray, rpy_rad: FloatArray) -> FloatArray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_rpy(rpy_rad)
    transform[:3, 3] = xyz_m
    return transform


def _axis_rotation(axis_xyz: FloatArray, angle_rad: float) -> FloatArray:
    x, y, z = (float(value) for value in axis_xyz)
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)
    sine = math.sin(angle_rad)
    cosine = math.cos(angle_rad)
    rotation = cosine * np.eye(3) + (1.0 - cosine) * np.outer(axis_xyz, axis_xyz)
    rotation += sine * skew
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def _one_child(element: ET.Element, tag: str, *, joint_name: str) -> ET.Element:
    children = element.findall(tag)
    if len(children) != 1:
        raise CandidateValidationError(
            f"URDF joint {joint_name!r} must contain exactly one <{tag}> element"
        )
    return children[0]


def _raw_joints(root: ET.Element, links: set[str]) -> tuple[_RawJoint, ...]:
    result: list[_RawJoint] = []
    names: set[str] = set()
    child_owners: dict[str, str] = {}
    for element in root.findall("joint"):
        name = element.get("name", "").strip()
        joint_type = element.get("type", "").strip()
        if not name or not joint_type:
            raise CandidateValidationError("every URDF joint needs non-empty name and type")
        if name in names:
            raise CandidateValidationError(f"duplicate URDF joint name {name!r}")
        names.add(name)
        parent_element = _one_child(element, "parent", joint_name=name)
        child_element = _one_child(element, "child", joint_name=name)
        parent = parent_element.get("link", "").strip()
        child = child_element.get("link", "").strip()
        if parent not in links or child not in links:
            raise CandidateValidationError(
                f"URDF joint {name!r} references an undeclared parent or child link"
            )
        if parent == child:
            raise CandidateValidationError(f"URDF joint {name!r} cannot join a link to itself")
        if child in child_owners:
            raise CandidateValidationError(
                "URDF does not define a unique chain: "
                f"link {child!r} has parents via {child_owners[child]!r} and {name!r}"
            )
        child_owners[child] = name
        result.append(_RawJoint(name, joint_type, parent, child, element))
    return tuple(result)


def _reject_cycles(joints: Sequence[_RawJoint], links: set[str]) -> None:
    adjacency: dict[str, list[str]] = {link: [] for link in links}
    for joint in joints:
        adjacency[joint.parent].append(joint.child)
    state: dict[str, int] = {}

    def visit(link: str) -> None:
        current = state.get(link, 0)
        if current == 1:
            raise CandidateValidationError("URDF joint graph contains a cycle")
        if current == 2:
            return
        state[link] = 1
        for child in adjacency[link]:
            visit(child)
        state[link] = 2

    for link in links:
        visit(link)


def _ordered_chain(
    joints: Sequence[_RawJoint], *, base_link: str, flange_link: str
) -> tuple[_RawJoint, ...]:
    if base_link == flange_link:
        return ()
    by_child = {joint.child: joint for joint in joints}
    reverse_chain: list[_RawJoint] = []
    visited = {flange_link}
    current = flange_link
    while current != base_link:
        joint = by_child.get(current)
        if joint is None:
            raise CandidateValidationError(
                f"no directed URDF chain exists from {base_link!r} to {flange_link!r}"
            )
        reverse_chain.append(joint)
        current = joint.parent
        if current in visited:
            raise CandidateValidationError("URDF chain contains a cycle")
        visited.add(current)
    reverse_chain.reverse()
    return tuple(reverse_chain)


def _directed_chain_exists(
    joints: Sequence[_RawJoint], *, base_link: str, target_link: str
) -> bool:
    if base_link == target_link:
        return True
    by_child = {joint.child: joint for joint in joints}
    visited = {target_link}
    current = target_link
    while current != base_link:
        joint = by_child.get(current)
        if joint is None:
            return False
        current = joint.parent
        if current in visited:
            raise CandidateValidationError("URDF chain contains a cycle")
        visited.add(current)
    return True


def _validated_chain_joint(raw: _RawJoint) -> ChainJoint:
    if raw.joint_type not in ALLOWED_CHAIN_JOINT_TYPES:
        allowed = ", ".join(sorted(ALLOWED_CHAIN_JOINT_TYPES))
        raise CandidateValidationError(
            f"chain joint {raw.name!r} has unsupported type {raw.joint_type!r}; allowed: {allowed}"
        )
    origins = raw.element.findall("origin")
    if len(origins) > 1:
        raise CandidateValidationError(f"joint {raw.name!r} has multiple <origin> elements")
    origin = origins[0] if origins else None
    xyz_m = _finite_vector(
        origin.get("xyz") if origin is not None else None,
        label=f"joint {raw.name!r} origin xyz",
        default=(0.0, 0.0, 0.0),
    )
    rpy_rad = _finite_vector(
        origin.get("rpy") if origin is not None else None,
        label=f"joint {raw.name!r} origin rpy",
        default=(0.0, 0.0, 0.0),
    )

    axis_xyz: FloatArray | None = None
    lower_rad: float | None = None
    upper_rad: float | None = None
    if raw.element.findall("mimic"):
        raise CandidateValidationError(
            f"chain joint {raw.name!r} uses an unsupported mimic relationship"
        )
    if raw.joint_type in {"revolute", "continuous"}:
        axes = raw.element.findall("axis")
        if len(axes) != 1:
            raise CandidateValidationError(
                f"movable joint {raw.name!r} must contain exactly one explicit <axis>"
            )
        axis_xyz = _finite_vector(
            axes[0].get("xyz"), label=f"joint {raw.name!r} axis xyz"
        )
        norm = float(np.linalg.norm(axis_xyz))
        if not math.isclose(norm, 1.0, abs_tol=1e-7, rel_tol=0.0):
            raise CandidateValidationError(f"joint {raw.name!r} axis must be a unit vector")
        axis_xyz = axis_xyz / norm

        limits = raw.element.findall("limit")
        if len(limits) > 1:
            raise CandidateValidationError(f"joint {raw.name!r} has multiple <limit> elements")
        limit = limits[0] if limits else None
        if limit is not None:
            for attribute in ("lower", "upper", "effort", "velocity"):
                if attribute in limit.attrib:
                    _finite_float(
                        limit.get(attribute), label=f"joint {raw.name!r} limit {attribute}"
                    )
        if raw.joint_type == "revolute":
            if limit is None:
                raise CandidateValidationError(
                    f"revolute joint {raw.name!r} requires finite lower and upper limits"
                )
            lower_rad = _finite_float(
                limit.get("lower"), label=f"joint {raw.name!r} lower limit"
            )
            upper_rad = _finite_float(
                limit.get("upper"), label=f"joint {raw.name!r} upper limit"
            )
            if lower_rad > upper_rad:
                raise CandidateValidationError(
                    f"joint {raw.name!r} lower limit exceeds its upper limit"
                )
        elif limit is not None and ({"lower", "upper"} & set(limit.attrib)):
            raise CandidateValidationError(
                f"continuous joint {raw.name!r} must not define position limits"
            )
    elif raw.element.findall("axis") or raw.element.findall("limit"):
        raise CandidateValidationError(
            f"fixed joint {raw.name!r} must not define an axis or motion limits"
        )

    return ChainJoint(
        name=raw.name,
        joint_type=raw.joint_type,
        parent=raw.parent,
        child=raw.child,
        origin_xyz_m=xyz_m,
        origin_rpy_rad=rpy_rad,
        axis_xyz=axis_xyz,
        lower_rad=lower_rad,
        upper_rad=upper_rad,
    )


def _load_expanded_urdf(
    urdf_path: Path | str,
) -> tuple[Path, ET.Element, set[str]]:
    path = Path(urdf_path).expanduser().resolve(strict=True)
    source = path.read_bytes()
    if b"${" in source or b"$(" in source or b"xacro:" in source:
        raise CandidateValidationError("URDF must be fully expanded; xacro content remains")
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise CandidateValidationError(f"invalid URDF XML: {exc}") from exc
    if root.tag != "robot":
        raise CandidateValidationError("expanded URDF root element must be <robot>")

    link_elements = root.findall("link")
    links = [element.get("name", "").strip() for element in link_elements]
    if any(not link for link in links):
        raise CandidateValidationError("every URDF link needs a non-empty name")
    if len(links) != len(set(links)):
        raise CandidateValidationError("URDF link names must be unique")
    link_set = set(links)
    return path, root, link_set


def forward_kinematics_from_urdf(
    urdf_path: Path | str,
    *,
    base_link: str,
    flange_link: str,
    joint_deg: Sequence[float],
) -> ForwardKinematicsResult:
    """Parse an expanded URDF and calculate base-to-target-link FK locally."""

    _path, root, link_set = _load_expanded_urdf(urdf_path)
    if base_link not in link_set or flange_link not in link_set:
        raise CandidateValidationError("requested base_link and flange_link must exist in the URDF")

    raw_joints = _raw_joints(root, link_set)
    _reject_cycles(raw_joints, link_set)
    chain = tuple(
        _validated_chain_joint(joint)
        for joint in _ordered_chain(
            raw_joints, base_link=base_link, flange_link=flange_link
        )
    )
    actuated = tuple(joint for joint in chain if joint.joint_type != "fixed")
    try:
        requested_joint_deg = np.asarray(joint_deg, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError("joint_deg must be a finite numeric sequence") from exc
    if requested_joint_deg.shape != (len(actuated),):
        raise CandidateValidationError(
            "joint_deg count must equal the number of revolute/continuous joints in chain order: "
            f"expected {len(actuated)}, received {requested_joint_deg.size}"
        )
    if not np.all(np.isfinite(requested_joint_deg)):
        raise CandidateValidationError("joint_deg must contain only finite values")

    transform = np.eye(4, dtype=np.float64)
    actuated_index = 0
    for joint in chain:
        transform = transform @ _origin_transform(joint.origin_xyz_m, joint.origin_rpy_rad)
        if joint.joint_type == "fixed":
            continue
        angle_rad = math.radians(float(requested_joint_deg[actuated_index]))
        actuated_index += 1
        if joint.joint_type == "revolute":
            assert joint.lower_rad is not None and joint.upper_rad is not None
            if angle_rad < joint.lower_rad - 1e-12 or angle_rad > joint.upper_rad + 1e-12:
                raise CandidateValidationError(
                    f"joint {joint.name!r} value is outside its URDF position limits"
                )
        assert joint.axis_xyz is not None
        transform = transform @ _axis_rotation(joint.axis_xyz, angle_rad)

    return ForwardKinematicsResult(
        T_base_flange=_validate_rigid_matrix(transform, label="derived T_base_flange"),
        chain=chain,
        actuated_joint_names=tuple(joint.name for joint in actuated),
        joint_deg=requested_joint_deg.copy(),
    )


def _npz_array(data: np.lib.npyio.NpzFile, key: str, *, shape: tuple[int, ...]) -> FloatArray:
    if key not in data.files:
        raise CandidateValidationError(f"reference NPZ is missing {key!r}")
    try:
        value = np.asarray(data[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError(f"reference NPZ {key!r} must be numeric") from exc
    if value.shape != shape:
        raise CandidateValidationError(
            f"reference NPZ {key!r} must have shape {shape}, received {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise CandidateValidationError(f"reference NPZ {key!r} must be finite")
    return value.copy()


def _npz_scalar_text(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise CandidateValidationError(f"reference NPZ {key!r} must be a text scalar")
    item = value.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _workspace_polygon(data: np.lib.npyio.NpzFile) -> FloatArray:
    key = "workspace_safe_boundary_plane_xy_m"
    if key not in data.files:
        raise CandidateValidationError(f"reference NPZ is missing canonical polygon {key!r}")
    try:
        polygon = np.asarray(data[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError(f"reference NPZ {key!r} must be numeric") from exc
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise CandidateValidationError(f"reference NPZ {key!r} must have shape (N>=3, 2)")
    if not np.all(np.isfinite(polygon)):
        raise CandidateValidationError(f"reference NPZ {key!r} must be finite")
    x = polygon[:, 0]
    y = polygon[:, 1]
    twice_area = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(twice_area) <= 1e-12:
        raise CandidateValidationError(f"reference NPZ {key!r} must have non-zero area")
    return polygon.copy()


def _chain_json(fk: ForwardKinematicsResult) -> list[dict[str, object]]:
    joint_values = dict(zip(fk.actuated_joint_names, fk.joint_deg.tolist(), strict=True))
    return [
        {
            "name": joint.name,
            "type": joint.joint_type,
            "parent": joint.parent,
            "child": joint.child,
            "origin_xyz_m": joint.origin_xyz_m.tolist(),
            "origin_rpy_rad": joint.origin_rpy_rad.tolist(),
            "axis_xyz": joint.axis_xyz.tolist() if joint.axis_xyz is not None else None,
            "joint_deg": joint_values.get(joint.name),
            "lower_rad": joint.lower_rad,
            "upper_rad": joint.upper_rad,
        }
        for joint in fk.chain
    ]


def build_reference_candidate(
    *,
    urdf_path: Path | str,
    reference_npz_path: Path | str,
    base_link: str = "base_link",
    flange_link: str = "link_6",
    camera_link: str = "camera_color_optical_frame",
    joint_deg: Sequence[float] = DEFAULT_JOINT_DEG,
) -> dict[str, object]:
    """Return a JSON-safe, explicitly non-motion candidate without writing files."""

    urdf = Path(urdf_path).expanduser().resolve(strict=True)
    reference_npz = Path(reference_npz_path).expanduser().resolve(strict=True)
    urdf_sha256 = _sha256(urdf)
    reference_npz_sha256 = _sha256(reference_npz)
    fk = forward_kinematics_from_urdf(
        urdf,
        base_link=base_link,
        flange_link=flange_link,
        joint_deg=joint_deg,
    )
    _urdf_path, urdf_root, urdf_links = _load_expanded_urdf(urdf)
    model_joints = _raw_joints(urdf_root, urdf_links)
    _reject_cycles(model_joints, urdf_links)
    camera_link_declared = camera_link in urdf_links
    camera_chain_available = camera_link_declared and _directed_chain_exists(
        model_joints,
        base_link=base_link,
        target_link=camera_link,
    )
    camera_fk: ForwardKinematicsResult | None = None
    if camera_chain_available:
        camera_fk = forward_kinematics_from_urdf(
            urdf,
            base_link=base_link,
            flange_link=camera_link,
            joint_deg=joint_deg,
        )
        if camera_fk.actuated_joint_names != fk.actuated_joint_names:
            raise CandidateValidationError(
                "base-to-camera and base-to-flange chains must use the same actuated joints "
                "in the same order"
            )

    try:
        with np.load(reference_npz, allow_pickle=False) as data:
            reference_joint_deg = _npz_array(
                data, "reference_joint_deg", shape=fk.joint_deg.shape
            )
            if not np.allclose(
                reference_joint_deg,
                fk.joint_deg,
                atol=JOINT_MATCH_ATOL_DEG,
                rtol=0.0,
            ):
                raise CandidateValidationError(
                    "reference NPZ reference_joint_deg does not match the requested chain-ordered "
                    "joint_deg"
                )
            T_camera_plane = _validate_rigid_matrix(
                _npz_array(data, "T_camera_plane", shape=(4, 4)),
                label="reference T_camera_plane",
            )
            T_tcp_camera = _validate_rigid_matrix(
                _npz_array(data, "T_tcp_camera", shape=(4, 4)),
                label="reference T_tcp_camera",
            )
            T_tcp_plane = _validate_rigid_matrix(
                _npz_array(data, "T_tcp_plane", shape=(4, 4)),
                label="reference T_tcp_plane",
            )
            composed_tcp_plane = _validate_rigid_matrix(
                T_tcp_camera @ T_camera_plane,
                label="composed reference T_tcp_plane",
            )
            if not np.allclose(
                composed_tcp_plane,
                T_tcp_plane,
                atol=COMPOSITION_ATOL,
                rtol=0.0,
            ):
                raise CandidateValidationError(
                    "reference T_tcp_plane is inconsistent with T_tcp_camera @ T_camera_plane"
                )
            polygon_plane_xy_m = _workspace_polygon(data)
            npz_metadata = {
                key: value
                for key in (
                    "schema_version",
                    "artifact_kind",
                    "status",
                    "frame_name",
                    "source_tcp_camera_sha256",
                )
                if (value := _npz_scalar_text(data, key)) is not None
            }
    except CandidateValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise CandidateValidationError(f"could not read reference NPZ safely: {exc}") from exc

    transforms: dict[str, object] = {
        "T_base_flange": fk.T_base_flange.tolist(),
        "reference_T_tcp_camera": T_tcp_camera.tolist(),
        "reference_T_camera_plane": T_camera_plane.tolist(),
        "reference_T_tcp_plane": T_tcp_plane.tolist(),
    }
    frames: dict[str, object] = {
        "base_link": base_link,
        "flange_link": flange_link,
        "requested_camera_link": camera_link,
        "legacy_npz_reference_tcp": "unspecified legacy TCP frame",
    }
    joint_state: dict[str, object] = {
        "actuated_joint_names_chain_order": list(fk.actuated_joint_names),
        "joint_deg_chain_order": fk.joint_deg.tolist(),
        "reference_joint_deg": reference_joint_deg.tolist(),
        "reference_joint_deg_matches_input": True,
        "flange_chain": _chain_json(fk),
    }

    if camera_fk is not None:
        candidate_mode = "urdf_camera_link"
        verification_status = "nominal_urdf_geometry_candidate_not_hardware_verified"
        prohibition_reason = (
            "The URDF camera/bracket geometry and nominal camera extrinsics are not a "
            "hardware-validated hand-eye calibration; this artifact must not command motion."
        )
        T_base_camera = camera_fk.T_base_flange
        T_base_plane = _validate_rigid_matrix(
            T_base_camera @ T_camera_plane,
            label="URDF-camera candidate T_base_plane",
        )
        transforms.update(
            {
                "T_base_camera_from_urdf_geometry": T_base_camera.tolist(),
                "T_base_plane_from_urdf_camera_and_reference_T_camera_plane": (
                    T_base_plane.tolist()
                ),
            }
        )
        frames.update(
            {
                "camera_link_present_in_urdf": True,
                "base_to_camera_chain_present_in_urdf": True,
                "camera_link_used": camera_link,
                "frame_mismatch_status": (
                    "legacy_npz_reference_tcp_not_used_for_base_camera_or_base_plane"
                ),
                "flange_to_legacy_tcp_identity_assumption_used": False,
                "urdf_camera_geometry_hardware_verified": False,
            }
        )
        joint_state["camera_chain"] = _chain_json(camera_fk)
        polygon_key = "workspace_polygon_base_xyz_m_from_urdf_camera_geometry_candidate"
        normal_key = "normal_base_xyz_from_urdf_camera_geometry_candidate"
        tilt_key = "surface_tilt_from_base_xy_plane_deg_urdf_camera_geometry_candidate"
        directed_angle_key = (
            "directed_normal_angle_from_base_positive_z_deg_urdf_camera_geometry_candidate"
        )
    else:
        candidate_mode = "conditional_flange_equals_legacy_npz_reference_tcp"
        verification_status = "conditional_frame_identity_candidate_not_hardware_verified"
        prohibition_reason = (
            "The URDF flange-to-legacy-reference-TCP frame identity is unverified; this "
            "artifact must not command robot motion."
        )
        # Fallback only: the NPZ names a legacy TCP, not the URDF flange.  No supplied
        # tool transform proves that those frames are identical.
        T_base_camera = _validate_rigid_matrix(
            fk.T_base_flange @ T_tcp_camera,
            label="conditional T_base_camera",
        )
        T_base_plane = _validate_rigid_matrix(
            T_base_camera @ T_camera_plane,
            label="conditional T_base_plane",
        )
        if not np.allclose(
            T_base_plane,
            fk.T_base_flange @ T_tcp_plane,
            atol=COMPOSITION_ATOL,
            rtol=0.0,
        ):
            raise CandidateValidationError(
                "conditional base transform composition is inconsistent"
            )
        conditional_suffix = "assuming_urdf_flange_equals_legacy_npz_reference_tcp"
        transforms.update(
            {
                f"conditional_T_base_camera_{conditional_suffix}": T_base_camera.tolist(),
                f"conditional_T_base_plane_{conditional_suffix}": T_base_plane.tolist(),
            }
        )
        frames.update(
            {
                "camera_link_present_in_urdf": camera_link_declared,
                "base_to_camera_chain_present_in_urdf": False,
                "camera_link_used": None,
                "frame_mismatch_status": "unresolved_flange_vs_legacy_reference_tcp",
                "frame_identity_verified": False,
                "flange_to_legacy_tcp_identity_assumption_used": True,
                "conditional_identity_assumption": (
                    "T_base_legacy_reference_tcp := T_base_flange; no flange-to-tool/TCP "
                    "transform was supplied or verified"
                ),
            }
        )
        polygon_key = f"conditional_workspace_polygon_base_xyz_m_{conditional_suffix}"
        normal_key = f"conditional_normal_base_xyz_{conditional_suffix}"
        tilt_key = (
            f"conditional_surface_tilt_from_base_xy_plane_deg_{conditional_suffix}"
        )
        directed_angle_key = (
            "conditional_directed_normal_angle_from_base_positive_z_deg_"
            f"{conditional_suffix}"
        )

    polygon_h = np.column_stack(
        (
            polygon_plane_xy_m,
            np.zeros(polygon_plane_xy_m.shape[0], dtype=np.float64),
            np.ones(polygon_plane_xy_m.shape[0], dtype=np.float64),
        )
    )
    polygon_base_xyz_m = (T_base_plane @ polygon_h.T).T[:, :3]
    if not np.all(np.isfinite(polygon_base_xyz_m)):
        raise CandidateValidationError("conditional transformed workspace polygon is not finite")

    normal_base = T_base_plane[:3, 2]
    directed_cosine = float(np.clip(normal_base[2], -1.0, 1.0))
    directed_angle_deg = math.degrees(math.acos(directed_cosine))
    surface_tilt_deg = math.degrees(math.acos(abs(directed_cosine)))
    if _sha256(urdf) != urdf_sha256 or _sha256(reference_npz) != reference_npz_sha256:
        raise CandidateValidationError("an input file changed while the candidate was built")

    return {
        "schema_version": "1.0",
        "artifact_kind": "urdf_aruco_reference_candidate",
        "status": "candidate",
        "verification_status": verification_status,
        "candidate_mode": candidate_mode,
        "usable_for_motion": False,
        "hardware_approved": False,
        "created_at_ns": time.time_ns(),
        "prohibition_reason": prohibition_reason,
        "source_hashes_sha256": {
            "expanded_urdf": urdf_sha256,
            "fixed_workspace_reference_npz": reference_npz_sha256,
        },
        "sources": {
            "expanded_urdf_path": str(urdf),
            "fixed_workspace_reference_npz_path": str(reference_npz),
            "fixed_workspace_reference_metadata": npz_metadata,
        },
        "frames": frames,
        "joint_state": joint_state,
        "transforms": transforms,
        "plane": {
            "workspace_polygon_plane_xy_m": polygon_plane_xy_m.tolist(),
            polygon_key: polygon_base_xyz_m.tolist(),
            normal_key: normal_base.tolist(),
            tilt_key: surface_tilt_deg,
            directed_angle_key: directed_angle_deg,
        },
    }


def write_candidate_create_only(payload: dict[str, object], output_json: Path | str) -> Path:
    """Serialize a candidate with exclusive creation; an existing target is never replaced."""

    output = Path(output_json).expanduser()
    if output.suffix.lower() != ".json":
        raise CandidateValidationError("output_json must end in .json")
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return output.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hardware-unapproved URDF/ArUco frame candidate without contacting a robot"
        )
    )
    parser.add_argument("--urdf", type=Path, required=True, help="Expanded URDF file")
    parser.add_argument(
        "--reference-npz", type=Path, required=True, help="Frozen ArUco workspace NPZ"
    )
    parser.add_argument("--base-link", default="base_link")
    parser.add_argument("--flange-link", default="link_6")
    parser.add_argument(
        "--camera-link",
        default="camera_color_optical_frame",
        help=(
            "Preferred URDF camera link; if absent, emit only the conditional flange/TCP "
            "fallback"
        ),
    )
    parser.add_argument(
        "--joint-deg",
        type=float,
        nargs="+",
        default=list(DEFAULT_JOINT_DEG),
        metavar="DEG",
        help="Actuated joints in base-to-flange chain order",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        payload = build_reference_candidate(
            urdf_path=args.urdf,
            reference_npz_path=args.reference_npz,
            base_link=args.base_link,
            flange_link=args.flange_link,
            camera_link=args.camera_link,
            joint_deg=args.joint_deg,
        )
        output = write_candidate_create_only(payload, args.output_json)
    except (CandidateValidationError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": "candidate",
                "usable_for_motion": False,
                "output_json": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
