from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from aruco import urdf_reference_candidate as candidate


def _transform(*, xyz: tuple[float, float, float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = xyz
    return result


def _write_reference(
    path: Path,
    *,
    joint_deg: list[float],
    T_camera_plane: np.ndarray | None = None,
    T_tcp_camera: np.ndarray | None = None,
    T_tcp_plane: np.ndarray | None = None,
) -> Path:
    camera_plane = (
        _transform(xyz=(0.0, 0.0, 0.5))
        if T_camera_plane is None
        else T_camera_plane
    )
    tcp_camera = (
        _transform(xyz=(0.1, -0.2, 0.3)) if T_tcp_camera is None else T_tcp_camera
    )
    tcp_plane = tcp_camera @ camera_plane if T_tcp_plane is None else T_tcp_plane
    np.savez(
        path,
        schema_version=np.asarray("2.0"),
        artifact_kind=np.asarray("fixed_aruco_workspace_reference"),
        status=np.asarray("fixed_geometry_reference_not_hardware_safety_approval"),
        frame_name=np.asarray("test_plane"),
        source_tcp_camera_sha256=np.asarray("a" * 64),
        reference_joint_deg=np.asarray(joint_deg, dtype=np.float64),
        T_camera_plane=np.asarray(camera_plane, dtype=np.float64),
        T_tcp_camera=np.asarray(tcp_camera, dtype=np.float64),
        T_tcp_plane=np.asarray(tcp_plane, dtype=np.float64),
        workspace_safe_boundary_plane_xy_m=np.asarray(
            ((0.0, 0.0), (0.2, 0.0), (0.2, 0.1), (0.0, 0.1)),
            dtype=np.float64,
        ),
    )
    return path


def _write_one_joint_urdf(
    path: Path,
    *,
    joint_type: str = "revolute",
    origin_xyz: str = "1 2 3",
    axis_xyz: str = "1 0 0",
    limit_xml: str = '<limit lower="-3.141592653589793" upper="3.141592653589793"/>',
    include_camera: bool = False,
) -> Path:
    camera_link = '<link name="camera_color_optical_frame"/>' if include_camera else ""
    camera_joint = (
        """
  <joint name="camera_mount" type="fixed">
    <parent link="link_6"/>
    <child link="camera_color_optical_frame"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>"""
        if include_camera
        else ""
    )
    path.write_text(
        f"""<robot name="test">
  <link name="base_link"/>
  <link name="moving_link"/>
  <link name="link_6"/>
  {camera_link}
  <joint name="joint_1" type="{joint_type}">
    <parent link="base_link"/>
    <child link="moving_link"/>
    <origin xyz="{origin_xyz}" rpy="0 0 {math.pi / 2}"/>
    <axis xyz="{axis_xyz}"/>
    {limit_xml}
  </joint>
  <joint name="flange_fixed" type="fixed">
    <parent link="moving_link"/>
    <child link="link_6"/>
    <origin xyz="0 1 0" rpy="0 0 0"/>
  </joint>
  {camera_joint}
</robot>
""",
        encoding="utf-8",
    )
    return path


def _write_six_joint_urdf(path: Path) -> Path:
    links = ["base_link", *(f"link_{index}" for index in range(1, 7))]
    link_xml = "\n".join(f'  <link name="{link}"/>' for link in links)
    joints = []
    for index in range(1, 7):
        parent = "base_link" if index == 1 else f"link_{index - 1}"
        joints.append(
            f"""  <joint name="joint_{index}" type="revolute">
    <parent link="{parent}"/>
    <child link="link_{index}"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.2" upper="3.2"/>
  </joint>"""
        )
    joint_xml = "\n".join(joints)
    path.write_text(
        f'<robot name="six">\n{link_xml}\n{joint_xml}\n</robot>\n',
        encoding="utf-8",
    )
    return path


def test_flange_fallback_is_explicitly_conditional_and_create_only(tmp_path: Path) -> None:
    urdf = _write_one_joint_urdf(tmp_path / "robot.urdf")
    reference = _write_reference(tmp_path / "reference.npz", joint_deg=[90.0])

    payload = candidate.build_reference_candidate(
        urdf_path=urdf,
        reference_npz_path=reference,
        joint_deg=[90.0],
    )

    expected_base_flange = np.asarray(
        (
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 1.0, 0.0, 4.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    np.testing.assert_allclose(
        payload["transforms"]["T_base_flange"], expected_base_flange, atol=1e-12
    )
    assert payload["status"] == "candidate"
    assert payload["usable_for_motion"] is False
    assert payload["hardware_approved"] is False
    assert payload["candidate_mode"] == "conditional_flange_equals_legacy_npz_reference_tcp"
    assert payload["frames"]["frame_identity_verified"] is False
    conditional_camera_key = (
        "conditional_T_base_camera_"
        "assuming_urdf_flange_equals_legacy_npz_reference_tcp"
    )
    conditional_plane_key = (
        "conditional_T_base_plane_"
        "assuming_urdf_flange_equals_legacy_npz_reference_tcp"
    )
    transforms = payload["transforms"]
    assert conditional_camera_key in transforms
    assert conditional_plane_key in transforms
    assert "T_base_camera_from_urdf_geometry" not in transforms

    output = tmp_path / "candidate.json"
    candidate.write_candidate_create_only(payload, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        candidate.write_candidate_create_only(payload, output)
    assert output.read_bytes() == original


def test_camera_link_mode_uses_urdf_camera_and_raw_camera_plane(tmp_path: Path) -> None:
    urdf = _write_one_joint_urdf(tmp_path / "robot_camera.urdf", include_camera=True)
    T_camera_plane = _transform(xyz=(0.0, 0.0, 0.5))
    deliberately_different_tcp_camera = _transform(xyz=(4.0, 5.0, 6.0))
    reference = _write_reference(
        tmp_path / "reference.npz",
        joint_deg=[90.0],
        T_camera_plane=T_camera_plane,
        T_tcp_camera=deliberately_different_tcp_camera,
    )

    payload = candidate.build_reference_candidate(
        urdf_path=urdf,
        reference_npz_path=reference,
        joint_deg=[90.0],
    )

    transforms = payload["transforms"]
    T_base_flange = np.asarray(transforms["T_base_flange"])
    expected_base_camera = T_base_flange @ _transform(xyz=(0.0, 0.0, 0.2))
    expected_base_plane = expected_base_camera @ T_camera_plane
    np.testing.assert_allclose(
        transforms["T_base_camera_from_urdf_geometry"], expected_base_camera, atol=1e-12
    )
    np.testing.assert_allclose(
        transforms["T_base_plane_from_urdf_camera_and_reference_T_camera_plane"],
        expected_base_plane,
        atol=1e-12,
    )
    assert not np.allclose(
        expected_base_camera,
        T_base_flange @ deliberately_different_tcp_camera,
    )
    assert payload["candidate_mode"] == "urdf_camera_link"
    assert (
        payload["verification_status"]
        == "nominal_urdf_geometry_candidate_not_hardware_verified"
    )
    assert payload["frames"]["flange_to_legacy_tcp_identity_assumption_used"] is False
    assert payload["usable_for_motion"] is False
    polygon_plane = np.column_stack(
        (
            np.asarray(payload["plane"]["workspace_polygon_plane_xy_m"]),
            np.zeros(4),
            np.ones(4),
        )
    )
    expected_polygon_base = (expected_base_plane @ polygon_plane.T).T[:, :3]
    np.testing.assert_allclose(
        payload["plane"][
            "workspace_polygon_base_xyz_m_from_urdf_camera_geometry_candidate"
        ],
        expected_polygon_base,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("joint_type", "origin_xyz", "axis_xyz", "limit_xml", "joint_deg", "error"),
    [
        ("prismatic", "1 2 3", "1 0 0", "", [0.0], "unsupported type"),
        (
            "revolute",
            "nan 2 3",
            "1 0 0",
            '<limit lower="-3.2" upper="3.2"/>',
            [0.0],
            "finite",
        ),
        (
            "revolute",
            "1 2 3",
            "0 0 0",
            '<limit lower="-3.2" upper="3.2"/>',
            [0.0],
            "unit vector",
        ),
        ("revolute", "1 2 3", "1 0 0", "", [0.0], "requires finite"),
        (
            "revolute",
            "1 2 3",
            "1 0 0",
            '<limit lower="-0.5" upper="0.5"/>',
            [90.0],
            "outside",
        ),
        (
            "continuous",
            "1 2 3",
            "1 0 0",
            '<limit lower="-3.2" upper="3.2"/>',
            [0.0],
            "must not define position limits",
        ),
    ],
)
def test_fk_rejects_unsafe_chain_values(
    tmp_path: Path,
    joint_type: str,
    origin_xyz: str,
    axis_xyz: str,
    limit_xml: str,
    joint_deg: list[float],
    error: str,
) -> None:
    urdf = _write_one_joint_urdf(
        tmp_path / "invalid.urdf",
        joint_type=joint_type,
        origin_xyz=origin_xyz,
        axis_xyz=axis_xyz,
        limit_xml=limit_xml,
    )
    with pytest.raises(candidate.CandidateValidationError, match=error):
        candidate.forward_kinematics_from_urdf(
            urdf,
            base_link="base_link",
            flange_link="link_6",
            joint_deg=joint_deg,
        )


def test_fk_rejects_non_unique_chain(tmp_path: Path) -> None:
    urdf = tmp_path / "ambiguous.urdf"
    urdf.write_text(
        """<robot name="ambiguous">
  <link name="base_link"/>
  <link name="other"/>
  <link name="link_6"/>
  <joint name="first" type="fixed">
    <parent link="base_link"/><child link="link_6"/>
  </joint>
  <joint name="second" type="fixed">
    <parent link="other"/><child link="link_6"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    with pytest.raises(candidate.CandidateValidationError, match="unique chain"):
        candidate.forward_kinematics_from_urdf(
            urdf,
            base_link="base_link",
            flange_link="link_6",
            joint_deg=[],
        )


def test_reference_rejects_joint_mismatch_nonrigid_and_bad_composition(
    tmp_path: Path,
) -> None:
    urdf = _write_one_joint_urdf(tmp_path / "robot.urdf")

    mismatch = _write_reference(tmp_path / "mismatch.npz", joint_deg=[89.0])
    with pytest.raises(candidate.CandidateValidationError, match="does not match"):
        candidate.build_reference_candidate(
            urdf_path=urdf,
            reference_npz_path=mismatch,
            joint_deg=[90.0],
        )

    nonrigid_matrix = np.eye(4)
    nonrigid_matrix[0, 0] = 2.0
    nonrigid = _write_reference(
        tmp_path / "nonrigid.npz",
        joint_deg=[90.0],
        T_camera_plane=nonrigid_matrix,
    )
    with pytest.raises(candidate.CandidateValidationError, match="not orthonormal"):
        candidate.build_reference_candidate(
            urdf_path=urdf,
            reference_npz_path=nonrigid,
            joint_deg=[90.0],
        )

    inconsistent = _write_reference(
        tmp_path / "inconsistent.npz",
        joint_deg=[90.0],
        T_tcp_plane=_transform(xyz=(9.0, 9.0, 9.0)),
    )
    with pytest.raises(candidate.CandidateValidationError, match="inconsistent"):
        candidate.build_reference_candidate(
            urdf_path=urdf,
            reference_npz_path=inconsistent,
            joint_deg=[90.0],
        )


def test_cli_defaults_accept_repository_fixed_reference_and_record_hashes(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    reference = root / "aruco" / "fixed_workspace_reference.npz"
    urdf = _write_six_joint_urdf(tmp_path / "six.urdf")
    output = tmp_path / "candidate.json"

    assert (
        candidate.main(
            [
                "--urdf",
                str(urdf),
                "--reference-npz",
                str(reference),
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["joint_state"]["joint_deg_chain_order"] == list(
        candidate.DEFAULT_JOINT_DEG
    )
    assert payload["candidate_mode"] == "conditional_flange_equals_legacy_npz_reference_tcp"
    assert payload["source_hashes_sha256"]["expanded_urdf"] == hashlib.sha256(
        urdf.read_bytes()
    ).hexdigest()
    assert payload["source_hashes_sha256"][
        "fixed_workspace_reference_npz"
    ] == hashlib.sha256(reference.read_bytes()).hexdigest()

    original = output.read_bytes()
    with pytest.raises(SystemExit):
        candidate.main(
            [
                "--urdf",
                str(urdf),
                "--reference-npz",
                str(reference),
                "--output-json",
                str(output),
            ]
        )
    assert output.read_bytes() == original
