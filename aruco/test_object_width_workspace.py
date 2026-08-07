#!/usr/bin/env python3
"""CPU-only numeric and fail-closed tests for the fixed ArUco workspace tools."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import freeze_fixed_aruco_workspace as freeze  # noqa: E402
import object_width_workspace as workspace  # noqa: E402


def _memory_reference(
    tcp_reference_z_m: float = 0.160, camera_reference_z_m: float = 0.400
) -> dict[str, object]:
    T_camera_plane = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, camera_reference_z_m],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T_plane_camera = np.linalg.inv(T_camera_plane)
    T_plane_tcp = np.eye(4, dtype=np.float64)
    T_plane_tcp[2, 3] = tcp_reference_z_m
    T_tcp_plane = np.linalg.inv(T_plane_tcp)
    T_tcp_camera = T_tcp_plane @ T_plane_camera
    return {
        "schema_version": "2.0",
        "frame_name": "test_fixed_plane",
        "frame_definition": "+z=toward_camera",
        "reference_joint_deg": workspace.EXPECTED_REFERENCE_JOINT_DEG.copy(),
        "T_camera_plane": T_camera_plane,
        "T_plane_camera": T_plane_camera,
        "T_tcp_camera": T_tcp_camera,
        "T_tcp_plane": T_tcp_plane,
        "T_plane_tcp_reference": T_plane_tcp,
        "tcp_reference_plane_xyz_m": T_plane_tcp[:3, 3].copy(),
        "camera_reference_plane_xyz_m": T_plane_camera[:3, 3].copy(),
        "workspace_xy_m": np.asarray(
            [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]],
            dtype=np.float64,
        ),
        "closed_tip_clearance_m": 0.184,
        "safety_margin_m": 0.005,
        "gripper_radius_m": 0.110,
        "default_width_model": "full-opening",
        "reference_created_at_ns": None,
    }


def _plane_payload(polygon_xy_m: np.ndarray | None = None) -> dict[str, object]:
    if polygon_xy_m is None:
        polygon_xy_m = np.asarray(
            [[-0.2, -0.15], [0.2, -0.15], [0.2, 0.15], [-0.2, 0.15]],
            dtype=np.float64,
        )
    polygon = np.asarray(polygon_xy_m, dtype=np.float64)

    # The table is 0.4 m in front of the optical camera.  Plane +Z points back
    # toward the camera, so its camera-frame direction is [0, 0, -1].
    T_camera_plane = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    plane_points_h = np.column_stack(
        (polygon, np.zeros(len(polygon)), np.ones(len(polygon)))
    )
    boundary_camera = (T_camera_plane @ plane_points_h.T).T[:, :3]
    return {
        "schema_version": "1.0",
        "status": "geometric_candidate_not_robot_safety_approved",
        "plane_camera": {
            "normal_xyz": [0.0, 0.0, -1.0],
            "d_m": 0.4,
        },
        "plane_frame": {
            "definition": "origin=test; x_axis=camera_positive_x; +z=toward_camera",
            "T_camera_plane": T_camera_plane.tolist(),
            "T_plane_camera": np.linalg.inv(T_camera_plane).tolist(),
        },
        "workspace_candidate": {
            "reference_frame": "aruco_plane",
            "workspace_margin_m": 0.0,
            "safe_boundary_xy_m": polygon.tolist(),
            "safe_boundary_camera_xyz_m": boundary_camera.tolist(),
            "surface_z_plane_m": 0.0,
            "limitations": ["test geometry only"],
        },
    }


def _freeze_args(
    plane_json: Path, calibration_npy: Path, output_npz: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        plane_result_json=str(plane_json),
        tcp_camera_npy=str(calibration_npy),
        tcp_camera_direction="camera-to-tcp",
        tcp_camera_translation_unit="m",
        safety_margin_mm=5.0,
        width_model="full-opening",
        output_npz=str(output_npz),
    )


def _create_frozen_reference(root: Path) -> tuple[Path, Path, Path]:
    plane_json = root / "plane_workspace_result.json"
    calibration_npy = root / "T_tcp_camera.npy"
    reference_npz = root / "fixed_workspace_reference.npz"
    payload = _plane_payload()
    plane_json.write_text(json.dumps(payload), encoding="utf-8")
    T_camera_plane = np.asarray(
        payload["plane_frame"]["T_camera_plane"], dtype=np.float64
    )
    T_plane_camera = np.linalg.inv(T_camera_plane)
    T_plane_tcp = np.eye(4, dtype=np.float64)
    T_plane_tcp[2, 3] = 0.160
    T_tcp_plane = np.linalg.inv(T_plane_tcp)
    T_tcp_camera = T_tcp_plane @ T_plane_camera
    np.save(calibration_npy, T_tcp_camera)
    with contextlib.redirect_stdout(io.StringIO()):
        freeze.freeze_reference(
            _freeze_args(plane_json, calibration_npy, reference_npz)
        )
    return plane_json, calibration_npy, reference_npz


class WidthGeometryTests(unittest.TestCase):
    def test_width_models_and_numeric_formula(self) -> None:
        theta = workspace.width_to_theta_rad(80.0, 0.110, "full-opening")
        expected_theta = math.asin(80.0 / 110.0)
        self.assertAlmostEqual(theta, expected_theta, places=14)

        legacy_theta = workspace.width_to_theta_rad(
            40.0, 0.110, "legacy-half-factor"
        )
        self.assertAlmostEqual(legacy_theta, expected_theta, places=14)
        self.assertEqual(
            workspace.width_to_theta_rad(0.0, 0.110, "full-opening"), 0.0
        )
        self.assertAlmostEqual(
            workspace.width_to_theta_rad(110.0, 0.110, "full-opening"),
            math.pi / 2.0,
        )
        self.assertAlmostEqual(
            workspace.width_to_theta_rad(55.0, 0.110, "legacy-half-factor"),
            math.pi / 2.0,
        )

    def test_invalid_widths_are_rejected_instead_of_clamped(self) -> None:
        for value in (-1.0, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                workspace.width_to_theta_rad(value, 0.110, "full-opening")
        with self.assertRaises(ValueError):
            workspace.width_to_theta_rad(110.001, 0.110, "full-opening")
        with self.assertRaises(ValueError):
            workspace.width_to_theta_rad(
                math.nextafter(110.0, math.inf), 0.110, "full-opening"
            )
        with self.assertRaises(ValueError):
            workspace.width_to_theta_rad(55.001, 0.110, "legacy-half-factor")
        with self.assertRaises(ValueError):
            workspace.width_to_theta_rad(10.0, math.nan, "full-opening")
        with self.assertRaises(ValueError):
            workspace.width_to_theta_rad(10.0, 0.110, "unknown")

    def test_runtime_z_bounds_preserve_five_mm_tip_clearance(self) -> None:
        reference = _memory_reference()
        runtime = workspace.build_runtime_workspace(reference, 80.0)
        expected_theta = math.asin(80.0 / 110.0)
        expected_offset_m = 0.110 * (1.0 - math.cos(expected_theta))
        expected_allowed_m = 0.184 + expected_offset_m - 0.005
        self.assertAlmostEqual(runtime.opening_offset_m, expected_offset_m, places=14)
        self.assertAlmostEqual(
            runtime.allowed_down_from_reference_m, expected_allowed_m, places=14
        )
        self.assertAlmostEqual(runtime.z_reference_tcp_plane_m, 0.160, places=14)
        self.assertAlmostEqual(runtime.z_max_plane_m, 0.400, places=14)
        self.assertAlmostEqual(
            runtime.z_min_plane_m, 0.160 - expected_allowed_m, places=14
        )
        self.assertAlmostEqual(
            runtime.predicted_tip_clearance_at_z_min_m, 0.005, places=14
        )


class PolygonAndTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = _memory_reference()
        self.runtime = workspace.build_runtime_workspace(self.reference, 80.0)

    def test_polygon_accepts_both_windings_and_uses_metre_tolerance(self) -> None:
        polygon = np.asarray(self.reference["workspace_xy_m"])
        self.assertTrue(
            workspace.point_in_convex_polygon(np.asarray([0.0, 0.0]), polygon)
        )
        self.assertTrue(
            workspace.point_in_convex_polygon(np.asarray([0.0, 0.0]), polygon[::-1])
        )
        self.assertTrue(
            workspace.point_in_convex_polygon(np.asarray([0.2, 0.0]), polygon)
        )
        self.assertFalse(
            workspace.point_in_convex_polygon(np.asarray([0.201, 0.0]), polygon)
        )
        self.assertFalse(
            workspace.point_in_convex_polygon(np.asarray([math.nan, 0.0]), polygon)
        )

    def test_nearly_collinear_or_concave_polygon_is_rejected(self) -> None:
        skinny = np.asarray(
            [[-0.15, 0.0], [0.0, 0.0005], [0.15, 0.0]], dtype=np.float64
        )
        with self.assertRaises(ValueError):
            workspace.point_in_convex_polygon(np.asarray([0.0, 0.0]), skinny)
        concave = np.asarray(
            [[0.0, 0.0], [0.2, 0.0], [0.1, 0.05], [0.2, 0.2], [0.0, 0.2]]
        )
        with self.assertRaises(ValueError):
            workspace.point_in_convex_polygon(np.asarray([0.05, 0.05]), concave)

    def test_safe_boundaries_and_all_unsafe_directions(self) -> None:
        for z_m in (self.runtime.z_min_plane_m, self.runtime.z_max_plane_m):
            safe, details = workspace.check_tcp_point_plane(
                self.reference, self.runtime, np.asarray([0.0, 0.0, z_m])
            )
            self.assertTrue(safe, details)
            self.assertFalse(details["target_was_clamped"])
        above_tcp_safe, above_tcp_details = workspace.check_tcp_point_plane(
            self.reference, self.runtime, np.asarray([0.0, 0.0, 0.300])
        )
        self.assertTrue(above_tcp_safe, above_tcp_details)

        unsafe_points = (
            np.asarray([0.0, 0.0, self.runtime.z_min_plane_m - 0.001]),
            np.asarray([0.0, 0.0, self.runtime.z_max_plane_m + 0.001]),
            np.asarray([0.3, 0.0, self.runtime.z_min_plane_m + 0.01]),
        )
        for point in unsafe_points:
            with self.subTest(point=point.tolist()):
                safe, details = workspace.check_tcp_point_plane(
                    self.reference, self.runtime, point
                )
                self.assertFalse(safe)
                self.assertTrue(details["rejection_reasons"])
                self.assertFalse(details["target_was_clamped"])
                with self.assertRaises(workspace.UnsafeWorkspaceTargetError):
                    workspace.require_tcp_point_plane(
                        self.reference, self.runtime, point
                    )

    def test_nan_and_inf_targets_fail_closed(self) -> None:
        valid_z_m = self.runtime.z_min_plane_m + 0.01
        for point in (
            np.asarray([math.nan, 0.0, valid_z_m]),
            np.asarray([math.inf, math.inf, valid_z_m]),
            np.asarray([0.0, 0.0, math.nan]),
        ):
            with self.subTest(point=point.tolist()):
                safe, details = workspace.check_tcp_point_plane(
                    self.reference, self.runtime, point
                )
                self.assertFalse(safe)
                self.assertFalse(details["valid_input"])

    def test_forged_runtime_bounds_fail_closed(self) -> None:
        forged = replace(
            self.runtime,
            z_min_plane_m=-math.inf,
            z_max_plane_m=math.inf,
        )
        safe, details = workspace.check_tcp_point_plane(
            self.reference, forged, np.asarray([0.0, 0.0, 0.0])
        )
        self.assertFalse(safe)
        self.assertFalse(details["valid_input"])
        self.assertIn("invalid runtime workspace", details["rejection_reasons"][0])

    def test_geometric_plane_to_camera_range_is_distinct_from_tcp_lower_bound(self) -> None:
        reference = _memory_reference(
            tcp_reference_z_m=0.350, camera_reference_z_m=0.583
        )
        runtime = workspace.build_runtime_workspace(reference, 0.0)
        self.assertAlmostEqual(runtime.z_min_plane_m, 0.171)
        self.assertAlmostEqual(runtime.z_max_plane_m, 0.583)

        geometric_safe, geometric_details = workspace.check_workspace_point_plane(
            reference, np.asarray([0.0, 0.0, 0.0])
        )
        self.assertTrue(geometric_safe, geometric_details)
        camera_boundary_safe, camera_boundary_details = (
            workspace.check_workspace_point_plane(
                reference, np.asarray([0.0, 0.0, 0.583])
            )
        )
        self.assertTrue(camera_boundary_safe, camera_boundary_details)
        tcp_safe, tcp_details = workspace.check_tcp_point_plane(
            reference, runtime, np.asarray([0.0, 0.0, 0.0])
        )
        self.assertFalse(tcp_safe)
        self.assertIn("below", " ".join(tcp_details["rejection_reasons"]))
        lower_safe, lower_details = workspace.check_tcp_point_plane(
            reference, runtime, np.asarray([0.0, 0.0, runtime.z_min_plane_m])
        )
        self.assertTrue(lower_safe, lower_details)

        above_camera, _ = workspace.check_workspace_point_plane(
            reference, np.asarray([0.0, 0.0, 0.584])
        )
        self.assertFalse(above_camera)
        below_plane, _ = workspace.check_workspace_point_plane(
            reference, np.asarray([0.0, 0.0, -0.001])
        )
        self.assertFalse(below_plane)

    def test_frozen_camera_to_plane_transform_and_check(self) -> None:
        T_plane_camera = np.asarray(self.reference["T_plane_camera"])
        camera_point = np.asarray([0.02, -0.03, 0.20])
        plane_point = workspace.camera_ref_xyz_to_plane_xyz(
            self.reference, camera_point
        )
        expected = (T_plane_camera @ np.r_[camera_point, 1.0])[:3]
        np.testing.assert_allclose(plane_point, expected)
        safe, details = workspace.check_tcp_point_camera_ref(
            self.reference, self.runtime, camera_point
        )
        self.assertTrue(safe, details)
        self.assertEqual(details["source_input_frame"], "frozen_reference_camera")


class FrozenArtifactTests(unittest.TestCase):
    def test_freeze_load_and_runtime_atomic_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plane_json, calibration_npy, reference_npz = _create_frozen_reference(root)
            calibration_before = hashlib.sha256(calibration_npy.read_bytes()).hexdigest()

            reference = workspace.load_reference(reference_npz)
            self.assertEqual(reference["schema_version"], "2.0")
            self.assertEqual(
                hashlib.sha256(calibration_npy.read_bytes()).hexdigest(),
                calibration_before,
            )
            with np.load(reference_npz, allow_pickle=False) as frozen:
                self.assertIn("workspace_safe_boundary_plane_xy", frozen.files)
                self.assertIn("reference_tcp_position_plane_xyz_m", frozen.files)

            runtime_path = root / "runtime" / "runtime_workspace.npz"
            runtime_80 = workspace.build_runtime_workspace(reference, 80.0)
            workspace.save_runtime_npz(runtime_path, reference, runtime_80)
            with np.load(runtime_path, allow_pickle=False) as runtime_npz:
                self.assertEqual(float(runtime_npz["object_width_mm"]), 80.0)
                self.assertTrue(bool(runtime_npz["usable_for_target_validation"]))
                self.assertFalse(bool(runtime_npz["unsafe_targets_are_clamped"]))
                np.testing.assert_allclose(
                    runtime_npz["workspace_safe_boundary_plane_xy"],
                    reference["workspace_xy_m"],
                )
                np.testing.assert_allclose(
                    runtime_npz["nominal_plane_to_camera_z_range_m"],
                    [0.0, 0.4],
                )
                self.assertAlmostEqual(float(runtime_npz["z_max_plane_m"]), 0.4)

            first_runtime_bytes = runtime_path.read_bytes()
            runtime_20 = workspace.build_runtime_workspace(reference, 20.0)
            workspace.save_runtime_npz(runtime_path, reference, runtime_20)
            self.assertNotEqual(runtime_path.read_bytes(), first_runtime_bytes)
            with np.load(runtime_path, allow_pickle=False) as runtime_npz:
                self.assertEqual(float(runtime_npz["object_width_mm"]), 20.0)

            self.assertTrue(plane_json.is_file())
            self.assertTrue(reference_npz.is_file())
            with self.assertRaises(ValueError):
                workspace.save_runtime_npz(reference_npz, reference, runtime_20)

    def test_frozen_reference_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plane_json, calibration_npy, reference_npz = _create_frozen_reference(root)
            before = reference_npz.read_bytes()
            with self.assertRaises(FileExistsError):
                with contextlib.redirect_stdout(io.StringIO()):
                    freeze.freeze_reference(
                        _freeze_args(plane_json, calibration_npy, reference_npz)
                    )
            self.assertEqual(reference_npz.read_bytes(), before)

    def test_actual_failure_shape_is_rejected_at_freeze(self) -> None:
        skinny_polygon = np.asarray(
            [
                [-0.1503543108701706, -0.002444296842440963],
                [-0.0002498928806744516, -0.0007128280121833086],
                [0.15060420334339142, 0.0031571248546242714],
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plane_json = root / "plane_workspace_result.json"
            calibration_npy = root / "T_tcp_camera.npy"
            reference_npz = root / "fixed_workspace_reference.npz"
            plane_json.write_text(
                json.dumps(_plane_payload(skinny_polygon)), encoding="utf-8"
            )
            valid_payload = _plane_payload()
            T_camera_plane = np.asarray(
                valid_payload["plane_frame"]["T_camera_plane"], dtype=np.float64
            )
            T_plane_camera = np.linalg.inv(T_camera_plane)
            T_plane_tcp = np.eye(4, dtype=np.float64)
            T_plane_tcp[2, 3] = 0.160
            T_tcp_camera = np.linalg.inv(T_plane_tcp) @ T_plane_camera
            np.save(calibration_npy, T_tcp_camera)
            with self.assertRaisesRegex(ValueError, "nearly collinear"):
                freeze.freeze_reference(
                    _freeze_args(plane_json, calibration_npy, reference_npz)
                )
            self.assertFalse(reference_npz.exists())

    def test_malformed_transform_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, reference_npz = _create_frozen_reference(root)
            with np.load(reference_npz, allow_pickle=False) as original:
                values = {key: np.asarray(original[key]) for key in original.files}
            values["T_plane_camera"] = np.eye(4, dtype=np.float64)
            corrupt_path = root / "corrupt_reference.npz"
            np.savez_compressed(corrupt_path, **values)
            with self.assertRaises(ValueError):
                workspace.load_reference(corrupt_path)

    def test_loaded_reference_mutation_cannot_expand_safety_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, reference_npz = _create_frozen_reference(root)
            reference = workspace.load_reference(reference_npz)
            with self.assertRaises(ValueError):
                np.asarray(reference["workspace_xy_m"])[0, 0] = -100.0

            reference["camera_reference_plane_xyz_m"] = np.asarray(
                [0.0, 0.0, 100.0]
            )
            reference["workspace_xy_m"] = np.asarray(
                [[-100.0, -100.0], [100.0, -100.0], [100.0, 100.0], [-100.0, 100.0]]
            )
            runtime = workspace.build_runtime_workspace(reference, 80.0)
            self.assertAlmostEqual(runtime.z_max_plane_m, 0.4)
            safe, _ = workspace.check_tcp_point_plane(
                reference, runtime, np.asarray([1.0, 0.0, 0.3])
            )
            self.assertFalse(safe)

            reference_npz.unlink()
            safe_after_delete, details = workspace.check_tcp_point_plane(
                reference, runtime, np.asarray([0.0, 0.0, 0.3])
            )
            self.assertFalse(safe_after_delete)
            self.assertFalse(details["valid_input"])
            camera_safe_after_delete, camera_details = (
                workspace.check_tcp_point_camera_ref(
                    reference, runtime, np.asarray([0.0, 0.0, 0.1])
                )
            )
            self.assertFalse(camera_safe_after_delete)
            self.assertFalse(camera_details["valid_input"])

    def test_schema_and_artifact_kind_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, reference_npz = _create_frozen_reference(root)
            with np.load(reference_npz, allow_pickle=False) as original:
                values = {key: np.asarray(original[key]) for key in original.files}

            missing_schema = root / "missing_schema.npz"
            np.savez_compressed(
                missing_schema,
                **{key: value for key, value in values.items() if key != "schema_version"},
            )
            with self.assertRaises(ValueError):
                workspace.load_reference(missing_schema)

            wrong_kind = root / "wrong_kind.npz"
            values["artifact_kind"] = np.asarray("runtime_workspace")
            np.savez_compressed(wrong_kind, **values)
            with self.assertRaises(ValueError):
                workspace.load_reference(wrong_kind)

    def test_no_runtime_writer_can_replace_another_frozen_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            _, _, first_reference_path = _create_frozen_reference(first_root)
            _, _, second_reference_path = _create_frozen_reference(second_root)
            second_before = second_reference_path.read_bytes()

            reference = workspace.load_reference(first_reference_path)
            runtime = workspace.build_runtime_workspace(reference, 80.0)
            checks = {
                "plane_target": {
                    "safe": False,
                    "rejection_reasons": ["test rejection"],
                }
            }
            with self.assertRaises(ValueError):
                workspace.save_runtime_npz(second_reference_path, reference, runtime)
            with self.assertRaises(ValueError):
                workspace.save_rejected_runtime_npz(
                    second_reference_path, reference, runtime, checks
                )
            with self.assertRaises(ValueError):
                workspace.save_runtime_generation_failure_npz(
                    second_reference_path,
                    first_reference_path,
                    "80 mm",
                    ValueError("test failure"),
                )
            self.assertEqual(second_reference_path.read_bytes(), second_before)

    def test_runtime_writers_refuse_corrupt_and_unknown_existing_npz(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, reference_path = _create_frozen_reference(root)
            reference = workspace.load_reference(reference_path)
            runtime = workspace.build_runtime_workspace(reference, 80.0)
            checks = {
                "plane_target": {
                    "safe": False,
                    "rejection_reasons": ["test rejection"],
                }
            }
            writers = {
                "runtime": lambda path: workspace.save_runtime_npz(
                    path, reference, runtime
                ),
                "target_rejection": lambda path: workspace.save_rejected_runtime_npz(
                    path, reference, runtime, checks
                ),
                "generation_failure": lambda path: (
                    workspace.save_runtime_generation_failure_npz(
                        path,
                        reference_path,
                        "80 mm",
                        ValueError("test failure"),
                    )
                ),
            }

            for existing_kind in ("corrupt", "unknown"):
                for writer_name, writer in writers.items():
                    with self.subTest(
                        existing_kind=existing_kind, writer=writer_name
                    ):
                        output = root / f"{existing_kind}_{writer_name}.npz"
                        if existing_kind == "corrupt":
                            output.write_bytes(b"not a valid NPZ archive")
                        else:
                            np.savez_compressed(
                                output,
                                artifact_kind=np.asarray("unrelated_npz_artifact"),
                                payload=np.asarray([1, 2, 3], dtype=np.int64),
                            )
                        before = output.read_bytes()
                        with self.assertRaises(ValueError):
                            writer(output)
                        self.assertEqual(output.read_bytes(), before)

    def test_known_prior_runtime_can_be_atomically_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, reference_path = _create_frozen_reference(root)
            reference = workspace.load_reference(reference_path)
            output = root / "runtime_workspace.npz"

            first_runtime = workspace.build_runtime_workspace(reference, 80.0)
            workspace.save_runtime_npz(output, reference, first_runtime)
            first_bytes = output.read_bytes()

            replacement_runtime = workspace.build_runtime_workspace(reference, 20.0)
            workspace.save_runtime_npz(output, reference, replacement_runtime)
            self.assertNotEqual(output.read_bytes(), first_bytes)
            with np.load(output, allow_pickle=False) as replaced:
                self.assertEqual(
                    str(replaced["artifact_kind"]),
                    "object_width_runtime_workspace",
                )
                self.assertTrue(bool(replaced["usable_for_target_validation"]))
                self.assertEqual(float(replaced["object_width_mm"]), 20.0)

    def test_atomic_runtime_failure_preserves_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, reference_npz = _create_frozen_reference(root)
            reference = workspace.load_reference(reference_npz)
            runtime = workspace.build_runtime_workspace(reference, 80.0)
            output = root / "runtime_workspace.npz"
            workspace.save_runtime_npz(output, reference, runtime)
            previous = output.read_bytes()
            with mock.patch.object(
                workspace.np, "savez_compressed", side_effect=OSError("injected")
            ):
                with self.assertRaises(OSError):
                    workspace.save_runtime_npz(output, reference, runtime)
            self.assertEqual(output.read_bytes(), previous)

    def test_cli_unsafe_target_returns_nonzero_and_does_not_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, reference_npz = _create_frozen_reference(root)
            output = root / "runtime_workspace.npz"
            reference = workspace.load_reference(reference_npz)
            previous_runtime = workspace.build_runtime_workspace(reference, 20.0)
            workspace.save_runtime_npz(output, reference, previous_runtime)
            previous = output.read_bytes()
            argv = [
                "object_width_workspace.py",
                "--reference-npz",
                str(reference_npz),
                "--width-cm",
                "8",
                "--output-npz",
                str(output),
                "--check-plane-xyz",
                "0",
                "0",
                "-10",
            ]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                io.StringIO()
            ):
                result = workspace.main()
            self.assertEqual(result, workspace.UNSAFE_TARGET_EXIT_CODE)
            self.assertNotEqual(output.read_bytes(), previous)
            with np.load(output, allow_pickle=False) as rejected:
                self.assertEqual(str(rejected["status"]), "unsafe_target_rejected")
                self.assertFalse(bool(rejected["usable_for_target_validation"]))
                self.assertNotIn("z_min_plane_m", rejected.files)

    def test_cli_generation_error_invalidates_stale_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, reference_npz = _create_frozen_reference(root)
            output = root / "runtime_workspace.npz"
            reference = workspace.load_reference(reference_npz)
            previous_runtime = workspace.build_runtime_workspace(reference, 20.0)
            workspace.save_runtime_npz(output, reference, previous_runtime)
            argv = [
                "object_width_workspace.py",
                "--reference-npz",
                str(reference_npz),
                "--width-mm",
                "1000",
                "--output-npz",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(
                io.StringIO()
            ):
                result = workspace.main()
            self.assertEqual(result, 1)
            with np.load(output, allow_pickle=False) as rejected:
                self.assertEqual(str(rejected["status"]), "runtime_generation_failed")
                self.assertFalse(bool(rejected["usable_for_target_validation"]))
                self.assertNotIn("z_min_plane_m", rejected.files)

    def test_cli_required_width_error_invalidates_default_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, reference_npz = _create_frozen_reference(root)
            temporary_home = root / "home"
            output = temporary_home / "Dittobot" / "aruco" / "runtime" / "runtime_workspace.npz"
            reference = workspace.load_reference(reference_npz)
            previous_runtime = workspace.build_runtime_workspace(reference, 20.0)
            workspace.save_runtime_npz(output, reference, previous_runtime)
            previous = output.read_bytes()
            argv = [
                "object_width_workspace.py",
                "--reference-npz",
                str(reference_npz),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, {"HOME": str(temporary_home)}),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = workspace.main()
            self.assertEqual(result, 2)
            self.assertNotEqual(output.read_bytes(), previous)
            with np.load(output, allow_pickle=False) as rejected:
                self.assertEqual(
                    str(rejected["artifact_kind"]),
                    "rejected_object_width_runtime_workspace",
                )
                self.assertFalse(bool(rejected["usable_for_target_validation"]))
                self.assertNotIn("z_min_plane_m", rejected.files)

    def test_cli_width_type_error_invalidates_custom_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, reference_npz = _create_frozen_reference(root)
            output = root / "custom_runtime_workspace.npz"
            reference = workspace.load_reference(reference_npz)
            previous_runtime = workspace.build_runtime_workspace(reference, 20.0)
            workspace.save_runtime_npz(output, reference, previous_runtime)
            previous = output.read_bytes()
            argv = [
                "object_width_workspace.py",
                "--reference-npz",
                str(reference_npz),
                "--output-npz",
                str(output),
                "--width-mm",
                "not-a-number",
            ]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(
                io.StringIO()
            ):
                result = workspace.main()
            self.assertEqual(result, 2)
            self.assertNotEqual(output.read_bytes(), previous)
            with np.load(output, allow_pickle=False) as rejected:
                self.assertEqual(
                    str(rejected["artifact_kind"]),
                    "rejected_object_width_runtime_workspace",
                )
                self.assertFalse(bool(rejected["usable_for_target_validation"]))
                self.assertNotIn("z_min_plane_m", rejected.files)

    def test_cli_help_does_not_invalidate_default_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, reference_npz = _create_frozen_reference(root)
            temporary_home = root / "home"
            output = temporary_home / "Dittobot" / "aruco" / "runtime" / "runtime_workspace.npz"
            reference = workspace.load_reference(reference_npz)
            previous_runtime = workspace.build_runtime_workspace(reference, 20.0)
            workspace.save_runtime_npz(output, reference, previous_runtime)
            previous = output.read_bytes()
            argv = ["object_width_workspace.py", "--help"]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, {"HOME": str(temporary_home)}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                try:
                    result = workspace.main()
                except SystemExit as exc:
                    result = int(exc.code)
            self.assertEqual(result, 0)
            self.assertEqual(output.read_bytes(), previous)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        reference = _memory_reference()
        runtime = workspace.build_runtime_workspace(reference, 80.0)
        print(
            "SANITY: 80 mm -> "
            f"theta={runtime.theta_deg:.3f} deg, "
            f"offset={runtime.opening_offset_m * 1000.0:.3f} mm, "
            f"allowed_down={runtime.allowed_down_from_reference_m * 1000.0:.3f} mm"
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
