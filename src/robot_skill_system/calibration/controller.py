"""Asynchronous, abortable eye-in-hand calibration session controller."""

from __future__ import annotations

import io
import json
import math
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from robot_skill_system.calibration.handeye import (
    CheckerboardObservation,
    EyeInHandCalibrationConfig,
    EyeInHandCalibrationResult,
    convert_legacy_tcp_camera_mm_to_flange_camera_m,
    detect_checkerboard_observation,
    solve_eye_in_hand,
    validate_eye_in_hand_transform,
)
from robot_skill_system.calibration.robot import (
    CalibrationWaypoint,
    HandEyeCalibrationRobot,
    JointVector,
    approved_eye_in_hand_waypoints,
)
from robot_skill_system.capture.interfaces import SynchronizedRGBDFrame
from robot_skill_system.capture.rgbd_recording import (
    RGBDCameraController,
    encode_rgb_jpeg,
)
from robot_skill_system.exceptions import HardwareExecutionDenied
from robot_skill_system.storage.artifact_store import ArtifactMetadata, LocalArtifactStore

ObservationDetector = Callable[
    [SynchronizedRGBDFrame, EyeInHandCalibrationConfig],
    tuple[np.ndarray[Any, np.dtype[np.float64]], float],
]
CalibrationSolver = Callable[
    [list[CheckerboardObservation], EyeInHandCalibrationConfig],
    EyeInHandCalibrationResult,
]


class CalibrationAborted(RuntimeError):
    """Raised internally after a user-requested stop."""


class HandEyeCalibrationController:
    """Own one calibration job and keep every hardware transition fail-closed."""

    def __init__(
        self,
        *,
        store: LocalArtifactStore,
        camera: RGBDCameraController,
        robot_factory: Callable[[], HandEyeCalibrationRobot],
        hardware_authorized: bool,
        gate_summary: dict[str, bool],
        config: EyeInHandCalibrationConfig | None = None,
        waypoints: tuple[CalibrationWaypoint, ...] | None = None,
        detector: ObservationDetector = detect_checkerboard_observation,
        solver: CalibrationSolver = solve_eye_in_hand,
        joint_velocity_rad_s: float = math.radians(10.0),
        joint_acceleration_rad_s2: float = math.radians(5.0),
        settle_time_s: float = 1.0,
        joint_lock_tolerance_rad: float = math.radians(0.25),
        target_tolerance_rad: float = math.radians(1.0),
        legacy_npy_path: Path | None = None,
        legacy_expected_tcp_name: str = "2FG_TCP",
    ) -> None:
        if joint_velocity_rad_s <= 0.0 or joint_acceleration_rad_s2 <= 0.0:
            raise ValueError("calibration joint speed limits must be positive")
        if settle_time_s < 0.0 or settle_time_s > 10.0:
            raise ValueError("calibration settle time must be in [0, 10]")
        self.store = store
        self.camera = camera
        self.robot_factory = robot_factory
        self.hardware_authorized = hardware_authorized
        self.gate_summary = dict(gate_summary)
        self.config = config or EyeInHandCalibrationConfig()
        self.waypoints = waypoints or approved_eye_in_hand_waypoints()
        if len(self.waypoints) < self.config.minimum_observation_count:
            raise ValueError("calibration plan has fewer poses than the validation minimum")
        self.detector = detector
        self.solver = solver
        self.joint_velocity_rad_s = joint_velocity_rad_s
        self.joint_acceleration_rad_s2 = joint_acceleration_rad_s2
        self.settle_time_s = settle_time_s
        self.joint_lock_tolerance_rad = joint_lock_tolerance_rad
        self.target_tolerance_rad = target_tolerance_rad
        self.legacy_npy_path = legacy_npy_path
        self.legacy_expected_tcp_name = legacy_expected_tcp_name
        self._lock = threading.RLock()
        self._abort_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_robot: HandEyeCalibrationRobot | None = None
        self._session: dict[str, Any] | None = None
        self._latest_result: dict[str, Any] | None = None
        self._latest_legacy_transform = self._discover_latest_legacy_transform()

    def capabilities(self) -> dict[str, Any]:
        failed_gates = sorted(name for name, passed in self.gate_summary.items() if not passed)
        reference = self.waypoints[0].joint_positions_rad
        maximum_excursion_rad = max(
            abs(value - reference[joint_index])
            for waypoint in self.waypoints
            for joint_index, value in enumerate(waypoint.joint_positions_rad)
        )
        maximum_step_rad = max(
            abs(current_value - previous.joint_positions_rad[joint_index])
            for previous, current in zip(
                self.waypoints[:-1], self.waypoints[1:], strict=True
            )
            for joint_index, current_value in enumerate(current.joint_positions_rad)
        )
        return {
            "mode": "eye_in_hand",
            "hardware_authorized": self.hardware_authorized,
            "failed_gates": failed_gates,
            "board": {
                "internal_corners": [self.config.board_columns, self.config.board_rows],
                "square_size_m": self.config.square_size_m,
            },
            "image_size_px": [self.config.image_width_px, self.config.image_height_px],
            "reference_joint_positions_deg": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0],
            "locked_joint_indices": [1, 2],
            "pose_count": len(self.waypoints),
            "maximum_joint_excursion_deg": math.degrees(maximum_excursion_rad),
            "maximum_joint_step_deg": math.degrees(maximum_step_rad),
            "joint_velocity_deg_s": math.degrees(self.joint_velocity_rad_s),
            "joint_acceleration_deg_s2": math.degrees(self.joint_acceleration_rad_s2),
            "motion_plan": [item.as_dict() for item in self.waypoints],
            "output_transform": "T_flange_camera",
            "legacy_npy": {
                "configured": self.legacy_npy_path is not None,
                "available": bool(
                    self.legacy_npy_path is not None and self.legacy_npy_path.is_file()
                ),
                "expected_tcp_name": self.legacy_expected_tcp_name,
                "source_convention": "T_tcp_camera",
                "source_translation_unit": "mm",
            },
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            session = dict(self._session) if self._session is not None else None
        return {
            "capabilities": self.capabilities(),
            "session": session,
            "latest_result": self._latest_result,
            "legacy_transform": self._latest_legacy_transform,
        }

    def import_legacy_npy(self, *, operator_id: str) -> dict[str, Any]:
        """Back up, convert, and validate a legacy TCP-camera NPY without motion."""

        if not self.hardware_authorized:
            failed = ", ".join(self.capabilities()["failed_gates"]) or "hardware gates"
            raise HardwareExecutionDenied(
                "legacy hand-eye import requires robot frame authorization: " + failed
            )
        source_path = self.legacy_npy_path
        if source_path is None or not source_path.is_file():
            raise ValueError("configured legacy T_gripper2camera.npy is unavailable")
        if source_path.suffix.lower() != ".npy":
            raise ValueError("legacy hand-eye source must be an .npy file")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("wait for the active hand-eye calibration session to finish")
        source_bytes = source_path.read_bytes()
        source_matrix = np.load(io.BytesIO(source_bytes), allow_pickle=False)
        if source_matrix.shape != (4, 4):
            raise ValueError("legacy hand-eye NPY must contain one 4x4 matrix")
        observations, source_session_id = self._load_latest_observations()
        robot = self.robot_factory()
        connected = False
        try:
            robot.connect()
            connected = True
            active_tcp_name = robot.get_active_tcp_name()
            base_to_flange = robot.get_base_to_flange_matrix()
            base_to_tcp = robot.get_base_to_tcp_matrix()
            flange_to_tcp = np.asarray(
                np.linalg.inv(base_to_flange) @ base_to_tcp,
                dtype=np.float64,
            )
        finally:
            if connected:
                robot.disconnect()
        flange_to_camera = convert_legacy_tcp_camera_mm_to_flange_camera_m(
            np.asarray(source_matrix, dtype=np.float64),
            flange_to_tcp,
        )
        validation = validate_eye_in_hand_transform(
            observations,
            self.config,
            flange_to_camera,
        )
        import_id = f"legacy_import_{uuid.uuid4().hex}"
        root = f"calibrations/{import_id}"
        source_artifact = self.store.put_bytes(
            f"{root}/source_T_gripper2camera.npy",
            source_bytes,
            media_type="application/x-npy",
        )
        matrix_buffer = io.BytesIO()
        np.save(matrix_buffer, flange_to_camera, allow_pickle=False)
        matrix_artifact = self.store.put_bytes(
            f"{root}/T_flange_camera_candidate.npy",
            matrix_buffer.getvalue(),
            media_type="application/x-npy",
        )
        tcp_name_matches = active_tcp_name == self.legacy_expected_tcp_name
        import_failures = list(validation.failures)
        if not tcp_name_matches:
            import_failures.append(
                "active TCP name does not match the legacy calibration provenance"
            )
        payload = {
            "schema_version": "1.0",
            "evidence_type": "legacy_handeye_transform_candidate",
            "import_id": import_id,
            "operator_id": operator_id,
            "created_at_ns": time.time_ns(),
            "source_session_id": source_session_id,
            "source_transform_convention": "T_tcp_camera",
            "source_translation_unit": "mm",
            "source_backup": {
                "uri": source_artifact.uri,
                "checksum_sha256": source_artifact.checksum_sha256,
                "original_preserved": True,
            },
            "active_tcp_name": active_tcp_name,
            "expected_tcp_name": self.legacy_expected_tcp_name,
            "tcp_name_matches": tcp_name_matches,
            "flange_to_tcp": flange_to_tcp.tolist(),
            "transform_convention": "T_flange_camera",
            "flange_to_camera": flange_to_camera.tolist(),
            "matrix_artifact": {
                "uri": matrix_artifact.uri,
                "checksum_sha256": matrix_artifact.checksum_sha256,
            },
            "metrics": validation.as_dict()["metrics"],
            "observation_count": validation.observation_count,
            "passed": validation.passed and tcp_name_matches,
            "failures": import_failures,
            "candidate_only": True,
            "hardware_validated": False,
            "runtime_authorized": False,
        }
        artifact = self.store.put_json(f"{root}/result.json", payload)
        self._latest_legacy_transform = {
            **payload,
            "artifact_uri": artifact.uri,
            "artifact_checksum_sha256": artifact.checksum_sha256,
        }
        return self.status()

    def start(self, *, operator_id: str) -> dict[str, Any]:
        if not self.hardware_authorized:
            failed = ", ".join(self.capabilities()["failed_gates"]) or "hardware gates"
            raise HardwareExecutionDenied(
                "hand-eye calibration hardware authorization is closed: " + failed
            )
        camera_status = self.camera.status()
        if camera_status["state"] != "streaming":
            raise ValueError("start the RealSense RGB-D preview before calibration")
        if camera_status.get("recording") is not None:
            raise ValueError("stop RGB-D motion recording before calibration")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("a hand-eye calibration session is already running")
            session_id = f"handeye_{uuid.uuid4().hex}"
            now_ns = time.time_ns()
            self._session = {
                "schema_version": "1.0",
                "session_id": session_id,
                "status": "starting",
                "operator_id": operator_id,
                "started_at_ns": now_ns,
                "ended_at_ns": None,
                "current_waypoint_index": 0,
                "current_waypoint_id": None,
                "planned_pose_count": len(self.waypoints),
                "accepted_observation_count": 0,
                "rejected_observation_count": 0,
                "message": "Doosan/RealSense preflight starting",
                "result_uri": None,
                "error": None,
            }
            self._latest_result = None
            self._abort_event.clear()
            self.store.put_json(
                f"calibrations/{session_id}/session_started.json",
                {**self._session, "capabilities": self.capabilities()},
            )
            self._thread = threading.Thread(
                target=self._run,
                name=f"handeye-calibration-{session_id}",
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def abort(self, *, reason: str) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            robot = self._active_robot
            if thread is None or not thread.is_alive():
                return self.status()
            self._abort_event.set()
            self._update_session_locked(status="aborting", message=f"stop requested: {reason}")
        if robot is not None:
            robot.stop(reason=reason)
        return self.status()

    def close(self) -> None:
        with self._lock:
            thread = self._thread
        if thread is None or not thread.is_alive():
            return
        self._abort_event.set()
        with self._lock:
            robot = self._active_robot
        if robot is not None:
            robot.stop(reason="application_shutdown")
        thread.join(timeout=5.0)

    def _run(self) -> None:
        robot: HandEyeCalibrationRobot | None = None
        observations: list[CheckerboardObservation] = []
        try:
            robot = self.robot_factory()
            with self._lock:
                self._active_robot = robot
            robot.connect()
            current = robot.get_joint_positions_rad()
            nominal_reference = self.waypoints[0].joint_positions_rad
            for index in (0, 1):
                if (
                    abs(current[index] - nominal_reference[index])
                    > self.joint_lock_tolerance_rad
                ):
                    raise ValueError(
                        f"J{index + 1} must already be 0deg; automatic reference motion "
                        "never rotates J1/J2"
                    )
            session_waypoints = self._waypoints_with_locked_joints(current[:2])
            reference = session_waypoints[0].joint_positions_rad
            self._update_session(
                status="moving_to_reference",
                current_waypoint_index=0,
                current_waypoint_id=session_waypoints[0].waypoint_id,
                locked_joint_positions_deg=[
                    math.degrees(current[0]),
                    math.degrees(current[1]),
                ],
                message="moving J3-J6 to J=[locked,locked,90,0,90,0]",
            )
            self._raise_if_aborted()
            robot.move_joints(
                reference,
                velocity_rad_s=self.joint_velocity_rad_s,
                acceleration_rad_s2=self.joint_acceleration_rad_s2,
            )
            self._raise_if_aborted()
            measured_reference = robot.get_joint_positions_rad()
            self._validate_joint_locks(measured_reference, reference)
            if self._abort_event.wait(self.settle_time_s):
                raise CalibrationAborted("operator requested calibration stop")
            preflight_frame = self.camera.get_latest_frame(
                after_timestamp_ns=time.time_ns(),
                timeout_s=3.0,
            )
            try:
                self.detector(preflight_frame, self.config)
            except ValueError as exc:
                raise ValueError(
                    "checkerboard was not detected after automatic reference motion"
                ) from exc
            self._update_session(
                status="running",
                message="preflight passed; collecting synchronized poses",
            )
            for index, waypoint in enumerate(session_waypoints):
                self._raise_if_aborted()
                self._update_session(
                    current_waypoint_index=index,
                    current_waypoint_id=waypoint.waypoint_id,
                    message=f"moving to {waypoint.waypoint_id}",
                )
                if index != 0:
                    robot.move_joints(
                        waypoint.joint_positions_rad,
                        velocity_rad_s=self.joint_velocity_rad_s,
                        acceleration_rad_s2=self.joint_acceleration_rad_s2,
                    )
                self._raise_if_aborted()
                measured = robot.get_joint_positions_rad()
                self._validate_joint_locks(measured, waypoint.joint_positions_rad)
                if self._abort_event.wait(self.settle_time_s):
                    raise CalibrationAborted("operator requested calibration stop")
                settled_at_ns = time.time_ns()
                base_to_flange = robot.get_base_to_flange_matrix().copy()
                observation = self._capture_observation(
                    waypoint=waypoint,
                    measured_joint_positions=measured,
                    base_to_flange=base_to_flange,
                    after_timestamp_ns=settled_at_ns,
                )
                if observation is None:
                    self._increment_rejected()
                    continue
                observations.append(observation)
                self._persist_observation(observation)
                self._update_session(
                    accepted_observation_count=len(observations),
                    message=f"captured {waypoint.waypoint_id}",
                )
            self._raise_if_aborted()
            result = self.solver(observations, self.config)
            result_payload = self._persist_result(result, observations)
            with self._lock:
                self._latest_result = result_payload
                self._update_session_locked(result_uri=result_payload["artifact_uri"])
            if not result.passed:
                raise ValueError("hand-eye validation failed: " + "; ".join(result.failures))
            robot.move_joints(
                reference,
                velocity_rad_s=self.joint_velocity_rad_s,
                acceleration_rad_s2=self.joint_acceleration_rad_s2,
            )
            self._validate_joint_locks(robot.get_joint_positions_rad(), reference)
            with self._lock:
                self._update_session_locked(
                    status="passed",
                    ended_at_ns=time.time_ns(),
                    result_uri=result_payload["artifact_uri"],
                    message="T_flange_camera validation passed",
                )
        except CalibrationAborted as exc:
            self._finish_failure("aborted", str(exc))
        except Exception as exc:
            status = "aborted" if self._abort_event.is_set() else "failed"
            self._finish_failure(status, f"{type(exc).__name__}: {exc}")
        finally:
            if robot is not None:
                try:
                    robot.disconnect()
                except Exception as exc:
                    with self._lock:
                        if self._session is not None and self._session["error"] is None:
                            self._session["error"] = f"disconnect failed: {type(exc).__name__}"
            with self._lock:
                self._active_robot = None

    def _waypoints_with_locked_joints(
        self, locked_joints: tuple[float, float]
    ) -> tuple[CalibrationWaypoint, ...]:
        joint_1, joint_2 = locked_joints
        return tuple(
            CalibrationWaypoint(
                waypoint_id=waypoint.waypoint_id,
                joint_positions_rad=(
                    joint_1,
                    joint_2,
                    *waypoint.joint_positions_rad[2:],
                ),
            )
            for waypoint in self.waypoints
        )

    def _capture_observation(
        self,
        *,
        waypoint: CalibrationWaypoint,
        measured_joint_positions: JointVector,
        base_to_flange: np.ndarray[Any, np.dtype[np.float64]],
        after_timestamp_ns: int,
    ) -> CheckerboardObservation | None:
        latest_timestamp = after_timestamp_ns
        for _attempt in range(3):
            self._raise_if_aborted()
            frame = self.camera.get_latest_frame(
                after_timestamp_ns=latest_timestamp,
                timeout_s=3.0,
            )
            latest_timestamp = frame.timestamp_ns
            try:
                camera_to_board, reprojection_rms_px = self.detector(frame, self.config)
            except ValueError:
                continue
            if reprojection_rms_px > self.config.maximum_single_view_reprojection_rms_px:
                continue
            rgb_artifact = self._persist_rgb_frame(waypoint.waypoint_id, frame)
            intrinsics = frame.color_intrinsics
            return CheckerboardObservation(
                waypoint_id=waypoint.waypoint_id,
                frame_number=frame.frame_number,
                captured_at_ns=frame.timestamp_ns,
                joint_positions_rad=measured_joint_positions,
                base_to_flange=np.asarray(base_to_flange, dtype=np.float64),
                camera_to_board=np.asarray(camera_to_board, dtype=np.float64),
                reprojection_rms_px=reprojection_rms_px,
                camera_reference_frame=frame.reference_frame,
                image_size_px=(intrinsics.width_px, intrinsics.height_px),
                camera_intrinsics_px=(
                    intrinsics.fx_px,
                    intrinsics.fy_px,
                    intrinsics.cx_px,
                    intrinsics.cy_px,
                ),
                distortion_model=intrinsics.distortion_model,
                distortion_coefficients=intrinsics.distortion_coefficients,
                rgb_artifact_uri=rgb_artifact.uri,
                rgb_checksum_sha256=rgb_artifact.checksum_sha256,
            )
        return None

    def _persist_rgb_frame(
        self, waypoint_id: str, frame: SynchronizedRGBDFrame
    ) -> ArtifactMetadata:
        session_id = self._session_id()
        return self.store.put_bytes(
            f"calibrations/{session_id}/rgb/{waypoint_id}.jpg",
            encode_rgb_jpeg(frame.color_image_rgb),
            media_type="image/jpeg",
        )

    def _persist_observation(self, observation: CheckerboardObservation) -> None:
        session_id = self._session_id()
        self.store.put_json(
            f"calibrations/{session_id}/observations/{observation.waypoint_id}.json",
            {
                "waypoint_id": observation.waypoint_id,
                "frame_number": observation.frame_number,
                "captured_at_ns": observation.captured_at_ns,
                "joint_positions_rad": list(observation.joint_positions_rad),
                "base_to_flange": observation.base_to_flange.tolist(),
                "camera_to_board": observation.camera_to_board.tolist(),
                "reprojection_rms_px": observation.reprojection_rms_px,
                "camera_reference_frame": observation.camera_reference_frame,
                "image_size_px": list(observation.image_size_px),
                "camera_intrinsics_px": {
                    "fx": observation.camera_intrinsics_px[0],
                    "fy": observation.camera_intrinsics_px[1],
                    "cx": observation.camera_intrinsics_px[2],
                    "cy": observation.camera_intrinsics_px[3],
                },
                "distortion_model": observation.distortion_model,
                "distortion_coefficients": list(observation.distortion_coefficients),
                "rgb_artifact_uri": observation.rgb_artifact_uri,
                "rgb_checksum_sha256": observation.rgb_checksum_sha256,
            },
        )

    def _persist_result(
        self,
        result: EyeInHandCalibrationResult,
        observations: list[CheckerboardObservation],
    ) -> dict[str, Any]:
        session_id = self._session_id()
        buffer = io.BytesIO()
        np.save(buffer, result.flange_to_camera, allow_pickle=False)
        matrix_artifact = self.store.put_bytes(
            f"calibrations/{session_id}/T_flange_camera.npy",
            buffer.getvalue(),
            media_type="application/x-npy",
        )
        payload = {
            "schema_version": "1.0",
            "session_id": session_id,
            "calibration_type": "eye_in_hand",
            "robot_model": "m0609",
            "camera_mount": "flange_bracket",
            "board": {
                "internal_corners": [self.config.board_columns, self.config.board_rows],
                "square_size_m": self.config.square_size_m,
                "fixed_during_capture": True,
            },
            "image_size_px": [self.config.image_width_px, self.config.image_height_px],
            "locked_joint_indices": [1, 2],
            "observation_ids": [item.waypoint_id for item in observations],
            "matrix_artifact": {
                "uri": matrix_artifact.uri,
                "checksum_sha256": matrix_artifact.checksum_sha256,
            },
            **result.as_dict(),
        }
        artifact = self.store.put_json(
            f"calibrations/{session_id}/result.json",
            payload,
        )
        return {
            **payload,
            "artifact_uri": artifact.uri,
            "artifact_checksum_sha256": artifact.checksum_sha256,
        }

    def _load_latest_observations(
        self,
    ) -> tuple[list[CheckerboardObservation], str]:
        result_paths = sorted(
            (self.store.root / "calibrations").glob("handeye_*/result.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not result_paths:
            raise ValueError("run one hand-eye capture before validating a legacy NPY")
        result_path = result_paths[-1]
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        source_session_id = str(result_payload["session_id"])
        observation_root = result_path.parent / "observations"
        observations: list[CheckerboardObservation] = []
        for waypoint_id in result_payload.get("observation_ids") or []:
            path = observation_root / f"{waypoint_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            joints = tuple(float(value) for value in payload["joint_positions_rad"])
            if len(joints) != 6:
                raise ValueError("persisted hand-eye observation has invalid joint count")
            intrinsics = payload.get("camera_intrinsics_px") or {}
            observations.append(
                CheckerboardObservation(
                    waypoint_id=str(payload["waypoint_id"]),
                    frame_number=int(payload["frame_number"]),
                    captured_at_ns=int(payload["captured_at_ns"]),
                    joint_positions_rad=joints,
                    base_to_flange=np.asarray(
                        payload["base_to_flange"], dtype=np.float64
                    ),
                    camera_to_board=np.asarray(
                        payload["camera_to_board"], dtype=np.float64
                    ),
                    reprojection_rms_px=float(payload["reprojection_rms_px"]),
                    camera_reference_frame=str(
                        payload.get("camera_reference_frame")
                        or "camera_color_optical_frame"
                    ),
                    image_size_px=cast(
                        tuple[int, int],
                        tuple(int(value) for value in payload.get("image_size_px") or (0, 0)),
                    ),
                    camera_intrinsics_px=(
                        float(intrinsics.get("fx") or 0.0),
                        float(intrinsics.get("fy") or 0.0),
                        float(intrinsics.get("cx") or 0.0),
                        float(intrinsics.get("cy") or 0.0),
                    ),
                    distortion_model=str(payload.get("distortion_model") or "none"),
                    distortion_coefficients=tuple(
                        float(value)
                        for value in payload.get("distortion_coefficients") or []
                    ),
                    rgb_artifact_uri=payload.get("rgb_artifact_uri"),
                    rgb_checksum_sha256=payload.get("rgb_checksum_sha256"),
                )
            )
        if len(observations) < self.config.minimum_observation_count:
            raise ValueError("latest hand-eye session has insufficient valid observations")
        return observations, source_session_id

    def _discover_latest_legacy_transform(self) -> dict[str, Any] | None:
        root = self.store.root / "calibrations"
        if not root.is_dir():
            return None
        candidates = sorted(
            root.glob("legacy_import_*/result.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not candidates:
            return None
        path = candidates[-1]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return {
            **payload,
            "artifact_uri": path.relative_to(self.store.root).as_posix(),
        }

    def _validate_joint_locks(self, measured: JointVector, target: JointVector) -> None:
        for index in (0, 1):
            if abs(measured[index] - target[index]) > self.joint_lock_tolerance_rad:
                raise ValueError(f"J{index + 1} moved outside the calibration lock tolerance")
        if max(
            abs(actual - expected)
            for actual, expected in zip(measured, target, strict=True)
        ) > (
            self.target_tolerance_rad
        ):
            raise ValueError("robot did not settle at the approved calibration waypoint")

    def _raise_if_aborted(self) -> None:
        if self._abort_event.is_set():
            raise CalibrationAborted("operator requested calibration stop")

    def _increment_rejected(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session["rejected_observation_count"] += 1
                self._session["message"] = "checkerboard not accepted; pose skipped"

    def _finish_failure(self, status: str, error: str) -> None:
        with self._lock:
            self._update_session_locked(
                status=status,
                ended_at_ns=time.time_ns(),
                message=error,
                error=error,
            )
            if self._session is not None:
                session_id = str(self._session["session_id"])
                self.store.put_json(
                    f"calibrations/{session_id}/session_{status}.json",
                    self._session,
                )

    def _update_session(self, **changes: Any) -> None:
        with self._lock:
            self._update_session_locked(**changes)

    def _update_session_locked(self, **changes: Any) -> None:
        if self._session is not None:
            self._session.update(changes)

    def _session_id(self) -> str:
        with self._lock:
            if self._session is None:
                raise RuntimeError("calibration session was not initialized")
            return str(self._session["session_id"])


__all__ = ["CalibrationAborted", "HandEyeCalibrationController"]
