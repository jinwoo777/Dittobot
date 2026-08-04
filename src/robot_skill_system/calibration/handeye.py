"""Local metric eye-in-hand calibration with explicit transform conventions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robot_skill_system.capture.interfaces import SynchronizedRGBDFrame
from robot_skill_system.exceptions import NotConfiguredError

Matrix44 = NDArray[np.float64]


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise NotConfiguredError(
            "hand-eye calibration requires the optional OpenCV dependency"
        ) from exc
    return cv2


@dataclass(frozen=True, slots=True)
class EyeInHandCalibrationConfig:
    """Approved fixed target and validation limits for this robot cell."""

    board_columns: int = 10
    board_rows: int = 7
    square_size_m: float = 0.025
    image_width_px: int = 640
    image_height_px: int = 480
    minimum_observation_count: int = 15
    maximum_reprojection_rms_px: float = 0.5
    maximum_single_view_reprojection_rms_px: float = 1.0
    maximum_translation_rms_m: float = 0.003
    maximum_translation_residual_m: float = 0.005
    maximum_rotation_rms_deg: float = 0.5
    maximum_rotation_residual_deg: float = 1.0
    minimum_translation_span_m: float = 0.05
    minimum_rotation_span_deg: float = 15.0

    def __post_init__(self) -> None:
        if self.board_columns < 2 or self.board_rows < 2:
            raise ValueError("checkerboard internal-corner dimensions must be at least 2x2")
        if self.square_size_m <= 0.0:
            raise ValueError("checkerboard square size must be positive")
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("calibration image dimensions must be positive")
        if self.minimum_observation_count < 3:
            raise ValueError("hand-eye calibration requires at least three observations")


@dataclass(frozen=True, slots=True)
class CheckerboardObservation:
    """One synchronized robot/camera observation.

    ``base_to_flange`` is :math:`T_base_flange`; ``camera_to_board`` is
    :math:`T_camera_board`, the target-to-camera transform returned by PnP.
    """

    waypoint_id: str
    frame_number: int
    captured_at_ns: int
    joint_positions_rad: tuple[float, float, float, float, float, float]
    base_to_flange: Matrix44
    camera_to_board: Matrix44
    reprojection_rms_px: float
    camera_reference_frame: str = "camera_color_optical_frame"
    image_size_px: tuple[int, int] = (0, 0)
    camera_intrinsics_px: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    distortion_model: str = "none"
    distortion_coefficients: tuple[float, ...] = ()
    rgb_artifact_uri: str | None = None
    rgb_checksum_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_rigid_matrix(self.base_to_flange, "base_to_flange")
        _validate_rigid_matrix(self.camera_to_board, "camera_to_board")
        if not math.isfinite(self.reprojection_rms_px) or self.reprojection_rms_px < 0.0:
            raise ValueError("reprojection_rms_px must be finite and non-negative")
        if not self.camera_reference_frame.strip():
            raise ValueError("camera_reference_frame must be non-empty")


@dataclass(frozen=True, slots=True)
class EyeInHandCalibrationResult:
    """Solved :math:`T_flange_camera` and held-constant-board diagnostics."""

    flange_to_camera: Matrix44
    base_to_board_mean: Matrix44
    observation_count: int
    reprojection_rms_px: float
    maximum_view_reprojection_rms_px: float
    translation_rms_m: float
    maximum_translation_residual_m: float
    rotation_rms_deg: float
    maximum_rotation_residual_deg: float
    translation_span_m: float
    rotation_span_deg: float
    passed: bool
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "transform_convention": "T_flange_camera",
            "flange_to_camera": self.flange_to_camera.tolist(),
            "base_to_board_mean": self.base_to_board_mean.tolist(),
            "observation_count": self.observation_count,
            "metrics": {
                "reprojection_rms_px": self.reprojection_rms_px,
                "maximum_view_reprojection_rms_px": (
                    self.maximum_view_reprojection_rms_px
                ),
                "translation_rms_m": self.translation_rms_m,
                "maximum_translation_residual_m": (
                    self.maximum_translation_residual_m
                ),
                "rotation_rms_deg": self.rotation_rms_deg,
                "maximum_rotation_residual_deg": self.maximum_rotation_residual_deg,
                "translation_span_m": self.translation_span_m,
                "rotation_span_deg": self.rotation_span_deg,
            },
            "passed": self.passed,
            "failures": list(self.failures),
        }


def checkerboard_object_points(config: EyeInHandCalibrationConfig) -> NDArray[np.float32]:
    """Return row-major internal corners in the board frame, in metres."""

    points = np.zeros((config.board_columns * config.board_rows, 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0 : config.board_columns, 0 : config.board_rows]
        .T.reshape(-1, 2)
        .astype(np.float32)
        * np.float32(config.square_size_m)
    )
    return points


def detect_checkerboard_observation(
    frame: SynchronizedRGBDFrame,
    config: EyeInHandCalibrationConfig,
) -> tuple[Matrix44, float]:
    """Detect the fixed 10x7 board and return ``T_camera_board`` plus pixel RMS."""

    cv2 = _load_cv2()
    image = np.ascontiguousarray(frame.color_image_rgb)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    board_size = (config.board_columns, config.board_rows)
    corners: NDArray[np.float32] | None = None
    find_sb = getattr(cv2, "findChessboardCornersSB", None)
    if callable(find_sb):
        found, detected = find_sb(
            gray,
            board_size,
            flags=int(cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE),
        )
        if found:
            corners = np.asarray(detected, dtype=np.float32)
    if corners is None:
        found, detected = cv2.findChessboardCorners(
            gray,
            board_size,
            flags=int(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            ),
        )
        if not found:
            raise ValueError("10x7 checkerboard was not detected in the current RGB frame")
        corners = np.asarray(
            cv2.cornerSubPix(
                gray,
                detected,
                (11, 11),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            ),
            dtype=np.float32,
        )
    intrinsics = frame.color_intrinsics
    if (intrinsics.width_px, intrinsics.height_px) != (
        config.image_width_px,
        config.image_height_px,
    ):
        raise ValueError(
            "hand-eye calibration requires the approved "
            f"{config.image_width_px}x{config.image_height_px} RGB profile"
        )
    camera_matrix = np.asarray(
        [
            [intrinsics.fx_px, 0.0, intrinsics.cx_px],
            [0.0, intrinsics.fy_px, intrinsics.cy_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(intrinsics.distortion_coefficients, dtype=np.float64)
    if distortion.size == 0:
        distortion = np.zeros((5, 1), dtype=np.float64)
    object_points = checkerboard_object_points(config)
    solved, rotation_vector, translation = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise ValueError("checkerboard PnP did not converge")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    camera_to_board = np.eye(4, dtype=np.float64)
    camera_to_board[:3, :3] = np.asarray(rotation, dtype=np.float64)
    camera_to_board[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        camera_matrix,
        distortion,
    )
    residual = np.asarray(projected, dtype=np.float64).reshape(-1, 2) - np.asarray(
        corners, dtype=np.float64
    ).reshape(-1, 2)
    rms_px = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    _validate_rigid_matrix(camera_to_board, "camera_to_board")
    return camera_to_board, rms_px


def solve_eye_in_hand(
    observations: list[CheckerboardObservation],
    config: EyeInHandCalibrationConfig,
) -> EyeInHandCalibrationResult:
    """Solve ``T_flange_camera`` and validate the fixed-board reconstruction."""

    if len(observations) < config.minimum_observation_count:
        raise ValueError(
            f"at least {config.minimum_observation_count} valid observations are required"
        )
    cv2 = _load_cv2()
    rotations_gripper_to_base = [item.base_to_flange[:3, :3] for item in observations]
    translations_gripper_to_base = [
        item.base_to_flange[:3, 3].reshape(3, 1) for item in observations
    ]
    rotations_target_to_camera = [item.camera_to_board[:3, :3] for item in observations]
    translations_target_to_camera = [
        item.camera_to_board[:3, 3].reshape(3, 1) for item in observations
    ]
    rotation, translation = cv2.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    flange_to_camera = np.eye(4, dtype=np.float64)
    flange_to_camera[:3, :3] = np.asarray(rotation, dtype=np.float64)
    flange_to_camera[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    _validate_rigid_matrix(flange_to_camera, "flange_to_camera")
    return validate_eye_in_hand_transform(observations, config, flange_to_camera)


def convert_legacy_tcp_camera_mm_to_flange_camera_m(
    tcp_to_camera_mm: Matrix44,
    flange_to_tcp_m: Matrix44,
) -> Matrix44:
    """Convert the tutorial's ``T_tcp_camera`` millimetre matrix to flange/metres."""

    tcp_to_camera = np.asarray(tcp_to_camera_mm, dtype=np.float64).copy()
    flange_to_tcp = np.asarray(flange_to_tcp_m, dtype=np.float64).copy()
    _validate_rigid_matrix(tcp_to_camera, "tcp_to_camera_mm")
    _validate_rigid_matrix(flange_to_tcp, "flange_to_tcp_m")
    tcp_to_camera[:3, 3] /= 1000.0
    flange_to_camera = np.asarray(flange_to_tcp @ tcp_to_camera, dtype=np.float64)
    _validate_rigid_matrix(flange_to_camera, "flange_to_camera_m")
    return flange_to_camera


def validate_eye_in_hand_transform(
    observations: list[CheckerboardObservation],
    config: EyeInHandCalibrationConfig,
    flange_to_camera: Matrix44,
) -> EyeInHandCalibrationResult:
    """Evaluate an existing transform against fixed-board observations without solving."""

    if len(observations) < config.minimum_observation_count:
        raise ValueError(
            f"at least {config.minimum_observation_count} valid observations are required"
        )
    flange_to_camera = np.asarray(flange_to_camera, dtype=np.float64).copy()
    _validate_rigid_matrix(flange_to_camera, "flange_to_camera")

    base_to_boards = [
        item.base_to_flange @ flange_to_camera @ item.camera_to_board
        for item in observations
    ]
    translations = np.asarray([matrix[:3, 3] for matrix in base_to_boards])
    translation_mean = np.mean(translations, axis=0)
    translation_residuals = np.linalg.norm(translations - translation_mean, axis=1)
    rotation_mean = _mean_rotation([matrix[:3, :3] for matrix in base_to_boards])
    rotation_residuals_deg = np.asarray(
        [
            math.degrees(_rotation_angle(rotation_mean.T @ matrix[:3, :3]))
            for matrix in base_to_boards
        ],
        dtype=np.float64,
    )
    base_to_board_mean = np.eye(4, dtype=np.float64)
    base_to_board_mean[:3, :3] = rotation_mean
    base_to_board_mean[:3, 3] = translation_mean

    flange_translations = np.asarray(
        [item.base_to_flange[:3, 3] for item in observations], dtype=np.float64
    )
    translation_span_m = _maximum_pairwise_distance(flange_translations)
    flange_rotations = [item.base_to_flange[:3, :3] for item in observations]
    rotation_span_deg = max(
        (
            math.degrees(_rotation_angle(left.T @ right))
            for index, left in enumerate(flange_rotations)
            for right in flange_rotations[index + 1 :]
        ),
        default=0.0,
    )
    reprojection_values = np.asarray(
        [item.reprojection_rms_px for item in observations], dtype=np.float64
    )
    reprojection_rms_px = float(np.sqrt(np.mean(reprojection_values**2)))
    translation_rms_m = float(np.sqrt(np.mean(translation_residuals**2)))
    rotation_rms_deg = float(np.sqrt(np.mean(rotation_residuals_deg**2)))
    failures: list[str] = []
    checks = (
        (
            reprojection_rms_px <= config.maximum_reprojection_rms_px,
            "overall reprojection RMS exceeds the approved limit",
        ),
        (
            float(np.max(reprojection_values))
            <= config.maximum_single_view_reprojection_rms_px,
            "one or more view reprojection errors exceed the approved limit",
        ),
        (
            translation_rms_m <= config.maximum_translation_rms_m,
            "fixed-board translation RMS exceeds the approved limit",
        ),
        (
            float(np.max(translation_residuals))
            <= config.maximum_translation_residual_m,
            "fixed-board translation residual exceeds the approved limit",
        ),
        (
            rotation_rms_deg <= config.maximum_rotation_rms_deg,
            "fixed-board rotation RMS exceeds the approved limit",
        ),
        (
            float(np.max(rotation_residuals_deg))
            <= config.maximum_rotation_residual_deg,
            "fixed-board rotation residual exceeds the approved limit",
        ),
        (
            translation_span_m >= config.minimum_translation_span_m,
            "robot translations do not sufficiently excite hand-eye calibration",
        ),
        (
            rotation_span_deg >= config.minimum_rotation_span_deg,
            "robot rotations do not sufficiently excite hand-eye calibration",
        ),
    )
    failures.extend(message for passed, message in checks if not passed)
    return EyeInHandCalibrationResult(
        flange_to_camera=flange_to_camera,
        base_to_board_mean=base_to_board_mean,
        observation_count=len(observations),
        reprojection_rms_px=reprojection_rms_px,
        maximum_view_reprojection_rms_px=float(np.max(reprojection_values)),
        translation_rms_m=translation_rms_m,
        maximum_translation_residual_m=float(np.max(translation_residuals)),
        rotation_rms_deg=rotation_rms_deg,
        maximum_rotation_residual_deg=float(np.max(rotation_residuals_deg)),
        translation_span_m=translation_span_m,
        rotation_span_deg=rotation_span_deg,
        passed=not failures,
        failures=tuple(failures),
    )


def _mean_rotation(rotations: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    accumulator = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    left, _, right_transposed = np.linalg.svd(accumulator)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = np.linalg.det(left @ right_transposed)
    return np.asarray(left @ correction @ right_transposed, dtype=np.float64)


def _rotation_angle(rotation: NDArray[np.float64]) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _maximum_pairwise_distance(points: NDArray[np.float64]) -> float:
    return max(
        (
            float(np.linalg.norm(left - right))
            for index, left in enumerate(points)
            for right in points[index + 1 :]
        ),
        default=0.0,
    )


def _validate_rigid_matrix(matrix: Matrix44, label: str) -> None:
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6):
        raise ValueError(f"{label} rotation determinant must be one")
