"""Checksum-verified, read-only replay of relocated RGB-D demonstrations.

Recorded manifests retain artifact-store URIs such as
``demonstrations/rgbd_<id>/rgb/000001.jpg``.  Operators commonly relocate the
complete recording directory and give it a semantic name (for example
``hammer1``).  This module deliberately does not follow those stale URIs.
Instead, each manifest index is mapped to the fixed local paths
``rgb/<index>.jpg`` and ``depth/<index>.npz`` beneath the supplied directory,
and both files are verified against the manifest checksums before use.

No data is written, and importing this module does not import optional image or
MediaPipe dependencies.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from robot_skill_system.capture.interfaces import (
    CameraIntrinsics,
    NotConfiguredError,
    SynchronizedRGBDFrame,
)
from robot_skill_system.perception.hand_pose import (
    FingerObservation,
    FingerStateTransition,
    FingerTrackingResult,
    MediaPipeHandPoseEstimator,
)

from .stage_segmentation import (
    GripActionEndSegmentationConfig,
    GripActionEndSegmentationResult,
    segment_grip_action_end,
)

_RECORDING_ID_PATTERN = re.compile(r"^rgbd_[A-Za-z0-9_-]{1,96}$")
_CHECKSUM_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_INDEX_FILE_PATTERN = re.compile(r"^[0-9]{6}$")
_MAXIMUM_MANIFEST_BYTES = 32 * 1024 * 1024
_MAXIMUM_FRAME_COUNT = 600 * 90
_MAXIMUM_RGB_BYTES = 64 * 1024 * 1024
_MAXIMUM_DEPTH_BYTES = 128 * 1024 * 1024
_MAXIMUM_IMAGE_PIXELS = 4096 * 2160
_MAXIMUM_TIMESTAMP_SKEW_NS = 20_000_000

RGBImageDecoder = Callable[[bytes], NDArray[np.uint8]]


class RGBDDatasetValidationError(ValueError):
    """A local demonstration or its manifest failed closed validation."""


class _FingerTracker(Protocol):
    def track(self, frame: SynchronizedRGBDFrame) -> FingerTrackingResult:
        """Return local fingertip evidence for one frame."""

        ...


@dataclass(frozen=True, slots=True)
class RGBDFrameArtifact:
    """Validated metadata and fixed local paths for one synchronized frame."""

    index: int
    frame_number: int
    timestamp_ns: int
    color_timestamp_ns: int
    depth_timestamp_ns: int
    timestamp_clock_domain: str
    raw_color_timestamp_ns: int | None
    raw_depth_timestamp_ns: int | None
    raw_color_timestamp_clock_domain: str | None
    raw_depth_timestamp_clock_domain: str | None
    reference_frame: str
    depth_scale_m: float
    color_intrinsics: CameraIntrinsics
    rgb_path: Path
    rgb_checksum_sha256: str
    depth_path: Path
    depth_checksum_sha256: str
    recorded_rgb_uri: str
    recorded_depth_uri: str


@dataclass(frozen=True, slots=True)
class RGBDDatasetManifest:
    """Small immutable summary of a validated recording manifest."""

    schema_version: str
    recording_id: str
    frame_count: int
    recording_fps: float
    manifest_path: Path
    manifest_checksum_sha256: str


@dataclass(frozen=True, slots=True)
class MediaPipeDatasetDiagnostic:
    """Configuration-only diagnostic; it never imports or initializes MediaPipe."""

    configured: bool
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class RGBDDatasetSegmentation:
    """Auditable output of persistent local tracking followed by segmentation."""

    manifest_checksum_sha256: str
    observations: tuple[FingerObservation, ...]
    transitions: tuple[FingerStateTransition, ...]
    segmentation: GripActionEndSegmentationResult


class RGBDDataset:
    """A validated local RGB-D dataset that re-verifies bytes during replay."""

    def __init__(
        self,
        manifest: RGBDDatasetManifest,
        frames: tuple[RGBDFrameArtifact, ...],
        *,
        rgb_decoder: RGBImageDecoder | None = None,
    ) -> None:
        if not frames:
            raise RGBDDatasetValidationError("RGB-D dataset contains no frames")
        if manifest.frame_count != len(frames):
            raise RGBDDatasetValidationError("manifest frame count does not match its index")
        self.manifest = manifest
        self.frames = frames
        self._rgb_decoder = rgb_decoder or _decode_rgb_jpeg

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[SynchronizedRGBDFrame]:
        return self.iter_frames()

    def iter_frames(self) -> Iterator[SynchronizedRGBDFrame]:
        """Yield chronological frames after per-use checksum verification."""

        for index in range(len(self.frames)):
            yield self.load_frame(index)

    def load_frame(self, index: int) -> SynchronizedRGBDFrame:
        """Reconstruct one owned frame without trusting a manifest URI as a path."""

        if isinstance(index, bool) or not 0 <= index < len(self.frames):
            raise IndexError(f"RGB-D frame index {index!r} is outside the dataset")
        artifact = self.frames[index]
        rgb_payload = _read_checksum_verified(
            artifact.rgb_path,
            artifact.rgb_checksum_sha256,
            maximum_bytes=_MAXIMUM_RGB_BYTES,
            label=f"RGB frame {index}",
        )
        depth_payload = _read_checksum_verified(
            artifact.depth_path,
            artifact.depth_checksum_sha256,
            maximum_bytes=_MAXIMUM_DEPTH_BYTES,
            label=f"depth frame {index}",
        )
        color_image_rgb = np.asarray(self._rgb_decoder(rgb_payload))
        expected_shape = (
            artifact.color_intrinsics.height_px,
            artifact.color_intrinsics.width_px,
            3,
        )
        if color_image_rgb.dtype != np.uint8 or color_image_rgb.shape != expected_shape:
            raise RGBDDatasetValidationError(
                f"RGB frame {index} must decode to uint8 shape {expected_shape}"
            )
        color_image_rgb = np.ascontiguousarray(color_image_rgb, dtype=np.uint8)
        depth_image_m = _decode_depth_npz(
            depth_payload,
            expected_shape=expected_shape[:2],
            frame_index=index,
        )
        return SynchronizedRGBDFrame(
            color_image_rgb=color_image_rgb,
            depth_image_m=depth_image_m,
            color_timestamp_ns=artifact.color_timestamp_ns,
            depth_timestamp_ns=artifact.depth_timestamp_ns,
            color_intrinsics=artifact.color_intrinsics,
            frame_number=artifact.frame_number,
            reference_frame=artifact.reference_frame,
            depth_scale_m=artifact.depth_scale_m,
            aligned_depth_to_color=True,
            maximum_timestamp_skew_ns=_MAXIMUM_TIMESTAMP_SKEW_NS,
            timestamp_clock_domain=artifact.timestamp_clock_domain,
            raw_color_timestamp_ns=artifact.raw_color_timestamp_ns,
            raw_depth_timestamp_ns=artifact.raw_depth_timestamp_ns,
            raw_color_timestamp_clock_domain=(
                artifact.raw_color_timestamp_clock_domain
            ),
            raw_depth_timestamp_clock_domain=(
                artifact.raw_depth_timestamp_clock_domain
            ),
        )


def _decode_rgb_jpeg(payload: bytes) -> NDArray[np.uint8]:
    """Decode JPEG bytes lazily with OpenCV, then Pillow as a local fallback."""

    if not payload.startswith(b"\xff\xd8"):
        raise RGBDDatasetValidationError("RGB artifact is not a JPEG stream")
    cv2_module: Any | None
    try:
        import cv2
    except ImportError:
        cv2_module = None
    else:
        cv2_module = cv2
    if cv2_module is not None:
        decoded_bgr = cv2_module.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2_module.IMREAD_COLOR
        )
        if decoded_bgr is None:
            raise RGBDDatasetValidationError("RGB artifact is not a valid JPEG")
        return np.asarray(
            cv2_module.cvtColor(decoded_bgr, cv2_module.COLOR_BGR2RGB),
            dtype=np.uint8,
        )

    try:
        from PIL import Image
    except ImportError as exc:
        raise NotConfiguredError(
            "RGB-D dataset replay requires optional OpenCV or Pillow JPEG decoding"
        ) from exc
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "JPEG":
                raise RGBDDatasetValidationError("RGB artifact is not a JPEG image")
            width_px, height_px = image.size
            if width_px * height_px > _MAXIMUM_IMAGE_PIXELS:
                raise RGBDDatasetValidationError("RGB image exceeds the supported pixel bound")
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except RGBDDatasetValidationError:
        raise
    except Exception as exc:
        raise RGBDDatasetValidationError("RGB artifact is not a valid JPEG") from exc


def _decode_depth_npz(
    payload: bytes,
    *,
    expected_shape: tuple[int, int],
    frame_index: int,
) -> NDArray[np.float32]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {"depth_m"}:
                raise RGBDDatasetValidationError(
                    f"depth frame {frame_index} must contain only depth_m"
                )
            raw_depth = archive["depth_m"]
            if raw_depth.dtype != np.float32:
                raise RGBDDatasetValidationError(
                    f"depth frame {frame_index} must use float32 metres"
                )
            if raw_depth.shape != expected_shape or raw_depth.size > _MAXIMUM_IMAGE_PIXELS:
                raise RGBDDatasetValidationError(
                    f"depth frame {frame_index} does not match colour dimensions"
                )
            depth_image_m = np.ascontiguousarray(raw_depth, dtype=np.float32)
    except RGBDDatasetValidationError:
        raise
    except Exception as exc:
        raise RGBDDatasetValidationError(
            f"depth frame {frame_index} is not a valid safe NumPy archive"
        ) from exc
    return depth_image_m


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RGBDDatasetValidationError(f"manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise RGBDDatasetValidationError(f"manifest contains non-finite JSON value {value}")


def _load_manifest_payload(manifest_path: Path) -> tuple[dict[str, Any], str]:
    try:
        size = manifest_path.stat().st_size
    except OSError as exc:
        raise RGBDDatasetValidationError("RGB-D manifest is not readable") from exc
    if size <= 0 or size > _MAXIMUM_MANIFEST_BYTES:
        raise RGBDDatasetValidationError("RGB-D manifest has an invalid file size")
    try:
        manifest_bytes = manifest_path.read_bytes()
        decoded = manifest_bytes.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except RGBDDatasetValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RGBDDatasetValidationError("RGB-D manifest is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise RGBDDatasetValidationError("RGB-D manifest root must be an object")
    return cast(dict[str, Any], payload), hashlib.sha256(manifest_bytes).hexdigest()


def _require_string(
    payload: Mapping[str, Any], key: str, *, pattern: re.Pattern[str] | None = None
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RGBDDatasetValidationError(f"manifest field {key!r} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise RGBDDatasetValidationError(f"manifest field {key!r} has an invalid format")
    return value


def _require_integer(payload: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RGBDDatasetValidationError(
            f"manifest field {key!r} must be an integer >= {minimum}"
        )
    return value


def _optional_integer(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return _require_integer(payload, key)


def _require_finite_number(
    payload: Mapping[str, Any], key: str, *, positive: bool = False
) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RGBDDatasetValidationError(f"manifest field {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise RGBDDatasetValidationError(f"manifest field {key!r} must be {qualifier}")
    return result


def _optional_clock_pair(
    payload: Mapping[str, Any], timestamp_key: str, domain_key: str
) -> tuple[int | None, str | None]:
    timestamp = _optional_integer(payload, timestamp_key)
    domain_value = payload.get(domain_key)
    if domain_value is None:
        domain = None
    elif (
        isinstance(domain_value, str)
        and domain_value.strip()
        and domain_value == domain_value.strip()
    ):
        domain = domain_value
    else:
        raise RGBDDatasetValidationError(
            f"manifest field {domain_key!r} must be a non-empty string or null"
        )
    if (timestamp is None) != (domain is None):
        raise RGBDDatasetValidationError(
            f"manifest fields {timestamp_key!r} and {domain_key!r} must occur together"
        )
    return timestamp, domain


def _camera_intrinsics(payload: Mapping[str, Any]) -> CameraIntrinsics:
    value = payload.get("color_intrinsics")
    if not isinstance(value, dict):
        raise RGBDDatasetValidationError("each frame requires color_intrinsics")
    width_px = _require_integer(value, "width_px", minimum=1)
    height_px = _require_integer(value, "height_px", minimum=1)
    if width_px * height_px > _MAXIMUM_IMAGE_PIXELS:
        raise RGBDDatasetValidationError("camera dimensions exceed the supported pixel bound")
    fx_px = _require_finite_number(value, "fx_px", positive=True)
    fy_px = _require_finite_number(value, "fy_px", positive=True)
    cx_px = _require_finite_number(value, "cx_px")
    cy_px = _require_finite_number(value, "cy_px")
    distortion_model = value.get("distortion_model", "none")
    if not isinstance(distortion_model, str) or not distortion_model.strip():
        raise RGBDDatasetValidationError("distortion_model must be a non-empty string")
    coefficients_value = value.get("distortion_coefficients", ())
    if not isinstance(coefficients_value, (list, tuple)) or len(coefficients_value) > 32:
        raise RGBDDatasetValidationError("distortion_coefficients must be a bounded sequence")
    coefficients: list[float] = []
    for coefficient in coefficients_value:
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
            raise RGBDDatasetValidationError("distortion coefficients must be numeric")
        numeric = float(coefficient)
        if not math.isfinite(numeric):
            raise RGBDDatasetValidationError("distortion coefficients must be finite")
        coefficients.append(numeric)
    return CameraIntrinsics(
        width_px=width_px,
        height_px=height_px,
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=cx_px,
        cy_px=cy_px,
        distortion_model=distortion_model,
        distortion_coefficients=tuple(coefficients),
    )


def _validate_recorded_uri(uri: str, *, directory: str, expected_name: str) -> None:
    if "\\" in uri or "://" in uri:
        raise RGBDDatasetValidationError("recorded frame URI must be a relative artifact URI")
    path = PurePosixPath(uri)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RGBDDatasetValidationError("recorded frame URI contains an unsafe path")
    if len(path.parts) < 2 or path.parts[-2:] != (directory, expected_name):
        raise RGBDDatasetValidationError(
            f"recorded frame URI must end with {directory}/{expected_name}"
        )


def _resolved_index_path(root: Path, directory: str, name: str) -> Path:
    candidate = root / directory / name
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RGBDDatasetValidationError(
            f"missing relocated RGB-D artifact {directory}/{name}"
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RGBDDatasetValidationError("RGB-D artifact resolves outside its dataset") from exc
    if not resolved.is_file():
        raise RGBDDatasetValidationError("RGB-D artifact must be a regular file")
    return resolved


def _read_checksum_verified(
    path: Path,
    expected_checksum_sha256: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    try:
        with path.open("rb") as handle:
            payload = handle.read(maximum_bytes + 1)
    except OSError as exc:
        raise RGBDDatasetValidationError(f"{label} is not readable") from exc
    if not payload or len(payload) > maximum_bytes:
        raise RGBDDatasetValidationError(f"{label} has an invalid file size")
    actual_checksum = hashlib.sha256(payload).hexdigest()
    if actual_checksum != expected_checksum_sha256:
        raise RGBDDatasetValidationError(f"{label} checksum does not match the manifest")
    return payload


def _frame_artifact(
    root: Path,
    payload: object,
    expected_index: int,
) -> RGBDFrameArtifact:
    if not isinstance(payload, dict):
        raise RGBDDatasetValidationError(f"frame {expected_index} must be an object")
    index = _require_integer(payload, "index")
    if index != expected_index:
        raise RGBDDatasetValidationError("manifest frame indices must be contiguous from zero")
    expected_stem = f"{index:06d}"
    if _INDEX_FILE_PATTERN.fullmatch(expected_stem) is None:
        raise RGBDDatasetValidationError("frame index exceeds the six-digit local index format")
    rgb_name = expected_stem + ".jpg"
    depth_name = expected_stem + ".npz"
    recorded_rgb_uri = _require_string(payload, "rgb_uri")
    recorded_depth_uri = _require_string(payload, "depth_uri")
    _validate_recorded_uri(recorded_rgb_uri, directory="rgb", expected_name=rgb_name)
    _validate_recorded_uri(recorded_depth_uri, directory="depth", expected_name=depth_name)
    rgb_checksum = _require_string(payload, "rgb_checksum_sha256", pattern=_CHECKSUM_PATTERN)
    depth_checksum = _require_string(
        payload, "depth_checksum_sha256", pattern=_CHECKSUM_PATTERN
    )
    color_timestamp_ns = _require_integer(payload, "color_timestamp_ns")
    depth_timestamp_ns = _require_integer(payload, "depth_timestamp_ns")
    if abs(color_timestamp_ns - depth_timestamp_ns) > _MAXIMUM_TIMESTAMP_SKEW_NS:
        raise RGBDDatasetValidationError(
            f"frame {index} RGB/depth timestamps exceed the synchronization bound"
        )
    timestamp_ns = _require_integer(payload, "timestamp_ns")
    if timestamp_ns != (color_timestamp_ns + depth_timestamp_ns) // 2:
        raise RGBDDatasetValidationError(f"frame {index} timestamp_ns is not the pair midpoint")
    raw_color_timestamp_ns, raw_color_domain = _optional_clock_pair(
        payload,
        "raw_color_timestamp_ns",
        "raw_color_timestamp_clock_domain",
    )
    raw_depth_timestamp_ns, raw_depth_domain = _optional_clock_pair(
        payload,
        "raw_depth_timestamp_ns",
        "raw_depth_timestamp_clock_domain",
    )
    return RGBDFrameArtifact(
        index=index,
        frame_number=_require_integer(payload, "frame_number"),
        timestamp_ns=timestamp_ns,
        color_timestamp_ns=color_timestamp_ns,
        depth_timestamp_ns=depth_timestamp_ns,
        timestamp_clock_domain=_require_string(payload, "timestamp_clock_domain"),
        raw_color_timestamp_ns=raw_color_timestamp_ns,
        raw_depth_timestamp_ns=raw_depth_timestamp_ns,
        raw_color_timestamp_clock_domain=raw_color_domain,
        raw_depth_timestamp_clock_domain=raw_depth_domain,
        reference_frame=_require_string(payload, "reference_frame"),
        depth_scale_m=_require_finite_number(payload, "depth_scale_m", positive=True),
        color_intrinsics=_camera_intrinsics(payload),
        rgb_path=_resolved_index_path(root, "rgb", rgb_name),
        rgb_checksum_sha256=rgb_checksum,
        depth_path=_resolved_index_path(root, "depth", depth_name),
        depth_checksum_sha256=depth_checksum,
        recorded_rgb_uri=recorded_rgb_uri,
        recorded_depth_uri=recorded_depth_uri,
    )


def load_rgbd_dataset(
    directory: Path | str,
    *,
    rgb_decoder: RGBImageDecoder | None = None,
) -> RGBDDataset:
    """Validate a relocated local dataset and return a read-only replay object.

    All frame artifacts are checksum-verified eagerly.  They are verified again
    when decoded so a file changed after initial validation also fails closed.
    """

    candidate = Path(directory)
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise RGBDDatasetValidationError("RGB-D dataset directory does not exist") from exc
    if not root.is_dir():
        raise RGBDDatasetValidationError("RGB-D dataset path must be a directory")
    manifest_path = root / "rgbd_manifest.json"
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
        resolved_manifest.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RGBDDatasetValidationError("RGB-D manifest must be inside the dataset") from exc
    if not resolved_manifest.is_file():
        raise RGBDDatasetValidationError("RGB-D manifest must be a regular file")

    payload, manifest_checksum = _load_manifest_payload(resolved_manifest)
    schema_version = _require_string(payload, "schema_version")
    if schema_version != "1.0":
        raise RGBDDatasetValidationError(f"unsupported RGB-D schema version {schema_version!r}")
    recording_id = _require_string(payload, "recording_id", pattern=_RECORDING_ID_PATTERN)
    if payload.get("status") != "finished" or payload.get("error") is not None:
        raise RGBDDatasetValidationError("only successfully finalized RGB-D recordings can load")
    if payload.get("depth_aligned_to_color") is not True:
        raise RGBDDatasetValidationError("RGB-D recording depth must be aligned to colour")
    if payload.get("timestamps_preserved") is not True:
        raise RGBDDatasetValidationError("RGB-D recording must preserve source timestamps")
    if payload.get("rgb_format") != "jpeg":
        raise RGBDDatasetValidationError("RGB-D recording must contain JPEG RGB artifacts")
    if payload.get("depth_format") != "numpy_npz_float32_metres":
        raise RGBDDatasetValidationError("RGB-D recording must contain float32 metre depth")
    recording_fps = _require_finite_number(payload, "recording_fps", positive=True)
    frames_payload = payload.get("frames")
    if not isinstance(frames_payload, list):
        raise RGBDDatasetValidationError("RGB-D manifest frames must be a list")
    if not 1 <= len(frames_payload) <= _MAXIMUM_FRAME_COUNT:
        raise RGBDDatasetValidationError("RGB-D frame count is outside the supported bound")
    declared_frame_count = _require_integer(payload, "frame_count", minimum=1)
    if declared_frame_count != len(frames_payload):
        raise RGBDDatasetValidationError("declared RGB-D frame count does not match frames")

    frames = tuple(
        _frame_artifact(root, frame_payload, index)
        for index, frame_payload in enumerate(frames_payload)
    )
    for previous, current in zip(frames, frames[1:], strict=False):
        if current.frame_number <= previous.frame_number:
            raise RGBDDatasetValidationError("RGB-D frame numbers must be strictly increasing")
        if current.color_timestamp_ns <= previous.color_timestamp_ns:
            raise RGBDDatasetValidationError("RGB timestamps must be strictly increasing")
        if current.depth_timestamp_ns <= previous.depth_timestamp_ns:
            raise RGBDDatasetValidationError("depth timestamps must be strictly increasing")
        if current.timestamp_ns <= previous.timestamp_ns:
            raise RGBDDatasetValidationError("frame timestamps must be strictly increasing")
        if current.color_intrinsics != previous.color_intrinsics:
            raise RGBDDatasetValidationError("colour intrinsics changed within one recording")

    for frame in frames:
        _read_checksum_verified(
            frame.rgb_path,
            frame.rgb_checksum_sha256,
            maximum_bytes=_MAXIMUM_RGB_BYTES,
            label=f"RGB frame {frame.index}",
        )
        _read_checksum_verified(
            frame.depth_path,
            frame.depth_checksum_sha256,
            maximum_bytes=_MAXIMUM_DEPTH_BYTES,
            label=f"depth frame {frame.index}",
        )

    manifest = RGBDDatasetManifest(
        schema_version=schema_version,
        recording_id=recording_id,
        frame_count=len(frames),
        recording_fps=recording_fps,
        manifest_path=resolved_manifest,
        manifest_checksum_sha256=manifest_checksum,
    )
    return RGBDDataset(manifest, frames, rgb_decoder=rgb_decoder)


def mediapipe_dataset_diagnostic() -> MediaPipeDatasetDiagnostic:
    """Report whether the fixed optional MediaPipe dependency can be imported."""

    if importlib.util.find_spec("mediapipe") is None:
        return MediaPipeDatasetDiagnostic(
            configured=False,
            code="mediapipe_not_configured",
            message=(
                "MediaPipe is not installed; RGB-D bytes can be validated and replayed, "
                "but automatic Grip/Action/End segmentation cannot run"
            ),
        )
    return MediaPipeDatasetDiagnostic(
        configured=True,
        code="mediapipe_available",
        message="MediaPipe is available for persistent local RGB-D hand tracking",
    )


def segment_rgbd_dataset(
    dataset_or_directory: RGBDDataset | Path | str,
    *,
    config: GripActionEndSegmentationConfig | None = None,
    estimator: _FingerTracker | None = None,
) -> RGBDDatasetSegmentation:
    """Track every frame locally, then split it into Grip, Action, and End.

    When ``estimator`` is omitted, exactly one persistent
    :class:`MediaPipeHandPoseEstimator` is owned for the complete recording.
    The optional dependency failure is converted to an explicit configuration
    error after the dataset itself has already passed integrity validation.
    """

    dataset = (
        dataset_or_directory
        if isinstance(dataset_or_directory, RGBDDataset)
        else load_rgbd_dataset(dataset_or_directory)
    )
    effective_config = config or GripActionEndSegmentationConfig()
    owns_estimator = estimator is None
    tracker: _FingerTracker = estimator or MediaPipeHandPoseEstimator(
        close_threshold_m=effective_config.finger_close_threshold_m,
        stable_frames=effective_config.finger_state_stable_frames,
    )
    observations: list[FingerObservation] = []
    transitions: list[FingerStateTransition] = []
    try:
        for frame in dataset:
            try:
                tracking = tracker.track(frame)
            except NotConfiguredError as exc:
                diagnostic = mediapipe_dataset_diagnostic()
                raise NotConfiguredError(
                    f"{diagnostic.message}; dataset checksum and frame artifacts are valid"
                ) from exc
            observations.append(tracking.observation)
            if tracking.transition is not None:
                transitions.append(tracking.transition)
    finally:
        if owns_estimator:
            close = getattr(tracker, "close", None)
            if callable(close):
                close()
    segmentation = segment_grip_action_end(observations, config=effective_config)
    return RGBDDatasetSegmentation(
        manifest_checksum_sha256=dataset.manifest.manifest_checksum_sha256,
        observations=tuple(observations),
        transitions=tuple(transitions),
        segmentation=segmentation,
    )


__all__ = [
    "MediaPipeDatasetDiagnostic",
    "RGBDDataset",
    "RGBDDatasetManifest",
    "RGBDDatasetSegmentation",
    "RGBDDatasetValidationError",
    "RGBDFrameArtifact",
    "load_rgbd_dataset",
    "mediapipe_dataset_diagnostic",
    "segment_rgbd_dataset",
]
