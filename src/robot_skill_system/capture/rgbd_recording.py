"""Threaded local RGB-D preview and bounded recording service."""

from __future__ import annotations

import io
import json
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from robot_skill_system.storage.artifact_store import LocalArtifactStore

from .interfaces import (
    CameraIntrinsics,
    NotConfiguredError,
    RGBDCapture,
    SynchronizedRGBDFrame,
)

CameraState = Literal["stopped", "starting", "streaming", "error"]
PreviewKind = Literal["rgb", "depth"]
_RECORDING_ID_PATTERN = re.compile(r"^rgbd_[A-Za-z0-9_-]{1,96}$")


class CameraStateError(RuntimeError):
    """A camera lifecycle request is incompatible with the current state."""


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise NotConfiguredError(
            "RGB-D preview/recording requires the optional OpenCV dependency"
        ) from exc
    return cv2


def _jpeg_bytes(image_bgr: NDArray[np.uint8], *, quality: int = 88) -> bytes:
    cv2 = _load_cv2()
    encoded_ok, encoded = cv2.imencode(
        ".jpg",
        np.ascontiguousarray(image_bgr),
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not encoded_ok:
        raise RuntimeError("OpenCV could not encode a JPEG preview frame")
    return bytes(encoded)


def encode_rgb_jpeg(image_rgb: NDArray[np.uint8]) -> bytes:
    """Encode one owned RGB array for a browser MJPEG stream or artifact."""

    cv2 = _load_cv2()
    return _jpeg_bytes(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def encode_depth_jpeg(
    depth_image_m: NDArray[np.float32], *, display_maximum_m: float = 3.0
) -> bytes:
    """Create a human-readable depth colormap without changing raw depth artifacts."""

    if display_maximum_m <= 0.0:
        raise ValueError("display_maximum_m must be positive")
    cv2 = _load_cv2()
    valid = np.isfinite(depth_image_m) & (depth_image_m > 0.0)
    normalized = np.zeros(depth_image_m.shape, dtype=np.uint8)
    normalized[valid] = np.asarray(
        np.clip(depth_image_m[valid] / display_maximum_m, 0.0, 1.0) * 255.0,
        dtype=np.uint8,
    )
    # Near pixels are warm and invalid pixels remain black.
    inverted = np.where(valid, 255 - normalized, 0).astype(np.uint8, copy=False)
    colorized = cv2.applyColorMap(inverted, cv2.COLORMAP_TURBO)
    colorized[~valid] = 0
    return _jpeg_bytes(colorized)


class RGBDRecordingWriter:
    """Persist owned RGB-D frames asynchronously beneath one demonstration URI."""

    def __init__(
        self,
        store: LocalArtifactStore,
        *,
        recording_id: str,
        frames_per_second: float,
        recording_frames_per_second: float | None = None,
        maximum_duration_s: float,
        monotonic_clock_ns: Callable[[], int] = time.monotonic_ns,
        epoch_clock_ns: Callable[[], int] = time.time_ns,
        queue_capacity: int = 30,
    ) -> None:
        if not recording_id or recording_id != recording_id.strip():
            raise ValueError("recording_id must be non-empty")
        effective_recording_fps = recording_frames_per_second or frames_per_second
        if frames_per_second <= 0.0 or effective_recording_fps <= 0.0:
            raise ValueError("recording rate and duration must be positive")
        if effective_recording_fps > frames_per_second:
            raise ValueError("recording rate cannot exceed the raw capture rate")
        if maximum_duration_s <= 0.0:
            raise ValueError("recording rate and duration must be positive")
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self.store = store
        self.recording_id = recording_id
        self.frames_per_second = frames_per_second
        self.recording_frames_per_second = effective_recording_fps
        self.maximum_duration_s = maximum_duration_s
        self._monotonic_clock_ns = monotonic_clock_ns
        self._epoch_clock_ns = epoch_clock_ns
        self._queue: queue.Queue[SynchronizedRGBDFrame] = queue.Queue(queue_capacity)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"rgbd-writer-{recording_id}",
            daemon=True,
        )
        self._started_monotonic_ns = monotonic_clock_ns()
        self._recording_period_ns = max(
            1, int(round(1.0e9 / self.recording_frames_per_second))
        )
        self._next_sample_monotonic_ns = self._started_monotonic_ns
        self._started_at_ns = epoch_clock_ns()
        self._ended_at_ns: int | None = None
        self._accepting = True
        self._finalized = False
        self._requested_status = "recording"
        self._error: str | None = None
        self._dropped_frames = 0
        self._entries: list[dict[str, Any]] = []
        self._manifest_uri: str | None = None
        self._manifest_checksum_sha256: str | None = None
        self._thread.start()

    def submit(self, frame: SynchronizedRGBDFrame) -> bool:
        """Queue one owned frame; return false when the bounded recording must stop."""

        with self._lock:
            now_ns = self._monotonic_clock_ns()
            elapsed_ns = now_ns - self._started_monotonic_ns
            expired = elapsed_ns >= int(self.maximum_duration_s * 1.0e9)
            if not self._accepting or self._error is not None or expired:
                return False
            if now_ns < self._next_sample_monotonic_ns:
                return True
            elapsed_periods = (
                (now_ns - self._next_sample_monotonic_ns)
                // self._recording_period_ns
            ) + 1
            self._next_sample_monotonic_ns += (
                elapsed_periods * self._recording_period_ns
            )
        owned = replace(
            frame,
            color_image_rgb=frame.color_image_rgb.copy(),
            depth_image_m=frame.depth_image_m.copy(),
        )
        try:
            self._queue.put_nowait(owned)
        except queue.Full:
            with self._lock:
                self._dropped_frames += 1
        return True

    def stop(self, *, status: Literal["finished", "failed"] = "finished") -> dict[str, Any]:
        with self._lock:
            if self._finalized:
                return self.summary()
            self._accepting = False
            self._requested_status = status
            self._stop_event.set()
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            with self._lock:
                self._error = "recording writer did not stop within 30 seconds"
        with self._lock:
            self._ended_at_ns = self._epoch_clock_ns()
            self._finalize_locked()
            return self.summary()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            status = (
                "failed"
                if self._error is not None
                else self._requested_status
                if self._finalized
                else "recording"
            )
            duration_s = (
                ((self._ended_at_ns or self._epoch_clock_ns()) - self._started_at_ns)
                / 1.0e9
            )
            return {
                "recording_id": self.recording_id,
                "status": status,
                "frame_count": len(self._entries),
                "dropped_frame_count": self._dropped_frames,
                "started_at_ns": self._started_at_ns,
                "ended_at_ns": self._ended_at_ns,
                "duration_s": max(0.0, duration_s),
                "raw_capture_fps": self.frames_per_second,
                "recording_fps": self.recording_frames_per_second,
                "maximum_duration_s": self.maximum_duration_s,
                "timestamps_preserved": True,
                "rgb_format": "jpeg",
                "depth_format": "numpy_npz_float32_metres",
                "depth_aligned_to_color": True,
                "manifest_uri": self._manifest_uri,
                "manifest_checksum_sha256": self._manifest_checksum_sha256,
                "error": self._error,
            }

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._write_frame(frame)
            except Exception as exc:  # pragma: no cover - defensive filesystem boundary
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._accepting = False
                    self._stop_event.set()
                break
            finally:
                self._queue.task_done()

    def _write_frame(self, frame: SynchronizedRGBDFrame) -> None:
        with self._lock:
            index = len(self._entries)
        prefix = f"demonstrations/{self.recording_id}"
        rgb = self.store.put_bytes(
            f"{prefix}/rgb/{index:06d}.jpg",
            encode_rgb_jpeg(frame.color_image_rgb),
            media_type="image/jpeg",
        )
        depth_buffer = io.BytesIO()
        np.savez_compressed(depth_buffer, depth_m=frame.depth_image_m)
        depth = self.store.put_bytes(
            f"{prefix}/depth/{index:06d}.npz",
            depth_buffer.getvalue(),
            media_type="application/x-npz",
        )
        entry = {
            "index": index,
            "frame_number": frame.frame_number,
            "timestamp_ns": frame.timestamp_ns,
            "color_timestamp_ns": frame.color_timestamp_ns,
            "depth_timestamp_ns": frame.depth_timestamp_ns,
            "timestamp_clock_domain": frame.timestamp_clock_domain,
            "raw_color_timestamp_ns": frame.raw_color_timestamp_ns,
            "raw_depth_timestamp_ns": frame.raw_depth_timestamp_ns,
            "raw_color_timestamp_clock_domain": frame.raw_color_timestamp_clock_domain,
            "raw_depth_timestamp_clock_domain": frame.raw_depth_timestamp_clock_domain,
            "reference_frame": frame.reference_frame,
            "depth_scale_m": frame.depth_scale_m,
            "rgb_uri": rgb.uri,
            "rgb_checksum_sha256": rgb.checksum_sha256,
            "depth_uri": depth.uri,
            "depth_checksum_sha256": depth.checksum_sha256,
            "color_intrinsics": {
                "width_px": frame.color_intrinsics.width_px,
                "height_px": frame.color_intrinsics.height_px,
                "fx_px": frame.color_intrinsics.fx_px,
                "fy_px": frame.color_intrinsics.fy_px,
                "cx_px": frame.color_intrinsics.cx_px,
                "cy_px": frame.color_intrinsics.cy_px,
                "distortion_model": frame.color_intrinsics.distortion_model,
                "distortion_coefficients": frame.color_intrinsics.distortion_coefficients,
            },
        }
        with self._lock:
            self._entries.append(entry)

    def _finalize_locked(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        try:
            manifest = {
                **self.summary(),
                "schema_version": "1.0",
                "frames": list(self._entries),
            }
            artifact = self.store.put_json(
                f"demonstrations/{self.recording_id}/rgbd_manifest.json", manifest
            )
            self._manifest_uri = artifact.uri
            self._manifest_checksum_sha256 = artifact.checksum_sha256
        except Exception:
            self._finalized = False
            raise


class RGBDCameraController:
    """Own one RGB-D device, browser previews, and at most one recording."""

    def __init__(
        self,
        capture_factory: Callable[[], RGBDCapture],
        store: LocalArtifactStore,
        *,
        frames_per_second: float = 30.0,
        recording_frames_per_second: float = 10.0,
        maximum_recording_duration_s: float = 300.0,
        startup_timeout_s: float = 10.0,
    ) -> None:
        if frames_per_second <= 0.0 or recording_frames_per_second <= 0.0:
            raise ValueError("camera rate and recording duration must be positive")
        if recording_frames_per_second > frames_per_second:
            raise ValueError("recording rate cannot exceed the camera rate")
        if maximum_recording_duration_s <= 0.0:
            raise ValueError("camera rate and recording duration must be positive")
        self._capture_factory = capture_factory
        self._store = store
        self._frames_per_second = frames_per_second
        self._recording_frames_per_second = recording_frames_per_second
        self._maximum_recording_duration_s = maximum_recording_duration_s
        self._startup_timeout_s = startup_timeout_s
        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: CameraState = "stopped"
        self._last_error: str | None = None
        self._latest_rgb_jpeg: bytes | None = None
        self._latest_depth_jpeg: bytes | None = None
        self._latest_frame: SynchronizedRGBDFrame | None = None
        self._latest_frame_number: int | None = None
        self._latest_timestamp_ns: int | None = None
        self._preview_sequence = 0
        self._recording: RGBDRecordingWriter | None = None
        self._recordings: dict[str, dict[str, Any]] = {}

    def start_preview(self) -> dict[str, Any]:
        with self._condition:
            if self._state == "streaming":
                return self.status()
            if self._thread is not None and self._thread.is_alive():
                raise CameraStateError("RealSense camera is already starting")
            self._state = "starting"
            self._last_error = None
            self._latest_rgb_jpeg = None
            self._latest_depth_jpeg = None
            self._latest_frame = None
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_capture,
                name="realsense-rgbd-preview",
                daemon=True,
            )
            self._thread.start()
            ready = self._condition.wait_for(
                lambda: self._state in {"streaming", "error", "stopped"},
                timeout=self._startup_timeout_s,
            )
            if not ready:
                self._stop_event.set()
                self._state = "error"
                self._last_error = "RealSense did not produce a frame before startup timeout"
                raise CameraStateError("RealSense did not produce a frame before startup timeout")
            if self.status()["state"] != "streaming":
                raise CameraStateError(self._last_error or "RealSense preview could not start")
            return self.status()

    def stop_preview(self) -> dict[str, Any]:
        with self._condition:
            self._stop_event.set()
            thread = self._thread
            recording = self._recording
        if recording is not None:
            self._finalize_recording(recording)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._startup_timeout_s + 2.0)
            if thread.is_alive():
                raise CameraStateError("RealSense preview thread did not stop")
        with self._condition:
            self._state = "stopped"
            self._thread = None
            self._condition.notify_all()
            return self.status()

    def start_recording(self, *, maximum_duration_s: float) -> dict[str, Any]:
        if not 1.0 <= maximum_duration_s <= self._maximum_recording_duration_s:
            raise ValueError(
                f"maximum_duration_s must be in [1, {self._maximum_recording_duration_s:g}]"
            )
        with self._condition:
            if self._state != "streaming":
                raise CameraStateError("start RealSense preview before recording")
            if self._recording is not None:
                raise CameraStateError("an RGB-D recording is already active")
            recording = RGBDRecordingWriter(
                self._store,
                recording_id=f"rgbd_{uuid.uuid4().hex}",
                frames_per_second=self._frames_per_second,
                recording_frames_per_second=self._recording_frames_per_second,
                maximum_duration_s=maximum_duration_s,
            )
            self._recording = recording
            return recording.summary()

    def stop_recording(self, recording_id: str) -> dict[str, Any]:
        with self._condition:
            recording = self._recording
            if recording is None or recording.recording_id != recording_id:
                known = self._recordings.get(recording_id)
                if known is None:
                    raise KeyError(f"unknown RGB-D recording {recording_id!r}")
                return dict(known)
        return self._finalize_recording(recording)

    def get_recording(self, recording_id: str) -> dict[str, Any]:
        with self._condition:
            if self._recording is not None and self._recording.recording_id == recording_id:
                return self._recording.summary()
            known = self._recordings.get(recording_id)
            if known is not None:
                return dict(known)
        return self.get_recording_manifest(recording_id, include_frames=False)

    def list_recordings(self) -> dict[str, Any]:
        """List finalized recordings discovered from checksum-bearing manifests on disk."""

        root = self._store.root / "demonstrations"
        recordings: list[dict[str, Any]] = []
        invalid_recording_count = 0
        if root.is_dir():
            for manifest_path in root.glob("rgbd_*/rgbd_manifest.json"):
                try:
                    manifest = self._load_recording_manifest(manifest_path.parent.name)
                except (KeyError, OSError, ValueError):
                    invalid_recording_count += 1
                    continue
                recordings.append(self._recording_summary(manifest))
        recordings.sort(
            key=lambda item: int(item.get("started_at_ns") or 0), reverse=True
        )
        return {
            "recordings": recordings,
            "invalid_recording_count": invalid_recording_count,
        }

    def get_recording_manifest(
        self, recording_id: str, *, include_frames: bool = True
    ) -> dict[str, Any]:
        manifest = self._load_recording_manifest(recording_id)
        return dict(manifest) if include_frames else self._recording_summary(manifest)

    def get_recording_frame_jpeg(
        self,
        recording_id: str,
        frame_index: int,
        kind: PreviewKind,
    ) -> bytes:
        """Load one persisted RGB frame or render one persisted depth frame as JPEG."""

        if kind not in {"rgb", "depth"}:
            raise ValueError("recording frame kind must be rgb or depth")
        frame = self._recording_frame(recording_id, frame_index)
        if kind == "rgb":
            return self._read_frame_artifact(
                recording_id,
                frame,
                uri_field="rgb_uri",
                checksum_field="rgb_checksum_sha256",
                directory="rgb",
            )
        payload = self._read_frame_artifact(
            recording_id,
            frame,
            uri_field="depth_uri",
            checksum_field="depth_checksum_sha256",
            directory="depth",
        )
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {"depth_m"}:
                raise ValueError("depth artifact must contain only the depth_m array")
            depth_image_m = np.asarray(archive["depth_m"], dtype=np.float32)
        if depth_image_m.ndim != 2 or depth_image_m.size > 4096 * 2160:
            raise ValueError("depth artifact has an invalid image shape")
        return encode_depth_jpeg(depth_image_m)

    def load_recording_rgbd_frame(
        self, recording_id: str, frame_index: int
    ) -> SynchronizedRGBDFrame:
        """Reconstruct one checksum-verified recorded frame for local geometry."""

        frame = self._recording_frame(recording_id, frame_index)
        rgb_payload = self._read_frame_artifact(
            recording_id,
            frame,
            uri_field="rgb_uri",
            checksum_field="rgb_checksum_sha256",
            directory="rgb",
        )
        depth_payload = self._read_frame_artifact(
            recording_id,
            frame,
            uri_field="depth_uri",
            checksum_field="depth_checksum_sha256",
            directory="depth",
        )
        cv2 = _load_cv2()
        decoded_bgr = cv2.imdecode(
            np.frombuffer(rgb_payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if decoded_bgr is None:
            raise ValueError("recorded RGB artifact is not a valid JPEG")
        color_image_rgb = np.asarray(
            cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )
        with np.load(io.BytesIO(depth_payload), allow_pickle=False) as archive:
            if set(archive.files) != {"depth_m"}:
                raise ValueError("depth artifact must contain only the depth_m array")
            depth_image_m = np.asarray(archive["depth_m"], dtype=np.float32)
        intrinsics_payload = frame.get("color_intrinsics")
        if not isinstance(intrinsics_payload, dict):
            raise ValueError("recorded frame has no colour intrinsics")
        intrinsics = CameraIntrinsics(
            width_px=int(intrinsics_payload["width_px"]),
            height_px=int(intrinsics_payload["height_px"]),
            fx_px=float(intrinsics_payload["fx_px"]),
            fy_px=float(intrinsics_payload["fy_px"]),
            cx_px=float(intrinsics_payload["cx_px"]),
            cy_px=float(intrinsics_payload["cy_px"]),
            distortion_model=str(intrinsics_payload.get("distortion_model") or "none"),
            distortion_coefficients=tuple(
                float(value)
                for value in intrinsics_payload.get("distortion_coefficients") or ()
            ),
        )
        return SynchronizedRGBDFrame(
            color_image_rgb=color_image_rgb,
            depth_image_m=depth_image_m,
            color_timestamp_ns=int(frame["color_timestamp_ns"]),
            depth_timestamp_ns=int(frame["depth_timestamp_ns"]),
            color_intrinsics=intrinsics,
            frame_number=int(frame["frame_number"]),
            reference_frame=str(frame.get("reference_frame") or "camera_color_optical_frame"),
            depth_scale_m=float(frame.get("depth_scale_m") or 1.0),
            aligned_depth_to_color=True,
            timestamp_clock_domain=str(frame.get("timestamp_clock_domain") or "source_clock"),
            raw_color_timestamp_ns=(
                int(frame["raw_color_timestamp_ns"])
                if frame.get("raw_color_timestamp_ns") is not None
                else None
            ),
            raw_depth_timestamp_ns=(
                int(frame["raw_depth_timestamp_ns"])
                if frame.get("raw_depth_timestamp_ns") is not None
                else None
            ),
            raw_color_timestamp_clock_domain=(
                str(frame["raw_color_timestamp_clock_domain"])
                if frame.get("raw_color_timestamp_clock_domain") is not None
                else None
            ),
            raw_depth_timestamp_clock_domain=(
                str(frame["raw_depth_timestamp_clock_domain"])
                if frame.get("raw_depth_timestamp_clock_domain") is not None
                else None
            ),
        )

    def select_recording_keyframes(
        self, recording_id: str, maximum_count: int
    ) -> list[tuple[int, Path]]:
        """Select evenly spaced, checksum-verified RGB keyframes in chronological order."""

        if maximum_count < 1:
            raise ValueError("maximum_count must be positive")
        manifest = self._load_recording_manifest(recording_id)
        frames = manifest["frames"]
        frame_count = len(frames)
        if frame_count == 0:
            raise ValueError("RGB-D recording contains no frames")
        selected_count = min(maximum_count, frame_count)
        if selected_count == 1:
            indices = [frame_count // 2]
        else:
            indices = sorted(
                {
                    round(position * (frame_count - 1) / (selected_count - 1))
                    for position in range(selected_count)
                }
            )
        result: list[tuple[int, Path]] = []
        for index in indices:
            frame = self._validated_frame(frames[index], index)
            self._read_frame_artifact(
                recording_id,
                frame,
                uri_field="rgb_uri",
                checksum_field="rgb_checksum_sha256",
                directory="rgb",
            )
            result.append((index, self._store.path_for(str(frame["rgb_uri"]))))
        return result

    def select_recording_rgbd_keyframes(
        self, recording_id: str, maximum_count: int
    ) -> list[tuple[int, Path, Path]]:
        """Select aligned RGB and rendered depth pairs with checksum verification."""

        if maximum_count < 1:
            raise ValueError("maximum_count must be positive")
        manifest = self._load_recording_manifest(recording_id)
        frames = manifest["frames"]
        frame_count = len(frames)
        if frame_count == 0:
            raise ValueError("RGB-D recording contains no frames")
        selected_count = min(maximum_count, frame_count)
        if selected_count == 1:
            indices = [frame_count // 2]
        else:
            indices = sorted(
                {
                    round(position * (frame_count - 1) / (selected_count - 1))
                    for position in range(selected_count)
                }
            )
        result: list[tuple[int, Path, Path]] = []
        for index in indices:
            frame = self._validated_frame(frames[index], index)
            self._read_frame_artifact(
                recording_id,
                frame,
                uri_field="rgb_uri",
                checksum_field="rgb_checksum_sha256",
                directory="rgb",
            )
            depth_payload = self._read_frame_artifact(
                recording_id,
                frame,
                uri_field="depth_uri",
                checksum_field="depth_checksum_sha256",
                directory="depth",
            )
            with np.load(io.BytesIO(depth_payload), allow_pickle=False) as archive:
                if set(archive.files) != {"depth_m"}:
                    raise ValueError("depth artifact must contain only the depth_m array")
                depth_image_m = np.asarray(archive["depth_m"], dtype=np.float32)
            if depth_image_m.ndim != 2 or depth_image_m.size > 4096 * 2160:
                raise ValueError("depth artifact has an invalid image shape")
            depth_preview = self._store.put_bytes(
                f"demonstrations/{recording_id}/analysis_depth_v1/{index:06d}.jpg",
                encode_depth_jpeg(depth_image_m),
                media_type="image/jpeg",
            )
            result.append(
                (
                    index,
                    self._store.path_for(str(frame["rgb_uri"])),
                    self._store.path_for(depth_preview.uri),
                )
            )
        return result

    def status(self) -> dict[str, Any]:
        with self._condition:
            recording = self._recording.summary() if self._recording is not None else None
            return {
                "backend": "realsense",
                "state": self._state,
                "preview_ready": self._latest_rgb_jpeg is not None,
                "frame_number": self._latest_frame_number,
                "timestamp_ns": self._latest_timestamp_ns,
                "frames_per_second": self._frames_per_second,
                "recording_frames_per_second": self._recording_frames_per_second,
                "rgb_stream_uri": "/camera/streams/rgb.mjpg",
                "depth_stream_uri": "/camera/streams/depth.mjpg",
                "recording": recording,
                "last_error": self._last_error,
            }

    def get_latest_frame(
        self,
        *,
        after_timestamp_ns: int | None = None,
        timeout_s: float = 2.0,
    ) -> SynchronizedRGBDFrame:
        """Return an owned RGB-D frame, optionally newer than a robot settle timestamp."""

        if timeout_s <= 0.0 or timeout_s > 30.0:
            raise ValueError("camera snapshot timeout_s must be in (0, 30]")
        with self._condition:
            if self._state != "streaming":
                raise CameraStateError("start RealSense preview before calibration")
            ready = self._condition.wait_for(
                lambda: (
                    self._latest_frame is not None
                    and (
                        after_timestamp_ns is None
                        or self._latest_frame.timestamp_ns > after_timestamp_ns
                    )
                )
                or self._state != "streaming",
                timeout=timeout_s,
            )
            if not ready or self._latest_frame is None:
                raise CameraStateError("RealSense did not provide a fresh calibration frame")
            if self._state != "streaming":
                raise CameraStateError("RealSense stopped before calibration capture")
            frame = self._latest_frame
            return replace(
                frame,
                color_image_rgb=frame.color_image_rgb.copy(),
                depth_image_m=frame.depth_image_m.copy(),
            )

    def iter_mjpeg(self, kind: PreviewKind) -> Iterator[bytes]:
        if kind not in {"rgb", "depth"}:
            raise ValueError("preview kind must be rgb or depth")
        with self._condition:
            if self._state != "streaming" or self._latest_rgb_jpeg is None:
                raise CameraStateError("RealSense preview is not streaming")
        last_sequence = -1
        while True:
            with self._condition:
                self._condition.wait_for(
                    partial(self._preview_changed, last_sequence),
                    timeout=2.0,
                )
                if self._preview_sequence == last_sequence and self._state != "streaming":
                    return
                payload = (
                    self._latest_rgb_jpeg if kind == "rgb" else self._latest_depth_jpeg
                )
                last_sequence = self._preview_sequence
            if payload is None:
                if self._state != "streaming":
                    return
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                + payload
                + b"\r\n"
            )

    def close(self) -> None:
        with suppress(CameraStateError):
            self.stop_preview()

    def _preview_changed(self, previous_sequence: int) -> bool:
        return self._preview_sequence != previous_sequence or self._state != "streaming"

    @staticmethod
    def _validate_recording_id(recording_id: str) -> str:
        if not _RECORDING_ID_PATTERN.fullmatch(recording_id):
            raise ValueError("recording_id has an invalid format")
        return recording_id

    def _load_recording_manifest(self, recording_id: str) -> dict[str, Any]:
        safe_id = self._validate_recording_id(recording_id)
        manifest_uri = f"demonstrations/{safe_id}/rgbd_manifest.json"
        path = self._store.path_for(manifest_uri)
        if not path.is_file():
            raise KeyError(f"unknown RGB-D recording {recording_id!r}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("recording_id") != safe_id:
            raise ValueError("RGB-D manifest recording_id does not match its directory")
        frames = payload.get("frames")
        if not isinstance(frames, list):
            raise ValueError("RGB-D manifest frames must be a list")
        if len(frames) > 600 * 90:
            raise ValueError("RGB-D manifest exceeds the supported frame bound")
        normalized = dict(payload)
        normalized["frame_count"] = len(frames)
        normalized["recording_fps"] = float(
            payload.get("recording_fps") or payload.get("raw_capture_fps") or 1.0
        )
        normalized["manifest_uri"] = manifest_uri
        normalized["frames"] = frames
        return normalized

    @staticmethod
    def _recording_summary(manifest: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in manifest.items() if key != "frames"}

    def _recording_frame(self, recording_id: str, frame_index: int) -> dict[str, Any]:
        manifest = self._load_recording_manifest(recording_id)
        frames = manifest["frames"]
        if frame_index < 0 or frame_index >= len(frames):
            raise KeyError(
                f"unknown frame {frame_index} for RGB-D recording {recording_id!r}"
            )
        return self._validated_frame(frames[frame_index], frame_index)

    @staticmethod
    def _validated_frame(value: object, expected_index: int) -> dict[str, Any]:
        if not isinstance(value, dict) or value.get("index") != expected_index:
            raise ValueError("RGB-D manifest frame index is invalid")
        return value

    def _read_frame_artifact(
        self,
        recording_id: str,
        frame: dict[str, Any],
        *,
        uri_field: str,
        checksum_field: str,
        directory: str,
    ) -> bytes:
        uri = frame.get(uri_field)
        checksum = frame.get(checksum_field)
        prefix = f"demonstrations/{recording_id}/{directory}/"
        if not isinstance(uri, str) or not uri.startswith(prefix):
            raise ValueError(f"RGB-D frame {uri_field} is outside its recording directory")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"RGB-D frame {checksum_field} is invalid")
        return self._store.read_bytes(uri, expected_checksum_sha256=checksum)

    def _run_capture(self) -> None:
        capture: RGBDCapture | None = None
        failure: str | None = None
        try:
            capture = self._capture_factory()
            capture.start()
            frame_period_s = 1.0 / self._frames_per_second
            next_frame_not_before = time.monotonic()
            for frame in capture.stream():
                if self._stop_event.is_set():
                    break
                now = time.monotonic()
                if now < next_frame_not_before and self._stop_event.wait(
                    next_frame_not_before - now
                ):
                    break
                next_frame_not_before = max(
                    next_frame_not_before + frame_period_s,
                    time.monotonic(),
                )
                rgb_jpeg = encode_rgb_jpeg(frame.color_image_rgb)
                depth_jpeg = encode_depth_jpeg(frame.depth_image_m)
                owned_frame = replace(
                    frame,
                    color_image_rgb=frame.color_image_rgb.copy(),
                    depth_image_m=frame.depth_image_m.copy(),
                )
                with self._condition:
                    self._latest_rgb_jpeg = rgb_jpeg
                    self._latest_depth_jpeg = depth_jpeg
                    self._latest_frame = owned_frame
                    self._latest_frame_number = frame.frame_number
                    self._latest_timestamp_ns = frame.timestamp_ns
                    self._preview_sequence += 1
                    self._state = "streaming"
                    recording = self._recording
                    self._condition.notify_all()
                if recording is not None and not recording.submit(frame):
                    self._finish_recording_if_current(recording)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            if capture is not None:
                try:
                    capture.stop()
                except Exception as exc:  # pragma: no cover - hardware cleanup boundary
                    failure = failure or f"{type(exc).__name__}: {exc}"
            with self._condition:
                recording = self._recording
            if recording is not None:
                self._finalize_recording(
                    recording,
                    status="failed" if failure else "finished",
                )
            with self._condition:
                self._last_error = failure
                self._state = "error" if failure else "stopped"
                self._condition.notify_all()

    def _finish_recording_if_current(self, recording: RGBDRecordingWriter) -> None:
        with self._condition:
            if self._recording is not recording:
                return
        self._finalize_recording(recording)

    def _finalize_recording(
        self,
        recording: RGBDRecordingWriter,
        *,
        status: Literal["finished", "failed"] = "finished",
    ) -> dict[str, Any]:
        summary = recording.stop(status=status)
        with self._condition:
            if self._recording is recording:
                self._recording = None
            self._recordings[str(summary["recording_id"])] = dict(summary)
            self._condition.notify_all()
        return summary

__all__ = [
    "CameraStateError",
    "RGBDCameraController",
    "RGBDRecordingWriter",
    "encode_depth_jpeg",
    "encode_rgb_jpeg",
]
