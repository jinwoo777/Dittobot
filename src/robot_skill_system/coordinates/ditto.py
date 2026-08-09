"""Fail-closed process boundary for the read-only ``ditto_ws`` capture package."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from robot_skill_system.exceptions import HardwareExecutionDenied, NotConfiguredError

_RESULT_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
_STAGE_SUFFIXES = {
    "raw": ".json",
    "smooth": (".json", ".png"),
    "verify": ".png",
}
_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_IMAGE_BYTES = 16 * 1024 * 1024


class DittoCoordinateController:
    """Run exactly one fixed coordinate recorder without a shell or user command input.

    The external package is treated as read-only.  A child-only path adapter redirects
    its legacy ``~/Desktop/Dittobot/ditto_ws`` outputs into an ephemeral directory.
    Only locally validated anchor-relative artifacts are persisted by this application.
    """

    def __init__(
        self,
        *,
        repository_root: Path,
        workspace_root: Path,
        artifact_root: Path,
        hardware_authorized: bool,
        gate_summary: Mapping[str, bool],
        process_factory: Callable[..., subprocess.Popen[str]] | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.workspace_root = workspace_root.resolve()
        if self.workspace_root != self.repository_root / "ditto_ws":
            raise ValueError("ditto workspace must be repo_root/ditto_ws")
        self.package_root = self.workspace_root / "src" / "ditto_system"
        self.output_root = (artifact_root / "ditto_coordinates").resolve()
        self.hardware_authorized = hardware_authorized
        self.gate_summary = dict(gate_summary)
        self._process_factory = process_factory or subprocess.Popen
        self._process: subprocess.Popen[str] | None = None
        self._started_at_ns: int | None = None
        self._temporary_output: tempfile.TemporaryDirectory[str] | None = None
        self._runtime_output_root: Path | None = None
        self._log: deque[str] = deque(maxlen=200)
        self._lock = threading.RLock()

    def _missing_gates(self) -> list[str]:
        return [name for name, passed in self.gate_summary.items() if not passed]

    def _source_problems(self) -> list[str]:
        required = (
            self.package_root / "ditto_system" / "record_trajectory.py",
            self.package_root / "ditto_system" / "smooth_trajectory.py",
            self.package_root / "ditto_system" / "verify_trajectory.py",
            self.package_root / "resource" / "T_gripper2camera.npy",
            self.package_root / "resource" / "mobile_sam.pt",
        )
        return [
            f"missing read-only ditto_ws file: {path}"
            for path in required
            if not path.is_file()
        ]

    def _running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._running_locked()
            return_code = None if self._process is None else self._process.poll()
            source_problems = self._source_problems()
            missing_gates = self._missing_gates()
            return {
                "integration": "ditto_ws.record_trajectory",
                "running": running,
                "ready": not source_problems and not missing_gates,
                "hardware_authorized": self.hardware_authorized,
                "missing_gates": missing_gates,
                "source_problems": source_problems,
                "workspace_root": str(self.workspace_root),
                "external_package_read_only": True,
                "output_root": str(self.output_root),
                "absolute_coordinate_source_is_ephemeral": True,
                "started_at_ns": self._started_at_ns,
                "return_code": return_code,
                "log_tail": list(self._log)[-40:],
            }

    def _child_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        source_root = self.repository_root / "src"
        external_source = self.package_root
        cobot_install = self.repository_root.parents[2] / "install"
        ros_prefix = Path("/opt/ros/humble")

        python_paths = [
            source_root,
            external_source,
            ros_prefix / "lib" / "python3.10" / "site-packages",
            ros_prefix / "local" / "lib" / "python3.10" / "dist-packages",
        ]
        ament_prefixes: list[Path] = [
            self.workspace_root / "install" / "ditto_system",
            ros_prefix,
        ]
        library_paths: list[Path] = [
            ros_prefix / "lib",
            ros_prefix / "lib" / "x86_64-linux-gnu",
        ]
        if cobot_install.is_dir():
            for prefix in sorted(path for path in cobot_install.iterdir() if path.is_dir()):
                ament_prefixes.append(prefix)
                library_paths.append(prefix / "lib")
                python_paths.extend(
                    (
                        prefix / "lib" / "python3.10" / "site-packages",
                        prefix / "local" / "lib" / "python3.10" / "dist-packages",
                    )
                )

        def prepend(name: str, paths: list[Path]) -> None:
            existing = environment.get(name, "")
            values = [str(path) for path in paths if path.is_dir()]
            if existing:
                values.append(existing)
            environment[name] = os.pathsep.join(values)

        prepend("PYTHONPATH", python_paths)
        prepend("AMENT_PREFIX_PATH", ament_prefixes)
        prepend("LD_LIBRARY_PATH", library_paths)
        environment["DITTO_PACKAGE_ROOT"] = str(self.package_root)
        if self._runtime_output_root is None:
            raise RuntimeError("ditto coordinate runtime output is not initialized")
        environment["DITTO_OUTPUT_ROOT"] = str(self._runtime_output_root)
        environment["PYTHONUNBUFFERED"] = "1"
        return environment

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            with self._lock:
                self._log.append(line.rstrip("\n"))

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._running_locked():
                raise ValueError("ditto coordinate capture is already running")
            if not self.hardware_authorized:
                missing = ", ".join(self._missing_gates()) or "hardware authorization"
                raise HardwareExecutionDenied(
                    f"ditto coordinate capture denied; closed gates: {missing}"
                )
            problems = self._source_problems()
            if problems:
                raise NotConfiguredError("; ".join(problems))
            self.output_root.mkdir(parents=True, exist_ok=True)
            if self._temporary_output is not None:
                self._synchronize_latest_artifacts()
                self._temporary_output.cleanup()
            self._temporary_output = tempfile.TemporaryDirectory(
                prefix="robot-skill-system-ditto-coordinates-"
            )
            self._runtime_output_root = Path(self._temporary_output.name).resolve()
            self._log.clear()
            command = [
                sys.executable,
                "-m",
                "robot_skill_system.coordinates.ditto_runner",
            ]
            try:
                process = self._process_factory(
                    command,
                    cwd=self.repository_root,
                    env=self._child_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                raise NotConfiguredError(
                    f"failed to start the fixed ditto coordinate runner: {exc}"
                ) from exc
            self._process = process
            self._started_at_ns = time.time_ns()
            threading.Thread(
                target=self._read_output,
                args=(process,),
                daemon=True,
                name="ditto-coordinate-log",
            ).start()
        time.sleep(0.05)
        status = self.status()
        if not status["running"] and status["return_code"] not in (None, 0):
            detail = "\n".join(status["log_tail"][-8:]) or "runner exited during startup"
            raise NotConfiguredError(detail)
        return status

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return self.status()
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                return self.status()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        return self.status()

    def close(self) -> None:
        self.stop()
        with self._lock:
            self._synchronize_latest_artifacts()
            if self._temporary_output is not None:
                self._temporary_output.cleanup()
                self._temporary_output = None
                self._runtime_output_root = None

    def _stage_file(self, stage: str, filename: str) -> Path:
        suffixes = _STAGE_SUFFIXES.get(stage)
        if suffixes is None or not _RESULT_FILENAME.fullmatch(filename):
            raise ValueError("invalid ditto coordinate result path")
        accepted = (suffixes,) if isinstance(suffixes, str) else suffixes
        if not filename.endswith(accepted):
            raise ValueError("invalid ditto coordinate result extension")
        stage_root = (self.output_root / stage).resolve()
        path = (stage_root / filename).resolve()
        if path.parent != stage_root:
            raise ValueError("ditto coordinate result must stay within its stage")
        return path

    @staticmethod
    def _anchor_relative_payload(
        payload: object,
        *,
        source_filename: str,
        anchor_id: str,
        origin_mm_yaw_deg: tuple[float, float, float, float],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
            raise ValueError("ditto coordinate JSON does not contain a frames array")
        origin_x, origin_y, origin_z, origin_yaw = origin_mm_yaw_deg
        normalized_frames: list[dict[str, Any]] = []
        for index, frame in enumerate(payload["frames"]):
            if not isinstance(frame, dict):
                raise ValueError(f"ditto coordinate frame {index} must be an object")
            try:
                time_s = float(frame["time"])
                x_mm = float(frame["x"])
                y_mm = float(frame["y"])
                z_mm = float(frame["z"])
                yaw_deg = float(frame["yaw"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"ditto coordinate frame {index} is invalid") from exc
            values = (time_s, x_mm, y_mm, z_mm, yaw_deg)
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"ditto coordinate frame {index} contains non-finite values")
            yaw_delta_deg = (yaw_deg - origin_yaw + 180.0) % 360.0 - 180.0
            normalized_frames.append(
                {
                    "time_s": time_s,
                    "position_anchor_m": {
                        "x_m": (x_mm - origin_x) / 1000.0,
                        "y_m": (y_mm - origin_y) / 1000.0,
                        "z_m": (z_mm - origin_z) / 1000.0,
                    },
                    "yaw_anchor_rad": math.radians(yaw_delta_deg),
                    "gripper_state": str(frame.get("gripper") or "unknown"),
                }
            )
        return {
            "schema_version": "1.0",
            "coordinate_source": "ditto_ws_record_trajectory",
            "source_filename": source_filename,
            "executable": False,
            "anchor": {
                "kind": "workspace_region",
                "anchor_id": anchor_id,
                "definition": "first validated raw tool waypoint",
            },
            "units": {
                "time": "seconds",
                "position": "metres",
                "yaw": "radians",
                "quaternion_convention": "xyzw",
            },
            "frames": normalized_frames,
        }

    @staticmethod
    def _read_external_json(path: Path) -> dict[str, Any]:
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise ValueError("ditto coordinate JSON exceeds the local size limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ditto coordinate JSON must be an object")
        return payload

    @staticmethod
    def _origin(payload: Mapping[str, Any]) -> tuple[float, float, float, float]:
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames or not isinstance(frames[0], dict):
            raise ValueError("ditto coordinate JSON has no origin frame")
        first = frames[0]
        try:
            origin = (
                float(first["x"]),
                float(first["y"]),
                float(first["z"]),
                float(first["yaw"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ditto coordinate origin frame is invalid") from exc
        if any(not math.isfinite(value) for value in origin):
            raise ValueError("ditto coordinate origin contains non-finite values")
        return origin

    @staticmethod
    def _persist_if_changed(path: Path, payload: bytes) -> None:
        if path.is_file() and path.read_bytes() == payload:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)

    def _synchronize_latest_artifacts(self) -> None:
        runtime_root = self._runtime_output_root
        if runtime_root is None:
            return
        raw_root = runtime_root / "skills" / "raw"
        raw_files = list(raw_root.glob("*.json")) if raw_root.is_dir() else []
        if not raw_files:
            return
        latest_raw = max(raw_files, key=lambda path: path.stat().st_mtime_ns)
        try:
            raw_payload = self._read_external_json(latest_raw)
            origin = self._origin(raw_payload)
            normalized_raw = self._anchor_relative_payload(
                raw_payload,
                source_filename=latest_raw.name,
                anchor_id=f"ditto_capture_{latest_raw.stem}",
                origin_mm_yaw_deg=origin,
            )
            self._persist_if_changed(
                self._stage_file("raw", latest_raw.name),
                json.dumps(normalized_raw, indent=2, ensure_ascii=False).encode("utf-8"),
            )
            base = latest_raw.stem
            smooth_path = runtime_root / "skills" / "smooth" / f"{base}_smoothed.json"
            if smooth_path.is_file():
                smooth_payload = self._read_external_json(smooth_path)
                normalized_smooth = self._anchor_relative_payload(
                    smooth_payload,
                    source_filename=smooth_path.name,
                    anchor_id=f"ditto_capture_{latest_raw.stem}",
                    origin_mm_yaw_deg=origin,
                )
                self._persist_if_changed(
                    self._stage_file("smooth", smooth_path.name),
                    json.dumps(
                        normalized_smooth, indent=2, ensure_ascii=False
                    ).encode("utf-8"),
                )
            image_sources = (
                (
                    "smooth",
                    runtime_root / "skills" / "smooth" / f"{base}_compare.png",
                ),
                (
                    "verify",
                    runtime_root
                    / "skills"
                    / "verify"
                    / f"{base}_smoothed_verify.png",
                ),
            )
            for stage, source in image_sources:
                if source.is_file() and source.stat().st_size <= _MAX_IMAGE_BYTES:
                    self._persist_if_changed(
                        self._stage_file(stage, source.name), source.read_bytes()
                    )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._log.append(f"artifact synchronization pending: {exc}")

    def latest(self) -> dict[str, Any]:
        with self._lock:
            self._synchronize_latest_artifacts()
        raw_root = self.output_root / "raw"
        raw_files = list(raw_root.glob("*.json")) if raw_root.is_dir() else []
        if not raw_files:
            return {"raw": None, "smooth_json": None, "compare_png": None, "verify_png": None}
        latest_raw = max(raw_files, key=lambda path: path.stat().st_mtime_ns)
        base = latest_raw.stem

        def entry(stage: str, filename: str) -> dict[str, Any]:
            path = self._stage_file(stage, filename)
            return {
                "stage": stage,
                "filename": filename,
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }

        smooth_name = f"{base}_smoothed.json"
        return {
            "raw": entry("raw", latest_raw.name),
            "smooth_json": entry("smooth", smooth_name),
            "compare_png": entry("smooth", f"{base}_compare.png"),
            "verify_png": entry("verify", f"{base}_smoothed_verify.png"),
        }

    def read_json_result(self, stage: str, filename: str) -> dict[str, Any]:
        path = self._stage_file(stage, filename)
        if not path.is_file():
            raise KeyError(filename)
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise ValueError("ditto coordinate JSON exceeds the local size limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
            raise ValueError("ditto coordinate JSON does not contain a frames array")
        return payload

    def read_image_result(self, stage: str, filename: str) -> bytes:
        path = self._stage_file(stage, filename)
        if not path.is_file():
            raise KeyError(filename)
        if path.stat().st_size > _MAX_IMAGE_BYTES:
            raise ValueError("ditto coordinate image exceeds the local size limit")
        return path.read_bytes()

    def read_live_frame(self) -> bytes:
        runtime_root = self._runtime_output_root
        if runtime_root is None:
            raise KeyError("coords_frame.jpg")
        path = runtime_root / ".runtime" / "coords_frame.jpg"
        if not path.is_file():
            raise KeyError("coords_frame.jpg")
        if path.stat().st_size > _MAX_IMAGE_BYTES:
            raise ValueError("ditto live frame exceeds the local size limit")
        return path.read_bytes()
