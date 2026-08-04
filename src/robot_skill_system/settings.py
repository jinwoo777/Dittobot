"""Environment-backed application settings without import-time side effects."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class OpenAIMode(str, Enum):
    """Available semantic-orchestration modes."""

    MOCK = "mock"
    LIVE = "live"


class ExecutionMode(str, Enum):
    """Robot execution backends."""

    MOCK = "mock"
    DRY_RUN = "dry_run"
    SIMULATION = "simulation"
    HARDWARE = "hardware"


class CaptureMode(str, Enum):
    """RGB-D acquisition modes."""

    SINGLE = "single"
    BURST = "burst"
    CONTINUOUS = "continuous"


def repository_root() -> Path:
    """Return the repository root derived from this installed source tree."""

    return Path(__file__).resolve().parents[2]


def _bool_value(value: str | bool | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


class Settings(BaseModel):
    """Validated runtime configuration loaded explicitly from environment variables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_root: Path
    openai_api_key: SecretStr | None = None
    openai_reasoning_model: str = "gpt-5.6"
    openai_transcribe_model: str = "gpt-transcribe"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_timeout_seconds: float = Field(default=300.0, gt=0, le=600)
    openai_api_max_retries: int = Field(default=3, ge=0, le=10)
    openai_max_keyframes: int = Field(default=300, ge=1, le=300)
    openai_image_detail: Literal["auto", "low", "high"] = "auto"
    stt_domain_prompt: str = (
        "Doosan M0609, RG2, MoveJ, MoveL, MoveC, Force Control, 힘 제어, 걸레질, "
        "그리퍼, 작업공간, 표면"
    )
    openai_mode: OpenAIMode = OpenAIMode.MOCK
    robot_execution_mode: ExecutionMode = ExecutionMode.MOCK
    enable_hardware_execution: bool = False
    robot_backend: Literal["mock", "simulation", "doosan"] = "mock"
    enable_real_robot: bool = False
    dry_run: bool = True
    enable_handeye_calibration: bool = False
    calibration_pose_plan_approved: bool = False
    calibration_cell_safety_verified: bool = False
    doosan_robot_id: str = Field(default="dsr01", pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
    doosan_robot_model: Literal["m0609"] = "m0609"
    handeye_legacy_npy_path: Path | None = None
    handeye_legacy_expected_tcp: str = Field(default="2FG_TCP", min_length=1, max_length=64)
    scene_capture_mode: CaptureMode = CaptureMode.BURST
    scene_burst_frame_count: int = Field(default=5, ge=1, le=30)
    rgbd_max_timestamp_delta_ms: float = Field(default=20.0, gt=0, le=1_000)
    pose_inference_fps: int = Field(default=10, ge=1, le=30)
    realsense_device_serial: str | None = None
    realsense_width_px: int = Field(default=640, ge=1, le=4096)
    realsense_height_px: int = Field(default=480, ge=1, le=2160)
    realsense_frames_per_second: int = Field(default=30, ge=1, le=90)
    realsense_recording_frames_per_second: int = Field(default=10, ge=1, le=90)
    realsense_maximum_recording_duration_s: float = Field(default=60.0, ge=1.0, le=600.0)
    scene_freshness_ms: int = Field(default=5_000, ge=1)
    database_url: str
    artifact_root: Path
    log_level: str = "INFO"

    @field_validator("repo_root", "artifact_root")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("paths must be absolute after configuration resolution")
        return value

    @field_validator("handeye_legacy_npy_path")
    @classmethod
    def _optional_absolute_path(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("configured legacy NPY path must be absolute after resolution")
        return value

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("unsupported log level")
        return normalized

    @model_validator(mode="after")
    def _recording_rate_not_above_capture_rate(self) -> Settings:
        if self.realsense_recording_frames_per_second > self.realsense_frames_per_second:
            raise ValueError(
                "REALSENSE_RECORDING_FRAMES_PER_SECOND cannot exceed "
                "REALSENSE_FRAMES_PER_SECOND"
            )
        return self

    @model_validator(mode="after")
    def _legacy_npy_stays_within_managed_roots(self) -> Settings:
        path = self.handeye_legacy_npy_path
        if path is None:
            return self
        if not (
            path == self.repo_root
            or path.is_relative_to(self.repo_root)
            or path == self.artifact_root
            or path.is_relative_to(self.artifact_root)
        ):
            raise ValueError("HANDEYE_LEGACY_NPY_PATH must stay under repo or artifact root")
        return self

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        root: Path | None = None,
    ) -> Settings:
        """Build settings from *environ*, resolving defaults relative to the repository."""

        env = os.environ if environ is None else environ
        resolved_root = (root or repository_root()).resolve()
        database_url = env.get("DATABASE_URL") or (
            f"sqlite:///{(resolved_root / 'data' / 'robot_skills.db').as_posix()}"
        )
        artifact_text = env.get("ARTIFACT_ROOT")
        artifact_root = (
            Path(artifact_text).expanduser() if artifact_text else resolved_root / "data"
        )
        if not artifact_root.is_absolute():
            artifact_root = resolved_root / artifact_root
        legacy_npy_text = env.get(
            "HANDEYE_LEGACY_NPY_PATH", "T_gripper2camera.npy"
        ).strip()
        legacy_npy_path = Path(legacy_npy_text).expanduser() if legacy_npy_text else None
        if legacy_npy_path is not None and not legacy_npy_path.is_absolute():
            legacy_npy_path = resolved_root / legacy_npy_path
        raw_key = env.get("OPENAI_API_KEY", "").strip()
        image_detail = env.get("OPENAI_IMAGE_DETAIL", "auto").lower()
        if image_detail not in {"auto", "low", "high"}:
            raise ValueError("OPENAI_IMAGE_DETAIL must be auto, low, or high")
        robot_backend = env.get("ROBOT_BACKEND", "mock").lower()
        if robot_backend not in {"mock", "simulation", "doosan"}:
            raise ValueError("ROBOT_BACKEND must be mock, simulation, or doosan")
        return cls(
            repo_root=resolved_root,
            openai_api_key=SecretStr(raw_key) if raw_key else None,
            openai_reasoning_model=env.get(
                "OPENAI_REASONING_MODEL",
                env.get("OPENAI_MULTIMODAL_MODEL", "gpt-5.6"),
            ),
            openai_transcribe_model=env.get("OPENAI_TRANSCRIBE_MODEL", "gpt-transcribe"),
            openai_embedding_model=env.get(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            openai_api_timeout_seconds=float(
                env.get("OPENAI_API_TIMEOUT_SECONDS", env.get("OPENAI_REQUEST_TIMEOUT_S", "300"))
            ),
            openai_api_max_retries=int(
                env.get("OPENAI_API_MAX_RETRIES", env.get("OPENAI_MAX_RETRIES", "3"))
            ),
            openai_max_keyframes=int(env.get("OPENAI_MAX_KEYFRAMES", "300")),
            openai_image_detail=cast(
                Literal["auto", "low", "high"], image_detail
            ),
            stt_domain_prompt=env.get(
                "STT_DOMAIN_PROMPT",
                "Doosan M0609, RG2, MoveJ, MoveL, MoveC, Force Control, 힘 제어, "
                "걸레질, 그리퍼, 작업공간, 표면",
            ),
            openai_mode=OpenAIMode(env.get("OPENAI_MODE", "mock").lower()),
            robot_execution_mode=ExecutionMode(
                env.get("ROBOT_EXECUTION_MODE", "mock").lower()
            ),
            enable_hardware_execution=_bool_value(
                env.get("ENABLE_HARDWARE_EXECUTION"), default=False
            ),
            robot_backend=cast(
                Literal["mock", "simulation", "doosan"], robot_backend
            ),
            enable_real_robot=_bool_value(env.get("ENABLE_REAL_ROBOT"), default=False),
            dry_run=_bool_value(env.get("DRY_RUN"), default=True),
            enable_handeye_calibration=_bool_value(
                env.get("ENABLE_HANDEYE_CALIBRATION"), default=False
            ),
            calibration_pose_plan_approved=_bool_value(
                env.get("CALIBRATION_POSE_PLAN_APPROVED"), default=False
            ),
            calibration_cell_safety_verified=_bool_value(
                env.get("CALIBRATION_CELL_SAFETY_VERIFIED"), default=False
            ),
            doosan_robot_id=env.get("DOOSAN_ROBOT_ID", "dsr01"),
            doosan_robot_model=cast(
                Literal["m0609"], env.get("DOOSAN_ROBOT_MODEL", "m0609").lower()
            ),
            handeye_legacy_npy_path=(
                legacy_npy_path.resolve() if legacy_npy_path is not None else None
            ),
            handeye_legacy_expected_tcp=env.get(
                "HANDEYE_LEGACY_EXPECTED_TCP", "2FG_TCP"
            ),
            scene_capture_mode=CaptureMode(env.get("SCENE_CAPTURE_MODE", "burst").lower()),
            scene_burst_frame_count=int(env.get("SCENE_BURST_FRAME_COUNT", "5")),
            rgbd_max_timestamp_delta_ms=float(
                env.get("RGBD_MAX_TIMESTAMP_DELTA_MS", "20")
            ),
            pose_inference_fps=int(env.get("POSE_INFERENCE_FPS", "10")),
            realsense_device_serial=(
                env.get("REALSENSE_DEVICE_SERIAL", "").strip() or None
            ),
            realsense_width_px=int(env.get("REALSENSE_WIDTH_PX", "640")),
            realsense_height_px=int(env.get("REALSENSE_HEIGHT_PX", "480")),
            realsense_frames_per_second=int(
                env.get("REALSENSE_FRAMES_PER_SECOND", "30")
            ),
            realsense_recording_frames_per_second=int(
                env.get("REALSENSE_RECORDING_FRAMES_PER_SECOND", "10")
            ),
            realsense_maximum_recording_duration_s=float(
                env.get("REALSENSE_MAXIMUM_RECORDING_DURATION_S", "60")
            ),
            scene_freshness_ms=int(env.get("SCENE_FRESHNESS_MS", "5000")),
            database_url=database_url,
            artifact_root=artifact_root.resolve(),
            log_level=env.get("LOG_LEVEL", "INFO"),
        )

    def safe_summary(self) -> dict[str, object]:
        """Return log-safe configuration data with credentials omitted."""

        return {
            "repo_root": str(self.repo_root),
            "openai_mode": self.openai_mode.value,
            "openai_reasoning_model": self.openai_reasoning_model,
            "robot_execution_mode": self.robot_execution_mode.value,
            "enable_hardware_execution": self.enable_hardware_execution,
            "robot_backend": self.robot_backend,
            "enable_real_robot": self.enable_real_robot,
            "dry_run": self.dry_run,
            "enable_handeye_calibration": self.enable_handeye_calibration,
            "calibration_pose_plan_approved": self.calibration_pose_plan_approved,
            "calibration_cell_safety_verified": self.calibration_cell_safety_verified,
            "doosan_robot_id": self.doosan_robot_id,
            "doosan_robot_model": self.doosan_robot_model,
            "handeye_legacy_npy_configured": self.handeye_legacy_npy_path is not None,
            "handeye_legacy_expected_tcp": self.handeye_legacy_expected_tcp,
            "scene_capture_mode": self.scene_capture_mode.value,
            "realsense_device_serial": self.realsense_device_serial,
            "realsense_resolution": (
                self.realsense_width_px,
                self.realsense_height_px,
            ),
            "realsense_frames_per_second": self.realsense_frames_per_second,
            "realsense_recording_frames_per_second": (
                self.realsense_recording_frames_per_second
            ),
            "realsense_maximum_recording_duration_s": (
                self.realsense_maximum_recording_duration_s
            ),
            "artifact_root": str(self.artifact_root),
        }

    @property
    def hardware_enabled(self) -> bool:
        """Return true only when both legacy and explicit hardware gates are open."""

        return (
            self.robot_execution_mode is ExecutionMode.HARDWARE
            and self.enable_hardware_execution
            and self.robot_backend == "doosan"
            and self.enable_real_robot
            and not self.dry_run
        )

    @property
    def handeye_hardware_enabled(self) -> bool:
        """Require cell-commissioning acknowledgements in addition to hardware gates."""

        return (
            self.hardware_enabled
            and self.enable_handeye_calibration
            and self.calibration_pose_plan_approved
            and self.calibration_cell_safety_verified
        )
