"""RGB-D capture protocols, mock backend, and optional RealSense adapter."""

from .image_sequence import (
    ArrayImageSequenceAdapter,
    ImageSequenceCapture,
    NumpyDirectorySequenceLoader,
    RGBDImageSequenceLoader,
)
from .interfaces import (
    CameraExtrinsics,
    CameraIntrinsics,
    CaptureError,
    CaptureMode,
    CaptureRequest,
    CaptureResult,
    NotConfiguredError,
    RGBDCapture,
    RGBDFrame,
    SynchronizedRGBDFrame,
)
from .mock_capture import MockCapture, MockCaptureConfig, MockSceneFrame
from .realsense_capture import (
    HOST_UNIX_EPOCH_CLOCK_DOMAIN,
    RealSenseCapture,
    RealSenseCaptureConfig,
    RealSenseTimestampMapper,
)
from .sequence import (
    InferenceSamplingConfig,
    RGBDSequence,
    SequenceMetadata,
    select_inference_frames,
    select_keyframes,
    sequence_from_frames,
)

__all__ = [
    "ArrayImageSequenceAdapter",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "CaptureError",
    "CaptureMode",
    "CaptureRequest",
    "CaptureResult",
    "ImageSequenceCapture",
    "HOST_UNIX_EPOCH_CLOCK_DOMAIN",
    "InferenceSamplingConfig",
    "MockCapture",
    "MockCaptureConfig",
    "MockSceneFrame",
    "NumpyDirectorySequenceLoader",
    "NotConfiguredError",
    "RGBDCapture",
    "RGBDFrame",
    "RGBDImageSequenceLoader",
    "RGBDSequence",
    "RealSenseCapture",
    "RealSenseCaptureConfig",
    "RealSenseTimestampMapper",
    "SequenceMetadata",
    "SynchronizedRGBDFrame",
    "select_inference_frames",
    "select_keyframes",
    "sequence_from_frames",
]
