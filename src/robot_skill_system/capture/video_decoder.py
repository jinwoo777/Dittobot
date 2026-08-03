"""Optional local RGB video decoding for recorded teaching sessions."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from .interfaces import CaptureError, NotConfiguredError


def _load_cv2() -> ModuleType:
    try:
        import cv2
    except ImportError as exc:
        raise NotConfiguredError(
            "video decoding requires the optional opencv-python-headless package"
        ) from exc
    return cv2


@dataclass(frozen=True)
class DecodedColorFrame:
    color_image_rgb: NDArray[np.uint8]
    timestamp_ns: int
    frame_number: int


class VideoDecoder:
    """Decode an RGB recording locally; no video is sent over a network."""

    def decode(self, path: Path | str) -> Iterator[DecodedColorFrame]:
        cv2 = _load_cv2()
        video_path = Path(path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise CaptureError(f"could not open video recording: {video_path}")
        frames_per_second = float(capture.get(cv2.CAP_PROP_FPS))
        if frames_per_second <= 0.0:
            capture.release()
            raise CaptureError("video recording reports an invalid frame rate")
        frame_number = 0
        try:
            while True:
                success, image_bgr = capture.read()
                if not success:
                    break
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                yield DecodedColorFrame(
                    color_image_rgb=np.asarray(image_rgb, dtype=np.uint8),
                    timestamp_ns=int(round(frame_number / frames_per_second * 1.0e9)),
                    frame_number=frame_number,
                )
                frame_number += 1
        finally:
            capture.release()


__all__ = ["DecodedColorFrame", "VideoDecoder"]
