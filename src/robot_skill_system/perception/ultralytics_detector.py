"""Optional local Ultralytics adapter used by the live RGB-D scene builder.

The adapter intentionally loads a caller-supplied, local weight file lazily.
It never downloads weights and it exposes only the typed 2-D detections that
the local depth/grasp code needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_skill_system.capture.interfaces import SynchronizedRGBDFrame
from robot_skill_system.scene.models import BoundingBox2D

from .interfaces import ObjectDetection2D


class UltralyticsObjectDetector:
    """Run a local YOLO weight file and return allowed semantic classes only."""

    def __init__(
        self,
        model_path: Path,
        *,
        minimum_confidence: float = 0.60,
        allowed_classes: frozenset[str] = frozenset({"hammer"}),
    ) -> None:
        if not 0.0 < minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in (0, 1]")
        if not allowed_classes:
            raise ValueError("allowed_classes must not be empty")
        self.model_path = model_path.expanduser().resolve()
        self.minimum_confidence = minimum_confidence
        self.allowed_classes = frozenset(name.strip() for name in allowed_classes)
        if not all(self.allowed_classes):
            raise ValueError("allowed_classes must contain non-empty names")
        self._model: Any | None = None

    def _loaded_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise ValueError(f"local object-detector weight is missing: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "live object detection requires the optional ultralytics package"
            ) from exc
        self._model = YOLO(str(self.model_path))
        return self._model

    def detect(self, frame: SynchronizedRGBDFrame) -> tuple[ObjectDetection2D, ...]:
        """Detect configured classes in one locally captured RGB frame."""

        result_set = self._loaded_model()(frame.color_image_rgb, verbose=False)
        detections: list[ObjectDetection2D] = []
        for result in result_set:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for index, (xyxy, confidence, class_index) in enumerate(
                zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True)
            ):
                score = float(confidence)
                class_name = str(names[int(class_index)]).strip().casefold()
                if score < self.minimum_confidence or class_name not in self.allowed_classes:
                    continue
                x_min, y_min, x_max, y_max = (float(value) for value in xyxy)
                if x_max <= x_min or y_max <= y_min:
                    continue
                detections.append(
                    ObjectDetection2D(
                        detection_id=f"{class_name}_{len(detections)}",
                        class_name=class_name,
                        bounding_box=BoundingBox2D(
                            x_min_px=x_min,
                            y_min_px=y_min,
                            x_max_px=x_max,
                            y_max_px=y_max,
                        ),
                        confidence=score,
                    )
                )
        return tuple(detections)


__all__ = ["UltralyticsObjectDetector"]
