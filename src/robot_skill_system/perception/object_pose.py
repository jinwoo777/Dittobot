"""Local metric object-pose estimation interfaces and depth-centroid MVP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from robot_skill_system.capture.interfaces import SynchronizedRGBDFrame
from robot_skill_system.scene.models import ObjectInstance, Pose, Quaternion, Vector3

from .interfaces import ObjectDetection2D, ObjectPoseEstimator


class DepthCentroidObjectPoseEstimator:
    """Estimate configured object centers from aligned depth inside detections."""

    def __init__(
        self,
        instance_ids_by_detection: Mapping[str, str] | None = None,
        *,
        minimum_valid_depth_fraction: float = 0.20,
    ) -> None:
        if not 0.0 <= minimum_valid_depth_fraction <= 1.0:
            raise ValueError("minimum_valid_depth_fraction must be in [0, 1]")
        self.instance_ids_by_detection = dict(instance_ids_by_detection or {})
        self.minimum_valid_depth_fraction = minimum_valid_depth_fraction

    def estimate(
        self,
        frame: SynchronizedRGBDFrame,
        detections: Sequence[ObjectDetection2D] = (),
    ) -> Sequence[ObjectInstance]:
        output: list[ObjectInstance] = []
        intrinsics = frame.color_intrinsics
        image_height, image_width = frame.depth_image_m.shape
        for detection in detections:
            box = detection.bounding_box
            x_min = max(0, min(image_width - 1, int(np.floor(box.x_min_px))))
            x_max = max(x_min + 1, min(image_width, int(np.ceil(box.x_max_px))))
            y_min = max(0, min(image_height - 1, int(np.floor(box.y_min_px))))
            y_max = max(y_min + 1, min(image_height, int(np.ceil(box.y_max_px))))
            patch = frame.depth_image_m[y_min:y_max, x_min:x_max]
            valid_mask = np.isfinite(patch) & (patch > 0.0)
            valid_fraction = float(np.mean(valid_mask)) if patch.size else 0.0
            if valid_fraction < self.minimum_valid_depth_fraction:
                continue
            depth_m = float(np.median(patch[valid_mask]))
            center_x_px = (x_min + x_max - 1) / 2.0
            center_y_px = (y_min + y_max - 1) / 2.0
            position = Vector3(
                x=(center_x_px - intrinsics.cx_px) / intrinsics.fx_px * depth_m,
                y=(center_y_px - intrinsics.cy_px) / intrinsics.fy_px * depth_m,
                z=depth_m,
            )
            confidence = detection.confidence * valid_fraction
            pose = Pose(
                frame_id=frame.reference_frame,
                position_m=position,
                orientation_xyzw=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                timestamp_ns=frame.timestamp_ns,
                source="depth_centroid",
                confidence=confidence,
                uncertainty={"depth_valid_fraction": 1.0 - valid_fraction},
            )
            instance_id = self.instance_ids_by_detection.get(
                detection.detection_id, detection.detection_id
            )
            output.append(
                ObjectInstance(
                    instance_id=instance_id,
                    class_name=detection.class_name,
                    pose=pose,
                    bounding_box_2d=box,
                    confidence=confidence,
                    visible_fraction=valid_fraction,
                    pose_source="depth_centroid",
                )
            )
        return tuple(output)


__all__ = ["DepthCentroidObjectPoseEstimator", "ObjectPoseEstimator"]
