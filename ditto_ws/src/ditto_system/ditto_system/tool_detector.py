"""
tool_detector.py

Grounding DINO(open-vocabulary bbox 검출) + SAM(마스크 생성)으로 텍스트
프롬프트를 3D 위치까지 검출하는 모듈.

- detect(): 프롬프트 1개 검출 (1단계에서 사용)
- detect_multi(): 프롬프트 여러 개를 DINO 1회 + SAM 1회 호출로 동시 검출
  (2단계: 도구+손 동시 인식에서 사용 — 따로 부르면 추론이 2배로 느려진다)
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


def pixel_to_camera_point(
    u: float,
    v: float,
    depth_frame: np.ndarray,
    intrinsics: Dict[str, float],
    depth_scale: float,
    patch_radius: int = 2,
) -> Optional[Tuple[float, float, float]]:
    """픽셀 좌표 (u, v)와 depth 프레임으로 카메라 좌표계 3D 포인트(x, y, z)[m] 계산.
    해당 픽셀 depth가 비어 있으면 patch_radius 안의 유효 depth 중앙값으로 대체."""
    h, w = depth_frame.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None

    r = patch_radius
    patch = depth_frame[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1].astype(np.float32)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None

    depth = float(np.median(valid)) * depth_scale
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return (x, y, depth)


def mask_to_camera_point(
    mask: np.ndarray,
    depth_frame: np.ndarray,
    intrinsics: Dict[str, float],
    depth_scale: float,
) -> Optional[Tuple[float, float, float]]:
    """마스크 내부 전체 픽셀의 depth 중앙값으로 3D 포인트를 계산.
    표면 일부 depth가 비어도(반사/유리 등) 나머지 유효 픽셀로 버틴다."""
    h, w = depth_frame.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None

    depths = depth_frame[ys, xs].astype(np.float32)
    valid = np.isfinite(depths) & (depths > 0)
    if not np.any(valid):
        return None

    depth = float(np.median(depths[valid])) * depth_scale
    u, v = float(np.mean(xs[valid])), float(np.mean(ys[valid]))
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return (x, y, depth)


def mask_yaw_deg(mask: np.ndarray) -> Optional[float]:
    """마스크(도구 실루엣) 픽셀들의 주축(PCA) 방향을 이미지 평면 각도(도)로
    반환한다. 값의 정의: 이미지 x축(오른쪽 방향) 기준, 반시계 방향 각도.

    주의: PCA 주축은 "방향 없는 직선"이라 180도 주기성을 가진다 (예: 손잡이가
    왼쪽/오른쪽 어느 쪽을 향해도 같은 값이 나온다). 그래서 이 값 하나만으로는
    도구의 "앞/뒤"는 구분 못 하고, 오직 "축이 얼마나 기울었는지"만 알려준다.
    기준값(TOOL_AXIS_REFERENCE_DEG)과 비교할 때도 이 180도 주기를 wrap해서
    비교해야 한다 (robot_replay.py의 compute_grasp_rz 참고).

    keypoint_tracker.ToolPointTracker._estimate_yaw_deg와 같은 방식(PCA)이지만,
    거긴 CoTracker가 추적 중인 keypoint 점들 기준이고 여긴 SAM 마스크 픽셀
    전체 기준이라 별도로 둔다.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    return float(np.degrees(np.arctan2(principal[1], principal[0])))


def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """두 bbox(x1,y1,x2,y2)의 IoU(교집합/합집합)를 계산."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class GroundedSamDetector:
    """Grounding DINO로 bbox를 찾고 SAM으로 마스크를 얻어 3D 위치까지 반환한다."""

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        overlap_iou_threshold: float = 0.85,
        grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny",
        sam_model: str = "mobile_sam.pt",
        depth_scale: float = 0.001,  # RealSense z16 raw depth(mm) -> m
    ):
        from transformers import pipeline
        from ultralytics import SAM

        self.confidence_threshold = confidence_threshold
        self.overlap_iou_threshold = overlap_iou_threshold  # 이 이상 겹치면 "같은 물체 이중 라벨"로 보고 1개만 채택
        # (0.85처럼 높게 잡아야 함: 손이 도구를 실제로 쥐어서 두 박스가 겹치는
        #  정상적인 경우는 IoU가 이보다 낮은 경우가 대부분이라 살아남아야 한다.
        #  너무 낮게 잡으면 "파지" 상황에서 손/도구 중 하나가 통째로 사라져버린다.)
        self.depth_scale = depth_scale

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[tool_detector] device = {self.device}")

        self._dino = pipeline(
            model=grounding_dino_model,
            task="zero-shot-object-detection",
            device=0 if self.device == "cuda" else -1,  # transformers 관례: GPU=0, CPU=-1
        )
        self._sam = SAM(sam_model)
        self._sam.to(self.device)

    def detect(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        intrinsics: Dict[str, float],
        prompt: str,
    ) -> Optional[Dict[str, object]]:
        """프롬프트 1개 검출. 성공 시 {"xyz", "box", "conf", "mask"}, 실패 시 None."""
        return self.detect_multi(rgb_frame, depth_frame, intrinsics, [prompt])[prompt]

    def detect_multi(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        intrinsics: Dict[str, float],
        prompts: List[str],
    ) -> Dict[str, Optional[Dict[str, object]]]:
        """프롬프트 여러 개를 한 번의 DINO 호출 + 한 번의 SAM batched 호출로 검출.
        반환: {원본 prompt: {"xyz","box","conf","mask"} 또는 None}"""
        boxes_by_prompt = self._detect_boxes(rgb_frame, prompts)

        found_prompts = [p for p, b in boxes_by_prompt.items() if b is not None]
        boxes = [boxes_by_prompt[p][0] for p in found_prompts]
        masks = self._segment(rgb_frame, boxes) if boxes else []
        masks_by_prompt = dict(zip(found_prompts, masks))

        results: Dict[str, Optional[Dict[str, object]]] = {}
        for prompt in prompts:
            entry = boxes_by_prompt.get(prompt)
            if entry is None:
                results[prompt] = None
                continue

            box, conf = entry
            mask = masks_by_prompt.get(prompt)
            xyz = mask_to_camera_point(mask, depth_frame, intrinsics, self.depth_scale) if mask is not None else None
            if xyz is None:
                u, v = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                xyz = pixel_to_camera_point(u, v, depth_frame, intrinsics, self.depth_scale)

            results[prompt] = {"xyz": xyz, "box": box, "conf": conf, "mask": mask} if xyz is not None else None

        return results

    def _detect_boxes(
        self, rgb_frame: np.ndarray, prompts: List[str]
    ) -> Dict[str, Optional[Tuple[Tuple[float, float, float, float], float]]]:
        """프롬프트마다 가장 점수가 높은 bbox 하나씩 반환 (없으면 None).

        DINO는 라벨마다 독립적으로 채점하기 때문에, 같은 물체가 서로 다른
        라벨(예: 도구가 "hand"로도) 로 동시에 threshold를 넘는 경우가 생길 수
        있다. 이를 막기 위해 박스끼리 많이 겹치면 confidence가 더 높은 라벨
        하나만 채택한다.
        """
        from PIL import Image

        texts = [p.strip().lower() + ("" if p.strip().endswith(".") else ".") for p in prompts]
        results = self._dino(Image.fromarray(rgb_frame), candidate_labels=texts, threshold=self.confidence_threshold)

        best_by_text: Dict[str, dict] = {}
        for r in results:
            if r["label"] not in best_by_text or r["score"] > best_by_text[r["label"]]["score"]:
                best_by_text[r["label"]] = r

        kept_texts = self._suppress_overlaps(best_by_text)

        out = {}
        for prompt, text in zip(prompts, texts):
            best = best_by_text.get(text) if text in kept_texts else None
            if best is None:
                out[prompt] = None
                continue
            b = best["box"]
            box = (float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"]))
            out[prompt] = (box, float(best["score"]))
        return out

    def _suppress_overlaps(self, best_by_text: Dict[str, dict]):
        """confidence가 높은 라벨부터 채택하고, 이미 채택된 박스와 많이
        겹치는(IoU > overlap_iou_threshold) 다른 라벨의 박스는 오검출로 버린다."""
        ordered = sorted(best_by_text.items(), key=lambda kv: kv[1]["score"], reverse=True)
        kept_boxes: List[Tuple[float, float, float, float]] = []
        kept_texts = set()
        for text, r in ordered:
            b = r["box"]
            box = (b["xmin"], b["ymin"], b["xmax"], b["ymax"])
            if any(_iou(box, kb) > self.overlap_iou_threshold for kb in kept_boxes):
                continue
            kept_boxes.append(box)
            kept_texts.add(text)
        return kept_texts

    def _segment(
        self, rgb_frame: np.ndarray, boxes: List[Tuple[float, float, float, float]]
    ) -> List[Optional[np.ndarray]]:
        """여러 bbox를 한 번의 SAM 호출로 마스크화 (실패하면 전부 None으로 대체)."""
        try:
            results = self._sam(rgb_frame, bboxes=[list(b) for b in boxes], device=self.device, verbose=False)
            masks = results[0].masks
            if masks is None:
                return [None] * len(boxes)
            data = masks.data
            return [
                (data[i].cpu().numpy() if hasattr(data[i], "cpu") else np.asarray(data[i]))
                if i < len(data) else None
                for i in range(len(boxes))
            ]
        except Exception as e:
            print(f"[tool_detector] SAM 마스크 생성 실패, 박스만 사용: {e}")
            return [None] * len(boxes)

