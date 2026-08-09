"""
hand_pinch.py

MediaPipe Hands로 손을 인식하고, 엄지-검지 사이 3D 거리(pinch distance)를
계산하는 모듈. 손을 최대 2개까지 인식한다 — 도구를 쥔 손과, 나중에 도구를
건네받을 다른 사람 손을 구분해야 하기 때문이다.

역할 분리(하이브리드 구조): 도구 인식(open-vocabulary)은 tool_detector.py가
담당하고, 이 모듈은 "손이 어디 있는지 / 쥐는 제스처를 하고 있는지"만 책임진다.

설치:
    pip install mediapipe opencv-python numpy
"""

from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np

from .tool_detector import pixel_to_camera_point

THUMB_TIP = 4
INDEX_TIP = 8

Hand = List[Tuple[float, float]]  # 21개 랜드마크의 픽셀 좌표 리스트


class HandPinchTracker:
    """MediaPipe Hands를 감싸서 프레임마다 손(들)의 랜드마크 픽셀 좌표를 뽑아준다.
    pinch distance는 필요한 손 하나를 골라서 별도로 계산한다 (여러 손이 있을
    수 있어서, "어느 손"의 pinch인지는 호출하는 쪽에서 정해야 한다)."""

    def __init__(self, max_num_hands: int = 2):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def process(self, rgb_frame: np.ndarray) -> List[Hand]:
        """이번 프레임에서 검출된 손들의 랜드마크 픽셀 좌표 리스트.
        손이 없으면 빈 리스트."""
        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return []
        h, w = rgb_frame.shape[:2]
        return [
            [(lm.x * w, lm.y * h) for lm in hand.landmark]
            for hand in result.multi_hand_landmarks
        ]

    @staticmethod
    def pinch_distance_m(
        hand: Hand,
        depth_frame: np.ndarray,
        intrinsics: Dict[str, float],
        depth_scale: float,
    ) -> Optional[float]:
        """주어진 손 하나의 엄지-검지 3D 거리[m]. depth를 못 재면 None."""
        p_thumb = pixel_to_camera_point(*hand[THUMB_TIP], depth_frame, intrinsics, depth_scale)
        p_index = pixel_to_camera_point(*hand[INDEX_TIP], depth_frame, intrinsics, depth_scale)
        if p_thumb is None or p_index is None:
            return None
        return float(np.linalg.norm(np.array(p_thumb) - np.array(p_index)))

    @staticmethod
    def pinch_point_px(hand: Hand) -> Tuple[float, float]:
        """엄지-검지 중간 지점 픽셀 좌표 (=실제로 쥔 지점)."""
        tx, ty = hand[THUMB_TIP]
        ix, iy = hand[INDEX_TIP]
        return ((tx + ix) / 2.0, (ty + iy) / 2.0)

    @staticmethod
    def center_px(hand: Hand) -> Tuple[float, float]:
        """손 전체 랜드마크의 중심 픽셀 좌표 (손이 어디 있는지 대략적인 위치)."""
        xs = [p[0] for p in hand]
        ys = [p[1] for p in hand]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @staticmethod
    def closest_hand(hands: List[Hand], point_px: Tuple[float, float]) -> Optional[Hand]:
        """여러 손 중 point_px(예: 도구를 쥔 지점)에 가장 가까운 손 하나를 고른다."""
        if not hands:
            return None
        px, py = point_px
        return min(hands, key=lambda h: (HandPinchTracker.center_px(h)[0] - px) ** 2
                                          + (HandPinchTracker.center_px(h)[1] - py) ** 2)

    def close(self) -> None:
        self._hands.close()
