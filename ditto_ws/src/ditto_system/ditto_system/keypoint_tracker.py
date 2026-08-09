"""
keypoint_tracker.py

CoTracker3 online 기반 도구 keypoint 추적기.

--------------------------------------------------------------------------
왜 필요한가
--------------------------------------------------------------------------
YOLOE-seg/YOLO는 매 프레임 새로 검출(re-detection)하는 방식이라, 손이 도구를
쥐는 순간 시야 가림(occlusion)이 생기면 그 프레임의 검출이 통째로 실패하고
좌표를 잃는다. 다른 제로샷 검출 모델(Grounding-SAM 등)로 바꿔도 "매 프레임
재검출"이라는 구조 자체는 같기 때문에 같은 문제가 재발한다.

해결책은 검출 모델을 바꾸는 게 아니라 파이프라인 구조를 바꾸는 것이다:

    1) 손이 도구를 쥐기 *전*, 도구가 완전히 보이는 마지막 프레임에서
       tool_detector로 딱 1번 검출/세그멘테이션한다.
    2) 그 결과(마스크 또는 박스)에서 keypoint 여러 개를 샘플링하고,
       파지가 감지된 프레임을 기준(t=0)으로 CoTracker 추적을 시작한다.
    3) 이후에는 재검출 없이 CoTracker가 매 프레임 keypoint의 2D 위치와
       visibility(가려짐 여부)를 갱신한다. CoTracker는 점들을 개별이 아니라
       공동(joint)으로 추적하기 때문에 일부 점이 손에 가려져도 나머지 점들과의
       상관관계로 위치를 추정할 수 있다.
    4) 3D 위치는 "보이는" keypoint들의 depth 중앙값으로 계산한다
       (tool_detector.mask_to_camera_point와 동일한 견고성 전략 — 일부가
       가려지거나 depth가 안 잡혀도 나머지로 버틴다).

--------------------------------------------------------------------------
그룹(도구/손) 동시 추적 — start_groups / update_groups
--------------------------------------------------------------------------
손을 "놓았는지"(release) 판정할 때, 도구는 CoTracker로 매 프레임 추적하고
손은 DINO+SAM으로 몇 프레임마다 따로 재검출해서 비교하면, 서로 다른 시점·
다른 방식으로 구한 두 값을 비교하는 셈이라 실전에서 자꾸 어긋났다
(손이 매번 다른 시점 프레임 기준이라 노이즈가 크고, 겹침/거리 기준이
매번 미묘하게 안 맞았다).

그래서 파지가 확정되는 순간 도구뿐 아니라 **손도 같이** keypoint를 심어서
하나의 CoTracker 세션으로 동시에 추적한다. 그러면 두 그룹 다 "같은 순간,
같은 방식"으로 위치가 나오므로 둘 사이 거리/겹침 비교가 훨씬 일관적이고,
release 판정에 더 이상 DINO+SAM을 쓸 필요가 없다 (파지 이후로는 완전히
CoTracker만으로 동작).

기존 start()/update()(단일 그룹)는 내부적으로 start_groups()/update_groups()를
그룹 1개("default")로 호출하는 얇은 래퍼로 남아 있어서, 이전 코드도 그대로 동작한다.

--------------------------------------------------------------------------
설치 / 주의사항
--------------------------------------------------------------------------
    pip install torch torchvision opencv-python numpy

CoTracker는 pypi 패키지가 아니라 torch.hub에서 받는다:

    torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")

로봇 PC가 인터넷이 막혀 있다면 최초 1회는 인터넷 되는 곳에서 받아 캐시
(~/.cache/torch/hub)를 옮기거나, repo를 git clone해서 로컬 경로로 로드해야
한다.

** 중요: 아래 스텝 처리(step/window) 로직은 CoTracker 공식 online_demo.py의
패턴을 그대로 따랐지만, "직접 지정한 query point + streaming" 조합은 공식
예제 대부분이 쓰는 "grid_size 자동 그리드" 방식과는 다르다. 실제 설치된
버전에서 CoTrackerOnlinePredictor.forward()의 정확한 시그니처를
(`python -c "import torch; m=torch.hub.load(...); help(m.forward)"`) 한 번
확인하고, GPU 없는 로봇 PC에서 프레임레이트가 실사용 가능한 수준인지
반드시 실측할 것. **
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class ToolPointTracker:
    """파지 직전 1회 검출 결과로 초기화되고, 이후 CoTracker로 keypoint를 추적한다."""

    def __init__(
        self,
        device: Optional[str] = None,
        num_query_points: int = 24,
        min_visible_points: int = 3,
        trail_length: int = 15,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_query_points = num_query_points  # yaw(PCA) 추정 안정성을 위해 12->24로 늘림
        self.min_visible_points = min_visible_points
        self.trail_length = trail_length  # 화면에 궤적으로 그려줄 최근 갱신 개수

        self._model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        self._model = self._model.to(self.device)
        self._model.eval()

        self._window_frames: List[np.ndarray] = []
        self._queries: Optional[torch.Tensor] = None
        self._active = False
        self._last_result: Optional[Dict[str, object]] = None
        self._frame_index = 0
        self._is_first_call = True
        self._trail: deque = deque(maxlen=trail_length)
        self._group_slices: Dict[str, slice] = {}

    # ------------------------------------------------------------------
    # keypoint 샘플링: 마스크가 있으면 마스크 내부, 없으면 박스 내부 격자
    # ------------------------------------------------------------------
    def _sample_query_points(
        self,
        box: Tuple[float, float, float, float],
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        if mask is not None and mask.any():
            ys, xs = np.nonzero(mask)
            n = min(self.num_query_points, len(xs))
            idx = np.linspace(0, len(xs) - 1, num=n).astype(int)
            pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
        else:
            x1, y1, x2, y2 = box
            n_side = max(2, int(round(np.sqrt(self.num_query_points))))
            gx = np.linspace(x1, x2, n_side)
            gy = np.linspace(y1, y2, n_side)
            grid = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
            pts = grid[: self.num_query_points].astype(np.float32)
        return pts

    def _to_video_chunk(self, frames: List[np.ndarray]) -> torch.Tensor:
        arr = np.stack(frames)  # (T, H, W, 3) RGB uint8
        chunk = torch.from_numpy(arr).permute(0, 3, 1, 2)[None].float()  # (1, T, 3, H, W)
        return chunk.to(self.device)

    # ------------------------------------------------------------------
    # 파지 감지 프레임에서 추적 시작 (여러 그룹을 하나의 세션으로 동시에)
    # ------------------------------------------------------------------
    def start_groups(
        self,
        frame_rgb: np.ndarray,
        groups: Dict[str, Tuple[Tuple[float, float, float, float], Optional[np.ndarray]]],
    ) -> Dict[str, np.ndarray]:
        """groups = {그룹명: (box, mask)}. 반환: {그룹명: 초기 keypoint 좌표(N,2)}."""
        all_pts = []
        self._group_slices = {}
        idx = 0
        for name, (box, mask) in groups.items():
            pts = self._sample_query_points(box, mask)
            n = pts.shape[0]
            self._group_slices[name] = slice(idx, idx + n)
            all_pts.append(pts)
            idx += n
        pts_xy = np.concatenate(all_pts, axis=0)

        n_total = pts_xy.shape[0]
        t0 = np.zeros((n_total, 1), dtype=np.float32)  # 전부 현재 프레임(t=0) 기준
        queries_np = np.concatenate([t0, pts_xy], axis=1)  # (N, 3) = (t, x, y)
        self._queries = torch.from_numpy(queries_np)[None].to(self.device)  # (1, N, 3)

        self._window_frames = [frame_rgb]
        self._active = True
        self._frame_index = 0     # 이 프레임이 window의 0번째 (아직 model은 호출하지 않음)
        self._is_first_call = True
        self._trail = deque(maxlen=self.trail_length)
        self._trail.append(pts_xy.copy())  # 궤적의 시작점

        self._last_result = {
            name: {
                "points": pts_xy[sl],
                "visible": np.ones(pts_xy[sl].shape[0], dtype=bool),
                "xyz": None,
                "yaw_deg": None,
                "trail": [pts_xy[sl].copy()],
            }
            for name, sl in self._group_slices.items()
        }

        # 주의: 여기서 model을 호출하지 않는다. CoTracker online은 step(=window_len//2)
        # 프레임이 쌓인 뒤에야 첫 forward를 해야 한다 (공식 online_demo.py 패턴).
        # 예전 버전은 여기서 1프레임짜리 청크로 is_first_step=True를 호출했는데,
        # 그게 내부 상태를 1프레임 윈도우로 잘못 초기화시켜서 이후 정상 크기(step*2)
        # 청크가 들어올 때 텐서 shape mismatch로 죽는 원인이었다.

        return {name: pts_xy[sl] for name, sl in self._group_slices.items()}

    def start(
        self,
        frame_rgb: np.ndarray,
        box: Tuple[float, float, float, float],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """단일 그룹(도구만) 추적용 얇은 래퍼. 하위 호환용."""
        return self.start_groups(frame_rgb, {"default": (box, mask)})["default"]

    # ------------------------------------------------------------------
    # 매 프레임 갱신
    # ------------------------------------------------------------------
    def update_groups(
        self,
        frame_rgb: np.ndarray,
        depth_frame: np.ndarray,
        intrinsics: Dict[str, float],
        depth_scale: float,
    ) -> Optional[Dict[str, Dict[str, object]]]:
        """모든 그룹의 keypoint를 한 번에 갱신. 반환: {그룹명: {"points","visible","xyz","yaw_deg","trail"}}.

        CoTracker3 online은 step(=window_len//2, 보통 8) 프레임마다 슬라이딩
        윈도우로 좌표를 갱신하는 구조라, 이 함수를 매 프레임 불러도 매번 새
        좌표가 나오는 건 아니다 — 트리거되지 않는 프레임에서는 직전 결과를
        그대로 돌려준다. 이는 버그가 아니라 온라인 트래커의 근본적인 동작
        방식이다.
        """
        if not self._active:
            return None

        step = self._model.step  # 보통 window_len // 2
        self._frame_index += 1

        # 공식 online_demo.py와 동일한 순서: "이번 인덱스가 step의 배수"이면
        # '현재까지 쌓인' window_frames(이번 프레임 append 이전)로 먼저 model을
        # 호출하고, 그 다음에 이번 프레임을 버퍼에 추가한다.
        if self._frame_index % step == 0:
            video_chunk = self._to_video_chunk(self._window_frames[-step * 2 :])
            try:
                pred_tracks, pred_visibility = self._model(
                    video_chunk, is_first_step=self._is_first_call, queries=self._queries
                )
                self._is_first_call = False

                tracks_xy = pred_tracks[0, -1].detach().cpu().numpy()          # (N, 2) 최신 프레임
                visible = (pred_visibility[0, -1].detach().cpu().numpy() > 0.5).reshape(-1)

                self._trail.append(tracks_xy.copy())  # 궤적 표시용 히스토리

                self._last_result = {
                    name: self._group_result(tracks_xy[sl], visible[sl], sl, depth_frame, intrinsics, depth_scale)
                    for name, sl in self._group_slices.items()
                }
            except Exception as e:
                # CoTracker 내부 오류로 시연 전체가 죽지 않도록 방어.
                # 대신 이번 트리거는 실패로 치고 직전 좌표를 그대로 쓴다.
                print(f"[keypoint_tracker] CoTracker 업데이트 실패, 직전 좌표 유지: {e}")

        self._window_frames.append(frame_rgb)
        return self._last_result

    def update(
        self,
        frame_rgb: np.ndarray,
        depth_frame: np.ndarray,
        intrinsics: Dict[str, float],
        depth_scale: float,
    ) -> Optional[Dict[str, object]]:
        """단일 그룹(도구만) 추적용 얇은 래퍼. 하위 호환용."""
        result = self.update_groups(frame_rgb, depth_frame, intrinsics, depth_scale)
        return result["default"] if result is not None else None

    def _group_result(
        self,
        tracks_xy: np.ndarray,
        visible: np.ndarray,
        group_slice: slice,
        depth_frame: np.ndarray,
        intrinsics: Dict[str, float],
        depth_scale: float,
    ) -> Dict[str, object]:
        result = self._compute_xyz(tracks_xy, visible, depth_frame, intrinsics, depth_scale)
        result["trail"] = [snap[group_slice] for snap in self._trail]
        return result

    def _compute_xyz(
        self,
        tracks_xy: np.ndarray,
        visible: np.ndarray,
        depth_frame: np.ndarray,
        intrinsics: Dict[str, float],
        depth_scale: float,
    ) -> Dict[str, object]:
        h, w = depth_frame.shape[:2]
        visible_pts = tracks_xy[visible]

        samples = []
        for x, y in visible_pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                d = float(depth_frame[yi, xi])
                if d > 0:
                    samples.append((x, y, d))

        yaw_deg = self._estimate_yaw_deg(visible_pts)

        if len(samples) < self.min_visible_points:
            # 대부분 가려짐 -> 이번 프레임은 3D 위치 계산 보류 (화면 표시용 2D는 반환)
            return {"points": tracks_xy, "visible": visible, "xyz": None, "yaw_deg": yaw_deg}

        arr = np.array(samples)
        u = float(np.median(arr[:, 0]))
        v = float(np.median(arr[:, 1]))
        depth_m = float(np.median(arr[:, 2])) * depth_scale

        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        x3 = (u - cx) * depth_m / fx
        y3 = (v - cy) * depth_m / fy

        return {
            "points": tracks_xy,
            "visible": visible,
            "xyz": (x3, y3, depth_m),
            "yaw_deg": yaw_deg,
        }

    @staticmethod
    def _estimate_yaw_deg(visible_pts: np.ndarray) -> Optional[float]:
        """추적 중인(보이는) keypoint 점군의 주축(PCA) 방향을 이미지 평면 기준
        각도(도)로 반환한다. MediaPipe palm landmark가 없는 상태에서 grasp
        yaw를 대신하는 근사치이며, "도구가 놓인 각도"를 반환한다 — 점이 적으면
        (특히 3개 미만) 노이즈가 커지므로 num_query_points를 너무 줄이지 말 것.
        """
        if visible_pts.shape[0] < 3:
            return None
        centered = visible_pts - visible_pts.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, int(np.argmax(eigvals))]
        return float(np.degrees(np.arctan2(principal[1], principal[0])))

    def stop(self) -> None:
        self._active = False
        self._window_frames = []
        self._queries = None
        self._last_result = None
        self._frame_index = 0
        self._is_first_call = True
        self._trail = deque(maxlen=self.trail_length)
        self._group_slices = {}

    @property
    def is_active(self) -> bool:
        return self._active
