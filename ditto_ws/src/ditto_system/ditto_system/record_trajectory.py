"""
record_trajectory.py (구 test_7_record_trajectory.py)

6번 파이프라인(도구 인식 + pinch 파지/해제 + CoTracker 추적)에 camera->robot
좌표 변환(handeye)과 JSON 저장(SkillRecorder)을 연결한다. "다른 사람 손"
구분 기능은 계속 역할이 뒤바뀌는 문제가 있어서 여기서는 빼고, 손은 1개만
다룬다. 로봇은 아직 안 움직인다 — 저장된 JSON이 그럴듯한지 확인하는 단계.

이번에 추가된 것 (6번에서 가져옴):
  - 엄지-검지 사이 선 + 거리 숫자 표시
  - 파지 확정 순간 "잡은 지점"(엄지-검지 중간)을 도구와 별개 keypoint
    그룹으로 CoTracker에 같이 심어서 추적 (빨간 점으로 표시)
  - 그 잡은 지점이 도구 중심에서 얼마나/어느 방향으로 떨어져 있는지
    (3D 오프셋, 카메라 좌표계, 단위 m)를 파지 확정 순간에 한 번 계산해서
    JSON 최상단에 "grasp_offset_camera_m"으로 저장. 도구를 쥔 채로는 이
    상대 위치가 안 변하므로 매 프레임 저장할 필요는 없다.

grip 필드는 이전과 동일하게, 파지 확정 순간의 pinch distance를 고정해서
모든 프레임에 같이 저장한다.

ESC로 종료, 'r'로 수동 리셋.
"""

import json
import os
import threading
import time
import traceback

import cv2
import numpy as np
import pyrealsense2 as rs

import rclpy
import DR_init
from ament_index_python.packages import get_package_share_directory

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
node = rclpy.create_node("record_trajectory", namespace=ROBOT_ID)
DR_init.__dsr__node = node

try:
    from DSR_ROBOT2 import get_current_posx
except ImportError as e:
    print(e)
    raise SystemExit(1)

from .tool_detector import GroundedSamDetector
from .keypoint_tracker import ToolPointTracker
from .hand_pinch import HandPinchTracker
from .handeye import handeye
from .recorder import SkillRecorder
from . import smooth_trajectory
from . import verify_trajectory

TOOL_PROMPT = "hammer"
DEPTH_SCALE = 0.001

GRASP_PINCH_THRESHOLD_M = 0.15
RELEASE_PINCH_THRESHOLD_M = 0.18
GRASP_CONFIRM_FRAMES = 5
RELEASE_CONFIRM_FRAMES = 5

SKILL_FILENAME_PREFIX = "hammering"
# colcon install share(빌드 산출물)가 아니라 고정된 실행 데이터 경로에 저장한다
# (smooth_trajectory.py 등 나머지 파이프라인 스크립트와 동일한 SKILLS_ROOT).
SKILLS_ROOT = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/src/ditto_system/skills")
RAW_OUTPUT_DIR = os.path.join(SKILLS_ROOT, "raw")

GROUP_COLORS = {"tool": (0, 255, 0), "grasp_point": (0, 0, 255)}  # 도구=초록, 잡은 지점=빨강
HAND_COLOR = (255, 128, 0)

# 웹 백엔드(backend/routes/coords.py)가 폴링해서 "좌표 생성" 페이지에 실시간 화면을
# 보여줄 수 있도록, robot_replay.py의 _publish_runtime_status()와 동일한 방식
# (임시 파일 + os.replace로 원자적 교체)으로 매 프레임 이미지를 남긴다.
RUNTIME_DIR = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/.runtime")
_COORDS_FRAME_PATH = os.path.join(RUNTIME_DIR, "coords_frame.jpg")
_COORDS_FRAME_TMP_PATH = _COORDS_FRAME_PATH + ".tmp"
os.makedirs(RUNTIME_DIR, exist_ok=True)


def _publish_coords_frame(frame_bgr):
    try:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with open(_COORDS_FRAME_TMP_PATH, "wb") as f:
                f.write(buf.tobytes())
            os.replace(_COORDS_FRAME_TMP_PATH, _COORDS_FRAME_PATH)
    except OSError as e:
        print(f"[좌표 생성 프레임 기록 오류] {e}")

# ------------------------------------
# RealSense
# ------------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

rs_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
intrinsics = {"fx": rs_intr.fx, "fy": rs_intr.fy, "cx": rs_intr.ppx, "cy": rs_intr.ppy}

_MOBILE_SAM_PATH = os.path.join(get_package_share_directory("ditto_system"), "resource", "mobile_sam.pt")
detector = GroundedSamDetector(depth_scale=DEPTH_SCALE, sam_model=_MOBILE_SAM_PATH)
point_tracker = ToolPointTracker(trail_length=40)
hand_pinch = HandPinchTracker(max_num_hands=1)
recorder = None  # 파지 확정마다 새로 생성 (이전 시연과 안 섞이게)

# ------------------------------------
# 도구 검출 스레드 (WAITING_FOR_GRASP 동안만 사용)
# ------------------------------------
lock = threading.Lock()
latest_rgb = None
latest_depth = None
latest_tool_det = None
running = True
detection_paused = False


def detection_worker():
    global latest_tool_det
    while running:
        if detection_paused:
            time.sleep(0.02)
            continue
        with lock:
            rgb, depth = latest_rgb, latest_depth
        if rgb is None:
            time.sleep(0.01)
            continue
        try:
            det = detector.detect(rgb, depth, intrinsics, TOOL_PROMPT)
        except Exception as e:
            print(f"[검출 오류] {e}")
            det = None
        with lock:
            latest_tool_det = det


threading.Thread(target=detection_worker, daemon=True).start()

# ------------------------------------
# 상태
# ------------------------------------
phase = "WAITING_FOR_GRASP"
grasp_pending = 0
release_pending = 0
grip_width_at_grasp = 0.0
grasp_offset_camera_m = None
recorded_count = 0
last_recorded_xyz = None


def reset():
    global phase, grasp_pending, release_pending, detection_paused
    global recorded_count, last_recorded_xyz, grasp_offset_camera_m
    point_tracker.stop()
    phase = "WAITING_FOR_GRASP"
    grasp_pending = 0
    release_pending = 0
    detection_paused = False
    recorded_count = 0
    last_recorded_xyz = None
    grasp_offset_camera_m = None
    print("\n[리셋] WAITING_FOR_GRASP로 복귀")


def draw_hand(frame, hand, color, depth_image):
    """손 랜드마크 + 엄지-검지 선/거리 표시. pinch 거리[m] 반환."""
    for (x, y) in hand:
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)

    pinch_dist = HandPinchTracker.pinch_distance_m(hand, depth_image, intrinsics, DEPTH_SCALE)

    thumb_px, index_px = hand[4], hand[8]
    cv2.line(frame, (int(thumb_px[0]), int(thumb_px[1])), (int(index_px[0]), int(index_px[1])), color, 2)
    mx, my = (thumb_px[0] + index_px[0]) / 2.0, (thumb_px[1] + index_px[1]) / 2.0
    dist_txt = f"{pinch_dist:.3f}m" if pinch_dist is not None else "N/A"
    cv2.putText(frame, dist_txt, (int(mx) + 8, int(my)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return pinch_dist


def exclude_region_from_mask(mask, center_px, radius_px=20):
    """마스크에서 center_px 주변 원형 영역을 제외한다. 손가락이 도구를 잡고
    있는 바로 그 지점 근처는 "도구" keypoint 샘플링에서 빼야, 그 점들이
    실제로는 손가락 위에 찍혀서(손이 살짝만 움직여도 도구 위치 전체가
    요동치는) 문제를 막을 수 있다."""
    if mask is None:
        return None
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    dist_sq = (xx - center_px[0]) ** 2 + (yy - center_px[1]) ** 2
    excluded = mask.astype(bool) & (dist_sq > radius_px ** 2)
    return excluded if excluded.any() else mask  # 다 지워지면 원본으로 폴백


print("=" * 50)
print(" A단계(재수정): 궤적 + 잡은 지점 오프셋 저장 테스트 (로봇 안 움직임)")
print(f" pinch < {GRASP_PINCH_THRESHOLD_M}m 쥠 / pinch > {RELEASE_PINCH_THRESHOLD_M}m 놓음")
print(" ESC 종료, 'r' 리셋")
print("=" * 50)

try:
    while True:
        try:
            frames = pipeline.wait_for_frames(5000)
        except RuntimeError as e:
            print(f"[프레임 타임아웃, 재시도] {e}")
            continue

        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands = hand_pinch.process(rgb)
        hand = hands[0] if hands else None  # 손 1개만 사용 (다른 손 구분 없음)

        pinch_dist = None
        if hand is not None:
            pinch_dist = draw_hand(frame, hand, HAND_COLOR, depth_image)
        pinch_txt = f"{pinch_dist:.3f}m" if pinch_dist is not None else "N/A"

        if phase == "WAITING_FOR_GRASP":

            with lock:
                latest_rgb, latest_depth = rgb, depth_image
                tool_det = latest_tool_det

            is_grasp_frame = pinch_dist is not None and pinch_dist <= GRASP_PINCH_THRESHOLD_M and tool_det is not None
            grasp_pending = grasp_pending + 1 if is_grasp_frame else 0

            if tool_det is not None:
                x1, y1, x2, y2 = tool_det["box"]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), GROUP_COLORS["tool"], 2)

            cv2.putText(frame, f"Phase: {phase}  pinch: {pinch_txt}  pending: {grasp_pending}/{GRASP_CONFIRM_FRAMES}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if grasp_pending >= GRASP_CONFIRM_FRAMES:
                # --- 파지 확정: 이 순간(rgb)으로 동기 재검출해서 정확한 seed를 얻는다 ---
                detection_paused = True
                fresh = detector.detect(rgb, depth_image, intrinsics, TOOL_PROMPT)
                seed = fresh if fresh is not None else tool_det

                px, py = HandPinchTracker.pinch_point_px(hand)
                # 정확히 손가락이 맞닿는 지점은 depth 카메라가 원래 잘 못 잰다
                # (그림자/노이즈). 박스를 넓게 잡아서 주변(손가락 표면 등)의
                # depth가 잡히는 점들이 섞이도록 한다 — 좁게 잡으면 점들이
                # 전부 한 곳에 몰려서 그 지점이 실패하면 다같이 실패한다.
                grasp_box = (px - 20, py - 20, px + 20, py + 20)

                # 손가락이 잡고 있는 지점 근처는 "도구" keypoint에서 제외 —
                # 안 그러면 그 점들이 실제론 손가락 위에 찍혀서 손이 살짝만
                # 움직여도 도구 위치 전체가 요동친다.
                tool_mask = exclude_region_from_mask(seed.get("mask"), (px, py), radius_px=20)

                point_tracker.start_groups(rgb, {
                    "tool": (seed["box"], tool_mask),
                    "grasp_point": (grasp_box, None),
                })

                grasp_offset_camera_m = None  # TRACKING 첫 실제 갱신 때 계산 (아래)
                phase = "TRACKING"
                release_pending = 0
                recorded_count = 0
                last_recorded_xyz = None
                grip_width_at_grasp = pinch_dist
                recorder = SkillRecorder()
                print(f"\n[파지 확정] pinch={pinch_dist:.3f}m -> 추적+기록 시작")

        else:  # TRACKING

            groups = point_tracker.update_groups(rgb, depth_image, intrinsics, depth_scale=DEPTH_SCALE)

            if groups is not None:
                tool_g = groups["tool"]
                trail = tool_g.get("trail")
                if trail and len(trail) > 1:
                    for pi in range(trail[0].shape[0]):
                        for a, b in zip(trail[:-1], trail[1:]):
                            cv2.line(frame, (int(a[pi, 0]), int(a[pi, 1])),
                                      (int(b[pi, 0]), int(b[pi, 1])), (0, 200, 255), 1)
                for (px, py), vis in zip(tool_g["points"], tool_g["visible"]):
                    color = GROUP_COLORS["tool"] if vis else (0, 0, 255)
                    cv2.circle(frame, (int(px), int(py)), 4, color, -1)

                gp_pts = groups["grasp_point"]["points"]
                gpx, gpy = float(np.mean(gp_pts[:, 0])), float(np.mean(gp_pts[:, 1]))
                cv2.circle(frame, (int(gpx), int(gpy)), 8, GROUP_COLORS["grasp_point"], -1)

                # --- 오프셋: CoTracker가 두 그룹 다 첫 실제 3D 위치를 낸 순간
                #     한 번만 계산 (단일 픽셀 depth보다 median 기반이라 더 안정적)
                if grasp_offset_camera_m is None:
                    gp_xyz = groups["grasp_point"]["xyz"]
                    if tool_g["xyz"] is not None and gp_xyz is not None:
                        grasp_offset_camera_m = (np.array(gp_xyz) - np.array(tool_g["xyz"])).tolist()
                        print(f"[오프셋 계산됨] {grasp_offset_camera_m}")

                # --- camera xyz -> robot base xyz 변환 + 기록 (도구가 "새로" 갱신된 프레임에만) ---
                tool_xyz = tool_g["xyz"]
                if tool_xyz is not None and tool_xyz != last_recorded_xyz:
                    last_recorded_xyz = tool_xyz
                    camera_xyz_mm = np.array(tool_xyz) * 1000.0
                    robot_xyz = handeye.transform_to_base(camera_xyz_mm)
                    yaw = tool_g.get("yaw_deg")
                    yaw = float(yaw) if yaw is not None else 0.0

                    pose = {
                        "x": float(robot_xyz[0]),
                        "y": float(robot_xyz[1]),
                        "z": float(robot_xyz[2]),
                        "yaw": yaw,
                        "grip": float(grip_width_at_grasp),
                    }
                    recorder.add_frame(pose)
                    recorded_count += 1

                    cv2.putText(frame, f"robot xyz: ({robot_xyz[0]:.0f},{robot_xyz[1]:.0f},{robot_xyz[2]:.0f})mm",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            is_release_frame = pinch_dist is not None and pinch_dist >= RELEASE_PINCH_THRESHOLD_M
            release_pending = release_pending + 1 if is_release_frame else 0

            cv2.putText(frame, f"Phase: {phase}  pinch: {pinch_txt}  recorded: {recorded_count}  "
                                f"release_pending: {release_pending}/{RELEASE_CONFIRM_FRAMES}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if release_pending >= RELEASE_CONFIRM_FRAMES:
                print(f"\n[해제 확정] pinch={pinch_dist:.3f}m -> {recorded_count}개 프레임 저장")
                filename = f"{SKILL_FILENAME_PREFIX}_{int(time.time())}.json"
                try:
                    # recorder.save()의 내부 저장 경로가 우리가 원하는 경로와 다를 수
                    # 있어서(예: skills/ 하위), recorder.save()에 의존하지 않고
                    # recorder.frames를 직접 가져다 우리가 원하는 경로/이름으로 쓴다.
                    output = {
                        "skill": TOOL_PROMPT,
                        "frames": recorder.frames,
                        "grasp_offset_camera_m": grasp_offset_camera_m,
                    }
                    os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
                    filepath = os.path.join(RAW_OUTPUT_DIR, filename)
                    with open(filepath, "w") as f:
                        json.dump(output, f, indent=4)
                    print(f"[저장 완료] {filepath}")

                    # 좌표 생성 웹 페이지에서 record -> smooth -> verify 결과를 한 번에
                    # 볼 수 있도록, 원본 저장 직후 그 자리에서 이어서 돌린다. 실패해도
                    # 녹화 루프 자체는 계속돼야 하니 별도 try로 감싼다.
                    try:
                        smoothed_path = smooth_trajectory.process(filepath)
                        verify_trajectory.process(smoothed_path)
                    except Exception as e:
                        print(f"[스무딩/검증 자동 실행 오류] {e}")
                except Exception as e:
                    print(f"[저장 오류] {e}")
                reset()

        _publish_coords_frame(frame)
        cv2.imshow("Stage A: Trajectory + Grasp Point", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("r"):
            reset()

except KeyboardInterrupt:
    print("\nKeyboard Interrupt")

except Exception:
    traceback.print_exc()

finally:
    running = False
    hand_pinch.close()
    pipeline.stop()
    cv2.destroyAllWindows()
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


def main():
    # 이 파일은 위에서부터 전부 모듈 최상단 코드로 실행되므로(원본 test_7_...
    # 스크립트 구조 그대로), import 시점에 이미 실제 동작이 다 끝난다.
    # console_scripts entry_point(`ros2 run ditto_system record_trajectory`)가
    # 요구하는 main() 심볼만 채워주는 자리표시자.
    pass


if __name__ == "__main__":
    main()
