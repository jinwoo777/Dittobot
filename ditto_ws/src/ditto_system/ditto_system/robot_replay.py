"""
robot_replay.py

전체 시퀀스:
  0) 조인트 홈 [0, 0, 90, 0, 90, 0] (deg)으로 이동
  1) 도구를 실시간 검출 -> 축정렬(rz) 계산 -> 같은 z에서 수평 이동 -> hover ->
     실제 파지 높이로 하강 -> 그리퍼 닫기 -> 다시 hover로 상승
  2) 기록된 궤적(frames)을 순서대로 재생
  3) 손 위치를 검출/안정화한 뒤 그쪽으로 딱 한 번만 이동하고 완전히 정지
  4) 정지한 채로 사람이 도구를 쥐는 동작(주먹을 쥐듯 손가락을 접는 동작)이
     감지되면 -> 그리퍼 열기
  5) 조인트 홈으로 복귀

카메라 창을 계속 띄워서 도구/손 인식 상태를 실시간으로 볼 수 있다.

--------------------------------------------------------------------------
손 핸드오버 방식: "계속 쫓아가기"를 버리고 "한 번 접근 후 정지 + 주먹(fist) 감지"로 변경
--------------------------------------------------------------------------
카메라가 로봇 손끝에 달려있다(eye-in-hand). 로봇이 손 위치를 계속 실시간
추적하며 움직이면, 로봇이 움직일 때마다 카메라 자신의 위치/각도도 같이
바뀌어서 같은 손인데도 "다른 곳"으로 보이고, 그걸 또 쫓아가느라 로봇이
움직이고... 하는 식으로 관측과 행동이 서로를 계속 오염시키는 폐루프가
생긴다. handeye 캘리브레이션에 아주 작은 오차만 있어도 이게 한쪽으로
편향되면 발산해서(예: Z가 끝없이 올라감) 데드밴드/필터/트래커를 아무리
정교하게 다듬어도 근본적으로 못 고친다.

그래서 접근 방식을 바꿨다:
  - 손 위치를 검출해서(여러 프레임 median으로 안정화) 그 위로 **딱 한 번만**
    이동한다.
  - 이동이 끝나면 로봇은 **완전히 멈추고 더 이상 위치를 갱신하지 않는다**
    (eye-in-hand 피드백 루프를 아예 차단).
  - 정지한 상태에서 사람이 도구를 실제로 쥐는지는 위치가 아니라 손 랜드마크
    21개로 계산한 "주먹 쥠(fist)" 여부로 판단한다 (도구 손잡이를 감싸 쥐는
    동작은 엄지-검지 pinch보다 네 손가락이 손바닥 쪽으로 말리는 fist에
    가깝다). 네 손가락(검지/중지/약지/소지) 끝이 팔목+MCP 관절들의 중심
    (palm_center)에서 각 MCP 관절보다 더 가까워지면 그 손가락은 "말림"으로
    보고, FIST_MIN_CURLED_FINGERS개 이상 말리면 주먹으로 판정한다. 이 판정이
    FIST_CONFIRM_SEC 동안 유지되면 "쥐었다"고 보고 그리퍼를 연다. 로봇이 안
    움직이므로 카메라도 안 움직이고, 따라서 이 단계에는 애초에 발산할
    여지가 없다.
  - 쥐는 순간 손가락이 도구에 살짝 가려져 검출이 잠깐 끊겨도 FIST_LOST_
    GRACE_SEC 동안은 직전 상태를 유지한다.

--------------------------------------------------------------------------
위치 보정
--------------------------------------------------------------------------
축정렬은 정상이지만 실제 도달 위치가 목표보다 조금씩 어긋나는 경우를 위해
보정 상수를 뒀다. 도구 픽업(TOOL_POSITION_OFFSET_*)과 손 접근
(HAND_POSITION_OFFSET_*)을 따로 뒀다 - 둘은 검출 방식도 다르고(도구는
GroundedSAM 마스크, 손은 MediaPipe 랜드마크) 카메라와의 거리/각도도 달라서
어긋나는 방향과 크기가 서로 다를 수 있다. 실제 로봇에서 목표 대비 어긋난
방향/거리를 측정해서 반대 부호로 채워 넣는다 (예: 목표보다 오른쪽(+Y)으로
6mm 더 갔으면 해당 _Y_MM = -6.0). 도구/손 양쪽 다, 그리고 여러 위치에서
비슷한 방향/크기로 계속 어긋난다면 이 상수보다 handeye.py 캘리브레이션
(특히 T_gripper2camera.npy의 방향 convention)을 다시 확인하는 게 근본
해결책이다.

--------------------------------------------------------------------------
movel() 좌표계 (mod=DR_MV_MOD_ABS 필수)
--------------------------------------------------------------------------
handeye.transform_to_base()는 base 원점 기준 절대좌표를 반환하므로,
move_and_verify()에서 mod=DR_MV_MOD_ABS를 명시해서 절대좌표로 해석되도록
고정한다.

** 실행 전 반드시 확인할 것 **
  import DSR_ROBOT2, inspect; print(inspect.signature(DSR_ROBOT2.movel))
  print(inspect.signature(DSR_ROBOT2.movej))
  로 실제 설치된 movel()/movej()의 mod 파라미터 이름/기본값이 여기서 가정한
  것과 일치하는지 먼저 확인할 것.

--------------------------------------------------------------------------
카메라 스레드
--------------------------------------------------------------------------
camera_worker() 스레드가 로봇 제어(메인 스레드)와 독립적으로 계속 카메라를
읽고 화면을 갱신한다. 메인 스레드는 이 스레드가 채워둔 latest_tool_det /
hand_det_history / latest_is_fist를 lock으로 읽기만 한다 (movel()/movej()이
블로킹되는 동안에도 카메라 창이 계속 갱신되도록).

사용:
    python3 robot_replay.py skills/smooth/hammering_1786071664_smoothed.json   # 궤적 재생 데모
    python3 robot_replay.py                                                    # 음성으로 도구 요청받아 fetch

--------------------------------------------------------------------------
음성 모드 (voice_processing 패키지 연동)
--------------------------------------------------------------------------
인자 없이 실행하면 run_voice_fetch_loop()가 돌면서 get_keyword 서비스(웨이크워드
대기 -> Whisper STT -> GPT-4o로 도구명 추출)를 호출해 도구 이름을 받고,
그 이름으로 CURRENT_TOOL_PROMPTS를 바꿔서 pick_tool() -> approach_hand_once() ->
wait_for_fist_and_release() 순서로 그대로 재사용한다. 지원 도구는 TOOL_PROMPTS
딕셔너리에 등록된 hammer/screwdriver/wrench/brush이고, 각 도구는 프롬프트 후보를
여러 개(동의어) 가질 수 있다 - 그 중 하나라도 검출되면 그 도구로 인정한다.

그리퍼로 도구를 잡은 직후, 몇 초간 짧게 STT를 다시 돌려서 "아니야" 류의 거부
표현이 있으면(REJECTION_PHRASES) put_back_current_tool()로 pick_tool()이 기록해둔
원래 grasp 위치(_last_grasp_pose)에 도로 내려놓고 같은 도구 이름으로 재검출을
시도한다 (MAX_REJECT_RETRIES까지). 별도로 실측해야 할 위치가 없어 워크스페이스가
바뀌어도 그대로 동작한다.
"""

# --------------------------------------------------------------------------
# 서드파티 라이브러리(tensorflow/tflite, mediapipe, huggingface_hub 등)가
# 시작할 때 찍는 로그/경고를 줄인다. 반드시 그 라이브러리들을 import하기
# "전"에 설정해야 효과가 있다 (import 시점에 이미 로거/환경변수를 읽어가는
# 경우가 많음).
#
# 아래로 해결이 안 되는 것: 터미널에 뜨는
#   "selected interface "lo" is not multicast-capable: disabling multicast"
# 는 ROS2 DDS 미들웨어(Cyclone DDS)가 Python 레벨보다 먼저, OS 네트워크
# 인터페이스 설정을 보고 찍는 경고라 여기서 못 끈다. 딱 한 번:
#   sudo ip link set lo multicast on
# 을 실행해두면(재부팅하면 다시 꺼질 수 있어 필요하면 시작 스크립트에 추가)
# 이 경고 자체가 안 뜬다. 무해한 경고라 안 없애도 동작에는 지장 없다.
# --------------------------------------------------------------------------
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")           # TensorFlow/TFLite: FATAL만 출력
os.environ.setdefault("GLOG_minloglevel", "2")                # mediapipe(glog/absl): ERROR 이상만
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")    # 허깅페이스 모델 로딩 진행바 끄기

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")  # matplotlib 3D 관련, 무해

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # "unauthenticated requests..." 경고

import json
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs

import rclpy
import DR_init
from ament_index_python.packages import get_package_share_directory
from std_srvs.srv import Trigger
from dotenv import load_dotenv

from .onrobot import RG
from .tool_detector import GroundedSamDetector, pixel_to_camera_point, mask_yaw_deg
from .hand_pinch import HandPinchTracker
from .handeye import handeye
from voice_processing.stt import STT

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
node = rclpy.create_node("robot_replay", namespace=ROBOT_ID)
DR_init.__dsr__node = node

# get_keyword 서비스(음성 도구 요청) 호출 전용 노드. DSR_ROBOT2가 내부적으로
# 위 node를 어떻게 스핀하는지 알 수 없으므로, 서로 간섭하지 않도록 별도
# 노드로 분리해서 그 노드만 동기적으로 스핀한다.
voice_node = rclpy.create_node("robot_replay_voice_client")
get_keyword_client = voice_node.create_client(Trigger, "get_keyword")

try:
    # DR_MV_MOD_ABS: movel의 좌표를 "절대좌표"로 해석하라는 모드 플래그.
    from DSR_ROBOT2 import movel, movej, mwait, get_current_posx, DR_BASE, DR_MV_MOD_ABS
except ImportError as e:
    print(e)
    print("[확인 필요] 설치된 DSR_ROBOT2.py에 movej/DR_MV_MOD_ABS가 있는지 확인하세요. "
          "이름이 다르다면 이 스크립트의 import와 관련 호출부를 실제 이름으로 맞춰야 합니다.")
    sys.exit(1)

# movec은 스킬화된 궤적(run_generated_skill)에서만 쓰인다. 없어도 음성 fetch
# 모드 등 나머지 기능은 그대로 동작해야 하니, 실패해도 전체를 죽이지 않고
# 플래그만 꺼서 run_generated_skill() 호출 시점에 명확한 에러로 알린다.
# ** 실행 전 반드시 확인할 것 **
#   import DSR_ROBOT2, inspect; print(inspect.signature(DSR_ROBOT2.movec))
#   로 실제 movec()의 파라미터 이름/순서가 아래 move_and_verify_arc()의 가정과
#   맞는지 확인할 것 (아직 미확인 상태).
try:
    from DSR_ROBOT2 import movec
    MOVEC_AVAILABLE = True
except ImportError as e:
    print(e)
    print("[확인 필요] DSR_ROBOT2.py에 movec가 없습니다. run_generated_skill()의 "
          "movec 세그먼트는 호출 시점에 에러가 납니다.")
    MOVEC_AVAILABLE = False

# --- read_current_pose.py로 실측한 값 ---
ORIENTATION_RX_DEG = 155.69
ORIENTATION_RY_DEG = 179.97
ORIENTATION_RZ_DEG = 156.20

# 도구 축 정렬 기준값. 잘 맞는 것으로 확인됐으니 건드리지 않는다.
TOOL_AXIS_REFERENCE_DEG = 89.13

# 음성으로 요청 가능한 도구 -> GroundedSAM 검출 프롬프트.
# get_keyword 노드(LLM)가 한글 도구명을 이 딕셔너리의 키(영어)로 정규화해서
# 돌려주도록 프롬프트를 맞춰뒀다 (get_keyword.py 참고).
TOOL_PROMPTS = {
    "hammer": ["hammer"],
    "screwdriver": ["screwdriver"],
    "wrench": ["wrench"],
    "brush": ["brush", "scrub brush", "cleaning brush", "hand brush"],
}
DEFAULT_TOOL_PROMPT = "hammer"  # 아직 음성으로 도구를 못 받았을 때의 자리표시자 값 (실제 검출에는 안 쓰임)
DEPTH_SCALE = 0.001

VEL_MM_S = 15.0
ACC_MM_S2 = 15.0
JOINT_VEL_DEG_S = 20.0
JOINT_ACC_DEG_S2 = 20.0
WAIT_BETWEEN_WAYPOINTS_SEC = 0.3
GRIPPER_SETTLE_SEC = 1.0

Z_GRASP_OFFSET_MM = -35.0     # 그리퍼가 물체 위에서 닫혀 못 잡는 문제 보정용 (실측 튜닝값)
POSITION_TOLERANCE_MM = 5.0   # 이동 후 도달 검증 허용 오차

# --- 위치가 조금 틀어지는 문제용 보정 상수 (설명은 파일 상단 참고) ---
# 실제 로봇에서 목표 대비 어긋난 방향/거리를 측정해서 채워 넣을 것. 지금은 0으로 비활성.
# 도구 픽업용과 손 접근용을 분리했다 (검출 방식/카메라 거리가 달라 어긋나는
# 정도가 다를 수 있음).
TOOL_POSITION_OFFSET_X_MM = 0.0
TOOL_POSITION_OFFSET_Y_MM = 0.0
TOOL_POSITION_OFFSET_Z_MM = 0.0

HAND_POSITION_OFFSET_X_MM = 0.0
HAND_POSITION_OFFSET_Y_MM = 0.0
HAND_POSITION_OFFSET_Z_MM = 0.0

# --- 항상 이 조인트 각도(deg)에서 시작하고, 끝나면 여기로 복귀 ---
HOME_JOINT_DEG = [0, 0, 90, 0, 90, 0]

# --- 손 접근(한 번만 이동) ---
HAND_APPROACH_Z_OFFSET_MM = 20.0   # 손 위 2cm에서 정지
HAND_DET_SAMPLES_REQUIRED = 8      # 이동 전 median으로 안정화할 프레임 수 (도구 검출과 동일 방식)

# --- 정지 상태에서 쥐는 동작(주먹, fist) 감지 ---
# 검지/중지/약지/소지 끝이 팔목+MCP 관절 중심(palm_center)에서 각자의 MCP
# 관절보다 이 비율만큼 더 가까우면 그 손가락은 "말림"으로 본다. 값이 작을수록
# (예: 0.7) 더 확실하게 말려야 인정하고, 클수록(예: 0.95) 살짝만 굽혀도 인정한다.
FIST_CURL_RATIO = 0.85
FIST_MIN_CURLED_FINGERS = 3     # 4개 손가락 중 이만큼 이상 말리면 주먹으로 판정
FIST_CONFIRM_SEC = 0.3          # 이 시간 동안 계속 주먹 상태가 유지되면 확정
FIST_LOST_GRACE_SEC = 0.5       # 쥐는 순간 손가락이 살짝 가려져도 무시할 유예시간

# --- 거부된 도구를 원래 위치에 도로 내려놓고 다시 찾기 ---
# 잘못된 도구를 잡았을 때(음성으로 "아니야" 등 거부) pick_tool()이 기록해둔
# 마지막 grasp 위치(_last_grasp_pose) 그대로 다시 내려놓는다. 실측이 필요한
# 별도 위치가 없어서 로봇 워크스페이스마다 다시 잴 필요가 없다.
MAX_REJECT_RETRIES = 3    # 같은 도구 요청에 대해 거부-재시도를 허용하는 최대 횟수

GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"

try:
    gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)
    gripper.move_gripper(800)
except Exception as e:
    print(f"[그리퍼 연결 실패] {e}")
    sys.exit(1)

# ------------------------------------
# 음성 관련: get_keyword 서비스 호출 + 거부("아니야") 확인용 짧은 STT.
# voice_processing 패키지(get_keyword.py, stt.py 등)는 건드리지 않고 재사용만 한다.
# ------------------------------------
# ditto_system는 colcon ament_python 패키지로 빌드/설치되므로
# get_package_share_directory()로 resource/를 정상적으로 찾을 수 있다.
VOICE_RESOURCE_PATH = os.path.join(get_package_share_directory("ditto_system"), "resource")
REJECTION_LISTEN_SEC = 3        # 그리퍼로 잡은 직후 "아니야" 등 거부 발화를 들을 시간
REJECTION_PHRASES = ["아니야", "아니에요", "아니예요", "그거 아니", "틀렸", "다른 거", "다른거", "아니 그게"]

_rejection_stt = None
try:
    load_dotenv(dotenv_path=os.path.join(VOICE_RESOURCE_PATH, ".env"))
    _openai_api_key = os.getenv("OPENAI_API_KEY")
    if _openai_api_key:
        _rejection_stt = STT(openai_api_key=_openai_api_key)
        _rejection_stt.duration = REJECTION_LISTEN_SEC
    else:
        print("[음성] OPENAI_API_KEY를 못 찾아 거부 확인(STT)이 비활성화됩니다.")
except Exception as e:
    print(f"[음성] .env/STT 초기화 실패 - 거부 확인이 비활성화됩니다: {e}")


def listen_for_rejection():
    """그리퍼로 도구를 잡은 직후 몇 초간 녹음해서, 사용자가 '아니야' 류의 거부
    발화를 했는지 확인한다. STT가 준비 안 됐으면(예: API 키 없음) 항상 False."""
    if _rejection_stt is None:
        return False
    print(f"[확인] 맞는 도구인지 {REJECTION_LISTEN_SEC}초간 들을게요 (아니면 '아니야'라고 말해주세요)")
    try:
        text = _rejection_stt.speech2text()
    except Exception as e:
        print(f"[확인] STT 오류로 거부 확인을 건너뜁니다: {e}")
        return False
    rejected = any(phrase in text for phrase in REJECTION_PHRASES)
    if rejected:
        print(f"[확인] 거부 표현 감지: '{text}'")
    return rejected


def request_tool_via_voice(timeout_sec=60.0):
    """get_keyword 서비스를 호출해서 도구 이름 리스트를 받는다. 그 노드가
    웨이크워드 대기 -> STT -> LLM 추출까지 전부 처리하고, 여기서는
    결과 문자열(공백 구분)만 받아서 리스트로 나눈다."""
    if not get_keyword_client.wait_for_service(timeout_sec=5.0):
        print("[음성] get_keyword 서비스를 찾을 수 없습니다. voice_processing 노드가 켜져 있는지 확인하세요.")
        return []
    future = get_keyword_client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(voice_node, future, timeout_sec=timeout_sec)
    if not future.done() or future.result() is None:
        print("[음성] 응답을 받지 못했습니다 (타임아웃).")
        return []
    result = future.result()
    if not result.success or not result.message.strip():
        print("[음성] 도구를 인식하지 못했습니다.")
        return []
    tools = result.message.split()
    print(f"[음성] 요청된 도구: {tools}")
    return tools


def close_gripper():
    print("[그리퍼] close")
    gripper.close_gripper()


def open_gripper():
    print("[그리퍼] open")
    gripper.open_gripper()


def apply_tool_position_offset(robot_xyz):
    """handeye 변환 결과(도구 픽업용)에 고정 보정값을 더한다. 전부 0이면 동작 없음."""
    return [
        robot_xyz[0] + TOOL_POSITION_OFFSET_X_MM,
        robot_xyz[1] + TOOL_POSITION_OFFSET_Y_MM,
        robot_xyz[2] + TOOL_POSITION_OFFSET_Z_MM,
    ]


def apply_hand_position_offset(robot_xyz):
    """handeye 변환 결과(손 접근용)에 고정 보정값을 더한다. 전부 0이면 동작 없음."""
    return [
        robot_xyz[0] + HAND_POSITION_OFFSET_X_MM,
        robot_xyz[1] + HAND_POSITION_OFFSET_Y_MM,
        robot_xyz[2] + HAND_POSITION_OFFSET_Z_MM,
    ]


# ------------------------------------
# 주먹(fist) 판정 - MediaPipe 21개 랜드마크 픽셀 좌표만으로 계산한다
# (깊이/실측 좌표 불필요, hand_pinch.py는 건드리지 않고 이 파일 안에서만 쓴다).
# 표준 MediaPipe Hand 랜드마크 인덱스: 0=WRIST, 5/9/13/17=검지/중지/약지/소지 MCP,
# 8/12/16/20=같은 손가락들의 TIP.
# ------------------------------------
_WRIST = 0
_FINGER_MCP_TIP_PAIRS = [(5, 8), (9, 12), (13, 16), (17, 20)]  # (MCP, TIP) - 검지/중지/약지/소지


def is_fist(hand):
    """네 손가락(검지/중지/약지/소지) 중 FIST_MIN_CURLED_FINGERS개 이상이
    "말린" 상태면 주먹으로 판정한다. 손바닥 중심(palm_center, 손목+4개 MCP
    평균)에서 각 손가락의 TIP까지 거리가 그 손가락 MCP까지 거리보다 충분히
    가까우면 그 손가락은 말린 것으로 본다."""
    mcp_indices = [mcp for mcp, _ in _FINGER_MCP_TIP_PAIRS]
    palm_pts = [hand[_WRIST]] + [hand[i] for i in mcp_indices]
    palm_center = (sum(p[0] for p in palm_pts) / len(palm_pts),
                   sum(p[1] for p in palm_pts) / len(palm_pts))

    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    curled = 0
    for mcp_idx, tip_idx in _FINGER_MCP_TIP_PAIRS:
        dist_tip = dist(hand[tip_idx], palm_center)
        dist_mcp = dist(hand[mcp_idx], palm_center)
        if dist_tip < dist_mcp * FIST_CURL_RATIO:
            curled += 1
    return curled >= FIST_MIN_CURLED_FINGERS


# ------------------------------------
# RealSense + 검출기 (도구/손 실시간 검출용)
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
hand_pinch = HandPinchTracker(max_num_hands=1)

WINDOW_NAME = "robot_replay - live recognition status"

# 웹 백엔드(backend/routes/status.py)가 폴링해서 UI에 보여줄 수 있도록, 매 프레임마다
# 최신 이미지/상태를 이 경로에 덮어쓴다. 로컬 파일 기반 MVP - ROS 토픽 publish보다
# 훨씬 간단하고, robot_replay.py 쪽 로직은 이 몇 줄 외에는 안 건드려도 된다.
RUNTIME_DIR = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/.runtime")
_STATUS_JSON_PATH = os.path.join(RUNTIME_DIR, "status.json")
_STATUS_JSON_TMP_PATH = _STATUS_JSON_PATH + ".tmp"
_FRAME_JPG_PATH = os.path.join(RUNTIME_DIR, "frame.jpg")
_FRAME_JPG_TMP_PATH = _FRAME_JPG_PATH + ".tmp"
os.makedirs(RUNTIME_DIR, exist_ok=True)


def _publish_runtime_status(frame_bgr, phase, status_text):
    """최신 프레임(JPEG)과 상태를 파일로 덮어쓴다. 쓰는 도중 읽히지 않도록 임시
    파일에 먼저 쓰고 os.replace()로 원자적으로 교체한다."""
    try:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with open(_FRAME_JPG_TMP_PATH, "wb") as f:
                f.write(buf.tobytes())
            os.replace(_FRAME_JPG_TMP_PATH, _FRAME_JPG_PATH)

        with open(_STATUS_JSON_TMP_PATH, "w") as f:
            json.dump({
                "phase": phase,
                "status_text": status_text,
                "updated_at": time.time(),
            }, f)
        os.replace(_STATUS_JSON_TMP_PATH, _STATUS_JSON_PATH)
    except OSError as e:
        print(f"[상태 파일 기록 오류] {e}")


def get_frame():
    """RealSense에서 aligned rgb/depth 한 쌍을 얻는다 (타임아웃 재시도 포함)."""
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
        frame_bgr = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_bgr, rgb, depth_image


# ------------------------------------
# 카메라 디버그 스레드
#
# display_phase 값:
#   "IDLE"         - 아직 아무 것도 안 함
#   "TOOL_SEARCH"  - 도구 인식 시도 중 (1단계: pick_tool)
#   "HAND_SEARCH"  - 손 위치 검출/안정화 중 (3단계: 한 번만 접근하기 위함)
#   "GRASP_WAIT"   - 로봇 정지 상태, 사람이 도구를 쥐는지(주먹 fist) 대기 중
#   "MOVING"       - 로봇이 movel/movej 중이라 새로 검출은 안 하고 라이브 영상만 표시
#   "DONE"         - 시퀀스 종료
# ------------------------------------
cam_lock = threading.Lock()
latest_rgb = None
latest_depth = None
latest_tool_det = None
latest_is_fist = None   # GRASP_WAIT 단계에서 이번 프레임의 주먹 판정 결과 (검출 실패 시 None)
display_phase = "IDLE"
display_status_text = "Waiting..."
display_status_color = (0, 255, 0)
cam_running = True
CURRENT_TOOL_PROMPTS = TOOL_PROMPTS[DEFAULT_TOOL_PROMPT]  # camera_worker가 TOOL_SEARCH 때 검출할 프롬프트 후보 리스트


def set_tool_prompt(name):
    """다음 TOOL_SEARCH부터 이 도구(TOOL_PROMPTS의 키)의 프롬프트 후보들로 찾도록
    전환한다 (음성 요청용). 후보 여러 개는 전부 같은 물체를 가리키는 동의어라,
    그 중 어느 문구로든 검출되면(가장 confidence가 높은 것) 그 도구로 인정한다."""
    global CURRENT_TOOL_PROMPTS
    prompts = TOOL_PROMPTS[name]
    with cam_lock:
        CURRENT_TOOL_PROMPTS = prompts
    print(f"[설정] 검출 대상 도구 -> '{name}' (프롬프트 후보: {prompts})")


TOOL_DET_HISTORY_MAXLEN = 15
tool_det_history = deque(maxlen=TOOL_DET_HISTORY_MAXLEN)  # [(xyz_m, conf), ...]

HAND_DET_HISTORY_MAXLEN = 15
hand_det_history = deque(maxlen=HAND_DET_HISTORY_MAXLEN)  # [xyz_m(카메라 좌표계), ...]


def set_phase(phase, status_text=None, status_color=(0, 255, 0)):
    global display_phase, display_status_text, display_status_color
    with cam_lock:
        if phase == "TOOL_SEARCH" and display_phase != "TOOL_SEARCH":
            tool_det_history.clear()
        if phase == "HAND_SEARCH" and display_phase != "HAND_SEARCH":
            hand_det_history.clear()
        display_phase = phase
        if status_text is not None:
            display_status_text = status_text
        display_status_color = status_color


def set_status(status_text, status_color=(0, 255, 0)):
    global display_status_text, display_status_color
    with cam_lock:
        display_status_text = status_text
        display_status_color = status_color


def camera_worker():
    global latest_rgb, latest_depth, latest_tool_det, latest_is_fist

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while cam_running:
        try:
            frame_bgr, rgb, depth_image = get_frame()
        except Exception as e:
            print(f"[카메라 스레드 오류] {e}")
            time.sleep(0.05)
            continue

        with cam_lock:
            phase = display_phase
            status_text = display_status_text
            status_color = display_status_color

        tool_det = None
        hand = None            # 이번 프레임 MediaPipe 검출 랜드마크 (그리기용, 실패하면 None)
        hand_xyz_m = None       # HAND_SEARCH: 손 중심의 카메라 좌표계 3D 위치
        fist_now = None         # GRASP_WAIT: 이번 프레임 주먹 판정 결과 (bool)

        if phase == "TOOL_SEARCH":
            with cam_lock:
                tool_prompts = CURRENT_TOOL_PROMPTS
            try:
                # 후보 프롬프트(동의어) 전부를 한 번에 검출해서, 그 중 가장
                # confidence가 높은 것을 이번 프레임의 검출 결과로 쓴다.
                results = detector.detect_multi(rgb, depth_image, intrinsics, tool_prompts)
                candidates = [r for r in results.values() if r is not None]
                tool_det = max(candidates, key=lambda r: r["conf"]) if candidates else None
            except Exception as e:
                print(f"[도구 검출 오류] {e}")
        elif phase == "HAND_SEARCH":
            try:
                hands = hand_pinch.process(rgb)
                hand = hands[0] if hands else None
            except Exception as e:
                print(f"[손 검출 오류] {e}")
                hand = None
            if hand is not None:
                hand_px = HandPinchTracker.center_px(hand)
                hand_xyz_m = pixel_to_camera_point(hand_px[0], hand_px[1], depth_image, intrinsics, DEPTH_SCALE)
        elif phase == "GRASP_WAIT":
            try:
                hands = hand_pinch.process(rgb)
                hand = hands[0] if hands else None
            except Exception as e:
                print(f"[손 검출 오류] {e}")
                hand = None
            if hand is not None:
                fist_now = is_fist(hand)

        if tool_det is not None:
            x1, y1, x2, y2 = tool_det["box"]
            cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"tool conf={tool_det['conf']:.2f}",
                        (int(x1), max(0, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if hand is not None:
            if phase == "GRASP_WAIT":
                color = (0, 255, 0) if fist_now else (0, 165, 255)
                for (x, y) in hand:
                    cv2.circle(frame_bgr, (int(x), int(y)), 2, color, -1)
                cx, cy = HandPinchTracker.center_px(hand)
                cv2.putText(frame_bgr, "FIST" if fist_now else "open", (int(cx) + 10, int(cy) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                for (x, y) in hand:
                    cv2.circle(frame_bgr, (int(x), int(y)), 2, (255, 128, 0), -1)
                cx, cy = HandPinchTracker.center_px(hand)
                cv2.putText(frame_bgr, "hand", (int(cx) + 10, int(cy) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)

        with cam_lock:
            latest_rgb, latest_depth = rgb, depth_image
            latest_tool_det = tool_det
            if phase == "TOOL_SEARCH":
                if tool_det is not None:
                    tool_det_history.append((tool_det["xyz"], tool_det["conf"]))
                else:
                    tool_det_history.clear()
            if phase == "HAND_SEARCH":
                if hand_xyz_m is not None:
                    hand_det_history.append(np.array(hand_xyz_m))
                else:
                    hand_det_history.clear()
            if phase == "GRASP_WAIT":
                latest_is_fist = fist_now

        cv2.putText(frame_bgr, f"phase: {phase}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame_bgr, status_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        _publish_runtime_status(frame_bgr, phase, status_text)
        cv2.imshow(WINDOW_NAME, frame_bgr)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


def move_and_verify(posx, label, vel=None, acc=None):
    """movel은 NOT REACHABLE이어도 예외를 안 던질 수 있어서, 이동 후 실제
    위치를 다시 읽어 목표와 비교한다."""
    vel = VEL_MM_S if vel is None else vel
    acc = ACC_MM_S2 if acc is None else acc
    movel(posx, vel=vel, acc=acc, ref=DR_BASE, mod=DR_MV_MOD_ABS)
    mwait()

    actual, _ = get_current_posx()
    dist = sum((a - b) ** 2 for a, b in zip(actual[:3], posx[:3])) ** 0.5
    if dist > POSITION_TOLERANCE_MM:
        raise RuntimeError(
            f"{label}: 목표 {[round(v, 1) for v in posx[:3]]} 근처로 못 갔습니다 "
            f"(실제 {[round(v, 1) for v in actual[:3]]}, 차이 {dist:.1f}mm). "
            f"IK 도달 불가(NOT REACHABLE)이거나, mod(절대/상대)가 실제 DSR_ROBOT2와 "
            f"안 맞을 가능성이 있습니다."
        )


def move_and_verify_arc(via_posx, target_posx, vel, acc, blend_radius, label):
    """movec()로 via_posx를 지나 target_posx까지 원호로 이동.
    [!] movec 실제 시그니처 미확인 상태의 가정 코드
        (movec(via, target, vel, acc, ref, mod) 순서로 가정). 실행 전 반드시
        import DSR_ROBOT2, inspect; print(inspect.signature(DSR_ROBOT2.movec))
    로 확인하고 아래 호출부 인자 순서/이름을 맞출 것 (특히 blend_radius를 movec가
    직접 받는지, 별도 파라미터인지 확인 필요 - 지금은 값만 받아두고 실제 호출에는
    안 쓰고 있다)."""
    if not MOVEC_AVAILABLE:
        raise RuntimeError("movec를 DSR_ROBOT2에서 import하지 못했습니다. 위 [확인 필요] 로그를 보세요.")

    movec(via_posx, target_posx, vel=vel, acc=acc, ref=DR_BASE, mod=DR_MV_MOD_ABS)
    mwait()

    actual, _ = get_current_posx()
    dist = sum((a - b) ** 2 for a, b in zip(actual[:3], target_posx[:3])) ** 0.5
    if dist > POSITION_TOLERANCE_MM:
        raise RuntimeError(
            f"{label}: 목표 근처로 못 갔습니다 (차이 {dist:.1f}mm). "
            f"movec 시그니처 가정이 실제와 다를 가능성이 있습니다."
        )


def go_home():
    """항상 같은 조인트 각도로 시작/복귀한다."""
    print(f"[홈] 조인트 위치로 이동: {HOME_JOINT_DEG} (deg)")
    movej(HOME_JOINT_DEG, vel=JOINT_VEL_DEG_S, acc=JOINT_ACC_DEG_S2)
    mwait()


def confirm_start(n_segments):
    print("=" * 50)
    print(f" 실제 로봇 재생: 조인트 홈 -> 도구 자동 픽업 -> 궤적 구간 {n_segments}개 -> 손으로 한 번 접근 후 정지 -> 주먹 감지 시 전달")
    print(f" 속도: {VEL_MM_S}mm/s, 가속도: {ACC_MM_S2}mm/s^2")
    print(" 도구가 카메라에 보이는 곳에 놓여 있는지 확인하세요.")
    print(" 로봇 주변에 사람/장애물이 없는지, 비상정지 버튼에 손이 닿는지 확인하세요.")
    print(" [주의] 손/쥐는 동작 대기 단계는 검출될 때까지 무한정 기다립니다 (타임아웃 없음).")
    print("=" * 50)
    if not sys.stdin.isatty():
        # 백엔드가 웹 UI "실행" 버튼을 눌러 서브프로세스로 띄운 경우 - 터미널이 없어서
        # input()이 즉시 EOFError로 죽는다 (표준입력이 연결 안 됨). "실행" 버튼을 누른
        # 것 자체가 사람의 확인이므로, 그 경우엔 위 안전 안내만 로그로 남기고 바로 진행한다.
        print("[자동 진행] 터미널 입력이 불가능한 환경(백엔드 서브프로세스)이라 확인 없이 바로 시작합니다.")
        return True
    answer = input("정말 실행할까요? 'yes'를 정확히 입력하세요: ")
    return answer.strip() == "yes"


# --- 안전: 이 높이(로봇 베이스 기준 z, mm)보다 아래로는 절대 안 내려간다. ---
MIN_SAFE_Z_MM = 20.0
APPROACH_HEIGHT_MM = 80.0  # 잡기 전, 목표 지점 위 이만큼 높이에서 먼저 정지 (hover)


def clamp_z_safe(z_mm: float, label: str) -> float:
    if z_mm < MIN_SAFE_Z_MM:
        raise RuntimeError(
            f"{label}: 목표 z={z_mm:.1f}mm가 안전 최소 높이 MIN_SAFE_Z_MM={MIN_SAFE_Z_MM}mm보다 "
            f"낮습니다. 바닥/테이블에 걸릴 위험이 있어 이동을 중단합니다."
        )
    return z_mm


def compute_grasp_rz(mask, base_rz):
    """축정렬이 잘 맞는 것으로 확인됐으니 로직을 건드리지 않는다."""
    if TOOL_AXIS_REFERENCE_DEG is None or mask is None:
        return base_rz
    current_angle = mask_yaw_deg(mask)
    if current_angle is None:
        return base_rz

    raw_delta = current_angle - TOOL_AXIS_REFERENCE_DEG
    delta = ((raw_delta + 90) % 180) - 90  # 180도 주기 wrap -> [-90, 90)

    print(f"[도구 축] 현재 각도={current_angle:.1f}deg (기준={TOOL_AXIS_REFERENCE_DEG:.1f}deg, "
          f"보정={delta:.1f}deg)")
    return base_rz + delta


DET_SAMPLES_REQUIRED = 8
DET_SAMPLE_TIMEOUT_SEC = 5.0
DET_MIN_CONF = 0.35


def sample_stable_tool_xyz():
    start = time.time()
    while time.time() - start < DET_SAMPLE_TIMEOUT_SEC:
        with cam_lock:
            samples = [xyz for xyz, conf in tool_det_history if conf >= DET_MIN_CONF]
        if len(samples) >= DET_SAMPLES_REQUIRED:
            arr = np.array(samples[-DET_SAMPLES_REQUIRED:])
            median_xyz = np.median(arr, axis=0)
            spread_mm = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0))) * 1000.0
            return median_xyz, len(arr), spread_mm
        time.sleep(0.05)

    with cam_lock:
        samples = [xyz for xyz, conf in tool_det_history if conf >= DET_MIN_CONF]
    if not samples:
        return None, 0, None
    arr = np.array(samples)
    median_xyz = np.median(arr, axis=0)
    spread_mm = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0))) * 1000.0 if len(arr) > 1 else 0.0
    return median_xyz, len(arr), spread_mm


# ------------------------------------
# 1) 도구 검출 -> 축정렬(rz) -> 잡기
#    도구 중심 좌표를 그대로 쓴다 (사람이 잡았던 지점 재현 로직은 없음).
#    handeye 변환 뒤 apply_tool_position_offset()으로 위치 보정 상수를 더한다.
# ------------------------------------
_last_grasp_pose = None  # 마지막으로 성공한 픽업의 hover/grasp posx (거부 시 원위치 복귀용)


def pick_tool(timeout_sec=30.0):
    global _last_grasp_pose
    print("\n[1단계] 도구 검출 중...")
    set_phase("TOOL_SEARCH", "[1] Searching for tool...", (0, 0, 255))
    start = time.time()
    base_rx, base_ry, base_rz = ORIENTATION_RX_DEG, ORIENTATION_RY_DEG, ORIENTATION_RZ_DEG

    while time.time() - start < timeout_sec:
        with cam_lock:
            det = latest_tool_det

        if det is None:
            time.sleep(0.05)
            continue

        set_status(f"[1] tool detected conf={det['conf']:.2f}, stabilizing...", (0, 255, 0))
        print(f"[1] 도구 검출됨 conf={det['conf']:.2f} -> 최근 {DET_SAMPLES_REQUIRED}프레임 중앙값으로 위치 안정화 중...")

        median_xyz_m, n_samples, spread_mm = sample_stable_tool_xyz()
        if median_xyz_m is None:
            time.sleep(0.05)
            continue

        if spread_mm is not None and spread_mm > 15.0:
            print(f"[경고] 최근 검출 위치 jitter={spread_mm:.1f}mm로 큽니다.")
        print(f"[1] {n_samples}개 프레임 중앙값 사용 (jitter={spread_mm:.1f}mm)  "
              f"카메라 xyz={median_xyz_m.round(4).tolist()} m")

        tool_xyz_mm = median_xyz_m * 1000.0
        robot_xyz = handeye.transform_to_base(tool_xyz_mm)
        robot_xyz = apply_tool_position_offset(robot_xyz)  # 위치 보정 (도구 픽업용)
        grasp_z = clamp_z_safe(robot_xyz[2] + Z_GRASP_OFFSET_MM, "도구 픽업")
        rz = compute_grasp_rz(det.get("mask"), base_rz)

        current_pose, _ = get_current_posx()
        current_z = current_pose[2]

        # 1a) 지금 z(높이)를 유지한 채로 XY만 이동
        horizontal_posx = [robot_xyz[0], robot_xyz[1], current_z, base_rx, base_ry, rz]
        print(f"[1a. 수평 이동] z={current_z:.1f} 유지, XY만 목표로 -> {[round(v, 1) for v in horizontal_posx]}")
        open_gripper()
        set_phase("MOVING", "[1a] Moving horizontally...", (0, 255, 255))
        move_and_verify(horizontal_posx, "픽업 수평 이동")

        # 1b) 그 자리에서 수직으로 접근 높이까지 하강
        hover_posx = [robot_xyz[0], robot_xyz[1], grasp_z + APPROACH_HEIGHT_MM, base_rx, base_ry, rz]
        print(f"[1b. 접근 높이로 하강] -> {[round(v, 1) for v in hover_posx]}")
        set_phase("MOVING", "[1b] Descending to approach height...", (0, 255, 255))
        move_and_verify(hover_posx, "픽업 접근(hover)")

        # 1c) 마지막으로 실제 파지 높이까지 하강 + 파지
        grasp_posx = [robot_xyz[0], robot_xyz[1], grasp_z, base_rx, base_ry, rz]
        print(f"[1c. 파지 하강] -> {[round(v, 1) for v in grasp_posx]}")
        set_phase("MOVING", "[1c] Descending to grasp position...", (0, 255, 255))
        move_and_verify(grasp_posx, "픽업 위치")
        close_gripper()
        time.sleep(GRIPPER_SETTLE_SEC)
        _last_grasp_pose = {"hover_posx": hover_posx, "grasp_posx": grasp_posx}

        print("[1d. 상승] 다시 접근 높이로 복귀")
        set_phase("MOVING", "[1d] Rising back to approach height...", (0, 255, 255))
        move_and_verify(hover_posx, "픽업 후 상승")
        return True

    print("[실패] 시간 안에 도구를 찾지 못했습니다.")
    return False


# ------------------------------------
# 2) 궤적 재생 (frames[0]은 픽업 위치와 거의 같은 지점이라 건너뛴다)
# ------------------------------------
def replay_trajectory(frames, base_rz, first_yaw):
    set_phase("MOVING", "[2] Replaying trajectory...", (0, 255, 255))
    prev_time = frames[0]["time"]
    for i, frame in enumerate(frames[1:], start=1):
        rz = base_rz + (frame["yaw"] - first_yaw)
        z = clamp_z_safe(frame["z"] + Z_GRASP_OFFSET_MM, f"waypoint {i + 1}")
        posx = [frame["x"], frame["y"], z, ORIENTATION_RX_DEG, ORIENTATION_RY_DEG, rz]
        print(f"[{i + 1}/{len(frames)}] t={frame['time']}s -> {[round(v, 1) for v in posx]}")
        set_status(f"[2] Replaying trajectory... ({i + 1}/{len(frames)})", (0, 255, 255))
        move_and_verify(posx, f"waypoint {i + 1}")

        dt = frame["time"] - prev_time
        prev_time = frame["time"]
        time.sleep(max(0.0, min(dt, 2.0)))
        time.sleep(WAIT_BETWEEN_WAYPOINTS_SEC)


# ------------------------------------
# 2') 스킬 실행 - "스킬 생성"/"순차 블록" 탭이 만든 move_plan.json(v2, 좌표까지 포함)을
#    그대로 읽어서 movel/movec을 실행한다. classify_trajectory/GPT는 그 탭들에서
#    이미 끝낸 작업이라 여기서는 다시 안 부른다 - smoothed json도 안 읽는다.
# ------------------------------------
def load_move_plan(path):
    """move_plan.json(v2) 경로 -> sequence 리스트. v1(좌표 없음)이거나 형식이 안
    맞으면 None (실행 불가 - "스킬 생성"에서 다시 만들어야 함)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get("format") == "v2" and isinstance(data.get("sequence"), list):
        return data["sequence"]
    return None


def _execute_sequence(sequence):
    """v2 시퀀스(각 항목에 move_type/vel/acc/blend와 end_pose/via_pose 좌표 - yaw는
    이미 rz 보정까지 구워진 최종값 - 가 다 들어있음)를 그 순서 그대로 실행한다."""
    set_phase("MOVING", "[skill] Running sequence...", (0, 255, 255))
    for i, item in enumerate(sequence):
        label = f"segment {i} ({item['move_type']}, {item.get('source', 'manual_edit')})"

        end_pose = item["end_pose"]
        end_posx = [
            end_pose["x"], end_pose["y"],
            clamp_z_safe(end_pose["z"] + Z_GRASP_OFFSET_MM, label),
            ORIENTATION_RX_DEG, ORIENTATION_RY_DEG,
            end_pose["yaw"],
        ]

        if item["move_type"] == "movec" and item.get("via_pose"):
            via_pose = item["via_pose"]
            via_posx = [
                via_pose["x"], via_pose["y"],
                clamp_z_safe(via_pose["z"] + Z_GRASP_OFFSET_MM, label + " via"),
                ORIENTATION_RX_DEG, ORIENTATION_RY_DEG,
                via_pose["yaw"],
            ]
            move_and_verify_arc(via_posx, end_posx, vel=item["vel_mm_s"], acc=item["acc_mm_s2"],
                                 blend_radius=item["blend_radius_mm"], label=label)
        else:
            move_and_verify(end_posx, label, vel=item["vel_mm_s"], acc=item["acc_mm_s2"])

    movel_count = sum(1 for s in sequence if s["move_type"] == "movel")
    movec_count = len(sequence) - movel_count
    print(f"\n[스킬변경완료] 총 {len(sequence)}개 세그먼트 실행 완료 "
          f"(movel {movel_count}개, movec {movec_count}개)")


# ------------------------------------
# 3a) 손 위치를 검출/안정화해서 딱 한 번만 그쪽으로 이동
#     (더 이상 실시간으로 계속 쫓아가지 않는다 - eye-in-hand 피드백 루프 차단)
# ------------------------------------
def approach_hand_once(base_rx, base_ry, base_rz):
    print("\n[3a단계] 손 위치 검출/안정화 - 손이 검출될 때까지 계속 기다립니다")
    set_phase("HAND_SEARCH", "[3a] Searching for hand...", (0, 0, 255))
    start = time.time()

    while True:
        with cam_lock:
            samples = list(hand_det_history)

        if len(samples) >= HAND_DET_SAMPLES_REQUIRED:
            arr = np.array(samples[-HAND_DET_SAMPLES_REQUIRED:])
            median_xyz_m = np.median(arr, axis=0)
            spread_mm = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0))) * 1000.0
            break

        set_status(f"[3a] Detecting/stabilizing hand... ({len(samples)}/{HAND_DET_SAMPLES_REQUIRED}, "
                   f"elapsed {time.time() - start:.0f}s)", (0, 0, 255))
        time.sleep(0.05)

    print(f"[3a] {len(arr)}개 프레임 중앙값 사용 (jitter={spread_mm:.1f}mm)  "
          f"카메라 xyz={median_xyz_m.round(4).tolist()} m")

    hand_xyz_mm = median_xyz_m * 1000.0
    robot_xyz = handeye.transform_to_base(hand_xyz_mm)
    robot_xyz = apply_hand_position_offset(robot_xyz)  # 위치 보정 (손 접근용)
    target_z = clamp_z_safe(robot_xyz[2] + HAND_APPROACH_Z_OFFSET_MM, "손 접근")

    posx = [robot_xyz[0], robot_xyz[1], target_z, base_rx, base_ry, base_rz]
    print(f"[3a. 손 위로 이동] -> {[round(v, 1) for v in posx]}")
    set_phase("MOVING", "[3a] Moving above hand...", (255, 128, 0))
    move_and_verify(posx, "손 접근")
    print("[3a] 도착 - 로봇은 이제 완전히 정지하고 더 이상 위치를 갱신하지 않습니다.")


# ------------------------------------
# 3b) 정지한 채로 pinch(쥐는 동작) 감지 대기 -> 감지되면 그리퍼 오픈
#     로봇이 안 움직이므로 카메라도 안 움직이고, 위치가 아니라 엄지-검지
#     거리만 보기 때문에 eye-in-hand 피드백 루프 문제가 생기지 않는다.
# ------------------------------------
def wait_for_fist_and_release():
    print("\n[3b단계] 정지 상태에서 쥐는 동작(주먹) 대기 - 계속 기다립니다")
    set_phase("GRASP_WAIT", "[3b] Waiting for grasp (fist)...", (0, 0, 255))
    start = time.time()
    close_since = None
    last_seen_time = None

    while True:
        with cam_lock:
            fist_now = latest_is_fist
        now = time.time()

        if fist_now is not None:
            last_seen_time = now
            if fist_now:
                if close_since is None:
                    close_since = now
            else:
                close_since = None
        elif last_seen_time is not None and (now - last_seen_time) <= FIST_LOST_GRACE_SEC:
            pass  # 유예시간 이내 - close_since 상태를 그대로 유지 (놓친 순간에도 시간은 흐름)
        else:
            close_since = None

        if close_since is not None:
            elapsed = now - close_since
            set_status(f"[3b] Fist forming... {elapsed:.1f}/{FIST_CONFIRM_SEC:.1f}s", (255, 128, 0))
            if elapsed >= FIST_CONFIRM_SEC:
                print(f"[쥠 감지] {elapsed:.1f}초 동안 주먹 상태 유지됨 -> 도구를 쥔 것으로 판단")
                set_status("[3b] Grasp detected! Opening gripper", (0, 255, 0))
                break
        else:
            set_status(f"[3b] Waiting for grasp (fist)... (elapsed {now - start:.0f}s)", (0, 0, 255))

        time.sleep(0.03)

    print("[놓기] 쥐는 동작 감지 -> 그리퍼 열기")
    open_gripper()


# ------------------------------------
# 거부된 도구를 원래 집었던 위치에 도로 내려놓기
# ------------------------------------
def put_back_current_tool():
    """방금 집은 도구를 pick_tool()이 기록해둔 원래 grasp 위치(_last_grasp_pose)
    그대로 도로 내려놓는다. 이후 다시 pick_tool()을 호출하면 그 자리에서 다시
    검출을 시도한다."""
    if _last_grasp_pose is None:
        raise RuntimeError("되돌려 놓을 위치 정보가 없습니다 (pick_tool() 성공 기록이 없음).")

    hover_posx = _last_grasp_pose["hover_posx"]
    grasp_posx = _last_grasp_pose["grasp_posx"]

    print(f"[되돌리기] 원래 집었던 위치 위로 이동 -> {[round(v, 1) for v in hover_posx]}")
    set_phase("MOVING", "[Reject] Putting tool back...", (0, 165, 255))
    move_and_verify(hover_posx, "도구 되돌리기(hover)")

    move_and_verify(grasp_posx, "도구 되돌리기(하강)")
    open_gripper()
    time.sleep(GRIPPER_SETTLE_SEC)

    move_and_verify(hover_posx, "도구 되돌리기 후 상승")


# ------------------------------------
# 도구 픽업 + 거부("아니야") 감지 시 원위치 복귀 후 재시도
# ------------------------------------
def pick_tool_with_reject_retry(label):
    """pick_tool()로 도구를 잡고, 잡은 직후 거부 발화가 감지되면 원래 집었던
    위치에 도로 내려놓고(put_back_current_tool) MAX_REJECT_RETRIES까지 다시
    찾는다. 성공하면 True, 끝내 실패하면 False."""
    for attempt in range(1, MAX_REJECT_RETRIES + 1):
        go_home()
        if not pick_tool():
            print(f"[실패] '{label}'을(를) 찾지 못했습니다.")
            return False

        if listen_for_rejection():
            print(f"[거부] {attempt}/{MAX_REJECT_RETRIES}번째 시도 - 원래 위치에 내려놓고 다시 찾습니다.")
            put_back_current_tool()
            continue

        return True

    print(f"[포기] {MAX_REJECT_RETRIES}번 시도했지만 올바른 '{label}'을(를) 못 찾았습니다.")
    return False


# ------------------------------------
# 음성 요청 하나(도구 이름 하나)를 처리: 픽업 -> (거부 시 원위치 복귀 후 재시도) -> 핸드오버
# ------------------------------------
def fetch_and_handover(tool_name):
    base_rx, base_ry, base_rz = ORIENTATION_RX_DEG, ORIENTATION_RY_DEG, ORIENTATION_RZ_DEG
    set_tool_prompt(tool_name)

    if not pick_tool_with_reject_retry(tool_name):
        return

    approach_hand_once(base_rx, base_ry, base_rz)
    wait_for_fist_and_release()

    print("[복귀] 조인트 홈으로 복귀")
    set_phase("MOVING", "[Return] Moving to home...", (0, 255, 255))
    go_home()
    set_phase("DONE", f"[Done] '{tool_name}' delivered", (0, 255, 0))


def wait_for_tool_via_voice():
    """웨이크워드+STT로 지원하는 도구 이름을 받을 때까지 반복 대기한다.
    한 번에 여러 도구가 인식되면 그 중 지원되는 첫 번째 도구만 쓴다."""
    print("\n[음성 대기] 'hello rokey'라고 부른 뒤 가져올 도구를 말씀하세요.")
    while True:
        tools = request_tool_via_voice()
        for tool_name in tools:
            if tool_name in TOOL_PROMPTS:
                return tool_name
        if tools:
            print(f"[음성] 지원하지 않는 도구입니다 (지원: {list(TOOL_PROMPTS)}). 다시 말씀해주세요.")


def run_voice_fetch_loop():
    """웨이크워드로 부른 뒤 도구 이름을 말하면 그 도구를 찾아다 건네준다.
    get_keyword 서비스가 웨이크워드 대기 -> STT -> LLM 추출까지 처리한다."""
    print("\n[음성 모드] 'hello rokey'라고 부른 뒤 원하는 도구를 말씀하세요. (Ctrl+C로 종료)")
    while True:
        tools = request_tool_via_voice()
        if not tools:
            continue
        for tool_name in tools:
            if tool_name not in TOOL_PROMPTS:
                print(f"[음성] '{tool_name}'은(는) 지원하지 않는 도구입니다. (지원: {list(TOOL_PROMPTS)})")
                continue
            fetch_and_handover(tool_name)


def replay(path: str):
    sequence = load_move_plan(path)
    if not sequence:
        print(f"[중단] '{path}'에 실행 가능한 시퀀스가 없습니다 "
              f"(v1 형식이거나 비어 있음 - '스킬 생성' 탭에서 다시 생성하세요).")
        return

    if not confirm_start(len(sequence)):
        print("[취소] 사용자가 확인하지 않아 중단합니다.")
        return

    skill_name = os.path.basename(path)
    base_rx, base_ry, base_rz = ORIENTATION_RX_DEG, ORIENTATION_RY_DEG, ORIENTATION_RZ_DEG

    try:
        tool_name = wait_for_tool_via_voice()
        set_tool_prompt(tool_name)

        if not pick_tool_with_reject_retry(tool_name):
            return

        print(f"\n[2단계] 궤적 재생 시작 - '{skill_name}' ({len(sequence)}개 세그먼트)")
        _execute_sequence(sequence)

        approach_hand_once(base_rx, base_ry, base_rz)
        wait_for_fist_and_release()

        print("\n[4단계] 조인트 홈으로 복귀")
        set_phase("MOVING", "[4] Returning to home position...", (0, 255, 255))
        go_home()

        set_phase("DONE", "[Done] Sequence complete", (0, 255, 0))
        print("\n[완료] 전체 시퀀스가 끝났습니다.")

    except KeyboardInterrupt:
        print("\n[중단] 사용자가 Ctrl+C로 중단했습니다.")
    except Exception as e:
        set_phase("DONE", f"[Aborted - error] {e}", (0, 0, 255))
        print(f"\n[오류로 중단] {e}")
        raise


def main():
    # 인자로 스킬(move_plan.json, v2) 경로를 주면 그 스킬 재생 데모, 안 주면 음성으로
    # 도구를 요청받아 찾아다 건네주는 모드로 동작한다.
    #   ros2 run ditto_system robot_replay skills/generate/hammering_..._move_plan.json  -> 스킬 재생 데모
    #   ros2 run ditto_system robot_replay                                               -> 음성 fetch 모드
    global cam_running
    cam_thread = threading.Thread(target=camera_worker, daemon=True)
    cam_thread.start()

    try:
        if len(sys.argv) >= 2:
            replay(sys.argv[1])
        else:
            run_voice_fetch_loop()
    except KeyboardInterrupt:
        print("\n[종료] 사용자가 Ctrl+C로 종료했습니다.")
    finally:
        cam_running = False
        cam_thread.join(timeout=2.0)
        hand_pinch.close()
        pipeline.stop()
        try:
            node.destroy_node()
            voice_node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
