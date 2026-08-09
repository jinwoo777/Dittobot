#!/usr/bin/env bash
set -euo pipefail

# Terminal 2 launcher for the local Dittobot API/UI, RealSense, Web Jog,
# fixed ArUco-plane experiment, and acknowledged hardware skill execution.

DITTOBOT_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DITTOBOT_ROS_SETUP_FILE="${DITTOBOT_ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
DITTOBOT_DOOSAN_SETUP_FILE="${DITTOBOT_DOOSAN_SETUP_FILE:-/home/rokey/cobot_ws/install/setup.bash}"
DITTOBOT_RG2_SITE="${DITTOBOT_RG2_SITE:-/home/rokey/wok_wark/ws_cobot_pjt/ws_dsr/install/rokey/lib/python3.10/site-packages}"
DITTOBOT_ROBOT_INTERFACE="${DITTOBOT_ROBOT_INTERFACE:-enp2s0}"
DITTOBOT_ROS_DOMAIN_ID="${DITTOBOT_ROS_DOMAIN_ID:-78}"
DITTOBOT_API_HOST="${DITTOBOT_API_HOST:-127.0.0.1}"
DITTOBOT_API_PORT="${DITTOBOT_API_PORT:-8001}"
DITTOBOT_UI_URL="${DITTOBOT_UI_URL:-http://127.0.0.1:${DITTOBOT_API_PORT}/ui/}"
DITTOBOT_FIXED_REFERENCE_NPZ="${DITTOBOT_REPO_ROOT}/aruco/fixed_workspace_reference.npz"
DITTOBOT_FIXED_BASE_WORKSPACE_JSON="${DITTOBOT_REPO_ROOT}/aruco/results/d435i_plane_scans/scan_20260807_123803/base_workspace_urdf_candidate.json"
DITTOBOT_RUNTIME_WORKSPACE_NPZ="${DITTOBOT_REPO_ROOT}/aruco/runtime/runtime_workspace.npz"

DITTOBOT_CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
  DITTOBOT_CHECK_ONLY=true
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "사용법: $0 [--check]" >&2
  exit 2
fi

for DITTOBOT_REQUIRED_FILE in \
  "$DITTOBOT_ROS_SETUP_FILE" \
  "$DITTOBOT_DOOSAN_SETUP_FILE" \
  "$DITTOBOT_FIXED_REFERENCE_NPZ" \
  "$DITTOBOT_FIXED_BASE_WORKSPACE_JSON"; do
  if [[ ! -r "$DITTOBOT_REQUIRED_FILE" ]]; then
    echo "필수 파일을 읽을 수 없습니다: $DITTOBOT_REQUIRED_FILE" >&2
    exit 1
  fi
done
if [[ ! -d "$DITTOBOT_RG2_SITE" ]]; then
  echo "설치된 RG2 Python 경로가 없습니다: $DITTOBOT_RG2_SITE" >&2
  exit 1
fi

if [[ ! -d "/sys/class/net/${DITTOBOT_ROBOT_INTERFACE}" ]]; then
  echo "로봇 네트워크 인터페이스가 없습니다: ${DITTOBOT_ROBOT_INTERFACE}" >&2
  echo "다른 장비에서는 DITTOBOT_ROBOT_INTERFACE 값을 지정하세요." >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "$DITTOBOT_ROS_SETUP_FILE"
# shellcheck disable=SC1090
source "$DITTOBOT_DOOSAN_SETUP_FILE"
set -u

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ROS 2 명령을 찾을 수 없습니다. setup.bash 경로를 확인하세요." >&2
  exit 1
fi

mkdir -p "$(dirname -- "$DITTOBOT_RUNTIME_WORKSPACE_NPZ")"
cd "$DITTOBOT_REPO_ROOT"

export ROS_DOMAIN_ID="$DITTOBOT_ROS_DOMAIN_ID"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="<CycloneDDS xmlns=\"https://cdds.io/config\"><Domain><General><Interfaces><NetworkInterface name=\"${DITTOBOT_ROBOT_INTERFACE}\"/></Interfaces></General></Domain></CycloneDDS>"
export PYTHONPATH="${DITTOBOT_REPO_ROOT}/src:${DITTOBOT_RG2_SITE}${PYTHONPATH:+:${PYTHONPATH}}"

if ! python3 -c 'import dsr_msgs2.srv, rokey.onrobot_rg2, uvicorn' >/dev/null 2>&1; then
  echo "Python에서 dsr_msgs2.srv, rokey.onrobot_rg2 또는 uvicorn을 import할 수 없습니다." >&2
  exit 1
fi

export OPENAI_MODE=mock
export ROBOT_EXECUTION_MODE=hardware
export ENABLE_HARDWARE_EXECUTION=true
export ROBOT_BACKEND=doosan
export ENABLE_REAL_ROBOT=true
export DRY_RUN=false
export DOOSAN_ROBOT_ID=dsr01
export DOOSAN_ROBOT_MODEL=m0609

export ENABLE_HANDEYE_CALIBRATION=false
export CALIBRATION_POSE_PLAN_APPROVED=false
export CALIBRATION_CELL_SAFETY_VERIFIED=false

export ENABLE_WEB_JOG=true
export JOG_CELL_SAFETY_VERIFIED=true

export ENABLE_ARUCO_EXPERIMENT=true
export ARUCO_EXPERIMENT_CELL_SAFETY_VERIFIED=true
export ARUCO_EXPERIMENT_EXPECTED_TCP=GripperDA_v1
export ARUCO_FIXED_REFERENCE_NPZ="$DITTOBOT_FIXED_REFERENCE_NPZ"
export ARUCO_RUNTIME_WORKSPACE_NPZ="$DITTOBOT_RUNTIME_WORKSPACE_NPZ"

export RG2_MODBUS_HOST="${RG2_MODBUS_HOST:-192.168.1.1}"
export RG2_MODBUS_PORT="${RG2_MODBUS_PORT:-502}"
export RG2_MODBUS_UNIT_ID="${RG2_MODBUS_UNIT_ID:-65}"
export RG2_GRIP_FORCE_N="${RG2_GRIP_FORCE_N:-20}"
export RGBD_MAX_TIMESTAMP_DELTA_MS=50
export LIVE_PICK_GRASP_DEPTH_OFFSET_M="${LIVE_PICK_GRASP_DEPTH_OFFSET_M:--0.013}"
export LIVE_PICK_WORKSPACE_XY_TOLERANCE_M="${LIVE_PICK_WORKSPACE_XY_TOLERANCE_M:-0.001}"

echo "Terminal 2 환경 확인 완료"
echo "  ROS domain : ${ROS_DOMAIN_ID}"
echo "  Robot NIC  : ${DITTOBOT_ROBOT_INTERFACE}"
echo "  API/UI     : ${DITTOBOT_UI_URL}"
echo "  Workspace  : ${DITTOBOT_FIXED_BASE_WORKSPACE_JSON}"
echo "  RG2        : ${RG2_MODBUS_HOST}:${RG2_MODBUS_PORT} (unit ${RG2_MODBUS_UNIT_ID})"

if [[ "$DITTOBOT_CHECK_ONLY" == true ]]; then
  exit 0
fi

echo
echo "고정 T_base_plane/workspace를 사용합니다. 스킬에서 ‘실제 실행’을 누르세요."

release_existing_api_port() {
  local DITTOBOT_WAIT_ATTEMPT
  local DITTOBOT_LISTENER_PID
  local -a DITTOBOT_LISTENER_PIDS=()

  # This launcher owns the local operator UI.  Restarting it must not leave a
  # previous Uvicorn instance holding the requested port and serving stale code.
  mapfile -t DITTOBOT_LISTENER_PIDS < <(
    lsof -nP -t -iTCP:"${DITTOBOT_API_PORT}" -sTCP:LISTEN 2>/dev/null || true
  )
  if [[ "${#DITTOBOT_LISTENER_PIDS[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "기존 API 포트 ${DITTOBOT_API_PORT} 리스너를 종료합니다: ${DITTOBOT_LISTENER_PIDS[*]}"
  kill -TERM "${DITTOBOT_LISTENER_PIDS[@]}" 2>/dev/null || true
  for ((DITTOBOT_WAIT_ATTEMPT = 1; DITTOBOT_WAIT_ATTEMPT <= 20; DITTOBOT_WAIT_ATTEMPT++)); do
    if ! lsof -nP -iTCP:"${DITTOBOT_API_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done

  echo "포트 ${DITTOBOT_API_PORT}가 종료되지 않아 강제 종료합니다." >&2
  for DITTOBOT_LISTENER_PID in "${DITTOBOT_LISTENER_PIDS[@]}"; do
    if kill -0 "$DITTOBOT_LISTENER_PID" 2>/dev/null; then
      kill -KILL "$DITTOBOT_LISTENER_PID" 2>/dev/null || true
    fi
  done
  for ((DITTOBOT_WAIT_ATTEMPT = 1; DITTOBOT_WAIT_ATTEMPT <= 20; DITTOBOT_WAIT_ATTEMPT++)); do
    if ! lsof -nP -iTCP:"${DITTOBOT_API_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done

  echo "포트 ${DITTOBOT_API_PORT}를 비우지 못했습니다." >&2
  return 1
}

release_existing_api_port

open_dittobot_ui_when_ready() {
  local DITTOBOT_UI_OPEN_ATTEMPT

  if ! command -v curl >/dev/null 2>&1 || ! command -v xdg-open >/dev/null 2>&1; then
    echo "브라우저 자동 열기를 사용할 수 없습니다. 직접 접속하세요: ${DITTOBOT_UI_URL}" >&2
    return 0
  fi

  for ((DITTOBOT_UI_OPEN_ATTEMPT = 1; DITTOBOT_UI_OPEN_ATTEMPT <= 100; DITTOBOT_UI_OPEN_ATTEMPT++)); do
    if curl --fail --silent --show-error --max-time 1 --output /dev/null "$DITTOBOT_UI_URL" 2>/dev/null; then
      if ! xdg-open "$DITTOBOT_UI_URL" >/dev/null 2>&1; then
        echo "브라우저를 열지 못했습니다. 직접 접속하세요: ${DITTOBOT_UI_URL}" >&2
      fi
      return 0
    fi
    sleep 0.2
  done

  echo "UI 준비를 기다리는 시간이 초과되었습니다: ${DITTOBOT_UI_URL}" >&2
}

open_dittobot_ui_when_ready &

python3 -m uvicorn robot_skill_system.api.app:create_app \
  --factory \
  --host "$DITTOBOT_API_HOST" \
  --port "$DITTOBOT_API_PORT"
