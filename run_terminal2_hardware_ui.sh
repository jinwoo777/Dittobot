#!/usr/bin/env bash
set -euo pipefail

# Terminal 2 launcher for the local Dittobot API/UI, RealSense, Web Jog, and
# fixed ArUco-plane experiment. Starting the server does not move the robot;
# every motion still requires the corresponding UI safety acknowledgements.

DITTOBOT_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DITTOBOT_ROS_SETUP_FILE="${DITTOBOT_ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
DITTOBOT_DOOSAN_SETUP_FILE="${DITTOBOT_DOOSAN_SETUP_FILE:-/home/rokey/cobot_ws/install/setup.bash}"
DITTOBOT_ROBOT_INTERFACE="${DITTOBOT_ROBOT_INTERFACE:-enp2s0}"
DITTOBOT_ROS_DOMAIN_ID="${DITTOBOT_ROS_DOMAIN_ID:-78}"
DITTOBOT_API_HOST="${DITTOBOT_API_HOST:-127.0.0.1}"
DITTOBOT_API_PORT="${DITTOBOT_API_PORT:-8001}"
DITTOBOT_FIXED_REFERENCE_NPZ="${DITTOBOT_REPO_ROOT}/aruco/fixed_workspace_reference.npz"
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
  "$DITTOBOT_FIXED_REFERENCE_NPZ"; do
  if [[ ! -r "$DITTOBOT_REQUIRED_FILE" ]]; then
    echo "필수 파일을 읽을 수 없습니다: $DITTOBOT_REQUIRED_FILE" >&2
    exit 1
  fi
done

if [[ ! -d "/sys/class/net/${DITTOBOT_ROBOT_INTERFACE}" ]]; then
  echo "로봇 네트워크 인터페이스가 없습니다: ${DITTOBOT_ROBOT_INTERFACE}" >&2
  echo "다른 장비에서는 DITTOBOT_ROBOT_INTERFACE 값을 지정하세요." >&2
  exit 1
fi

# ROS-generated setup files may legitimately inspect unset variables.
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
if ! python3 -c 'import dsr_msgs2.srv, uvicorn' >/dev/null 2>&1; then
  echo "Python에서 dsr_msgs2.srv 또는 uvicorn을 import할 수 없습니다." >&2
  exit 1
fi

mkdir -p "$(dirname -- "$DITTOBOT_RUNTIME_WORKSPACE_NPZ")"
cd "$DITTOBOT_REPO_ROOT"

export ROS_DOMAIN_ID="$DITTOBOT_ROS_DOMAIN_ID"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="<CycloneDDS xmlns=\"https://cdds.io/config\"><Domain><General><Interfaces><NetworkInterface name=\"${DITTOBOT_ROBOT_INTERFACE}\"/></Interfaces></General></Domain></CycloneDDS>"
export PYTHONPATH="${DITTOBOT_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Common hardware gates. These expose the audited Doosan adapters; they do not
# bypass the per-session UI acknowledgements or controller-side validation.
export OPENAI_MODE=mock
export ROBOT_EXECUTION_MODE=hardware
export ENABLE_HARDWARE_EXECUTION=true
export ROBOT_BACKEND=doosan
export ENABLE_REAL_ROBOT=true
export DRY_RUN=false
export DOOSAN_ROBOT_ID=dsr01
export DOOSAN_ROBOT_MODEL=m0609

# The existing hand-eye result is already frozen into the ArUco reference.
# Recalibration stays closed unless it is deliberately commissioned separately.
export ENABLE_HANDEYE_CALIBRATION=false
export CALIBRATION_POSE_PLAN_APPROVED=false
export CALIBRATION_CELL_SAFETY_VERIFIED=false

# Web Jog / six-axis MoveJ gates.
export ENABLE_WEB_JOG=true
export JOG_CELL_SAFETY_VERIFIED=true

# Fixed ArUco-plane experiment gates and portable artifacts.
export ENABLE_ARUCO_EXPERIMENT=true
export ARUCO_EXPERIMENT_CELL_SAFETY_VERIFIED=true
export ARUCO_EXPERIMENT_EXPECTED_TCP=GripperDA_v1
export ARUCO_FIXED_REFERENCE_NPZ="$DITTOBOT_FIXED_REFERENCE_NPZ"
export ARUCO_RUNTIME_WORKSPACE_NPZ="$DITTOBOT_RUNTIME_WORKSPACE_NPZ"

# D435i RGB/depth frameset synchronization limit at 30 FPS.
export RGBD_MAX_TIMESTAMP_DELTA_MS=50

echo "Terminal 2 환경 확인 완료"
echo "  ROS domain : ${ROS_DOMAIN_ID}"
echo "  Robot NIC  : ${DITTOBOT_ROBOT_INTERFACE}"
echo "  API/UI     : http://${DITTOBOT_API_HOST}:${DITTOBOT_API_PORT}/ui/"
echo "  ArUco ref  : ${ARUCO_FIXED_REFERENCE_NPZ}"

if [[ "$DITTOBOT_CHECK_ONLY" == true ]]; then
  exit 0
fi

echo
echo "주의: 서버 실행 후 UI에서 안전 항목을 확인하면 실제 M0609가 움직일 수 있습니다."
read -r -p "작업공간이 비어 있고 E-stop을 즉시 사용할 수 있으면 RUN을 입력하세요: " DITTOBOT_START_CONFIRMATION
if [[ "$DITTOBOT_START_CONFIRMATION" != "RUN" ]]; then
  echo "실행을 취소했습니다."
  exit 1
fi

python3 -m uvicorn robot_skill_system.api.app:create_app \
  --factory \
  --host "$DITTOBOT_API_HOST" \
  --port "$DITTOBOT_API_PORT"
