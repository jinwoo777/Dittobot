#!/usr/bin/env bash
set -euo pipefail

# Terminal 1 launcher for the local Doosan M0609 ROS 2 bringup and RViz.
# It connects the driver/controller stack but does not request robot motion.

DITTOBOT_ROS_SETUP_FILE="${DITTOBOT_ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
DITTOBOT_DOOSAN_SETUP_FILE="${DITTOBOT_DOOSAN_SETUP_FILE:-/home/rokey/cobot_ws/install/setup.bash}"
DITTOBOT_ROBOT_INTERFACE="${DITTOBOT_ROBOT_INTERFACE:-enp2s0}"
DITTOBOT_ROS_DOMAIN_ID="${DITTOBOT_ROS_DOMAIN_ID:-78}"
DITTOBOT_ROBOT_ID="${DITTOBOT_ROBOT_ID:-dsr01}"
DITTOBOT_ROBOT_HOST="${DITTOBOT_ROBOT_HOST:-192.168.1.100}"
DITTOBOT_ROBOT_PORT="${DITTOBOT_ROBOT_PORT:-12345}"
DITTOBOT_ROBOT_MODEL="${DITTOBOT_ROBOT_MODEL:-m0609}"
DITTOBOT_ALLOW_INACTIVE_INTERFACE="${DITTOBOT_ALLOW_INACTIVE_INTERFACE:-false}"

usage() {
  cat <<'EOF'
사용법:
  ./run_terminal1_doosan_rviz.sh [--check]

옵션:
  --check  설정·NIC·ROS package만 확인하고 로봇에 연결하지 않음
  -h, --help  이 도움말 표시

환경변수 기본값:
  DITTOBOT_ROBOT_INTERFACE=enp2s0
  DITTOBOT_ROS_DOMAIN_ID=78
  DITTOBOT_ROBOT_ID=dsr01
  DITTOBOT_ROBOT_HOST=192.168.1.100
  DITTOBOT_ROBOT_PORT=12345
  DITTOBOT_ROBOT_MODEL=m0609
EOF
}

fail() {
  echo "오류: $*" >&2
  exit 1
}

DITTOBOT_CHECK_ONLY=false
case "${1:-}" in
  "")
    ;;
  --check)
    DITTOBOT_CHECK_ONLY=true
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
if [[ "$#" -gt 1 ]]; then
  usage >&2
  exit 2
fi

for DITTOBOT_REQUIRED_FILE in \
  "$DITTOBOT_ROS_SETUP_FILE" \
  "$DITTOBOT_DOOSAN_SETUP_FILE"; do
  if [[ ! -r "$DITTOBOT_REQUIRED_FILE" ]]; then
    fail "필수 setup 파일을 읽을 수 없습니다: ${DITTOBOT_REQUIRED_FILE}"
  fi
done

if [[ ! "$DITTOBOT_ROBOT_INTERFACE" =~ ^[A-Za-z0-9_.-]{1,15}$ ]]; then
  fail "잘못된 NIC 이름입니다: ${DITTOBOT_ROBOT_INTERFACE}"
fi
if [[ ! -d "/sys/class/net/${DITTOBOT_ROBOT_INTERFACE}" ]]; then
  fail "로봇 네트워크 인터페이스가 없습니다: ${DITTOBOT_ROBOT_INTERFACE}"
fi

if [[ "$DITTOBOT_ALLOW_INACTIVE_INTERFACE" != "true" && \
      "$DITTOBOT_ALLOW_INACTIVE_INTERFACE" != "false" ]]; then
  fail "DITTOBOT_ALLOW_INACTIVE_INTERFACE는 true 또는 false여야 합니다."
fi
DITTOBOT_INTERFACE_STATE="$(<"/sys/class/net/${DITTOBOT_ROBOT_INTERFACE}/operstate")"
if [[ "$DITTOBOT_INTERFACE_STATE" != "up" ]]; then
  if [[ "$DITTOBOT_ALLOW_INACTIVE_INTERFACE" != "true" ]]; then
    fail "NIC ${DITTOBOT_ROBOT_INTERFACE} 상태가 ${DITTOBOT_INTERFACE_STATE}입니다. 케이블·IP를 확인하세요."
  fi
  echo "경고: NIC ${DITTOBOT_ROBOT_INTERFACE} 상태가 ${DITTOBOT_INTERFACE_STATE}이지만 명시적 우회로 계속합니다." >&2
fi

if [[ ! "$DITTOBOT_ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] ||
   ((10#${DITTOBOT_ROS_DOMAIN_ID} > 232)); then
  fail "ROS domain은 0..232 정수여야 합니다: ${DITTOBOT_ROS_DOMAIN_ID}"
fi
if [[ ! "$DITTOBOT_ROBOT_ID" =~ ^[A-Za-z][A-Za-z0-9_-]{0,31}$ ]]; then
  fail "잘못된 로봇 ID입니다: ${DITTOBOT_ROBOT_ID}"
fi
if [[ "$DITTOBOT_ROBOT_MODEL" != "m0609" ]]; then
  fail "이 launcher는 m0609만 허용합니다: ${DITTOBOT_ROBOT_MODEL}"
fi
if [[ ! "$DITTOBOT_ROBOT_PORT" =~ ^[0-9]+$ ]] ||
   ((10#${DITTOBOT_ROBOT_PORT} < 1 || 10#${DITTOBOT_ROBOT_PORT} > 65535)); then
  fail "로봇 port는 1..65535 정수여야 합니다: ${DITTOBOT_ROBOT_PORT}"
fi
if [[ ! "$DITTOBOT_ROBOT_HOST" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  fail "로봇 host는 IPv4 주소여야 합니다: ${DITTOBOT_ROBOT_HOST}"
fi
IFS='.' read -r -a DITTOBOT_HOST_OCTETS <<<"$DITTOBOT_ROBOT_HOST"
for DITTOBOT_HOST_OCTET in "${DITTOBOT_HOST_OCTETS[@]}"; do
  if ((10#${DITTOBOT_HOST_OCTET} > 255)); then
    fail "잘못된 로봇 IPv4 주소입니다: ${DITTOBOT_ROBOT_HOST}"
  fi
done

# ROS-generated setup files may legitimately inspect unset variables.
set +u
# shellcheck disable=SC1090
source "$DITTOBOT_ROS_SETUP_FILE"
# shellcheck disable=SC1090
source "$DITTOBOT_DOOSAN_SETUP_FILE"
set -u

if ! command -v ros2 >/dev/null 2>&1; then
  fail "ROS 2 명령을 찾을 수 없습니다. setup.bash 경로를 확인하세요."
fi
if ! ros2 pkg prefix dsr_bringup2 >/dev/null 2>&1; then
  fail "source된 workspace에서 dsr_bringup2 package를 찾을 수 없습니다."
fi

export ROS_DOMAIN_ID="$DITTOBOT_ROS_DOMAIN_ID"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="<CycloneDDS xmlns=\"https://cdds.io/config\"><Domain><General><Interfaces><NetworkInterface name=\"${DITTOBOT_ROBOT_INTERFACE}\"/></Interfaces></General></Domain></CycloneDDS>"

echo "Terminal 1 환경 확인 완료"
echo "  ROS domain : ${ROS_DOMAIN_ID}"
echo "  Robot NIC  : ${DITTOBOT_ROBOT_INTERFACE} (${DITTOBOT_INTERFACE_STATE})"
echo "  Robot      : ${DITTOBOT_ROBOT_ID} / ${DITTOBOT_ROBOT_MODEL}"
echo "  Endpoint   : ${DITTOBOT_ROBOT_HOST}:${DITTOBOT_ROBOT_PORT}"

if [[ "$DITTOBOT_CHECK_ONLY" == true ]]; then
  echo "점검 모드이므로 ros2 launch를 실행하지 않습니다."
  exit 0
fi

echo
echo "Doosan driver/controller와 RViz를 시작합니다. 종료는 Ctrl+C입니다."
echo "이 launcher 자체는 motion 명령을 보내지 않습니다."

exec ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
  name:="$DITTOBOT_ROBOT_ID" \
  mode:=real \
  host:="$DITTOBOT_ROBOT_HOST" \
  port:="$DITTOBOT_ROBOT_PORT" \
  model:="$DITTOBOT_ROBOT_MODEL"
