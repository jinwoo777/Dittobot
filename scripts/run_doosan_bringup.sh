#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_doosan_bringup.sh
  scripts/run_doosan_bringup.sh --virtual
  scripts/run_doosan_bringup.sh --real --confirm-real-robot

With no arguments, starts the Doosan M0609 ROS 2 bringup in the local virtual
controller mode. Real hardware remains opt-in and requires both --real and the
explicit confirmation flag. For backwards compatibility, --confirm-real-robot
alone also selects real mode.

Optional environment overrides:
  ROS_SETUP_FILE          default: /opt/ros/humble/setup.bash
  COBOT_WS_ROOT           default: /home/rokey/cobot_ws
  ROS_DOMAIN_ID           default: 78
  ROBOT_NETWORK_IFACE     default: enp3s0 (real mode only)
  DOOSAN_ROBOT_ID         default: dsr01
  DOOSAN_ROBOT_MODEL      default: m0609
  DOOSAN_VIRTUAL_HOST     default: 127.0.0.1
  DOOSAN_ROBOT_HOST       default: 192.168.1.100 (real mode only)
  DOOSAN_ROBOT_PORT       default: 12345
  DOOSAN_ROBOT_RT_HOST    default: 192.168.137.50
  DOOSAN_ROBOT_COLOR      default: white
  DOOSAN_RVIZ             default: true
EOF
}

operation_mode=virtual
confirmed_real_robot=false

while (($#)); do
  case "$1" in
    --virtual)
      operation_mode=virtual
      shift
      ;;
    --real)
      operation_mode=real
      shift
      ;;
    --confirm-real-robot)
      operation_mode=real
      confirmed_real_robot=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$operation_mode" == real && "$confirmed_real_robot" != true ]]; then
  echo "Real Doosan bringup requires --confirm-real-robot" >&2
  exit 2
fi

ros_setup_file="${ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
cobot_ws_root="${COBOT_WS_ROOT:-/home/rokey/cobot_ws}"
cobot_setup_file="${cobot_ws_root}/install/setup.bash"
robot_network_iface="${ROBOT_NETWORK_IFACE:-enp3s0}"
robot_id="${DOOSAN_ROBOT_ID:-dsr01}"
robot_model="${DOOSAN_ROBOT_MODEL:-m0609}"
robot_port="${DOOSAN_ROBOT_PORT:-12345}"
robot_rt_host="${DOOSAN_ROBOT_RT_HOST:-192.168.137.50}"
robot_color="${DOOSAN_ROBOT_COLOR:-white}"
robot_rviz="${DOOSAN_RVIZ:-true}"

if [[ "$operation_mode" == real ]]; then
  robot_host="${DOOSAN_ROBOT_HOST:-192.168.1.100}"
else
  robot_host="${DOOSAN_VIRTUAL_HOST:-127.0.0.1}"
fi

if [[ ! "$robot_port" =~ ^[0-9]+$ ]] || ((robot_port < 1 || robot_port > 65535)); then
  echo "DOOSAN_ROBOT_PORT must be an integer in [1, 65535]" >&2
  exit 2
fi
if [[ ! "$robot_id" =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]]; then
  echo "DOOSAN_ROBOT_ID contains unsupported characters" >&2
  exit 2
fi
if [[ -z "$robot_model" || -z "$robot_host" ]]; then
  echo "Doosan model and host must be non-empty" >&2
  exit 2
fi

for setup_file in "$ros_setup_file" "$cobot_setup_file"; do
  if [[ ! -r "$setup_file" ]]; then
    echo "Required ROS setup file is not readable: $setup_file" >&2
    exit 1
  fi
done
if [[ "$operation_mode" == real ]] \
  && ! ip link show "$robot_network_iface" >/dev/null 2>&1; then
  echo "Robot network interface does not exist: $robot_network_iface" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "$ros_setup_file"
# shellcheck disable=SC1090
source "$cobot_setup_file"
set -u

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 is unavailable after sourcing the configured workspaces" >&2
  exit 1
fi
if ! ros2 pkg prefix dsr_bringup2 >/dev/null 2>&1; then
  echo "The dsr_bringup2 package is unavailable in COBOT_WS_ROOT=$cobot_ws_root" >&2
  exit 1
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-78}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export DOOSAN_ROBOT_ID="$robot_id"
export DOOSAN_ROBOT_MODEL="$robot_model"

if [[ "$operation_mode" == real ]]; then
  export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
  if [[ -z "${CYCLONEDDS_URI:-}" || "${CYCLONEDDS_URI}" == *'name=""'* ]]; then
    export CYCLONEDDS_URI="<CycloneDDS xmlns=\"https://cdds.io/config\"><Domain><General><Interfaces><NetworkInterface name=\"${robot_network_iface}\"/></Interfaces></General></Domain></CycloneDDS>"
  fi
  export ROBOT_EXECUTION_MODE=hardware
  export ENABLE_HARDWARE_EXECUTION=true
  export ROBOT_BACKEND=doosan
  export ENABLE_REAL_ROBOT=true
  export DRY_RUN=false
else
  export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
  if [[ "${CYCLONEDDS_URI:-}" == *'name=""'* ]]; then
    unset CYCLONEDDS_URI
  fi
  export ROBOT_EXECUTION_MODE=mock
  export ENABLE_HARDWARE_EXECUTION=false
  export ROBOT_BACKEND=mock
  export ENABLE_REAL_ROBOT=false
  export DRY_RUN=true
fi

echo "Starting Doosan bringup in ${operation_mode} mode"
echo "robot=${robot_id} model=${robot_model} host=${robot_host}:${robot_port}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID} ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}"
if [[ "$operation_mode" == real ]]; then
  echo "network_interface=${robot_network_iface}"
fi

launch_arguments=(
  "name:=${robot_id}"
  "mode:=${operation_mode}"
  "host:=${robot_host}"
  "port:=${robot_port}"
  "rt_host:=${robot_rt_host}"
  "model:=${robot_model}"
  "color:=${robot_color}"
  "gui:=${robot_rviz}"
)

if [[ "$operation_mode" == virtual ]]; then
  virtual_container_name="${robot_id}_emulator"

  cleanup_virtual_emulator() {
    local -a container_ids=()
    if command -v docker >/dev/null 2>&1; then
      mapfile -t container_ids < <(
        docker ps -aq --filter "name=^/${virtual_container_name}$" 2>/dev/null
      )
      if ((${#container_ids[@]})); then
        docker rm -f "${container_ids[@]}" >/dev/null
        echo "Stopped virtual Doosan emulator: ${virtual_container_name}"
      fi
    fi
  }

  trap cleanup_virtual_emulator EXIT
  ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py "${launch_arguments[@]}"
else
  exec ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py "${launch_arguments[@]}"
fi
