#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_dittobot_api.sh [--host HOST] [--port PORT]
  scripts/run_dittobot_api.sh --hardware --confirm-real-robot [--host HOST] [--port PORT]
  scripts/run_dittobot_api.sh --hardware-calibration --confirm-real-robot [options]

The default mode starts the API/UI for recording and Semantic Candidate creation
with robot execution and calibration disabled. ``--hardware`` enables the real
Doosan runtime and calibration adapters; all server-side preflight checks still apply.
``--hardware-calibration`` remains as a backwards-compatible alias.

Before startup, a Dittobot Uvicorn process already listening on the selected
port is stopped and replaced. An unrelated port owner is never terminated.

Optional environment overrides:
  DITTOBOT_ENV_FILE    default: <repository>/.env when present
  DITTOBOT_PYTHON      default: <repository>/.venv/bin/python, then python3
  DITTOBOT_API_HOST    default: 127.0.0.1
  DITTOBOT_API_PORT    default: 8001
  ROS_SETUP_FILE       default: /opt/ros/humble/setup.bash
  COBOT_WS_ROOT        default: /home/rokey/cobot_ws
  ROBOT_NETWORK_IFACE  default: enp3s0
  ROS_DOMAIN_ID        default: 78
EOF
}

listener_snapshot() {
  ss -H -ltnp "sport = :${api_port}" 2>/dev/null || true
}

stop_existing_dittobot_listener() {
  local snapshot remaining matched_pid process_cwd process_command owner_uid
  local -a matched_pids=()
  snapshot="$(listener_snapshot)"
  [[ -z "$snapshot" ]] && return 0

  remaining="$snapshot"
  while [[ "$remaining" =~ pid=([0-9]+) ]]; do
    matched_pid="${BASH_REMATCH[1]}"
    matched_pids+=("$matched_pid")
    remaining="${remaining#*pid=${matched_pid}}"
  done
  if ((${#matched_pids[@]} == 0)); then
    echo "Port ${api_port} is occupied, but its owner PID is not visible." >&2
    echo "$snapshot" >&2
    exit 1
  fi

  for matched_pid in "${matched_pids[@]}"; do
    if [[ ! -r "/proc/${matched_pid}/cmdline" ]]; then
      echo "Port owner disappeared while it was being inspected: PID ${matched_pid}" >&2
      continue
    fi
    owner_uid="$(stat -c '%u' "/proc/${matched_pid}")"
    process_cwd="$(readlink -f "/proc/${matched_pid}/cwd" 2>/dev/null || true)"
    process_command="$(tr '\0' ' ' <"/proc/${matched_pid}/cmdline")"
    if [[ "$owner_uid" != "$(id -u)" \
      || "$process_cwd" != "$repo_root" \
      || "$process_command" != *"uvicorn"* \
      || "$process_command" != *"robot_skill_system.api.app:create_app"* ]]; then
      echo "Port ${api_port} belongs to a non-Dittobot process; it was not stopped." >&2
      echo "PID ${matched_pid}: ${process_command}" >&2
      exit 1
    fi
  done

  echo "Stopping existing Dittobot API on port ${api_port}: PID ${matched_pids[*]}"
  kill -TERM "${matched_pids[@]}"
  for _attempt in {1..50}; do
    [[ -z "$(listener_snapshot)" ]] && return 0
    sleep 0.1
  done

  echo "Existing Dittobot API did not stop in 5 seconds; forcing its verified PID down." >&2
  kill -KILL "${matched_pids[@]}" 2>/dev/null || true
  for _attempt in {1..20}; do
    [[ -z "$(listener_snapshot)" ]] && return 0
    sleep 0.1
  done
  echo "Port ${api_port} is still occupied after stopping the verified Dittobot process." >&2
  exit 1
}

hardware_calibration=false
confirmed_real_robot=false
api_host="${DITTOBOT_API_HOST:-127.0.0.1}"
api_port="${DITTOBOT_API_PORT:-8001}"

while (($#)); do
  case "$1" in
    --hardware|--hardware-calibration)
      hardware_calibration=true
      shift
      ;;
    --confirm-real-robot)
      confirmed_real_robot=true
      shift
      ;;
    --port)
      if (($# < 2)); then
        echo "--port requires a value" >&2
        exit 2
      fi
      api_port="$2"
      shift 2
      ;;
    --host)
      if (($# < 2)); then
        echo "--host requires a value" >&2
        exit 2
      fi
      api_host="$2"
      shift 2
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

if [[ ! "$api_port" =~ ^[0-9]+$ ]] || ((api_port < 1024 || api_port > 65535)); then
  echo "Port must be an integer in [1024, 65535]" >&2
  exit 2
fi
if [[ -z "$api_host" || "$api_host" == *[[:space:]]* ]]; then
  echo "Host must be a non-empty address without whitespace" >&2
  exit 2
fi
if [[ "$hardware_calibration" == true && "$confirmed_real_robot" != true ]]; then
  echo "Hardware calibration requires --confirm-real-robot" >&2
  exit 2
fi
if [[ "$hardware_calibration" != true && "$confirmed_real_robot" == true ]]; then
  echo "--confirm-real-robot is valid only with --hardware" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
env_file="${DITTOBOT_ENV_FILE:-${repo_root}/.env}"
if [[ -n "${DITTOBOT_PYTHON:-}" ]]; then
  python_bin="$DITTOBOT_PYTHON"
elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
  python_bin="${repo_root}/.venv/bin/python"
else
  python_bin="python3"
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable was not found: $python_bin" >&2
  exit 1
fi

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
uvicorn_env_args=()
if [[ -r "$env_file" ]]; then
  uvicorn_env_args=(--env-file "$env_file")
elif [[ -n "${DITTOBOT_ENV_FILE:-}" ]]; then
  echo "Configured environment file is not readable: $env_file" >&2
  exit 1
else
  echo "No .env file found; using the application's safe mock defaults."
fi
if ! "$python_bin" -c 'import uvicorn, robot_skill_system' >/dev/null 2>&1; then
  echo "Dittobot API dependencies are unavailable in: $python_bin" >&2
  echo "Install them with: ${repo_root}/.venv/bin/python -m pip install -e '.[api]'" >&2
  exit 1
fi

if [[ "$hardware_calibration" == true ]]; then
  ros_setup_file="${ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
  cobot_ws_root="${COBOT_WS_ROOT:-/home/rokey/cobot_ws}"
  cobot_setup_file="${cobot_ws_root}/install/setup.bash"
  robot_network_iface="${ROBOT_NETWORK_IFACE:-enp3s0}"
  for setup_file in "$ros_setup_file" "$cobot_setup_file"; do
    if [[ ! -r "$setup_file" ]]; then
      echo "Required ROS setup file is not readable: $setup_file" >&2
      exit 1
    fi
  done
  if ! ip link show "$robot_network_iface" >/dev/null 2>&1; then
    echo "Robot network interface does not exist: $robot_network_iface" >&2
    exit 1
  fi
  set +u
  # shellcheck disable=SC1090
  source "$ros_setup_file"
  # shellcheck disable=SC1090
  source "$cobot_setup_file"
  set -u
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-78}"
  export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
  export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
  if [[ -z "${CYCLONEDDS_URI:-}" || "${CYCLONEDDS_URI}" == *'name=""'* ]]; then
    export CYCLONEDDS_URI="<CycloneDDS xmlns=\"https://cdds.io/config\"><Domain><General><Interfaces><NetworkInterface name=\"${robot_network_iface}\"/></Interfaces></General></Domain></CycloneDDS>"
  fi
  export ROBOT_EXECUTION_MODE=hardware
  export ENABLE_HARDWARE_EXECUTION=true
  export ROBOT_BACKEND=doosan
  export ENABLE_REAL_ROBOT=true
  export DRY_RUN=false
  export ENABLE_HANDEYE_CALIBRATION=true
  export CALIBRATION_POSE_PLAN_APPROVED=true
  export CALIBRATION_CELL_SAFETY_VERIFIED=true
  export DOOSAN_ROBOT_ID="${DOOSAN_ROBOT_ID:-dsr01}"
  export DOOSAN_ROBOT_MODEL="${DOOSAN_ROBOT_MODEL:-m0609}"
  export HANDEYE_LEGACY_NPY_PATH="${HANDEYE_LEGACY_NPY_PATH:-${repo_root}/T_gripper2camera.npy}"
  export HANDEYE_LEGACY_EXPECTED_TCP="${HANDEYE_LEGACY_EXPECTED_TCP:-2FG_TCP}"
  mode_label="Doosan hardware runtime + calibration"
else
  export ROBOT_EXECUTION_MODE=mock
  export ENABLE_HARDWARE_EXECUTION=false
  export ROBOT_BACKEND=mock
  export ENABLE_REAL_ROBOT=false
  export DRY_RUN=true
  export ENABLE_HANDEYE_CALIBRATION=false
  export CALIBRATION_POSE_PLAN_APPROVED=false
  export CALIBRATION_CELL_SAFETY_VERIFIED=false
  mode_label="safe skill creation"
fi

cd "$repo_root"
stop_existing_dittobot_listener
echo "Starting Dittobot API/UI in ${mode_label} mode"
echo "Python: ${python_bin}"
echo "UI: http://${api_host}:${api_port}/ui/"
echo "API docs: http://${api_host}:${api_port}/docs"

exec "$python_bin" -m uvicorn robot_skill_system.api.app:create_app \
  --factory \
  "${uvicorn_env_args[@]}" \
  --host "$api_host" \
  --port "$api_port"
