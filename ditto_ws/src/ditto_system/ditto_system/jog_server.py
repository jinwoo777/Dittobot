"""
jog_server.py

티칭 펜던트 없이 관절 단위로 로봇을 수동 조작(jog)할 수 있게 하는 최소 HTTP 서버.
robot_replay.py와 동일한 방식으로 rclpy/DSR_ROBOT2에 연결하고, 표준 라이브러리
http.server로 로컬 전용 REST 엔드포인트 4개만 제공한다. backend/routes/jog.py가
dittobot-design/app.js가 기대하는 응답 모양 그대로 이 서버에 그대로 프록시한다
(프론트 코드는 한 글자도 안 고침).

robot_replay/record_trajectory와 같은 물리 로봇을 쓰므로, backend의
process_manager가 이 스크립트도 같은 'tool' 슬롯으로 상호배타적으로 관리한다
(둘 중 하나만 동시에 실행 가능).

사용:
    ros2 run ditto_system jog_server
"""

import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from sensor_msgs.msg import JointState
import DR_init

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
node = rclpy.create_node("jog_server", namespace=ROBOT_ID)
DR_init.__dsr__node = node

try:
    from DSR_ROBOT2 import movej, mwait
except ImportError as e:
    print(e)
    sys.exit(1)

# ThreadingHTTPServer는 요청마다 새 스레드를 쓰는데, movej()는 DSR_ROBOT2 내부에서
# 공유 rclpy 노드(node)로 spin_until_future_complete()를 호출한다 - 이 spin 호출 자체가
# 재진입 불가능해서, 다른 spin(아래 상태 리스너 포함)과 겹치면 "generator already
# executing"으로 죽는다. rclpy 컨텍스트가 프로세스 전체에서 공유되는 탓에 서로 다른
# 노드를 스핀해도 겹치면 똑같이 죽는 걸 실측으로 확인함 - 그래서 이 프로세스 안에서
# 스핀하는 모든 곳(movej/mwait, 아래 상태 리스너)이 이 락 하나를 공유한다.
_rclpy_lock = threading.Lock()

# DSR_ROBOT2.get_current_posj()(서비스 /aux_control/get_current_posj)는 이 로봇에서
# 가끔 좌표 대신 전부 0.0을 success=True로 반환한다(ros2 service call로 직접 검증함 -
# 드라이버 쪽 문제로 추정, ditto_system 코드로 고칠 수 없음). 반면 ros2_control이 발행하는
# /dsr01/joint_states 토픽은 실시간으로 정확한 값을 준다 - 그래서 상태 조회는 이 토픽
# 구독으로 대체한다. movej()/mwait()는 여전히 DSR_ROBOT2 서비스를 그대로 쓴다(다른 문제 없음).
_JOINT_STATE_ORDER = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
_joint_state_lock = threading.Lock()
_latest_joint_positions_deg = None


def _on_joint_state(msg):
    global _latest_joint_positions_deg
    try:
        by_name = dict(zip(msg.name, msg.position))
        deg = [by_name[j] * 180.0 / math.pi for j in _JOINT_STATE_ORDER]
    except KeyError:
        return  # 아직 6개 조인트가 다 안 들어온 첫 메시지 등 - 다음 메시지를 기다린다
    with _joint_state_lock:
        _latest_joint_positions_deg = deg


_state_node = rclpy.create_node("jog_server_state_listener", namespace=ROBOT_ID)
_state_node.create_subscription(JointState, "joint_states", _on_joint_state, 10)


def _spin_state_node():
    while rclpy.ok():
        with _rclpy_lock:
            rclpy.spin_once(_state_node, timeout_sec=0.05)


threading.Thread(target=_spin_state_node, daemon=True).start()

PORT = 8020

# 옛 robot_skill_system/jog/controller.py의 M0609_JOINT_LIMITS_DEG를 그대로 재사용
# (M-series 컨트롤러가 공개한 보수적인 안전 각도 범위 - git 이력에서 확인함).
JOINT_LIMITS_DEG = [
    {"minimum": -360.0, "maximum": 360.0},
    {"minimum": -95.0, "maximum": 95.0},
    {"minimum": -135.0, "maximum": 135.0},
    {"minimum": -360.0, "maximum": 360.0},
    {"minimum": -135.0, "maximum": 135.0},
    {"minimum": -360.0, "maximum": 360.0},
]

JOG_VEL_DEG_S = 10.0   # 수동 조그라 robot_replay 홈 이동(20deg/s)보다 보수적으로
JOG_ACC_DEG_S2 = 10.0

_state = {"enabled": False, "operator_id": None}


def _joint_positions_deg():
    with _joint_state_lock:
        positions = _latest_joint_positions_deg
    if positions is None:
        raise RuntimeError("아직 /joint_states 토픽을 못 받았습니다 - 브링업 연결을 확인하세요.")
    return positions


def _status_payload():
    return {
        "enabled": _state["enabled"],
        "operator_id": _state["operator_id"],
        "adapter_name": "ditto_system_jog",
        "joint_positions_deg": _joint_positions_deg(),
        "capabilities": {
            "mode": "hardware",
            "failed_gates": [],
            "motion_profile_id": "ditto_jog",
            "joint_limits_deg": JOINT_LIMITS_DEG,
        },
    }


def _check_limits(targets):
    for i, v in enumerate(targets):
        limit = JOINT_LIMITS_DEG[i]
        if not (limit["minimum"] <= v <= limit["maximum"]):
            raise ValueError(
                f"J{i + 1} 목표각 {v}가 허용 범위({limit['minimum']}~{limit['maximum']})를 벗어났습니다."
            )


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def log_message(self, fmt, *args):
        print(f"[jog_server] {self.address_string()} {fmt % args}")

    def do_GET(self):
        if self.path == "/status":
            try:
                self._send_json(200, _status_payload())
            except Exception as e:
                self._send_json(500, {"detail": str(e)})
        else:
            self._send_json(404, {"detail": "not found"})

    def do_POST(self):
        try:
            body = self._read_json()
        except Exception as e:
            self._send_json(400, {"detail": f"잘못된 요청 본문: {e}"})
            return

        try:
            if self.path == "/enable":
                if not (
                    body.get("workspace_cleared")
                    and body.get("estop_ready")
                    and body.get("acknowledge_direct_motion")
                ):
                    self._send_json(400, {"detail": "안전 확인 3가지를 모두 true로 보내야 합니다."})
                    return
                _state["enabled"] = True
                _state["operator_id"] = body.get("operator_id") or "operator"
                self._send_json(200, _status_payload())

            elif self.path == "/movej":
                if not _state["enabled"]:
                    self._send_json(409, {"detail": "조그가 활성화되어 있지 않습니다."})
                    return
                targets = body.get("target_joint_positions_deg")
                if not isinstance(targets, list) or len(targets) != 6:
                    self._send_json(400, {"detail": "target_joint_positions_deg는 길이 6 리스트여야 합니다."})
                    return
                targets = [float(v) for v in targets]
                _check_limits(targets)
                with _rclpy_lock:
                    movej(targets, vel=JOG_VEL_DEG_S, acc=JOG_ACC_DEG_S2)
                    mwait()
                self._send_json(200, _status_payload())

            elif self.path == "/stop":
                _state["enabled"] = False
                self._send_json(200, _status_payload())

            else:
                self._send_json(404, {"detail": "not found"})
        except ValueError as e:
            self._send_json(400, {"detail": str(e)})
        except Exception as e:
            self._send_json(500, {"detail": str(e)})


def main():
    # 요청마다 get_current_posj() 등 ROS2 서비스 응답을 기다리며 블로킹될 수 있어서
    # (특히 브링업 전이면 오래 걸리거나 안 끝날 수 있음), 스레드 기반 서버로 요청 하나가
    # 다른 요청까지 막지 않게 한다.
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[jog_server] listening on http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            node.destroy_node()
            _state_node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
