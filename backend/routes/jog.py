"""
routes/jog.py

dittobot-design/api-client.js가 이미 부르고 있는 /jog/status, /jog/enable,
/jog/movej, /jog/stop 경로를 그대로(접두사 없이) 구현한다 - 프론트는 손대지 않는다.
실제 로봇 제어는 ditto_system의 jog_server(ros2 run ditto_system jog_server, 로컬
포트 8020)가 하고, 여기서는 그 로컬 서버로 그대로 프록시만 한다.
jog_server는 robot_replay/record_trajectory와 같은 물리 로봇을 쓰므로
process_manager의 'tool' 슬롯을 공유한다(동시 실행 불가).
"""

import json
import time
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException

from ..process_manager import manager

router = APIRouter()

JOG_SERVER_URL = "http://127.0.0.1:8020"
JOG_SERVER_START_TIMEOUT_SEC = 8.0


def _idle_status():
    return {
        "enabled": False,
        "operator_id": None,
        "adapter_name": None,
        "joint_positions_deg": [0.0] * 6,
        "capabilities": {
            "mode": "mock",
            "failed_gates": ["jog_server_not_running"],
            "motion_profile_id": None,
            "joint_limits_deg": [],
        },
    }


def _call_jog_server(path: str, method: str = "GET", body: dict | None = None, timeout: float = 15.0) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{JOG_SERVER_URL}{path}",
        data=data,
        method=method,
        headers={} if data is None else {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except json.JSONDecodeError:
            pass
        raise HTTPException(e.code, detail)
    except urllib.error.URLError as e:
        raise HTTPException(503, f"jog_server에 연결할 수 없습니다: {e}")


def _wait_for_jog_server(timeout: float = JOG_SERVER_START_TIMEOUT_SEC):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{JOG_SERVER_URL}/status", timeout=1.0)
            return
        except Exception:
            if not manager.tool.is_running() or manager.tool.exe != "jog_server":
                raise HTTPException(500, "jog_server가 시작 중 종료되었습니다. 로그를 확인하세요.")
            time.sleep(0.3)
    raise HTTPException(504, "jog_server 기동이 시간 내에 끝나지 않았습니다.")


@router.get("/status")
def jog_status():
    if not (manager.tool.is_running() and manager.tool.exe == "jog_server"):
        return _idle_status()
    return _call_jog_server("/status")


@router.post("/enable")
def jog_enable(body: dict):
    if manager.tool.is_running() and manager.tool.exe != "jog_server":
        raise HTTPException(409, f"로봇이 이미 다른 작업 중입니다: {manager.tool.exe}")
    if not manager.tool.is_running():
        manager.start_tool("jog_server")
        _wait_for_jog_server()
    return _call_jog_server("/enable", method="POST", body=body)


@router.post("/movej")
def jog_movej(body: dict):
    if not (manager.tool.is_running() and manager.tool.exe == "jog_server"):
        raise HTTPException(409, "조그가 활성화되어 있지 않습니다.")
    return _call_jog_server("/movej", method="POST", body=body, timeout=30.0)


@router.post("/stop")
def jog_stop(body: dict | None = None):
    if manager.tool.is_running() and manager.tool.exe == "jog_server":
        try:
            _call_jog_server("/stop", method="POST", body=body or {})
        except HTTPException:
            pass
    manager.stop_tool()
    return _idle_status()
