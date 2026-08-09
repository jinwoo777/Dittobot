"""
process_manager.py

ditto_system(ros2 ament_python 패키지)의 각 노드를 subprocess로 실행/추적/종료한다.
robot_replay.py 등은 모듈 최상단에서 rclpy.init()과 카메라/그리퍼를 직접 잡는 구조라
FastAPI 프로세스 안으로 직접 import하면 rclpy 중복 초기화, 이벤트루프 블로킹, 하드웨어
이중 연결 문제가 생긴다 - 그래서 실행 명령어.txt와 동일한 "ros2 run ditto_system <exe>"
쉘 명령을 그대로 서브프로세스로 띄우는 방식을 쓴다.
"""

import os
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from typing import Deque, List, Optional

DITTO_WS = os.path.expanduser("~/Desktop/Dittobot/ditto_ws")

# 실행명령어.txt와 동일한 환경 설정 (ROS2 + cobot_ws + ditto_ws 소싱, 실물 로봇과
# 통신하는 도메인/DDS 설정). bash -lc로 실행해야 source가 먹는다.
ROS_SETUP_CMD = (
    "source /opt/ros/humble/setup.bash && "
    "source ~/cobot_ws/install/setup.bash && "
    f"source {DITTO_WS}/install/setup.bash && "
    "export ROS_DOMAIN_ID=78 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && "
    "export CYCLONEDDS_URI='<CycloneDDS xmlns=\"https://cdds.io/config\"><Domain><General>"
    "<Interfaces><NetworkInterface name=\"enp3s0\"/></Interfaces></General></Domain></CycloneDDS>'"
)

LOG_LINES_KEPT = 200


class ManagedProcess:
    """ros2 run 서브프로세스 하나(로봇/카메라를 오래 붙잡는 프로세스)를 감싸서
    시작/종료/로그/상태를 관리한다."""

    def __init__(self, name: str):
        self.name = name
        self._proc: Optional[subprocess.Popen] = None
        self._log: Deque[str] = deque(maxlen=LOG_LINES_KEPT)
        self._lock = threading.Lock()
        self.started_at: Optional[float] = None
        self.exe: Optional[str] = None
        self.args: List[str] = []

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, exe: str, args: Optional[List[str]] = None) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError(f"'{self.name}' 슬롯이 이미 '{self.exe}'로 실행 중입니다.")
            args = args or []
            arg_str = " ".join(shlex.quote(a) for a in args)
            cmd = f"{ROS_SETUP_CMD} && exec ros2 run ditto_system {exe} {arg_str}"
            self._log.clear()
            self._proc = subprocess.Popen(
                ["bash", "-lc", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,  # 프로세스 그룹째로 종료하기 위해
            )
            self.exe = exe
            self.args = args
            self.started_at = time.time()
            threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._log.append(line.rstrip("\n"))

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            if not self.is_running():
                return
            pid = self._proc.pid
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                return
            os.killpg(pgid, signal.SIGINT)
            try:
                self._proc.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                pass
            os.killpg(pgid, signal.SIGTERM)
            try:
                self._proc.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                pass
            os.killpg(pgid, signal.SIGKILL)

    def status(self) -> dict:
        return {
            "name": self.name,
            "running": self.is_running(),
            "exe": self.exe,
            "args": self.args,
            "started_at": self.started_at,
            "log_tail": list(self._log)[-40:],
        }


class ProcessManager:
    """robot_replay/record_trajectory는 카메라+그리퍼를 동시에 못 쓰므로 'tool' 슬롯
    하나만 두고 상호배타적으로 관리한다. get_keyword(음성 서비스)는 별도 상시 프로세스."""

    def __init__(self):
        self.tool = ManagedProcess("tool")
        self.get_keyword = ManagedProcess("get_keyword")

    def start_tool(self, exe: str, args: Optional[List[str]] = None) -> None:
        self.tool.start(exe, args)

    def stop_tool(self) -> None:
        self.tool.stop()

    def ensure_get_keyword_running(self) -> None:
        if not self.get_keyword.is_running():
            self.get_keyword.start("get_keyword")

    def run_pipeline_sync(self, exe: str, args: List[str], timeout: float = 120.0) -> dict:
        """smooth/verify/classify/generate처럼 하드웨어 없이 짧게 끝나는 작업을
        완료까지 기다렸다가 결과를 반환한다."""
        arg_str = " ".join(shlex.quote(a) for a in args)
        cmd = f"{ROS_SETUP_CMD} && exec ros2 run ditto_system {exe} {arg_str}"
        result = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-4000:],
        }


manager = ProcessManager()
