"""
verify_trajectory.py

C단계: 다듬어진 궤적(hammering_..._smoothed.json)을 실제 로봇에 보내기 전에,
로봇 없이 좌표만 검증한다. hand-eye 캘리브레이션이나 오프셋이 잘못됐을 때
로봇이 이상한 곳으로 튀는 걸 하드웨어에 명령 내리기 전에 걸러내는 게 목적.

검증 항목:
  1) 작업영역: 베이스 원점에서의 거리가 M0609 도달범위 안인지, z가 바닥/
     테이블 아래로 안 내려가는지 (MAX_REACH_MM/MIN_REACH_MM/Z_MIN_MM은
     실제 로봇 스펙에 맞춰 조정 필요 — 지금은 안전 쪽으로 보수적인 값)
  2) 웨이포인트 간 속도: 두 waypoint 사이 시간 간격 대비 이동 거리가
     로봇이 낼 수 있는 속도를 넘는지 (스무딩 후에도 남아있는 튐이 있으면
     여기서 걸린다)
  3) 시각화: 위에서 본(XY) 궤적 + 옆에서 본(XZ) 궤적을 그려서 실제로
     그럴듯한 경로인지 눈으로 확인

로봇 명령은 전혀 안 보낸다 (DSR_ROBOT2 import도 안 함) — 순수 좌표 검증.

사용:
    ros2 run ditto_system verify_trajectory skills/smooth/hammering_1786006891_smoothed.json
    (출력: <SKILLS_ROOT>/verify/hammering_1786006891_smoothed_verify.png,
     입력 파일이 어느 폴더에 있든 출력은 항상 SKILLS_ROOT/verify/ 밑에 파일명만 써서 저장)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SKILLS_ROOT = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/src/ditto_system/skills")
OUTPUT_DIR = os.path.join(SKILLS_ROOT, "verify")

# --- M0609 스펙에 맞춰 실제 값으로 조정할 것 (지금은 보수적인 임시값) ---
MAX_REACH_MM = 900.0     # 베이스 원점 기준 최대 도달거리
MIN_REACH_MM = 150.0     # 너무 가까우면 특이점/자기충돌 위험
Z_MIN_MM = -50.0         # 테이블/바닥 기준 이보다 낮으면 충돌 위험
Z_MAX_MM = 900.0

MAX_LINEAR_SPEED_MM_S = 500.0   # 웨이포인트 간 평균 속도가 이걸 넘으면 경고
MAX_YAW_SPEED_DEG_S = 300.0


def load_frames(path):
    with open(path) as f:
        data = json.load(f)
    return data["frames"]


def check_workspace(frames):
    problems = []
    for i, f in enumerate(frames):
        reach = float(np.linalg.norm([f["x"], f["y"], f["z"]]))
        if reach > MAX_REACH_MM:
            problems.append(f"[{i}] t={f['time']}s reach={reach:.0f}mm > MAX_REACH_MM({MAX_REACH_MM})")
        elif reach < MIN_REACH_MM:
            problems.append(f"[{i}] t={f['time']}s reach={reach:.0f}mm < MIN_REACH_MM({MIN_REACH_MM})")
        if not (Z_MIN_MM <= f["z"] <= Z_MAX_MM):
            problems.append(f"[{i}] t={f['time']}s z={f['z']:.0f}mm 범위 밖 ({Z_MIN_MM}~{Z_MAX_MM})")
    return problems


def check_speed(frames):
    problems = []
    for i in range(1, len(frames)):
        prev, cur = frames[i - 1], frames[i]
        dt = cur["time"] - prev["time"]
        if dt <= 0:
            continue
        dist = float(np.linalg.norm([cur["x"] - prev["x"], cur["y"] - prev["y"], cur["z"] - prev["z"]]))
        speed = dist / dt
        yaw_diff = abs(((cur["yaw"] - prev["yaw"]) + 180) % 360 - 180)
        yaw_speed = yaw_diff / dt
        if speed > MAX_LINEAR_SPEED_MM_S:
            problems.append(f"[{i-1}->{i}] linear speed {speed:.0f}mm/s > {MAX_LINEAR_SPEED_MM_S}")
        if yaw_speed > MAX_YAW_SPEED_DEG_S:
            problems.append(f"[{i-1}->{i}] yaw speed {yaw_speed:.0f}deg/s > {MAX_YAW_SPEED_DEG_S}")
    return problems


def plot_paths(frames, out_png):
    x = [f["x"] for f in frames]
    y = [f["y"] for f in frames]
    z = [f["z"] for f in frames]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x, y, "o-", color="steelblue")
    axes[0].plot(x[0], y[0], "go", markersize=10, label="start")
    axes[0].plot(x[-1], y[-1], "rs", markersize=10, label="end")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].set_title("Top view (XY)")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, z, "o-", color="darkorange")
    axes[1].plot(x[0], z[0], "go", markersize=10, label="start")
    axes[1].plot(x[-1], z[-1], "rs", markersize=10, label="end")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("z (mm)")
    axes[1].set_title("Side view (XZ)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"[저장] {out_png}")


def process(path: str):
    frames = load_frames(path)
    print(f"총 {len(frames)}개 waypoint 검증 중...\n")

    ws_problems = check_workspace(frames)
    print(f"[작업영역 체크] 문제 {len(ws_problems)}건")
    for p in ws_problems:
        print(f"  - {p}")

    speed_problems = check_speed(frames)
    print(f"\n[속도 체크] 문제 {len(speed_problems)}건")
    for p in speed_problems:
        print(f"  - {p}")

    if not ws_problems and not speed_problems:
        print("\n모든 waypoint가 검증을 통과했습니다. 실제 로봇 재생을 시도해도 좋습니다 (D단계).")
    else:
        print("\n문제가 있는 waypoint가 있습니다. 실제 로봇에 보내기 전에 확인이 필요합니다.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_png = os.path.join(OUTPUT_DIR, os.path.basename(path).replace(".json", "_verify.png"))
    plot_paths(frames, out_png)


def main():
    if len(sys.argv) < 2:
        print("사용법: ros2 run ditto_system verify_trajectory <hammering_..._smoothed.json>")
        sys.exit(1)
    process(sys.argv[1])


if __name__ == "__main__":
    main()
