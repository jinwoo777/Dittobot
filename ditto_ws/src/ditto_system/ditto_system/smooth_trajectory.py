"""
smooth_trajectory.py

B단계: 저장된 궤적 JSON(hammering_<timestamp>.json)을 후처리한다. 로봇은
아직 안 움직인다 — 다듬은 결과를 원본과 비교해서 눈으로 확인하는 게 목표.

처리 순서:
  1) 이상치 제거: x/y/z 각각에 롤링 median 필터를 적용해 순간적으로 튄
     값(예: keypoint가 잠깐 다른 표면으로 넘어간 프레임)을 눌러준다.
     yaw는 각도라 그냥 median을 취하면 -179도/179도 근처에서 wrap-around
     오류가 나므로, (cos, sin)으로 변환해 각각 median을 적용한 뒤 다시
     각도로 되돌린다.
  2) 스무딩: 이상치 제거된 값에 가벼운 이동평균을 한 번 더 적용해 잔residual
     지터를 줄인다.
  3) 다운샘플링: 로봇이 안 움직이는 구간(예: 끝에서 가만히 있던 구간)에서
     거의 같은 위치가 여러 프레임 연속 기록된 걸 그대로 waypoint로 넘기면
     불필요하게 많은 movel 호출이 생긴다. 직전에 남긴 점에서 일정 거리/각도
     이상 움직였을 때만 남기는 방식으로 줄인다.

사용:
    ros2 run ditto_system smooth_trajectory skills/raw/hammering_1786006891.json
    (출력: <SKILLS_ROOT>/smooth/hammering_1786006891_smoothed.json + 비교 플롯 PNG,
     입력 파일이 어느 폴더에 있든 출력은 항상 SKILLS_ROOT/smooth/ 밑에 파일명만 써서 저장)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # 서버/헤드리스 환경에서도 저장은 되도록
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

MEDIAN_WINDOW = 5        # 이상치 제거용 롤링 median 윈도우 (홀수 권장)
SMOOTH_WINDOW = 3        # 이동평균 스무딩 윈도우
DOWNSAMPLE_MIN_DIST_MM = 5.0   # 이 거리 이상 움직였을 때만 다음 점을 남김
DOWNSAMPLE_MIN_YAW_DEG = 3.0   # 또는 이 각도 이상 회전했을 때

SKILLS_ROOT = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/src/ditto_system/skills")
OUTPUT_DIR = os.path.join(SKILLS_ROOT, "smooth")


def load_frames(path):
    with open(path) as f:
        data = json.load(f)
    return data, data["frames"]


def smooth_xyz(values: np.ndarray) -> np.ndarray:
    """(N,) 배열 하나에 median 이상치 제거 -> 이동평균 스무딩."""
    despiked = median_filter(values, size=MEDIAN_WINDOW, mode="nearest")
    return uniform_filter1d(despiked, size=SMOOTH_WINDOW, mode="nearest")


def smooth_yaw_deg(yaw_deg: np.ndarray) -> np.ndarray:
    """각도는 wrap-around(-180/180 경계) 때문에 그냥 median/평균을 취하면
    틀린 값이 나온다. (cos, sin)으로 바꿔서 각각 스무딩한 뒤 각도로 복원."""
    rad = np.radians(yaw_deg)
    cos_s = smooth_xyz(np.cos(rad))
    sin_s = smooth_xyz(np.sin(rad))
    return np.degrees(np.arctan2(sin_s, cos_s))


def downsample(frames: list) -> list:
    """직전에 남긴 점에서 충분히 움직였을 때만 남긴다 (정지 구간의 중복
    waypoint 제거). 첫 프레임과 마지막 프레임은 항상 남긴다."""
    if not frames:
        return frames

    kept = [frames[0]]
    for f in frames[1:-1]:
        last = kept[-1]
        dist = np.linalg.norm([f["x"] - last["x"], f["y"] - last["y"], f["z"] - last["z"]])
        yaw_diff = abs(((f["yaw"] - last["yaw"]) + 180) % 360 - 180)  # wrap-around 고려한 각도 차
        if dist >= DOWNSAMPLE_MIN_DIST_MM or yaw_diff >= DOWNSAMPLE_MIN_YAW_DEG:
            kept.append(f)
    kept.append(frames[-1])
    return kept


def process(path: str):
    data, frames = load_frames(path)
    n = len(frames)
    if n < MEDIAN_WINDOW:
        print(f"[경고] 프레임이 {n}개뿐이라 스무딩 윈도우({MEDIAN_WINDOW})보다 적습니다. 그대로 둡니다.")
        smoothed_frames = frames
    else:
        x = np.array([f["x"] for f in frames])
        y = np.array([f["y"] for f in frames])
        z = np.array([f["z"] for f in frames])
        yaw = np.array([f["yaw"] for f in frames])

        x_s, y_s, z_s = smooth_xyz(x), smooth_xyz(y), smooth_xyz(z)
        yaw_s = smooth_yaw_deg(yaw)

        smoothed_frames = [
            {**f, "x": float(x_s[i]), "y": float(y_s[i]), "z": float(z_s[i]), "yaw": float(yaw_s[i])}
            for i, f in enumerate(frames)
        ]

    downsampled = downsample(smoothed_frames)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.basename(path)
    out_path = os.path.join(OUTPUT_DIR, base.replace(".json", "_smoothed.json"))
    out_data = {**data, "frames": downsampled}
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=4)

    print(f"[완료] 원본 {n}개 프레임 -> 스무딩 {len(smoothed_frames)}개 -> 다운샘플링 후 {len(downsampled)}개")
    print(f"[저장] {out_path}")

    plot_comparison(frames, smoothed_frames, os.path.join(OUTPUT_DIR, base.replace(".json", "_compare.png")))
    return out_path


def plot_comparison(original: list, smoothed: list, out_png: str):
    t_orig = [f["time"] for f in original]
    t_smooth = [f["time"] for f in smoothed]

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    labels = ["x (mm)", "y (mm)", "z (mm)", "yaw (deg)"]
    for i, key in enumerate(["x", "y", "z", "yaw"]):
        axes[i].plot(t_orig, [f[key] for f in original], "o--", color="lightcoral", label="raw", alpha=0.6)
        axes[i].plot(t_smooth, [f[key] for f in smoothed], "o-", color="steelblue", label="smoothed")
        axes[i].set_ylabel(labels[i])
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Trajectory: raw vs smoothed")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"[저장] {out_png}")


def main():
    if len(sys.argv) < 2:
        print("사용법: ros2 run ditto_system smooth_trajectory <hammering_....json>")
        sys.exit(1)
    process(sys.argv[1])


if __name__ == "__main__":
    main()
