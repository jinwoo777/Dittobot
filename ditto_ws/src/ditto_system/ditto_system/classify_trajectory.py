"""
classify_trajectory.py

1단계: 다듬어진 궤적(hammering_..._smoothed.json)을 line(직선) / arc(호) 구간으로
분류한다. GPT는 아직 전혀 등장하지 않는다 — 순수 기하학적 판단만 한다.
이유: 원의 중심/반지름 같은 정밀 수치를 LLM이 직접 만들게 하면 실제 로봇
명령에 들어갈 좌표가 부정확해질 위험이 있다. 이 단계에서 "이 구간은 직선이다/
호다"라는 구조만 결정하고, 좌표 자체는 항상 실제 기록된 frame 값을 그대로 쓴다.

처리 순서:
  1) 국소 곡률 추정: 연속된 3점(p0,p1,p2)마다 menger curvature로 국소 반경(radius)을
     계산한다. 반경이 작을수록 급하게 휘는 구간(호에 가까움), 크거나 무한대면
     직선에 가까움.
  2) 라벨링: 반경이 ARC_RADIUS_THRESHOLD_MM보다 작으면 "arc", 아니면 "line".
     노이즈로 인한 라벨 flicker를 줄이기 위해 median filter로 스무딩.
  3) 세그먼트 병합: 같은 라벨이 연속된 구간을 하나의 세그먼트로 묶는다.
     너무 짧은 세그먼트(MIN_SEGMENT_POINTS 미만)는 이웃 세그먼트에 합친다.
  4) arc 세그먼트는 진단용으로 3D 원 피팅(Kasa method)을 해서 반경/잔차를
     같이 보여준다 (이 반경/중심 값은 좌표 생성에 쓰지 않고, "이 구간이 실제로
     원에 얼마나 가까운가"를 사람이 확인하기 위한 참고용).
  5) 시각화: 위/옆에서 본 궤적을 세그먼트 타입별로 색칠해서 플롯으로 저장.

사용:
    ros2 run ditto_system classify_trajectory skills/smooth/hammering_1786006891_smoothed.json
    (출력: <SKILLS_ROOT>/classify/hammering_1786006891_smoothed_segments.json + _segments.png,
     입력 파일이 어느 폴더에 있든 출력은 항상 SKILLS_ROOT/classify/ 밑에 파일명만 써서 저장)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

ARC_RADIUS_THRESHOLD_MM = 200.0   # 이보다 국소 반경이 작으면 "arc"로 라벨링
LABEL_SMOOTH_WINDOW = 5           # 라벨 flicker 제거용 median filter 윈도우 (홀수)
MIN_SEGMENT_POINTS = 3            # 이보다 짧은 세그먼트는 이웃에 합침

SKILLS_ROOT = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/src/ditto_system/skills")
OUTPUT_DIR = os.path.join(SKILLS_ROOT, "classify")


def load_frames(path):
    with open(path) as f:
        data = json.load(f)
    return data["frames"]


def local_radius(points: np.ndarray) -> np.ndarray:
    """각 점의 국소 곡률 반경(mm)을 menger curvature로 추정.
    양 끝점은 가장 가까운 내부 점의 값을 그대로 사용."""
    n = len(points)
    radius = np.full(n, np.inf)
    for i in range(1, n - 1):
        p0, p1, p2 = points[i - 1], points[i], points[i + 1]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        d01 = np.linalg.norm(p1 - p0)
        d12 = np.linalg.norm(p2 - p1)
        d20 = np.linalg.norm(p0 - p2)
        denom = d01 * d12 * d20
        if denom < 1e-6 or area < 1e-9:
            continue  # 사실상 직선 -> 반경 무한대 유지
        curvature = 4 * area / denom
        radius[i] = 1.0 / curvature
    if n >= 3:
        radius[0] = radius[1]
        radius[-1] = radius[-2]
    return radius


def labels_from_radius(radius: np.ndarray) -> np.ndarray:
    """반경 -> arc(1)/line(0) 이진 라벨, median filter로 flicker 제거."""
    raw = (radius < ARC_RADIUS_THRESHOLD_MM).astype(int)
    if len(raw) >= LABEL_SMOOTH_WINDOW:
        raw = median_filter(raw, size=LABEL_SMOOTH_WINDOW, mode="nearest")
    return raw


def merge_short_runs(labels: np.ndarray, min_len: int) -> np.ndarray:
    """min_len보다 짧은 연속 구간을 이웃 라벨로 흡수시킨다."""
    labels = labels.copy()
    n = len(labels)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            j = i
            while j < n and labels[j] == labels[i]:
                j += 1
            run_len = j - i
            if run_len < min_len and n > min_len:
                if i == 0:
                    labels[i:j] = labels[j] if j < n else labels[i]
                elif j == n:
                    labels[i:j] = labels[i - 1]
                else:
                    # 앞뒤 중 더 긴 쪽에 흡수
                    left_label = labels[i - 1]
                    labels[i:j] = left_label
                changed = True
            i = j
    return labels


def runs_from_labels(labels: np.ndarray):
    """[(start_idx, end_idx, label), ...] - end_idx는 inclusive."""
    runs = []
    n = len(labels)
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append((i, j - 1, int(labels[i])))
        i = j
    return runs


def fit_circle_3d(points: np.ndarray):
    """3D 점들에 원을 피팅 (평면 추정 + 2D Kasa fit). 진단용.
    반환: (radius_mm, residual_rms_mm) 또는 점이 부족하면 (None, None)."""
    if len(points) < 3:
        return None, None
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered)
    u, v = vt[0], vt[1]  # 평면 내 2개 축 (법선은 vt[2], 여기선 안 씀)
    pts_2d = np.column_stack([centered @ u, centered @ v])

    x, y = pts_2d[:, 0], pts_2d[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    a, bb, c = sol
    radius = float(np.sqrt(max(c + a ** 2 + bb ** 2, 0.0)))
    dist = np.sqrt((x - a) ** 2 + (y - bb) ** 2)
    residual = float(np.sqrt(np.mean((dist - radius) ** 2)))
    return radius, residual


def path_length_mm(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def build_segments(frames):
    xyz = np.array([[f["x"], f["y"], f["z"]] for f in frames])
    radius = local_radius(xyz)
    labels = labels_from_radius(radius)
    labels = merge_short_runs(labels, MIN_SEGMENT_POINTS)
    runs = runs_from_labels(labels)

    segments = []
    for idx, (s, e, lab) in enumerate(runs):
        seg_points = xyz[s:e + 1]
        seg_type = "arc" if lab == 1 else "line"
        via_idx = None
        via_frame = None
        fit_radius = fit_residual = None

        if seg_type == "arc":
            fit_radius, fit_residual = fit_circle_3d(seg_points)
            via_idx = s + (e - s) // 2  # 실제 기록된 프레임 중 세그먼트 중간점
            via_frame = frames[via_idx]

        segments.append({
            "index": idx,
            "type": seg_type,
            "start_idx": s,
            "end_idx": e,
            "point_count": e - s + 1,
            "start_pose": frames[s],
            "end_pose": frames[e],
            "via_idx": via_idx,
            "via_pose": via_frame,
            "path_length_mm": round(path_length_mm(seg_points), 1),
            "fit_radius_mm": round(fit_radius, 1) if fit_radius is not None else None,
            "fit_residual_mm": round(fit_residual, 2) if fit_residual is not None else None,
        })
    return segments, radius


def print_summary(segments):
    print(f"\n총 {len(segments)}개 세그먼트로 분류됨\n")
    header = f"{'#':>3} {'type':>5} {'frames':>10} {'points':>7} {'length(mm)':>11} {'fit_r(mm)':>10} {'residual(mm)':>13}"
    print(header)
    print("-" * len(header))
    for seg in segments:
        fit_r = f"{seg['fit_radius_mm']:.1f}" if seg["fit_radius_mm"] is not None else "-"
        resid = f"{seg['fit_residual_mm']:.2f}" if seg["fit_residual_mm"] is not None else "-"
        print(f"{seg['index']:>3} {seg['type']:>5} "
              f"{seg['start_idx']:>4}-{seg['end_idx']:<4} {seg['point_count']:>7} "
              f"{seg['path_length_mm']:>11} {fit_r:>10} {resid:>13}")
    print()
    print("[참고] fit_r/residual은 arc 세그먼트에 대한 진단용 원 피팅 결과입니다.")
    print("       residual이 크면(대략 반경의 10~20% 이상) 실제로는 원이 아니라")
    print("       불규칙한 곡선일 수 있으니 2단계에서 moveC 대신 moveL 연쇄로")
    print("       처리하는 게 나을 수 있습니다.")


def plot_segments(frames, segments, out_png):
    xyz = np.array([[f["x"], f["y"], f["z"]] for f in frames])
    colors = {"line": "steelblue", "arc": "darkorange"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for seg in segments:
        s, e = seg["start_idx"], seg["end_idx"]
        seg_xyz = xyz[s:e + 1]
        color = colors[seg["type"]]
        axes[0].plot(seg_xyz[:, 0], seg_xyz[:, 1], "o-", color=color, markersize=4)
        axes[1].plot(seg_xyz[:, 0], seg_xyz[:, 2], "o-", color=color, markersize=4)
        if seg["via_idx"] is not None:
            vp = xyz[seg["via_idx"]]
            axes[0].plot(vp[0], vp[1], "k*", markersize=12)
            axes[1].plot(vp[0], vp[2], "k*", markersize=12)

    axes[0].plot(xyz[0, 0], xyz[0, 1], "go", markersize=10, label="start")
    axes[0].plot(xyz[-1, 0], xyz[-1, 1], "rs", markersize=10, label="end")
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)")
    axes[0].set_title("Top view (XY)"); axes[0].axis("equal"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(xyz[0, 0], xyz[0, 2], "go", markersize=10)
    axes[1].plot(xyz[-1, 0], xyz[-1, 2], "rs", markersize=10)
    axes[1].set_xlabel("x (mm)"); axes[1].set_ylabel("z (mm)")
    axes[1].set_title("Side view (XZ)"); axes[1].grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=colors["line"], marker="o", label="line segment"),
        Line2D([0], [0], color=colors["arc"], marker="o", label="arc segment"),
        Line2D([0], [0], color="k", marker="*", linestyle="None", markersize=10, label="via point (arc)"),
        Line2D([0], [0], color="g", marker="o", linestyle="None", label="start"),
        Line2D([0], [0], color="r", marker="s", linestyle="None", label="end"),
    ]
    axes[0].legend(handles=legend_elems, fontsize=8, loc="best")

    fig.suptitle("Trajectory segmentation: line (blue) vs arc (orange)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"[저장] {out_png}")


def process(path: str):
    frames = load_frames(path)
    if len(frames) < 3:
        print("[경고] 프레임이 3개 미만이라 곡률 계산이 불가능합니다.")
        return

    segments, radius = build_segments(frames)
    print_summary(segments)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.basename(path)
    out_json = os.path.join(OUTPUT_DIR, base.replace(".json", "_segments.json"))
    with open(out_json, "w") as f:
        json.dump({"source": path, "segments": segments}, f, indent=2)
    print(f"[저장] {out_json}")

    out_png = os.path.join(OUTPUT_DIR, base.replace(".json", "_segments.png"))
    plot_segments(frames, segments, out_png)


def main():
    if len(sys.argv) < 2:
        print("사용법: ros2 run ditto_system classify_trajectory <hammering_..._smoothed.json>")
        sys.exit(1)
    process(sys.argv[1])


if __name__ == "__main__":
    main()
