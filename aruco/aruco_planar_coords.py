"""
ZED 2i RGB+Depth로 ArUco 마커 6개를 검출하고,
마커 코너의 3D 포인트(카메라 좌표계)로 평면을 피팅해
평면 좌표계(원점/축)를 산출한다.

평면 좌표계 정의:
- 원점: ORIGIN_ID 마커 중심을 평면에 투영한 점
- X축: ORIGIN_ID -> X_AXIS_ID 방향(평면에 투영, 정규화)
- Z축: 평면 법선(카메라를 향하도록 부호 결정)
- Y축: Z x X

결과:
- plane_frame.npy: T_cam_plane (4x4, 평면좌표계 -> 카메라좌표계)
- 콘솔에 각 마커의 평면좌표(x, y, z~=0) 출력
"""
import cv2
import numpy as np
import pyzed.sl as sl

from marker_config import ARUCO_DICT, MARKER_IDS

ORIGIN_ID = MARKER_IDS[0]
X_AXIS_ID = MARKER_IDS[1]


def fit_plane(points):
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[-1]
    return centroid, normal


def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED open failed")

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    xyz = sl.Mat()

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv2.aruco.DetectorParameters(),
    )

    detected = {}
    n_tries = 30
    for _ in range(n_tries):
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(xyz, sl.MEASURE.XYZ)

        rgb = image.get_data()[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            continue

        xyz_np = xyz.get_data()  # (H, W, 4) -> x, y, z, color

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in MARKER_IDS or marker_id in detected:
                continue
            pts_2d = marker_corners[0]  # (4, 2)
            pts_3d = []
            for (u, v) in pts_2d:
                x, y, z, _ = xyz_np[int(round(v)), int(round(u))]
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    pts_3d.append([x, y, z])
            if len(pts_3d) == 4:
                pts_3d = np.array(pts_3d)
                detected[marker_id] = {
                    "corners_3d": pts_3d,
                    "center_3d": pts_3d.mean(axis=0),
                }

        if len(detected) == len(MARKER_IDS):
            break

    zed.close()

    missing = set(MARKER_IDS) - set(detected.keys())
    if missing:
        raise RuntimeError(f"마커 검출 실패: {sorted(missing)} (조명/각도/거리 확인)")

    all_corners = np.concatenate([d["corners_3d"] for d in detected.values()], axis=0)
    centroid, normal = fit_plane(all_corners)

    if np.dot(normal, -centroid) < 0:
        normal = -normal

    origin = detected[ORIGIN_ID]["center_3d"]
    origin = origin - np.dot(origin - centroid, normal) * normal

    x_target = detected[X_AXIS_ID]["center_3d"]
    x_axis = x_target - np.dot(x_target - centroid, normal) * normal - origin
    x_axis = x_axis / np.linalg.norm(x_axis)

    z_axis = normal
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])  # plane -> camera
    T_cam_plane = np.eye(4)
    T_cam_plane[:3, :3] = R
    T_cam_plane[:3, 3] = origin

    np.save("plane_frame.npy", T_cam_plane)
    print("T_cam_plane (평면좌표계 -> 카메라좌표계):")
    print(T_cam_plane)

    T_plane_cam = np.linalg.inv(T_cam_plane)
    print("\n각 마커의 평면좌표 (x, y, z~=0) [m]:")
    for mid in MARKER_IDS:
        p_cam = np.append(detected[mid]["center_3d"], 1.0)
        p_plane = T_plane_cam @ p_cam
        print(f"  ID {mid}: x={p_plane[0]:.4f} y={p_plane[1]:.4f} z={p_plane[2]:.4f}")


if __name__ == "__main__":
    main()
