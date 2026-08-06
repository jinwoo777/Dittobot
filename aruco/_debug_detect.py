import cv2
import numpy as np
import pyzed.sl as sl

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError("open failed")

runtime_params = sl.RuntimeParameters()
image = sl.Mat()

params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params
)

seen = {}
last_rgb = None
for i in range(60):
    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        continue
    zed.retrieve_image(image, sl.VIEW.LEFT)
    rgb = image.get_data()[:, :, :3].copy()
    last_rgb = rgb
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None:
        for mid in ids.flatten().tolist():
            seen[mid] = seen.get(mid, 0) + 1

zed.close()

print("전체 프레임에서 관측된 마커 ID별 검출 횟수 (60프레임 중):")
for mid in sorted(seen):
    print(f"  ID {mid}: {seen[mid]}회")

annotated = last_rgb.copy()
gray = cv2.cvtColor(last_rgb, cv2.COLOR_BGR2GRAY)
corners, ids, _ = detector.detectMarkers(gray)
if ids is not None:
    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
cv2.imwrite("debug_frame_annotated.png", annotated)
print("Saved debug_frame_annotated.png")
