"""ZED 2i 카메라 오픈 후 RGB + depth 프레임을 몇 장 캡처해서 정상 동작을 확인하는 스크립트."""
import sys

import cv2
import numpy as np
import pyzed.sl as sl


def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ZED open failed: {status}")
        sys.exit(1)

    print("ZED SDK version:", sl.Camera.get_sdk_version())
    info = zed.get_camera_information()
    print("Camera model:", info.camera_model)
    print("Resolution:", info.camera_configuration.resolution.width,
          "x", info.camera_configuration.resolution.height)

    calib = info.camera_configuration.calibration_parameters.left_cam
    print(f"Left cam intrinsics: fx={calib.fx:.3f} fy={calib.fy:.3f} "
          f"cx={calib.cx:.3f} cy={calib.cy:.3f}")

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    n_frames = 5
    for i in range(n_frames):
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            print("grab failed")
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        rgb = image.get_data()[:, :, :3]
        depth_np = depth.get_data()

        cy, cx = rgb.shape[0] // 2, rgb.shape[1] // 2
        center_depth = depth_np[cy, cx]
        print(f"[{i}] rgb shape={rgb.shape} depth center={center_depth:.3f} m")

    cv2.imwrite("last_rgb.png", rgb)
    np.save("last_depth.npy", depth_np)
    print("Saved last_rgb.png, last_depth.npy")

    zed.close()


if __name__ == "__main__":
    main()
