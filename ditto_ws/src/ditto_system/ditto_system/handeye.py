import os

import numpy as np
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation


class HandEye:

    def __init__(self):
        # 예전에는 프로젝트 밖 절대경로(~/Calibration_Tutorial/T_gripper2camera.npy)를
        # 하드코딩해서 읽었다. 이 패키지(ditto_system) 소유의 캘리브레이션 파일로
        # 옮겨와서, colcon으로 설치된 패키지 share 디렉터리 기준으로 찾는다.
        calib_path = os.path.join(
            get_package_share_directory("ditto_system"), "resource", "T_gripper2camera.npy"
        )
        self.gripper2cam = np.load(calib_path)

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):

        R = Rotation.from_euler(
            "ZYZ",
            [rx, ry, rz],
            degrees=True
        ).as_matrix()

        T = np.eye(4)

        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T

    def transform_to_base(self, camera_xyz):

        from DSR_ROBOT2 import get_current_posx

        coord = np.append(
            np.array(camera_xyz),
            1
        )

        current_pose = get_current_posx()[0]

        base2gripper = self.get_robot_pose_matrix(
            *current_pose
        )

        base2cam = base2gripper @ self.gripper2cam
        print(base2gripper)
        print(self.gripper2cam)     

        robot_xyz = base2cam @ coord

        return robot_xyz[:3]


handeye = HandEye()
