import json
import time
import os
import math


class SkillRecorder:

    def __init__(self):

        self.frames = []
        self.start_time = time.time()
        self.prev = None

    def add_frame(self, pose):

        # 첫 프레임은 무조건 저장
        if self.prev is not None:

            d = math.sqrt(
                (pose["x"] - self.prev["x"])**2 +
                (pose["y"] - self.prev["y"])**2 +
                (pose["z"] - self.prev["z"])**2
            )

            # 5 mm보다 작으면 저장 안 함
            if d < 0.005:
                return

        frame = {
            "time": round(time.time() - self.start_time, 3),
            "x": round(pose["x"], 4),
            "y": round(pose["y"], 4),
            "z": round(pose["z"], 4),
            "yaw": round(pose["yaw"], 4),
            "gripper": "close" if pose["grip"] < 0.04 else "open"
        }

        self.frames.append(frame)
        self.prev = pose.copy()

    def save(self, filename="hammer.json"):

        skills_root = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/src/ditto_system/skills")
        os.makedirs(os.path.join(skills_root, "raw"), exist_ok=True)

        path = os.path.join(skills_root, "raw", filename)

        with open(path, "w") as f:
            json.dump(
                {
                    "skill": "hammer",
                    "frames": self.frames
                },
                f,
                indent=4
            )

        print(f"\nSaved {len(self.frames)} frames")
        print(path)