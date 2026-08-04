# Hardware setup (not yet physically validated)

## ROS 2 status

The skill runtime still has no commissioned Doosan ROS 2 service/action client,
`realsense2_camera` subscriber, TF listener, MoveIt client, or rosbag adapter. A separate,
calibration-only adapter now lazily creates one `rclpy` node and binds the fixed DSR functions used
by the supplied M0609 tutorial. It remains disabled until all runtime and calibration gates are
open. Sourcing ROS alone does not enable motion. On the Ubuntu 22.04 target cell, source the
verified Humble installation and that cell's built workspace before starting the API:

```bash
source /opt/ros/humble/setup.bash
source /home/rokey/cobot_ws/install/setup.bash
```

The supplied `dsr_bringup2_rviz.launch.py` declares `name`, `host`, `port`, `mode`, and `model`.
For this cell the explicit command is:

```bash
export CYCLONEDDS_URI='<CycloneDDS xmlns="https://cdds.io/config"><Domain><General><Interfaces><NetworkInterface name="enp3s0"/></Interfaces></General></Domain></CycloneDDS>'
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
  name:=dsr01 mode:=real host:=192.168.1.100 port:=12345 model:=m0609
```

The shorter alias shown by the operator is equivalent because the launch default for `name` is
`dsr01`. In `mode:=real` the emulator node is not started; `ros2_control_node`,
`joint_state_broadcaster`, `dsr_controller2`, robot-state publisher, and RViz are started under the
robot namespace. Start the FastAPI process from another shell that has sourced the same workspace.
When adding this repository to `PYTHONPATH`, preserve the sourced value instead of replacing it;
replacing it hides `DR_init`, `DSR_ROBOT2`, and the generated `dsr_msgs2` Python modules:

```bash
cd /home/rokey/Dittobot
source /opt/ros/humble/setup.bash
source /home/rokey/cobot_ws/install/setup.bash
export CYCLONEDDS_URI='<CycloneDDS xmlns="https://cdds.io/config"><Domain><General><Interfaces><NetworkInterface name="enp3s0"/></Interfaces></General></Domain></CycloneDDS>'
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m uvicorn \
  robot_skill_system.api.app:create_app --factory --env-file .env \
  --host 127.0.0.1 --port 8000
```

On this workstation `enp3s0` is the active `192.168.1.0/24` robot-network interface. The current
shell startup value uses an empty `NetworkInterface name`, which makes CycloneDDS reject node
creation. If the physical interface name changes, update this deployment value after checking
`ip -brief address`; do not blindly reuse `enp3s0` on another cell.

The calibration adapter waits up to two seconds for the exact `/dsr01/aux_control/*`,
`/dsr01/system/get_robot_state`, and `/dsr01/motion/*` services and requires
`STATE_STANDBY` before any move.

The inspected workstation currently resolves `dsr_bringup2`, `dsr_controller2`,
`dsr_description2`, `dsr_msgs2`, `rclpy`, `DR_init`, `DR_common2`, and `DSR_ROBOT2` from
`/home/rokey/cobot_ws`. The archive itself contains only `dsr_bringup2`; copying only that ZIP to a
new machine is not sufficient.

Package names, namespaces, and signatures must be taken from the packages actually installed on
the target cell; this repository does not guess them or run `sudo` automatically.

## RealSense status

`RealSenseCapture` lazily imports `pyrealsense2`, requests aligned RGB/depth frames, supports burst
median and a low-level stream iterator, maps device time onto a host Unix-epoch anchor, and keeps
raw per-stream timestamps/clock domains for audit. The UI Camera API owns one background pipeline,
serves local RGB/depth MJPEG previews, and records RGB JPEG plus compressed float32-metre depth NPZ
with SHA-256 metadata. Camera access begins only after an explicit UI/API start request. Capture and
preview default to 30 FPS while artifact recording is time-sampled at 10 FPS; configure the latter
with `REALSENSE_RECORDING_FRAMES_PER_SECOND`.

The UI can replay finalized RGB/depth sequences from their manifests after an API restart. A
recording may also be selected for bounded semantic drafting: only evenly spaced, checksum-verified
RGB JPEG and locally rendered aligned-depth colormap pairs are sent to OpenAI after the operator
presses the analysis button. Raw Depth NPZ stays local. This review does not produce robot-base
geometry or authorize hardware motion.

This development machine was checked with a D435i, firmware `5.17.0.10`, librealsense `2.58.3`,
and 640×480@30fps. Device serials are not committed. The startup synchronizer discarded early pairs with
200ms-class skew before accepting a pair below the configured 20ms limit. This is workstation
evidence, not calibration evidence for another cell. Install the `realsense` project extra and set
`REALSENSE_DEVICE_SERIAL` in deployment configuration rather than source constants.

The finalized RGB-D sequence can now be used in the UI to create local, operator-confirmed geometry
evidence. On one aligned RGB/depth frame the operator selects a surface origin, a point on its +X
axis, and a point toward +Y. The server deprojects those pixels with the recorded intrinsics/depth
and stores `T_camera_surface`. The operator may then mark the two extended fingertips in at least
four frames; their 3-D midpoint is stored as a surface-relative qualitative TCP path. These artifacts
can produce a Candidate SkillGraph and run compile/Mock validation, but they are not a calibrated
robot-base trajectory and cannot authorize hardware motion. Continuous Scene refresh, ROS topics,
rosbag, and runtime obstacle monitoring remain unconnected. The CLI `capture-scene --backend` still
accepts only `mock`.

## TF hierarchy and Chessboard calibration

Use the convention `T_A_B` for the transform that maps a point expressed in frame `B` into frame
`A`. The UI-created path is stored relative to the task surface, so it only needs
`T_camera_surface` while it remains a local Candidate. Actual replay additionally needs a verified
`T_base_camera` (or an equivalent chain such as `T_base_surface`) so that runtime can compute:

```text
T_base_tcp(t) = T_base_camera * T_camera_surface * T_surface_tcp(t)
```

Joint angles are not used as Cartesian coordinates directly. Each synchronized joint sample must
first pass through the robot's verified forward kinematics to obtain `T_base_flange`. A fixed
Chessboard can then be used in either of these standard arrangements:

- **Eye-in-hand (camera attached to the flange/wrist):** keep the Chessboard fixed in the cell,
  move the robot through diverse poses, estimate `T_camera_board` from each image using calibrated
  camera intrinsics, and pair every observation with the same-time `T_base_flange` from FK. A
  hand-eye solver estimates the constant `T_flange_camera`; a separately established board-to-base
  transform completes the chain.
- **Eye-to-hand (camera fixed outside the robot):** attach the Chessboard rigidly to the flange/TCP
  and move it through diverse poses. Pair `T_camera_board` with `T_base_flange` and solve the
  robot-world/hand-eye problem for constant `T_base_camera` and, when it is not already measured,
  `T_flange_board`.

If both the external camera and Chessboard remain fixed, changing only the robot joints provides no
new camera/board observations and cannot identify `T_base_camera`. In that arrangement, attach the
board to the robot for calibration or use a calibrated TCP to touch multiple known board points.

The UI `Calibrate` action implements the agreed eye-in-hand arrangement: a RealSense rigidly mounted
on the flange/gripper bracket, a fixed 10×7 internal-corner board with 25mm squares, and a 640×480
RGB stream. J1/J2 must already be within 0.25deg of zero. Pressing `Calibrate` anchors their
measured values and first moves only J3–J6 to `[90,0,90,0]`; it refuses to rotate J1/J2 into place.
The reviewed 21-pose micro-motion plan keeps those J1/J2 anchors fixed. J3–J6 stay within ±5deg of the reference and each joint changes by at
most 5deg between adjacent poses, at 10deg/s and 5deg/s². Every pose records the current joint vector,
controller-provided flange pose, a fresh RGB timestamp, PnP `T_camera_board`, and reprojection error.
OpenCV PARK hand-eye calibration produces `T_flange_camera`; the fixed-board reconstruction checks
reprojection, translation/rotation residuals, and motion span before marking the result passed.

Results are saved under `calibrations/handeye_<id>/`, including source RGB images, per-pose JSON,
`result.json`, and `T_flange_camera.npy`. The matrix is not automatically published to ROS TF and
does not enable SkillGraph hardware execution. Physical held-out-point verification remains required
before using it in a robot motion chain.

### Legacy `T_gripper2camera.npy`

The supplied tutorial calls `set_tcp("2FG_TCP")`, records `get_current_posx()`, and later evaluates
`T_base_tcp @ T_gripper2camera @ p_camera`. Despite the filename, its operational convention is
therefore `T_tcp_camera`, and its translation is in millimetres. Do not load it directly as this
repository's metre-valued `T_flange_camera`. With a verified active TCP, convert it as follows:

```text
T_tcp_camera[m]      = convert_translation_mm_to_m(npy)
T_flange_tcp         = inverse(T_base_flange) * T_base_tcp
T_flange_camera      = T_flange_tcp * T_tcp_camera[m]
T_base_camera(t)     = T_base_flange(t) * T_flange_camera
p_base               = T_base_camera(t) * p_camera
```

The active TCP name and geometry must match the TCP used to produce the file. Validate the converted
candidate against fixed-board observations and held-out measured points before publishing it to TF;
an `.npy` filename alone is not frame or unit evidence.

The monitoring UI's `NPY TF 복사·검증` action performs only this read/convert/validate flow; it
does not move the robot. It copies the original bytes into a new immutable artifact directory,
reads the current flange and active-TCP poses, converts the translation from millimetres to metres,
and cross-validates the candidate against the most recent accepted Chessboard observations. A
candidate that passes is attached to a Mock SkillGraph only as hand-eye provenance. It still does
not replace the separately measured `T_camera_surface` or surface-relative TCP trajectory, publish
ROS TF, or authorize hardware execution. A failed candidate is visible in the promotion checklist
but is never attached as usable SkillGraph geometry.

On the 2026-08-03 workstation check, the source SHA-256
`27cf64e46b2ddcdb5803a8d6c513f46879baddd359c4b03deb50d08c4c524146` was unchanged after import.
The active TCP was `GripperDA_v1`, not the tutorial's `2FG_TCP`, and the 18-observation held-board
check exceeded translation/rotation residual limits and the required translation span. That import
is therefore intentionally retained as a failed candidate, not calibration authority.

## Doosan M0609 and OnRobot RG2

The repository discovered no `DSR_ROBOT2`, Doosan ROS package, or RG2 driver in the initial
environment. Runtime `DoosanM0609Adapter` and `OnRobotRG2Adapter` therefore still fail closed. The
calibration-only adapter checks for the exact tutorial functions (`get_current_posj`,
`get_current_tool_flange_posx`, `get_robot_state`, `movej`, `mwait`, and `posj`) after initializing
`DR_init`; it uses the installed `dsr_msgs2/MoveStop` service directly because this DSR Python
module has no global `stop()` wrapper. A missing function or service fails before motion. This does
not physically validate the installed
driver's robot state, stop behavior, units, or collision safety. Those checks must be commissioned
on the target cell before opening the calibration gates.

There is no MoveIt integration or calibrated robot-link collision model. Offline geometry checks
cover only explicit bound TCP targets/path segments and supported Scene obstacle/workspace
volumes. Mock-labelled IK, collision, joint-limit, clearance, and singularity checks cannot be
used as hardware evidence.

## Enabling hardware

`Settings.hardware_enabled` becomes true only when all five environment gates match exactly:

```bash
export ROBOT_EXECUTION_MODE=hardware
export ENABLE_HARDWARE_EXECUTION=true
export ROBOT_BACKEND=doosan
export ENABLE_REAL_ROBOT=true
export DRY_RUN=false
```

`ENABLE_HARDWARE_EXECUTION` is the legacy enable gate and remains mandatory. The reusable
orchestrator additionally requires hardware-verified workspace/obstacle and Scene monitors;
current monitors report `hardware_verified = False`. The application endpoint explicitly rejects
hardware because the Doosan adapter is not configured. Therefore even all five variables do not
make the current repository hardware-capable.

The separate hand-eye workflow additionally requires three explicit commissioning gates:

```bash
export ENABLE_HANDEYE_CALIBRATION=true
export CALIBRATION_POSE_PLAN_APPROVED=true
export CALIBRATION_CELL_SAFETY_VERIFIED=true
export DOOSAN_ROBOT_ID=dsr01
export DOOSAN_ROBOT_MODEL=m0609
```

Set them only after reviewing all 21 joint poses in the physical cell, confirming the fixed board
is visible from every pose, and testing the vendor `stop()` path. Keep these values in an untracked
deployment `.env`; never commit cell credentials or device serials. The API has no authentication
and must remain bound to `127.0.0.1` during commissioning.

`simulation` is currently only an alias of `MockRobotAdapter`, not a physics simulator or MoveIt
planning scene. Use it to exercise software flow labels only.

## Blocking-call safety limit

The adapter protocol's motion and force operations are synchronous/blocking. Workspace polling and
force reads occur before and after a primitive/vendor call. They cannot inspect collision or force
continuously while that call blocks, and an application abort cannot guarantee that an unknown
vendor function will be interrupted promptly. Hardware commissioning therefore requires either
an independently safety-rated force/obstacle/stop system or a verified interruptible/streaming
vendor transport with watchdog behavior.

Only after those gaps are closed should commissioning progress from a real simulator to dry-run,
safety-reviewed low-speed motion, and supervised contact testing. Physical risk assessment,
guarding/scanners, safety-rated E-stop/protective stop, vendor limits, and payload/TCP calibration
remain outside this software.
