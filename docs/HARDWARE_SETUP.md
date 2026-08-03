# Hardware setup (not yet physically validated)

## ROS 2 status

No `rclpy` node, Doosan ROS 2 service/action client, `realsense2_camera` subscriber, TF listener,
MoveIt client, or rosbag adapter is implemented in this repository. Sourcing ROS does not enable
hardware by itself. On a future Ubuntu 22.04 target cell, source the verified Humble installation
and then that cell's built workspace before testing a separately implemented adapter:

```bash
source /opt/ros/humble/setup.bash
source <your-ros-workspace>/install/setup.bash
```

Package names, namespaces, and signatures must be taken from the packages actually installed on
the target cell; this repository does not guess them or run `sudo` automatically.

## RealSense status

`RealSenseCapture` lazily imports `pyrealsense2`, requests aligned RGB/depth frames, supports burst
median and a low-level stream iterator, maps device time onto a host Unix-epoch anchor, and keeps
raw per-stream timestamps/clock domains for audit. Tests use a fake `pyrealsense2` module only.
No D435i was connected, and the CLI/API `capture-scene --backend` currently accepts only `mock`.
The stream iterator is not wired to continuous Scene refresh, storage, ROS topics, or runtime
obstacle monitoring.

Install the Intel SDK/`pyrealsense2` using documentation appropriate to the target machine, then
verify serial selection, intrinsics/extrinsics, depth scale, depth-to-color alignment, timestamp
domains/skew, filters, and burst consensus against recorded ground truth. Set device serials in
deployment configuration rather than source constants.

## Doosan M0609 and OnRobot RG2

The repository discovered no `DSR_ROBOT2`, Doosan ROS package, or RG2 driver in the initial
environment. Consequently `DoosanM0609Adapter` and `OnRobotRG2Adapter` fail closed: they raise an
authorization error while locally unauthorized and `NotConfiguredError` even after that local
authorization is supplied. This does not replace the five application-level gates below. They do
not guess function or service signatures. On the target cell, record and test the exact package
version, Python import path, namespace,
service/action signatures, robot mode, controller state, RG2 interface, TCP/load calibration,
force axes/units, stop semantics, and error behavior before implementing those adapters.

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
