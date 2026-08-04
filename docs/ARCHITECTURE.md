# Architecture

## Data and control flow

```text
mock RGB-D -> capture -> local perception/geometry -> SceneSnapshot
                                            +-> demonstration preprocessing/fitting
audio file -> separate TranscriptionService -> TranscriptResult
STT/keyframes/IDs/local summaries -> OpenAI schema-only semantics
local fits + semantic proposal -> validated SkillGraph -> deterministic AST compiler
SQLite metadata + artifact files <- validation/mock execution

text command -> bounded intent -> validated active-skill retrieval -> fresh mock capture
-> entity binding -> mock-labelled TCP/path geometry and safety preflight
-> fixed runtime/supervisor hooks -> MockRobotAdapter -> immutable execution events
```

The OpenAI boundary never crosses into geometry or execution. It can label phases, connect
natural language to existing IDs, choose registered primitive/profile IDs, and propose a graph.
Pydantic, the primitive whitelist, graph invariants, the compiler, and the runtime all reject an
invalid proposal independently.

This diagram is the intended compositional boundary. In the current wipe application,
`DemonstrationAnalyzer` supplies bounded labels/reteach status while a local deterministic builder
materializes a fixed safe wipe topology and applies locally measured, surface-relative path
samples to its motion nodes. `SkillGraphComposer` exists and is tested as a strict service, but is
not yet called by application induction. Transcription is also a separate CLI/service; there is no
command that automatically pipes its output into runtime execution.

The final two arrows are the implemented offline path, not evidence of a commissioned robot
cell. The geometry validator checks explicit bound TCP points/path segments against supported
scene volumes. Its IK, joint-limit, self-collision, environment-collision, and singularity checks
are explicitly marked mock; there is no MoveIt or robot-link model. Hardware preflight refuses
mock geometry and requires hardware-verified dynamic workspace and scene monitors.

## Coordinate convention

Positions use metres. Orientations are normalized quaternions in `x, y, z, w` order. Canonical
capture timestamps are Unix-epoch nanoseconds and must increase within a sequence. Device-clock
timestamps and their clock-domain labels are retained as raw metadata. A stored target is
`T_anchor_target`; runtime binding uses:

```text
T_base_target = T_base_anchor_current * T_anchor_target_stored
```

Only the final hardware adapter may convert quaternion orientation to a vendor convention.

## Dependency direction

Scene and SkillGraph models are Pydantic. Demonstration fitting depends on scene types but not
runtime. The compiler depends only on SkillGraph and the primitive registry. Runtime depends on
validated graphs, scenes, profiles, and adapter protocols. `pyrealsense2`, OpenCV, MediaPipe, and
FastAPI are lazy/optional edges where their modules are used. The official OpenAI Python package
is a declared dependency, but network client construction happens only for `OPENAI_MODE=live`.
Mock CLI flows do not contact the network.

Eye-in-hand calibration is a separate local boundary. A gated calibration controller pairs fresh
RealSense RGB frames with DSR joint/flange observations, detects a fixed 10×7 board locally, solves
`T_flange_camera`, and persists validation artifacts. It does not use OpenAI, register a skill,
publish ROS TF, or grant runtime hardware authority.
The legacy-NPY path is part of the same boundary: it preserves the source bytes, derives
`T_flange_camera` from read-only live flange/TCP poses, and validates against persisted board
observations. Only a passing result may be linked as candidate provenance; no result can grant
runtime authority.

`SimulationRobotAdapter` is currently an alias of `MockRobotAdapter`; it is not Gazebo, Isaac,
MoveIt, or a physics simulator. Runtime Doosan/RG2 classes remain unconfigured protocol boundaries.
Only the separately gated calibration adapter lazily creates an `rclpy` node and calls a fixed set
of DSR tutorial functions. No general ROS 2 topic/action/service runtime or rosbag adapter is
implemented.

## Monitoring boundary

The reusable orchestrator can check Scene freshness and poll an injected obstacle monitor around
each primitive. The application/CLI path performs preflight freshness checks and uses mock
workspace hooks. Adapter motion calls are blocking, so the hook named `during_primitive` runs
after the awaited/blocking call returns; it is not continuous collision monitoring. Force-active
motions likewise read force immediately before and after the adapter call, not during it. A real
deployment needs independently verified continuous monitors or an interruptible/streaming vendor
API.

## Artifacts

By default SQLite is `data/robot_skills.db` and the artifact root is `data/`. The application
writes teaching metadata/capture JSON under `demonstrations/`, SceneSnapshot JSON under `scenes/`,
and graph/code/report/manifest files under `skills/<skill_id>/<version>/`. Execution runs and
events are currently SQLite rows. The explicitly triggered UI Camera API writes RGB JPEG,
color-aligned depth NPZ, and a checksum manifest beneath `demonstrations/rgbd_<id>/`; these bulk
payloads never enter SQLite. Finalized manifests are the only source for UI sequence replay and
server-selected OpenAI RGB plus aligned-depth-colormap keyframe pairs. OpenAI returns a strict
non-executable semantic draft with frame-complete normalized fingertip landmarks (or explicit
failure), hand/tool shapes, and a work-surface ROI. Explicit UI actions can fit
`T_camera_surface` from local raw depth with deterministic RANSAC/SVD or use the three-point method,
then deproject GPT or manually selected fingertip evidence with recorded intrinsics and materialize a
hardware-incompatible Candidate for compile/Mock validation. This does not activate the skill or
provide `T_base_camera`, robot FK, or a hardware-verified TCP trajectory. Audio, ROS TF, robot joint
states, video containers, and point clouds are not automatically recorded by the CLI/API MVP.
Calibration sessions are the explicit exception for robot joint/flange evidence and are written
under `calibrations/`; they are not demonstration frames or SkillGraph motion targets.
