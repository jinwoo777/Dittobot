# Scene and Skill schema

`SceneSnapshot` contains timestamp/freshness, reference frame, calibrated objects, tools,
surfaces, workspaces, obstacles, occupancy URI, camera metadata, and confidence summary. Every
pose records source, confidence, frame, timestamp, optional covariance, metres, and normalized
`xyzw` quaternion.

`SkillGraph` is an immutable versioned workflow. Nodes refer to whitelist operations and typed
arguments, timeout, checkpoint, explicit success/failure edges, and required scene freshness.
The graph records required tools/entity roles, symbolic bindings, source demonstrations,
approved motion/force profile IDs, policies, uncertainty, validation status, semantic version,
and parent version metadata.

Motion targets are represented as anchor frame plus relative pose. A literal target in the robot
base frame is invalid. Contact graphs must search contact before enabling force and must disable
force on every terminal path. Graph cycles are rejected in the MVP because bounded loop semantics
are not yet modeled.

Compilation uses a Python AST builder. It emits an import-free asynchronous `run(runtime)` that
contains only the validated order and literal data required to call `runtime.execute_primitive`.
The validator rejects unknown operations, dangerous names (`eval`, `exec`, `compile`, dynamic
imports), unapproved syntax, and all imports; it performs AST parsing/subset validation,
`py_compile`, and checksum generation without executing the source. Immediately before runtime,
the loader accepts only a registry-issued relative artifact URI, rechecks manifest association,
SHA-256 and AST, and then imports that compiler-owned module. User/model-selected modules and
free-form source are never accepted.

Compilation and Mock execution do not establish hardware compatibility. Relative target binding
and offline TCP/path clearance are validated locally, but real IK, robot-link collision, MoveIt,
vendor timing, and continuous monitor evidence remain outside the current schema/compiler path.
