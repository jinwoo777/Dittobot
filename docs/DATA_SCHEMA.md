# Data schema and version rules

Core Scene, Skill, and OpenAI documents carry `schema_version` and reject unknown fields. Teaching
metadata deliberately allows extension fields. JSON is UTF-8, timestamps are nanoseconds unless
the field ends in `_s`/`_ms`, distances are metres, force is newtons, and orientation is a
normalized quaternion in `xyzw` order.

## SceneSnapshot

A scene declares identity, capture timestamp, reference frame, validity window, calibration ID,
camera metadata, confidence summary, objects, tools, surfaces, workspace regions, and static/dynamic
obstacles. Point clouds and occupancy maps are URI/checksum metadata, never inline bulk JSON.
Unknown workspace regions must be forbidden or occupied. Entity IDs are unique within a scene.

## Demonstration

A teaching session links timestamped RGB/depth/audio/TF/intrinsics artifacts, raw and filtered pose
samples, phase intervals, local fit residuals, quality issues, operator metadata, and initial/final
scenes. A low-confidence or unsynchronized interval remains explicit and can force
`reteach_required`; it is not silently converted into motion.

That is the target artifact model. The current CLI/API teaching session writes metadata plus
captured Scene JSON and final status/transcript JSON. Separately, the UI Camera API records
`rgb/<index>.jpg`, color-aligned `depth/<index>.npz`, and `rgbd_manifest.json` beneath a generated
`demonstrations/rgbd_<id>/` URI. Every frame entry retains source/mapped timestamps, clock domains,
intrinsics, depth scale, artifact URIs, and SHA-256 checksums. The manifest distinguishes the raw
capture rate from the configured recording rate; recording defaults to 10 FPS without reducing the
live preview rate. Audio, TF, inferred pose, and success
labels are not added by that raw recorder. The standalone pose recorder still writes
`metadata.json` and `poses.jsonl` under a caller-supplied session directory.

Recording-based OpenAI review writes immutable `skill_drafts/<draft_id>.json` below the selected
RGB-D recording. It contains the source recording ID, selected frame indices, OpenAI mode/model and
trace metadata, RGB-then-depth transport metadata, and a strict semantic draft. The draft includes
two-finger TCP-proxy states for every supplied keyframe, normalized image landmarks or an explicit
failure reason, plus semantic hand/tool/work-surface regions. It contains no camera-metric or robot
coordinates. It contains neither image bytes nor
API credentials. The draft is explicitly non-executable and requires a separately validated pose
trajectory. Draft-list API responses add a derived fail-closed promotion checklist without
modifying the immutable artifact.

Explicit promotion actions write immutable evidence below
`skill_drafts/<draft_id>_evidence/`. A surface-calibration artifact records the selected RGB frame,
three source pixels or a semantic ROI, locally derived `T_camera_surface`, quality diagnostics, and
operator confirmation. Automatic calibration stores deterministic raw-depth RANSAC/SVD inlier,
residual, normal, and extent diagnostics. A TCP-trajectory artifact records the calibration ID,
per-frame fingertip evidence, locally depth-derived midpoint poses, surface-relative path, quality
summary, and its manual, GPT-normalized-plus-local-depth, or local-MediaPipe method. Candidate
registration adds a result artifact linking those evidence
checksums to the generated graph and Mock validation run. These artifacts contain no API key and
do not claim `T_base_camera` or hardware calibration.

## Eye-in-hand calibration

An explicitly confirmed calibration session writes
`calibrations/handeye_<id>/session_started.json`, accepted RGB images, and one JSON observation per
pose. Each observation contains a waypoint ID, capture timestamp, six joint angles in radians,
`T_base_flange`, `T_camera_board`, and reprojection RMS. A completed solve writes `result.json` plus
checksum-addressed `T_flange_camera.npy`. The result records the 10×7/0.025m fixed-board definition,
locked J1/J2 policy, validation metrics, accepted observation IDs, and pass/fail state. Failed
results remain evidence but are not publishable calibration authority.

A confirmed legacy import never overwrites its configured source NPY. It creates a unique
`calibrations/legacy_import_<id>/` directory containing the byte-identical
`source_T_gripper2camera.npy`, a metre-valued `T_flange_camera_candidate.npy`, and `result.json`.
The result records both checksums, source session, active/expected TCP names, measured
`T_flange_tcp`, fixed-board residual metrics, and explicit `candidate_only`,
`hardware_validated=false`, and `runtime_authorized=false` flags. A failed candidate remains audit
evidence but is not included as geometry provenance in a generated SkillGraph.

## SkillGraph and manifest

The graph records immutable semantic version, optional parent version, variant, source sessions,
symbolic entity bindings, anchor-relative targets, whitelist nodes/edges, profile IDs, policies,
uncertainty, and lifecycle/validation state. A manifest stores graph/code/report URIs and SHA-256
checksums. Runtime compares checksums before loading a compiled artifact.

Version states are `draft`, `candidate`, `validated`, `active`, `retired`, and `rejected`.
`deprecated` is not a current enum value. A candidate never changes the active pointer. Promotion
and rollback update the pointer transactionally while retaining prior versions.

The checksum-addressed graph keeps the state in which that immutable proposal was authored.
Current lifecycle/validation state is authoritative in registry columns and immutable validation
run rows; activation does not rewrite a graph merely to change a status label.

## RuntimeIntent

Intent contains a semantic action/skill query, target/tool descriptions constrained to current
entity IDs, style, bounded repetitions, approved profile requests, confidence/ambiguity, and a
short rationale. It has no pose, joint, force, velocity, code, file path, or execution permission.

## SkillUpdateProposal

An update proposal can select independent spatial, orientation, timing, gripper, force, and
recovery components. The updater preserves the parent's force profile unless force evidence or
explicit profile approval is supplied; the current CLI update fixture does not infer new force
values from RGB-D. Materially different route topology becomes a style variant instead of an
averaged path.

## File paths

`LocalArtifactStore` accepts only normalized relative POSIX URIs below `ARTIFACT_ROOT`; absolute
paths, `..`, and traversal are rejected. Writes use a same-directory temporary file followed by
atomic replace. The compiled loader also accepts only registry-issued relative `.py` URIs and
rechecks the manifest/checksum/AST immediately before import.

Demonstration input is a separate allowlisted boundary: application paths may be relative or
absolute only when they resolve under `tests/fixtures` or
`<ARTIFACT_ROOT>/demonstrations`. The default database is `data/robot_skills.db`; current execution
runs/events are database rows. The schema includes URI/checksum fields and an embedding table; the
application persists validated skill-version embeddings there. Bulk camera payloads stay outside
SQLite and are recorded only after an explicit Camera API start; audio is not automatically
recorded by the current application flow.
