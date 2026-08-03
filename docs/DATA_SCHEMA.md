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

That is the target artifact model. The current CLI/API session writes metadata plus captured Scene
JSON and final status/transcript JSON; it does not yet record raw RGB/depth/audio/TF files. The
standalone demonstration recorder writes `metadata.json` and `poses.jsonl` under a caller-supplied
session directory.

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
application persists validated skill-version embeddings there. Raw camera/audio payloads are
not automatically recorded by the current application flow.
