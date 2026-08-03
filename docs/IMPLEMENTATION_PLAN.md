# Implementation plan

## Repository assessment

The repository was empty except for Git metadata on 2026-08-03. Searches for the required
Doosan, RG2, RealSense, OpenAI, FastAPI, skill, workspace, and collision terms found no source
to reuse. Hardware signatures therefore remain behind protocols and optional adapters rather
than being guessed.

## Delivery phases

1. Establish typed settings, Scene/SkillGraph schemas, profile configs, SQLite storage,
   primitive whitelist, mock capture/OpenAI/robot adapters, deterministic compiler, and tests.
2. Add Responses API structured parsing, transcription, embeddings, retry/trace metadata,
   FastAPI routes, and command-line flows while keeping tests offline.
3. Add optional RealSense burst capture and local scene-building adapter boundaries. Missing
   native dependencies must degrade to a clear `NotConfiguredError` and leave mock mode usable.
4. Define ROS 2/Doosan/RG2 adapter boundaries and global workspace/force supervisors. Hardware
   execution stays disabled until actual imports, signatures, robot state, and safety gates are
   verified on the target cell.
5. Add candidate-only expert updates, quality comparison, style variants, activation, and
   rollback.

Phases 1, 2, and the offline portions of 3–5 are implemented. The RealSense Python class has only
fake-module tests and is not selected by the CLI/API. ROS 2, MoveIt, Doosan/RG2 transport,
hardware-verified continuous monitors, and physical commissioning remain future deployment work.
`simulation` currently names the Mock adapter rather than a separate simulator.

## Vertical slice acceptance

- A synthetic teaching session is smoothed, segmented, locally fitted, analyzed by the Mock
  semantic boundary, materialized through the deterministic wipe graph builder, validated,
  compiled, AST checked, mock-executed, and registered in SQLite. The standalone
  `SkillGraphComposer` is not yet part of this application flow.
- A Korean text command retrieves a validated active skill, captures/loads a fresh scene,
  binds relative targets, passes mock-labelled preflight checks, executes on a MockRobot, and
  records structured events.
- An expert demonstration creates a candidate child version without altering force profiles
  when force measurements are absent; it can be promoted and rolled back.
- `compileall`, offline `pytest`, Ruff, mypy, and the CLI mock E2E command are run before handoff;
  documentation intentionally avoids a test-count snapshot that will become stale.

## Safety boundaries

Model output is schema-only and cannot call motion functions. The primitive registry rejects
unknown operations. The compiler emits a fixed `run(runtime)` function with literal validated
arguments and no imports; only the checksum/AST-verified compiler artifact is loaded. Runtime
supervisor hooks wrap every primitive and force cleanup is attempted in `finally`. Skills store
anchor-relative transforms and numeric motion/force limits are resolved from approved profiles.

These are offline software invariants, not hardware validation. Deterministic TCP/path checks do
not provide IK, robot-link, self-collision, or MoveIt evidence. Supervisor hooks and force reads do
not run continuously inside a blocking vendor call. The five hardware environment gates and real
validators/adapters must all exist before hardware can be considered.
