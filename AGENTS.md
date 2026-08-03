# Repository guidance

This repository implements a safety-bounded robot skill teaching and execution system.

## Invariants

- Default all execution and OpenAI integrations to `mock`; tests must never contact hardware or the network.
- Never use `eval`, `exec`, dynamic user-selected imports, shell execution, or free-form model-generated robot code.
- OpenAI may return only validated semantic schemas; local code owns geometry, profiles, safety, compilation, and execution.
- Persist motion targets relative to object, tool, surface, fixture, or workspace-region anchors. Do not persist robot-base absolute targets in skills.
- Hardware execution requires `ROBOT_EXECUTION_MODE=hardware`, `ENABLE_HARDWARE_EXECUTION=true`,
  `ROBOT_BACKEND=doosan`, `ENABLE_REAL_ROBOT=true`, and `DRY_RUN=false`, plus every preflight
  supervisor and hardware-verification gate.
- Force values, velocity, and acceleration come only from approved configuration profiles, never model-provided numbers.
- Optional ROS 2, RealSense, Doosan, RG2, FastAPI, and OpenAI dependencies must not prevent core/mock imports.
- Use metres, seconds, newtons, nanoseconds, and quaternion `xyzw`; make units explicit in identifiers.
- Preserve generated artifacts outside the database; store only URI/checksum metadata in SQLite.

## Commands

Run from the repository root with Python 3.10:

```bash
python3 -m compileall src tests
PYTHONPATH=src python3 -m pytest -q
ruff check .
mypy src
PYTHONPATH=src python3 -m robot_skill_system.cli demo-e2e
```

When optional tools are unavailable, report that honestly; do not claim hardware validation without hardware.
