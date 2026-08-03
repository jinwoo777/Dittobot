# Safety model

Mock execution is the default, and `simulation` currently aliases the same Mock adapter. The
configuration-level hardware predicate requires all five gates:

```text
ROBOT_EXECUTION_MODE=hardware
ENABLE_HARDWARE_EXECUTION=true
ROBOT_BACKEND=doosan
ENABLE_REAL_ROBOT=true
DRY_RUN=false
```

The legacy `ENABLE_HARDWARE_EXECUTION` gate is still mandatory. Connection, E-stop, scene, graph,
tool, supervisor, profile, IK, limit, singularity, collision, and clearance checks must also pass
using hardware evidence. The current Doosan/RG2 adapters always reject use, and the reusable
orchestrator also rejects monitors whose `hardware_verified` flag is false. No current
configuration can turn the Mock validators into hardware approval.

Unknown/unsupported relevant space fails closed. Preflight and explicit workspace primitives can
validate bound targets and path segments. `GlobalWorkspaceSupervisor` polls an injected obstacle
monitor before and after each primitive. The reusable orchestrator can also check Scene freshness
around primitives, but the CLI/API application path currently relies on preflight freshness and
does not attach a live camera heartbeat provider.

An observed obstacle raises an error after requesting stop. If force mode is active, cleanup
releases force/compliance and retracts along the trusted surface normal when that adapter method is
available. The current code does **not** automatically recapture, replan, or resume;
`workspace.request_replan` aborts because no verified planner exists. Recovery must be an explicit,
validated higher-level workflow.

The deterministic offline geometry gate checks bound TCP points and explicit path segments against
supported static/dynamic sphere, box/oriented-box, and AABB volumes. It applies configured
clearance, conservatively inflates dynamic obstacles over the Scene validity window, and rejects
unsupported relevant geometry, unreadable occupancy maps, and frame mismatches. These checks are
Mock TCP/path evidence only: there is no calibrated robot-link geometry, kinematics solver,
self-collision model, or MoveIt planning scene. Mock-labelled IK/joint/collision/singularity checks
cannot authorize hardware. `workspace.validate_target` and `workspace.validate_path` repeat the
same deterministic check at execution time. Preflight also constructs segments between consecutive
Cartesian targets, and runtime automatically rechecks current-to-target `move_l` paths and
current/via/target `move_c` paths immediately before the Mock adapter call.

The force supervisor checks contact-search state, target role/link, profile compatibility, maximum
and minimum contact force, tangential limit, and normal direction. Compiler/runtime `finally`
blocks attempt to release force/compliance and safe-retract on success or failure. Cleanup still
depends on the adapter call returning and the adapter's release/retract methods behaving as
specified; this has not been verified against Doosan/RG2 hardware. Force magnitudes come only from
reviewed configuration profiles.

Every force-active `move_l`, `move_c`, `move_periodic`, and `contact.follow_path` adapter call is
checked immediately before and after the call for total force, normal contact force, approved
normal direction, and the profile's `tangential_force_limit_n`. Current vendor adapter motion calls
are blocking, so this is **not continuous polling during motion**. Hardware enablement therefore
still requires a separate verified safety-rated force/torque monitor or an interruptible/streaming
vendor motion interface; the offline runtime does not claim to close that hardware-time gap.

Workspace hooks have the same blocking-call limitation: the method named `during_primitive` runs
after the awaited blocking operation returns, not concurrently inside it. Runtime abort can call
the adapter's stop method from another control path, but no unimplemented vendor binding can be
assumed interruptible. Continuous collision/force/Scene monitoring therefore remains a hardware
integration requirement, not an implemented capability.

These controls are software architecture guards, not a certified safety function. A physical cell
still requires risk assessment, safety-rated stop devices, guarding/scanners, vendor limits, tool
payload/TCP calibration, independently verified monitors, and supervised low-speed commissioning.
