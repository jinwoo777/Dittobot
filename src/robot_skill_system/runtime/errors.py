"""Safety and execution failures surfaced by the fixed runtime."""


class RuntimeErrorBase(RuntimeError):
    code = "runtime_error"


class RuntimeSafetyError(RuntimeErrorBase):
    code = "safety_gate_failed"


class SceneStaleError(RuntimeSafetyError):
    code = "scene_stale"


class BindingError(RuntimeSafetyError):
    code = "binding_failed"


class AmbiguousBindingError(BindingError):
    code = "ambiguous_binding"


class PreflightError(RuntimeSafetyError):
    code = "preflight_failed"


class ProfileNotApprovedError(RuntimeSafetyError):
    code = "profile_not_approved"


class SkillHashMismatchError(RuntimeSafetyError):
    code = "skill_hash_mismatch"


class ObstacleDetectedError(RuntimeSafetyError):
    code = "obstacle_detected"


class ForceSafetyError(RuntimeSafetyError):
    code = "force_safety_violation"


class ExecutionAbortedError(RuntimeErrorBase):
    code = "execution_aborted"
