"""Domain exceptions for the robot skill system.

The exception hierarchy deliberately contains no dependency on optional robot,
camera, web, or OpenAI packages.  Core validation code can therefore be used in
offline tooling and tests.
"""


class RobotSkillError(Exception):
    """Base class for expected robot-skill-system failures."""


class ConfigurationError(RobotSkillError):
    """Raised when approved local configuration is missing or invalid."""


class NotConfiguredError(ConfigurationError):
    """Raised when an optional adapter is not configured in this environment."""


class SchemaValidationError(RobotSkillError):
    """Raised when externally supplied semantic data fails schema validation."""


class SceneValidationError(SchemaValidationError):
    """Raised when a scene snapshot is internally inconsistent."""


class FrameValidationError(SceneValidationError):
    """Raised when a coordinate-frame identifier or transform is invalid."""


class BindingError(SceneValidationError):
    """Raised when an anchor-relative target cannot be bound to a scene."""


class PrimitiveValidationError(SchemaValidationError):
    """Raised when an operation or its arguments violate the primitive catalog."""


class UnknownPrimitiveError(PrimitiveValidationError):
    """Raised when a SkillGraph names an operation outside the whitelist."""


class ProfileNotFoundError(PrimitiveValidationError):
    """Raised when a graph references an undeclared or unapproved profile."""


class SkillGraphValidationError(SchemaValidationError):
    """Raised when graph topology or safety invariants are invalid."""


class CompilationError(RobotSkillError):
    """Raised when deterministic SkillGraph compilation cannot complete."""


class CodeValidationError(CompilationError):
    """Raised when generated source falls outside the compiler subset."""


class SafetyViolation(RobotSkillError):
    """Raised when a safety policy blocks planning or execution."""


class HardwareExecutionDenied(SafetyViolation):
    """Raised when any mandatory hardware-enable or supervisor gate is closed."""


class OpenAIIntegrationError(RobotSkillError):
    """Raised for a handled OpenAI orchestration or schema failure."""


class SemanticCatalogViolationError(OpenAIIntegrationError):
    """Raised when semantic output references an ID outside local catalogs."""


# Backwards-compatible descriptive alias used by a few integrations.
RobotSkillSystemError = RobotSkillError
