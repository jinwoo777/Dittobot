"""Adapter-specific failures."""


class AdapterError(RuntimeError):
    """Base class for adapter failures."""


class NotConfiguredError(AdapterError):
    """Raised when an optional hardware binding has not been verified/configured."""


class HardwareExecutionDisabledError(AdapterError):
    """Raised when both hardware authorization switches are not active."""


def require_hardware_authorization(*, execution_mode: str, enabled: bool) -> None:
    if execution_mode != "hardware" or not enabled:
        raise HardwareExecutionDisabledError(
            "hardware requires ROBOT_EXECUTION_MODE=hardware and "
            "ENABLE_HARDWARE_EXECUTION=true"
        )
