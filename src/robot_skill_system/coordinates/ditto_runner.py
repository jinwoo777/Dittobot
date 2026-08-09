"""Child-only adapter that runs the fixed read-only ditto coordinate recorder."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    package_text = os.environ.get("DITTO_PACKAGE_ROOT", "")
    output_text = os.environ.get("DITTO_OUTPUT_ROOT", "")
    if not package_text or not output_text:
        raise RuntimeError("DITTO_PACKAGE_ROOT and DITTO_OUTPUT_ROOT are required")
    package_root = Path(package_text).resolve(strict=True)
    output_root = Path(output_text).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # ditto_system was moved from its original Desktop workspace.  Redirect only
    # that exact legacy data prefix inside this isolated child process.
    original_expanduser = os.path.expanduser
    legacy_prefix = "~/Desktop/Dittobot/ditto_ws"

    def expanduser_with_artifact_redirect(path: str) -> str:
        if path == legacy_prefix:
            return str(output_root)
        if path.startswith(f"{legacy_prefix}/"):
            return str(output_root / path[len(legacy_prefix) + 1 :])
        return original_expanduser(path)

    os.path.expanduser = expanduser_with_artifact_redirect

    # The copied install tree contains stale symlinks.  Resolve this package's
    # read-only resources from its source tree, while leaving every other ament
    # package lookup untouched.
    from ament_index_python import packages as ament_packages

    original_share_lookup = ament_packages.get_package_share_directory

    def get_package_share_directory(package_name: str, *, print_warning: bool = True) -> str:
        if package_name == "ditto_system":
            return str(package_root)
        return original_share_lookup(package_name, print_warning=print_warning)

    ament_packages.get_package_share_directory = get_package_share_directory

    # This is a fixed import, not a user-selected module.  record_trajectory owns
    # its main loop at module scope and performs smoothing and verification after
    # each completed hand demonstration.
    from ditto_system import record_trajectory  # noqa: F401, PLC0415


if __name__ == "__main__":
    main()
