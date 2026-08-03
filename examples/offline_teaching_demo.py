"""Capture a mock scene and induce the baseline wiping skill."""

from __future__ import annotations

import json
from pathlib import Path

from robot_skill_system.application import create_application
from robot_skill_system.cli import offline_settings
from robot_skill_system.settings import Settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = offline_settings(Settings.from_env(root=REPOSITORY_ROOT))
    app = create_application(settings)
    try:
        scene = app.capture_scene({"mode": "mock"})
        skill = app.induce_skill(
            {
                "demo_path": "tests/fixtures/demonstrations/novice_wipe.json",
                "name": "wipe_surface",
                "variant": "safe",
            }
        )
    finally:
        app.close()
    print(
        json.dumps(
            {"ok": True, "scene": scene, "skill": skill},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
