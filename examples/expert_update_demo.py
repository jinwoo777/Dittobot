"""Create, validate, and explicitly promote an expert candidate version."""

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
        try:
            baseline = app.get_skill("wipe_surface")
        except KeyError:
            baseline = app.induce_skill(
                {
                    "demo_path": "tests/fixtures/demonstrations/novice_wipe.json",
                    "name": "wipe_surface",
                    "variant": "safe",
                }
            )
        update = app.update_skill(
            "wipe_surface",
            {
                "demo_path": "tests/fixtures/demonstrations/expert_wipe.json",
                "operator_role": "expert",
                "has_force_measurements": False,
            },
        )
        candidate_version = str(update["candidate_version"])
        validation = app.validate_skill(
            "wipe_surface", {"version": candidate_version}
        )
        activation = app.activate_skill(
            "wipe_surface", {"version": candidate_version}
        )
    finally:
        app.close()
    print(
        json.dumps(
            {
                "ok": True,
                "baseline": baseline,
                "candidate": update,
                "validation": validation,
                "activation": activation,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
