"""Run the complete offline teaching-to-runtime vertical slice."""

from __future__ import annotations

import json
from pathlib import Path

from robot_skill_system.cli import offline_settings
from robot_skill_system.settings import Settings
from robot_skill_system.vertical_slice import run_offline_demo

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = offline_settings(Settings.from_env(root=REPOSITORY_ROOT))
    result = run_offline_demo(settings)
    print(
        json.dumps(
            {"ok": True, "result": result.model_dump(mode="json")},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
