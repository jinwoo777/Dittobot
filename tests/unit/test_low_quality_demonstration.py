from __future__ import annotations

from pathlib import Path

from robot_skill_system.application import MVPApplication
from robot_skill_system.demonstrations.models import DemonstrationTrajectory
from robot_skill_system.demonstrations.recorder import DemonstrationRecorder
from robot_skill_system.demonstrations.synthetic import generate_novice_wipe_trajectory
from robot_skill_system.settings import Settings


def test_low_quality_demonstration_is_saved_inactive_with_warnings(
    tmp_path: Path,
) -> None:
    settings = Settings.from_env(
        {
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
            "OPENAI_MODE": "mock",
            "ROBOT_EXECUTION_MODE": "mock",
            "DRY_RUN": "true",
        },
        root=Path(__file__).resolve().parents[2],
    )
    source = generate_novice_wipe_trajectory()
    low_quality = DemonstrationTrajectory(
        samples=tuple(
            sample.model_copy(update={"confidence": 0.1})
            for sample in source.samples
        ),
        session_id="low_quality_demo",
        operator_id="operator",
        operator_role="novice",
        successful=True,
    )
    demo_path = DemonstrationRecorder(
        settings.artifact_root / "demonstrations"
    ).record(low_quality)
    service = MVPApplication(settings)
    try:
        result = service.induce_skill(
            {
                "demo_path": str(demo_path),
                "name": "low_quality_wipe",
            }
        )

        assert result["active"] is False
        assert result["status"] == "validated"
        assert result["promotion_warnings"]
        version = service.get_skill("low_quality_wipe", result["version"])
        assert version["status"] == "validated"
        assert version["skill_graph"]["uncertainty"]["promotion_warnings"]
        assert all(
            item["status"] != "active"
            for item in service.get_skill_versions("low_quality_wipe")["versions"]
        )
    finally:
        service.close()
