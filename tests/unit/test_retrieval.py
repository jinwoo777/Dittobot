from robot_skill_system.skills.retrieval import (
    SkillCandidate,
    SkillSearchQuery,
    rank_skills,
)


def candidate(
    skill_id: str, *, status: str = "active", validated: str = "passed"
) -> SkillCandidate:
    return SkillCandidate(
        skill_id=skill_id,
        version="1.0.0",
        name="wipe surface",
        description="wipe table",
        embedding=(1.0, 0.0),
        intent="wipe_surface",
        tool_classes=("wiper",),
        target_types=("table",),
        contact=True,
        operator_style="expert",
        status=status,
        validation_status=validated,
        hardware_compatible=False,
    )


def test_search_uses_only_active_validated_versions() -> None:
    results = rank_skills(
        SkillSearchQuery(
            embedding=(1.0, 0.0),
            intent="wipe_surface",
            tool_class="wiper",
            target_type="table",
            contact=True,
            operator_style="expert",
        ),
        [
            candidate("valid"),
            candidate("candidate", status="candidate"),
            candidate("invalid", validated="failed"),
        ],
    )
    assert [item.candidate.skill_id for item in results] == ["valid"]


def test_hardware_filter_rejects_mock_only_candidate() -> None:
    results = rank_skills(
        SkillSearchQuery(embedding=(1.0, 0.0), require_hardware_compatibility=True),
        [candidate("mock-only")],
    )
    assert results == []
