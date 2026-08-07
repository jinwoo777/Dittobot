from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_block_editor_builds_typed_binding_specs_from_placeholders() -> None:
    app_source = (REPOSITORY_ROOT / "dittobot-design/app.js").read_text(
        encoding="utf-8"
    )
    html_source = (REPOSITORY_ROOT / "dittobot-design/index.html").read_text(
        encoding="utf-8"
    )

    assert 'id="block-binding-list"' in html_source
    assert "collectBlockBindingHints" in app_source
    assert "serializedBlockBindings" in app_source
    assert 'entity_kind: draft.entity_kind' in app_source
    assert "minimum_confidence" in app_source
    assert "minimum_visible_fraction" in app_source
    assert "must_be_attached" in app_source
    assert "bindings," in app_source


def test_block_editor_uses_local_blockly_and_exposes_skill_deletion() -> None:
    app_source = (REPOSITORY_ROOT / "dittobot-design/app.js").read_text(
        encoding="utf-8"
    )
    api_source = (REPOSITORY_ROOT / "dittobot-design/api-client.js").read_text(
        encoding="utf-8"
    )
    html_source = (REPOSITORY_ROOT / "dittobot-design/index.html").read_text(
        encoding="utf-8"
    )

    assert 'id="blocklyDiv"' in html_source
    assert 'src="./vendor/blockly/blockly.min.js"' in html_source
    assert "Blockly.inject" in app_source
    assert "syncBlocksFromBlockly" in app_source
    assert "renderBlocklyParameterEditor" in app_source
    assert "deleteInactiveSkill" in app_source
    assert "api.deleteSkill(skill.id)" in app_source
    assert 'method: "DELETE"' in api_source


def test_aruco_experiment_ui_exposes_only_ordered_reference_and_z_test() -> None:
    app_source = (REPOSITORY_ROOT / "dittobot-design/app.js").read_text(
        encoding="utf-8"
    )
    api_source = (REPOSITORY_ROOT / "dittobot-design/api-client.js").read_text(
        encoding="utf-8"
    )
    html_source = (REPOSITORY_ROOT / "dittobot-design/index.html").read_text(
        encoding="utf-8"
    )

    assert 'id="page-aruco-experiment"' in html_source
    assert 'id="aruco-move-reference"' in html_source
    assert 'id="aruco-move-z-test"' in html_source
    assert "평면에서 기준 카메라 위치까지" in html_source
    assert "[0,0,90,0,90,-90]" in html_source
    assert 'api.moveArucoReference()' in app_source
    assert 'api.moveArucoPlaneZTest()' in app_source
    assert 'payload?.reference_captured' in app_source
    assert 'payload?.z_test_completed' in app_source
    assert '"/aruco-experiment/move-reference"' in api_source
    assert '"/aruco-experiment/move-plane-z-test"' in api_source


def test_jog_ui_builds_one_validated_six_axis_movej_target() -> None:
    app_source = (REPOSITORY_ROOT / "dittobot-design/app.js").read_text(
        encoding="utf-8"
    )
    api_source = (REPOSITORY_ROOT / "dittobot-design/api-client.js").read_text(
        encoding="utf-8"
    )
    html_source = (REPOSITORY_ROOT / "dittobot-design/index.html").read_text(
        encoding="utf-8"
    )

    assert 'id="jog-movej"' in html_source
    assert 'id="jog-load-current"' in html_source
    assert "executeJogMoveJ" in app_source
    assert "targets.length !== 6" in app_source
    assert "api.moveJogJoints(targets)" in app_source
    assert '"/jog/movej"' in api_source
    assert "target_joint_positions_deg" in api_source
