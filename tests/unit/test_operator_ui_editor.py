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
