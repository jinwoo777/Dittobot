from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_operator_ui_uses_local_mediapipe_rgbd_and_renders_complete_overlay() -> None:
    app_source = (REPOSITORY_ROOT / "dittobot-design/app.js").read_text(
        encoding="utf-8"
    )
    html_source = (REPOSITORY_ROOT / "dittobot-design/index.html").read_text(
        encoding="utf-8"
    )

    assert 'method: "mediapipe_rgbd"' in app_source
    assert 'method: "openai_rgbd"' not in app_source
    for evidence_field in (
        "thumb_pixel_xy",
        "index_pixel_xy",
        "thumb_depth_m",
        "index_depth_m",
        "distance_m",
        "candidate_state",
        "stabilization_progress_frames",
        "required_stable_frames",
        "stable_state",
        "invalid_reason",
    ):
        assert f"observation.{evidence_field}" in app_source

    assert "fingerTrackingRecordingId" in app_source
    assert "fingerObservations" in app_source
    assert "result.manual_click_assist" in app_source
    assert "surfaceHint" in app_source
    assert "RANSAC 보조점" in app_source
    assert "MediaPipe RGB-D 경로 적용" in html_source
    assert ".finger-marker.thumb" in html_source
    assert ".finger-marker.index" in html_source
    assert ".finger-readout.uncertain" in html_source
