from __future__ import annotations

from fastapi.testclient import TestClient

from robot_skill_system.api.app import create_app
from robot_skill_system.runtime.errors import PreflightError


def test_preflight_error_is_reported_to_the_ui_not_as_server_error() -> None:
    app = create_app(service=object())

    @app.get("/_test/preflight-error")
    def preflight_error() -> None:
        raise PreflightError("live Doosan IK rejected target move_spline_001_000")

    response = TestClient(app).get("/_test/preflight-error")

    assert response.status_code == 422
    assert response.json() == {
        "detail": "live Doosan IK rejected target move_spline_001_000",
        "code": "preflight_failed",
    }
