from __future__ import annotations

import asyncio
import json

from robot_skill_system.api.app import create_app
from robot_skill_system.runtime.errors import PreflightError, RuntimeErrorBase, SceneStaleError


def test_preflight_error_is_reported_to_the_ui_not_as_server_error() -> None:
    app = create_app(service=object())

    @app.get("/_test/preflight-error")
    def preflight_error() -> None:
        raise PreflightError("live Doosan IK rejected target move_spline_001_000")

    handler = app.exception_handlers[PreflightError]
    response = asyncio.run(handler(None, PreflightError("live Doosan IK rejected target move_spline_001_000")))

    assert response.status_code == 422
    assert json.loads(response.body) == {
        "detail": "live Doosan IK rejected target move_spline_001_000",
        "code": "preflight_failed",
    }


def test_runtime_safety_error_is_reported_to_the_ui_not_as_server_error() -> None:
    app = create_app(service=object())

    @app.get("/_test/runtime-error")
    def runtime_error() -> None:
        raise SceneStaleError("scene is stale during execution")

    handler = app.exception_handlers[RuntimeErrorBase]
    response = asyncio.run(handler(None, SceneStaleError("scene is stale during execution")))

    assert response.status_code == 409
    assert json.loads(response.body) == {
        "detail": "scene is stale during execution",
        "code": "scene_stale",
    }
