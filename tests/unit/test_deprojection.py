from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_skill_system.capture.interfaces import CameraIntrinsics
from robot_skill_system.perception import deprojection
from robot_skill_system.perception.deprojection import (
    deproject_color_pixel,
    deproject_color_pixels,
)


def _intrinsics(
    *,
    distortion_model: str = "none",
    distortion_coefficients: tuple[float, ...] = (),
) -> CameraIntrinsics:
    return CameraIntrinsics(
        width_px=101,
        height_px=81,
        fx_px=100.0,
        fy_px=200.0,
        cx_px=50.0,
        cy_px=40.0,
        distortion_model=distortion_model,
        distortion_coefficients=distortion_coefficients,
    )


def test_no_distortion_uses_hardware_independent_pinhole_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deprojection,
        "_load_pyrealsense2",
        lambda: pytest.fail("no-distortion path must not import pyrealsense2"),
    )

    point = deproject_color_pixel(_intrinsics(), 60.0, 20.0, 2.0)
    points = deproject_color_pixels(
        _intrinsics(distortion_model="distortion.none"),
        np.asarray(((60.0, 20.0), (40.0, 60.0)), dtype=np.float64),
        np.asarray((2.0, 1.0), dtype=np.float64),
    )

    assert point == pytest.approx((0.2, -0.2, 2.0))
    np.testing.assert_allclose(
        points,
        np.asarray(((0.2, -0.2, 2.0), (-0.1, 0.1, 1.0))),
    )


def test_distortion_fails_closed_when_pyrealsense2_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> object:
        raise ValueError(
            "distortion-aware metric deprojection requires optional pyrealsense2"
        )

    monkeypatch.setattr(deprojection, "_load_pyrealsense2", unavailable)
    intrinsics = _intrinsics(
        distortion_model="inverse_brown_conrady",
        distortion_coefficients=(0.1, 0.01, 0.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="requires optional pyrealsense2"):
        deproject_color_pixel(intrinsics, 60.0, 20.0, 2.0)


def test_distortion_uses_realsense_model_and_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, list[float], float]] = []

    class FakeIntrinsics:
        pass

    def fake_deproject(
        sdk_intrinsics: object, pixel: list[float], depth_m: float
    ) -> list[float]:
        calls.append((sdk_intrinsics, pixel, depth_m))
        return [0.123, -0.456, depth_m]

    fake_rs = SimpleNamespace(
        distortion=SimpleNamespace(brown_conrady="brown-conrady-enum"),
        intrinsics=FakeIntrinsics,
        rs2_deproject_pixel_to_point=fake_deproject,
    )
    monkeypatch.setattr(deprojection, "_load_pyrealsense2", lambda: fake_rs)
    coefficients = (0.1, 0.01, 0.001, 0.002, 0.0001)
    intrinsics = _intrinsics(
        distortion_model="distortion.brown_conrady",
        distortion_coefficients=coefficients,
    )

    point = deproject_color_pixel(intrinsics, 60.0, 20.0, 2.0)

    assert point == pytest.approx((0.123, -0.456, 2.0))
    [(sdk_intrinsics, pixel, depth_m)] = calls
    assert pixel == [60.0, 20.0]
    assert depth_m == 2.0
    assert sdk_intrinsics.model == "brown-conrady-enum"
    assert sdk_intrinsics.coeffs == list(coefficients)
    assert sdk_intrinsics.width == 101
    assert sdk_intrinsics.height == 81
    assert sdk_intrinsics.fx == 100.0
    assert sdk_intrinsics.fy == 200.0
    assert sdk_intrinsics.ppx == 50.0
    assert sdk_intrinsics.ppy == 40.0


def test_unsupported_distortion_model_fails_without_importing_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deprojection,
        "_load_pyrealsense2",
        lambda: pytest.fail("unsupported model must fail before loading the SDK"),
    )
    intrinsics = _intrinsics(
        distortion_model="custom_fisheye",
        distortion_coefficients=(0.1, 0.01, 0.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="unsupported camera distortion model"):
        deproject_color_pixel(intrinsics, 60.0, 20.0, 2.0)
