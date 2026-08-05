from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from robot_skill_system.capture import (
    CaptureMode,
    CaptureRequest,
    InferenceSamplingConfig,
    MockCapture,
    RGBDSequence,
    SequenceMetadata,
    select_inference_frames,
    select_keyframes,
)
from robot_skill_system.capture.image_sequence import ArrayImageSequenceAdapter
from robot_skill_system.demonstrations.preprocessing import preprocess_trajectory
from robot_skill_system.demonstrations.primitive_fitter import (
    PrimitiveFittingError,
    fit_periodic_primitive_geometry,
    recommend_primitive,
)
from robot_skill_system.demonstrations.quality import validate_repeat_consistency
from robot_skill_system.demonstrations.recorder import load_demonstration
from robot_skill_system.demonstrations.segmentation import segment_trajectory
from robot_skill_system.demonstrations.synthetic import (
    generate_arc_trajectory,
    generate_expert_wipe_trajectory,
    generate_line_trajectory,
    generate_novice_wipe_trajectory,
    generate_periodic_trajectory,
)
from robot_skill_system.perception import MockPerception


def test_preprocessing_cleans_before_derivatives_and_normalizes_quaternions() -> None:
    raw = list(generate_line_trajectory(sample_count=31).samples)
    del raw[8]
    raw[11] = raw[11].model_copy(update={"confidence": 0.1})
    raw[19] = raw[19].model_copy(update={"position_m": (0.8, -0.5, 0.4)})
    raw[4] = raw[4].model_copy(update={"orientation_xyzw": (0.0, 0.0, 0.0, -1.0)})
    processed = preprocess_trajectory(tuple(reversed(raw)))

    timestamps = [sample.timestamp_ns for sample in processed.samples]
    assert timestamps == sorted(timestamps)
    assert processed.report.interpolated_sample_count >= 1
    assert processed.report.dropped_low_confidence_count >= 1
    assert processed.report.dropped_outlier_count >= 1
    assert len(processed.velocity_mps) == len(processed.samples)
    assert len(processed.acceleration_mps2) == len(processed.samples)
    assert len(processed.jerk_mps3) == len(processed.samples)
    assert len(processed.curvature_per_m) == len(processed.samples)
    for sample in processed.samples:
        assert math.sqrt(sum(value * value for value in sample.orientation_xyzw)) == pytest.approx(
            1.0, abs=1.0e-9
        )


def test_straight_trajectory_recommends_move_l() -> None:
    trajectory = preprocess_trajectory(generate_line_trajectory())
    recommendation = recommend_primitive(trajectory)

    assert recommendation.recommended_primitive_id == "motion.move_l"
    assert recommendation.selected_fit.fit_type == "line"
    assert recommendation.selected_fit.residuals.maximum_m < 1.0e-8


def test_arc_trajectory_recommends_move_c_with_radius_and_via() -> None:
    trajectory = preprocess_trajectory(generate_arc_trajectory(radius_m=0.12))
    recommendation = recommend_primitive(trajectory)

    assert recommendation.recommended_primitive_id == "motion.move_c"
    assert recommendation.selected_fit.radius_m == pytest.approx(0.12, rel=2.0e-3)
    assert recommendation.selected_fit.via_m != recommendation.selected_fit.start_m
    assert recommendation.selected_fit.residuals.root_mean_square_m < 1.0e-4


def test_periodic_trajectory_recommends_move_periodic() -> None:
    trajectory = preprocess_trajectory(generate_periodic_trajectory(cycles=3.0))
    recommendation = recommend_primitive(trajectory)

    assert recommendation.recommended_primitive_id == "motion.move_periodic"
    assert recommendation.selected_fit.cycle_count == pytest.approx(3.0, rel=0.03)
    assert recommendation.selected_fit.amplitude_m == pytest.approx(0.045, rel=0.06)
    assert recommendation.selected_fit.period_s is not None


def test_fixed_center_periodic_geometry_is_derived_only_from_local_samples() -> None:
    samples = generate_periodic_trajectory(cycles=3.0, drift_m=0.0).samples
    recommendation = recommend_primitive(samples)

    geometry = fit_periodic_primitive_geometry(
        samples, recommendation.selected_fit
    )

    assert geometry.center_m == pytest.approx((0.2, -0.1, 0.002), abs=1.0e-6)
    assert geometry.amplitude_vector_m == pytest.approx((0.045, 0.0, 0.0), abs=1.0e-6)
    assert geometry.repetitions == 3
    assert geometry.observed_cycle_count == pytest.approx(3.0, rel=0.03)
    assert geometry.residuals.maximum_m <= 0.004


def test_translating_periodic_fit_is_not_forced_into_fixed_center_schema() -> None:
    samples = generate_periodic_trajectory(cycles=3.0, drift_m=0.025).samples
    recommendation = recommend_primitive(samples)
    assert recommendation.recommended_primitive_id == "motion.move_periodic"

    with pytest.raises(PrimitiveFittingError, match="fixed-centre periodic"):
        fit_periodic_primitive_geometry(samples, recommendation.selected_fit)


def test_noise_is_smoothed_before_line_classification() -> None:
    noisy = generate_line_trajectory(noise_std_m=0.0015, seed=23)
    recommendation = recommend_primitive(preprocess_trajectory(noisy))

    assert recommendation.recommended_primitive_id == "motion.move_l"
    assert recommendation.selected_fit.residuals.root_mean_square_m < 0.004


def test_low_confidence_interval_requires_reteach_with_timestamp_range() -> None:
    raw = list(generate_line_trajectory(sample_count=31).samples)
    for index in range(9, 18):
        raw[index] = raw[index].model_copy(update={"confidence": 0.15})

    result = segment_trajectory(raw)

    assert result.reteach_required is True
    assert result.uncertain_intervals
    assert result.recommended_retake_range is not None
    assert result.reason is not None
    assert result.uncertain_intervals[0].start_timestamp_ns == raw[9].timestamp_ns


def test_fixture_wipe_is_loaded_and_classified_offline() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "wipe_demo"
    demonstration = load_demonstration(fixture)
    recommendation = recommend_primitive(preprocess_trajectory(demonstration))

    assert demonstration.session_id == "wipe_demo"
    assert recommendation.recommended_primitive_id == "motion.move_periodic"


def test_novice_and_expert_repeats_have_explicit_consistency_result() -> None:
    repeated_expert = validate_repeat_consistency(
        [generate_expert_wipe_trajectory(), generate_expert_wipe_trajectory()]
    )
    novice_vs_expert = validate_repeat_consistency(
        [generate_novice_wipe_trajectory(), generate_expert_wipe_trajectory()]
    )

    assert repeated_expert.consistent is True
    assert repeated_expert.confidence > 0.9
    assert novice_vs_expert.comparisons
    if not novice_vs_expert.consistent:
        assert novice_vs_expert.issues
        assert all(issue.timestamp_ns >= 0 and issue.reason for issue in novice_vs_expert.issues)


def test_rgbd_timestamp_tolerance_and_original_timestamp_sampling() -> None:
    burst = MockCapture().capture(CaptureRequest(mode=CaptureMode.BURST, frame_count=31))
    sequence = RGBDSequence(
        frames=burst.frames,
        metadata=SequenceMetadata(raw_capture_fps=30.0, timestamps_preserved=True),
    )
    config = InferenceSamplingConfig(inference_fps=10, keyframe_count=8)
    inference = select_inference_frames(sequence, config)
    keyframes = select_keyframes(sequence, config)

    source_timestamps = {frame.timestamp_ns for frame in sequence.frames}
    assert 10 <= len(inference) <= 12
    assert len(keyframes) == 8
    assert all(frame.timestamp_ns in source_timestamps for frame in inference)
    assert all(frame.timestamp_ns in source_timestamps for frame in keyframes)

    first = burst.frames[0]
    with pytest.raises(ValueError, match="synchronization skew"):
        ArrayImageSequenceAdapter(
            color_images_rgb=[first.color_image_rgb],
            depth_images_m=[first.depth_image_m],
            color_timestamps_ns=[0],
            depth_timestamps_ns=[30_000_000],
            color_intrinsics=first.color_intrinsics,
            maximum_timestamp_skew_ns=20_000_000,
        ).load()


def test_mock_capture_and_perception_return_complete_scene_outputs() -> None:
    single = MockCapture().capture(CaptureRequest(mode=CaptureMode.SINGLE, frame_count=5))
    perception = MockPerception().analyze(single)

    assert len(single.frames) == 1
    assert perception.markers
    assert perception.tools[0].attached is True
    assert perception.tools[0].pose.source == "robot_tf_mock"
    assert perception.objects and perception.surfaces and perception.workspace_regions
    assert any(region.role.value == "unknown_region" for region in perception.workspace_regions)
    assert single.frames[0].depth_intrinsics is not None
    assert single.frames[0].depth_to_color_extrinsics is not None
    assert np.isfinite(single.consensus_depth_m).all()
