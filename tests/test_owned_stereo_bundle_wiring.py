import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared_slam import (
    MappingConfig,
    SharedSlam,
    StereoCamera,
    _filter_owned_training_edges,
    _reserved_training_selected_as_edge,
)
from slam_state import Landmark, MappingKeyframe, Observation
from stereo_pose_arbitration import SupportedStereoFrame


REPO = Path(__file__).resolve().parents[1]


def test_owned_image_bundle_is_default_off_and_requires_stereo_arbitration():
    assert MappingConfig().stereo_owned_image_bundle is False
    camera = StereoCamera(object(), np.eye(4), 0.54)
    K = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
    with pytest.raises(ValueError, match="requires calibrated stereo and pose arbitration"):
        SharedSlam(K, stereo=camera,
                   config=MappingConfig(stereo_owned_image_bundle=True))


def test_owned_edge_filter_rejects_mismatched_observation_even_when_feature_claim_is_minus_one():
    forward = np.array([[0, 10], [1, 11]], dtype=np.int64)
    reverse = np.array([[0, 10], [1, 11]], dtype=np.int64)
    target_claims = {
        # This row has no feature-level LM claim, but the actual map pixel has
        # an owner whose Observation does not match the measured right-u.
        "10": {"classification": "ambiguous_or_measurement_mismatch", "claims": []},
        "11": {"classification": "existing_single_view_target", "claims": [
            {"landmark_id": 42, "existing_observation_certificate": {"frame_id": 5}}
        ]},
    }
    safe_forward, safe_reverse, rejected = _filter_owned_training_edges(
        forward, reverse, target_claims, {}, {42}
    )
    np.testing.assert_array_equal(safe_forward, [[1, 11]])
    np.testing.assert_array_equal(safe_reverse, [[1, 11]])
    assert rejected == 1


def test_reserved_edge_requires_selected_role_and_no_full_pool_attempt():
    measurement = np.eye(4)
    assert _reserved_training_selected_as_edge(
        "reserved_stereo_arbitration", "independent", 12, 12,
        measurement, measurement.copy(), False,
    )
    assert not _reserved_training_selected_as_edge(
        "map", "map", 12, 12, measurement, measurement, False,
    )
    assert not _reserved_training_selected_as_edge(
        "reserved_stereo_arbitration", "independent", 12, 12,
        measurement, measurement, True,
    )


@pytest.mark.parametrize(
    "script,required,extra",
    [
        ("src/main.py", ["--stereo-owned-image-bundle"], []),
        ("scripts/evaluate_shared_slam.py", ["--output", "unused", "--stereo-owned-image-bundle"], []),
    ],
)
def test_owned_image_bundle_cli_rejects_missing_stereo_arbitration(script, required, extra):
    result = subprocess.run(
        [sys.executable, str(REPO / script), *required, *extra],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "--stereo-owned-image-bundle requires" in result.stderr


def test_development_mapping_identity_records_owned_bundle(monkeypatch):
    scripts = REPO / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_development_tests as runner

    configuration = runner.current_mapping_configuration(
        "bundle", "verified_fallback", True, False, True
    )
    assert configuration["stereo_owned_image_bundle"] is True
    assert runner.current_mapping_configuration(
        "bundle", "verified_fallback", True
    )["stereo_owned_image_bundle"] is False


def _synthetic_provider(pose_source="reserved_stereo_arbitration", full_pool=False):
    K = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
    camera = StereoCamera(object(), np.eye(4), 0.54)
    slam = SharedSlam(
        K, stereo=camera,
        config=MappingConfig(stereo_pose_arbitration=True, stereo_owned_image_bundle=True),
    )
    width, height = 640, 480
    pixels = np.array([
        [90., 65.], [540., 70.], [100., 410.], [530., 415.],
        [210., 140.], [430., 150.], [220., 330.], [420., 340.],
    ], dtype=np.float32)
    z = np.full(len(pixels), 10., dtype=float)
    points = np.c_[((pixels[:, 0] - K[0, 2]) / K[0, 0]) * z,
                   ((pixels[:, 1] - K[1, 2]) / K[1, 1]) * z, z]
    right_u = pixels[:, 0] - K[0, 0] * camera.baseline / z
    descriptors = np.zeros((len(pixels), 128), dtype=np.float32)
    source = SupportedStereoFrame(
        pixels, descriptors, points, right_u, np.full(len(pixels), -1, dtype=int),
        1, (width, height), slam.stereo_calibration_identity,
    )
    target = SupportedStereoFrame(
        pixels.copy(), descriptors.copy(), points.copy(), right_u.copy(),
        np.full(len(pixels), -1, dtype=int), 2, (width, height),
        slam.stereo_calibration_identity,
    )
    eye = np.eye(4)
    slam.map.poses = [eye.copy(), eye.copy(), eye.copy()]
    slam.map.statuses = ["tracking", "tracking", "tracking"]
    slam.map.pose_anchors = [0, 0, 1]
    slam.map.revision = 7
    slam.map.geometry_revision = 0
    slam.map.keyframes = {
        0: MappingKeyframe(0, 0, eye.copy(), np.empty((0, 2), np.float32),
                           np.empty((0, 128), np.float32), np.empty(0, int)),
        1: MappingKeyframe(1, 2, eye.copy(), pixels.copy(), descriptors.copy(),
                           np.array([10, 11, 12, 13, -1, -1, -1, -1], dtype=int)),
    }
    prepared_points = {}
    for row, landmark_id in enumerate((10, 11, 12, 13)):
        position = points[row].copy()
        slam.map.landmarks[landmark_id] = Landmark(
            landmark_id, position.copy(), descriptors[row].copy(), 1,
            {1: Observation(pixels[row].copy(), float(right_u[row]))},
        )
        prepared_points[landmark_id] = position.tolist()

    fit = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64)
    held = np.array([[4, 4], [5, 5], [6, 6], [7, 7]], dtype=np.int64)
    training = {
        "fit_pairs": fit.copy(), "forward_inlier_pairs": fit.copy(),
        "reverse_inlier_pairs": fit.copy(),
        "forward_fit_row_indices": np.arange(4, dtype=np.int64),
        "reverse_fit_row_indices": np.arange(4, dtype=np.int64),
        "refinement_attempted": False, "refinement_applied": False,
        "reverse_status": "verified",
    }
    arbitration = {
        "previous": source, "current": target, "training_rows": training,
        "fit_pairs_snapshot": fit.copy(), "held_pairs": held.copy(),
        "excluded_targets": {4, 5, 6, 7}, "excluded_landmarks": set(),
        "fit_source_epoch": (6, 0),
        "fit_source_state": {
            "source_pose": eye.copy(), "source_status": "tracking",
            "source_anchor_keyframe_id": 0, "source_anchor_pose": eye.copy(),
        },
        "verified": {"measurement": eye.copy()},
    }
    payload = {
        "revision": 7, "geometry_revision": 0,
        "calibration": {"matrix": K.copy(), "baseline": camera.baseline,
                        "disparity_offset": camera.disparity_offset},
        "selection": {"selected_landmarks": [
            {"landmark_id": ident, "initial_world_position": point}
            for ident, point in prepared_points.items()
        ]},
        "single_view_propagations": [],
        "camera_poses": [
            {"keyframe_id": 0, "frame_id": 0, "camera_to_world": eye.copy()},
            {"keyframe_id": 1, "frame_id": 2, "camera_to_world": eye.copy()},
        ],
        "frame_statuses": ["tracking", "tracking", "tracking"],
    }
    provider = slam._owned_stereo_training_provider(
        2, (width, height), arbitration, {"choice": "independent"},
        {"pose_source": pose_source}, (1, eye.copy()), full_pool,
    )
    return slam, provider, payload


def test_provider_builds_factors_and_revalidates_exact_live_ownership():
    slam, provider, payload = _synthetic_provider()
    factors, report = provider(deepcopy(payload))
    assert report["status"] == "accepted"
    assert len(factors) == 4
    assert all(factor.reused_landmark_id in {10, 11, 12, 13} for factor in factors)
    validate = deepcopy(payload)
    validate["training_factor_phase"] = "validate"
    _none, checked = provider(validate)
    assert checked["status"] == "validated"

@pytest.mark.parametrize(
    "mutation,expected_reason",
    [
        ("point", "reused_landmark_changed_before_factor_apply"),
        ("observation", "observation_ownership_changed_before_factor_apply"),
        ("calibration", "calibration_changed_before_factor_apply"),
        ("source_pose", "endpoint_changed_before_factor_apply"),
    ],
)
def test_provider_rejects_direct_live_mutations_without_revision_bump(mutation, expected_reason):
    slam, provider, payload = _synthetic_provider()
    factors, report = provider(deepcopy(payload))
    assert report["status"] == "accepted" and factors
    validate = deepcopy(payload)
    validate["training_factor_phase"] = "validate"
    landmark_id = factors[0].reused_landmark_id
    if mutation == "point":
        slam.map.landmarks[landmark_id].position[0] += 0.01
    elif mutation == "observation":
        slam.map.landmarks[landmark_id].observations[1].right_u += 0.01
    elif mutation == "calibration":
        slam.K[0, 0] += 1.0
    elif mutation == "source_pose":
        slam.map.poses[1][0, 3] += 0.01
    _none, checked = provider(validate)
    assert checked["status"] == "rejected"
    assert checked["reason"] == expected_reason


@pytest.mark.parametrize("pose_source,full_pool", [("map", False), ("reserved_stereo_arbitration", True)])
def test_provider_fails_closed_when_reserved_source_was_not_final(pose_source, full_pool):
    _slam, provider, payload = _synthetic_provider(pose_source, full_pool)
    _factors, report = provider(payload)
    assert report["status"] == "rejected"
    assert report["reason"] == "reserved_reference_not_selected_or_full_pool_consumed"
    assert _factors == ()
