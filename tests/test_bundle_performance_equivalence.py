"""The precomputed bundle residual path must preserve the reliability BA."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import local_bundle as bundle
from local_bundle import local_bundle_adjustment
from mapping_geometry import project, right_pixel
from slam_state import MapState, MappingKeyframe, Observation


MATRIX = np.array([[250.0, 0.0, 320.0], [0.0, 250.0, 240.0], [0.0, 0.0, 1.0]])
BASELINE = 0.2
DISPARITY_OFFSET = 6.5


def _pose(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def _disconnected_mixed_state():
    """Two observed components, one monocular component, and held-out points."""
    state = MapState(metric=True)
    poses = [_pose(0.4 * index) for index in range(6)]
    for ident, pose in enumerate(poses):
        state.keyframes[ident] = MappingKeyframe(
            ident,
            ident,
            pose.copy(),
            np.empty((0, 2)),
            np.empty((0, 128)),
            np.empty(0, int),
        )

    rng = np.random.default_rng(812)
    groups = [
        (range(0, 26), (0, 1, 2), True),
        (range(26, 46), (3, 4, 5), False),
    ]
    for ids, observers, stereo_group in groups:
        for ident in ids:
            point = rng.uniform([-1.5, -1.4, 5.5], [1.5, 1.4, 11.0])
            observations = {}
            for camera_id in observers:
                pixel, depth = project(point[None], poses[camera_id], MATRIX)
                # The stereo component deliberately mixes left-only and stereo
                # rows; the disconnected component has no metric scale support.
                has_right = stereo_group and (ident + camera_id) % 2 == 0
                right = (
                    float(right_pixel(pixel[0, 0], depth[0], 250.0, BASELINE,
                                      DISPARITY_OFFSET))
                    if has_right
                    else None
                )
                observations[camera_id] = Observation(pixel[0], right)
            state.add_landmark(point, np.ones(128), observers[0], observations)
    return state


def _snapshot(state):
    return (
        state.revision,
        state.geometry_revision,
        {ident: keyframe.pose.copy() for ident, keyframe in state.keyframes.items()},
        {ident: landmark.position.copy() for ident, landmark in state.landmarks.items()},
    )


def _assert_snapshots_equal(first, second):
    assert first[:2] == second[:2]
    assert first[2].keys() == second[2].keys()
    assert first[3].keys() == second[3].keys()
    for ident in first[2]:
        np.testing.assert_allclose(first[2][ident], second[2][ident], atol=1e-9)
    for ident in first[3]:
        np.testing.assert_allclose(first[3][ident], second[3][ident], atol=1e-9)


def test_precomputed_path_matches_reference_residuals_across_gauges_and_held_out(monkeypatch):
    captures = []

    def inspect_solver(residual, initial, **kwargs):
        shifted = initial.copy()
        shifted[:6] += np.array([0.004, -0.003, 0.002, 0.015, -0.01, 0.005])
        shifted[-3:] += np.array([0.02, -0.01, 0.03])
        captures.append(
            {
                "initial": residual(initial),
                "shifted": residual(shifted),
                "pattern": kwargs["jac_sparsity"].toarray(),
            }
        )
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True)

    monkeypatch.setattr(bundle, "least_squares", inspect_solver)
    reports = []
    for optimized in (True, False):
        state = _disconnected_mixed_state()
        report = local_bundle_adjustment(
            state,
            MATRIX,
            BASELINE,
            window=5,
            max_landmarks=34,
            disparity_offset=DISPARITY_OFFSET,
            optimized=optimized,
        )
        reports.append(report)

    assert len(captures) == 2
    np.testing.assert_allclose(captures[0]["initial"], captures[1]["initial"], atol=1e-10)
    np.testing.assert_allclose(captures[0]["shifted"], captures[1]["shifted"], atol=1e-10)
    np.testing.assert_array_equal(captures[0]["pattern"], captures[1]["pattern"])
    assert reports[0]["observation_components"] == 2
    assert reports[0]["held_out_observations"] > 0
    assert {3, 4}.issubset(reports[0]["fixed_keyframes"])
    for field in (
        "initial_cost",
        "final_cost",
        "held_out_initial_cost",
        "held_out_final_cost",
        "affected_initial_cost",
        "affected_final_cost",
    ):
        assert reports[0][field] == pytest.approx(reports[1][field], abs=1e-10)


def _motion_state(change):
    rng = np.random.default_rng(941)
    world = rng.uniform([-2.0, -2.0, 6.0], [2.0, 2.0, 12.0], (24, 3))
    state = MapState(metric=True)
    proposed = {}
    for ident, x in enumerate([0.0, 0.5, 1.0]):
        pose = _pose(x)
        state.keyframes[ident] = MappingKeyframe(
            ident,
            2 * ident,
            pose.copy(),
            np.empty((0, 2)),
            np.empty((0, 128)),
            np.empty(0, int),
        )
        candidate = pose.copy()
        if ident == 2 or (change == "common" and ident == 1):
            candidate[0, 3] += 0.7
        proposed[ident] = candidate
    for frame in range(5):
        pose = _pose(0.25 * frame)
        state.record(pose, "tracking", frame // 2)
    for ident, point in enumerate(world):
        observations = {}
        for camera_id, pose in proposed.items():
            pixel, depth = project(point[None], pose, MATRIX)
            right = float(
                right_pixel(pixel[0, 0], depth[0], 250.0, BASELINE, DISPARITY_OFFSET)
            )
            if (ident + camera_id) % 3 == 1:
                right = None
            observations[camera_id] = Observation(pixel[0], right)
        state.add_landmark(point, np.ones(128), 0, observations)
    state.add_stereo_motion(2, 4, np.linalg.inv(state.poses[2]) @ state.poses[4])
    return state, proposed


@pytest.mark.parametrize(("change", "accepted"), [("translation", False), ("common", True)])
def test_acceptance_and_independent_motion_guard_match_reference(monkeypatch, change, accepted):
    reports = []
    snapshots = []
    for optimized in (True, False):
        state, proposed = _motion_state(change)

        def propose(_residual, initial, **_kwargs):
            value = initial.copy()
            for offset, ident in ((0, 1), (6, 2)):
                value[offset : offset + 3] = Rotation.from_matrix(
                    proposed[ident][:3, :3]
                ).as_rotvec()
                value[offset + 3 : offset + 6] = proposed[ident][:3, 3]
            return SimpleNamespace(x=value, nfev=1, success=True)

        monkeypatch.setattr(bundle, "least_squares", propose)
        reports.append(
            local_bundle_adjustment(
                state,
                MATRIX,
                BASELINE,
                window=3,
                disparity_offset=DISPARITY_OFFSET,
                optimized=optimized,
            )
        )
        snapshots.append(_snapshot(state))

    assert reports[0]["applied"] is accepted
    assert reports[1]["applied"] is accepted
    assert reports[0].get("reason") == reports[1].get("reason")
    if not accepted:
        assert reports[0]["reason"] == "independent_stereo_motion_inconsistency"
    assert reports[0]["max_stereo_motion_translation_error_m"] == pytest.approx(
        reports[1]["max_stereo_motion_translation_error_m"], abs=1e-10
    )
    assert reports[0]["max_stereo_motion_rotation_error_deg"] == pytest.approx(
        reports[1]["max_stereo_motion_rotation_error_deg"], abs=1e-10
    )
    _assert_snapshots_equal(*snapshots)
