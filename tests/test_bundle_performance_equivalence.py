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


def _disconnected_mixed_state(*, rotated=False):
    """Two observed components, one monocular component, and held-out points."""
    state = MapState(metric=True)
    poses = [_pose(0.4 * index) for index in range(6)]
    if rotated:
        for index, pose in enumerate(poses):
            pose[:3, :3] = Rotation.from_euler(
                "zyx", [0.025 * index, -0.018 * index, 0.012 * index]
            ).as_matrix()
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


def _oracle_residual_for_record(point, pose, observation):
    """Original scalar projection expressed directly, without bundle helpers."""
    camera_point = pose[:3, :3].T @ (point - pose[:3, 3])
    z = float(camera_point[2])
    dimensions = 3 if observation.right_u is not None else 2
    if z <= 0.0:
        return [1e4] * dimensions

    homogeneous = MATRIX @ camera_point
    predicted = homogeneous[:2] / homogeneous[2]
    left = np.clip(predicted - observation.pixel, -1e4, 1e4)
    rows = [float(left[0]), float(left[1])]
    if observation.right_u is not None:
        # Rectified right image has a different principal point by the supplied
        # offset: u_r = u_l - f*b/z - offset.
        predicted_right = predicted[0] - MATRIX[0, 0] * BASELINE / z - DISPARITY_OFFSET
        rows.append(float(predicted_right - observation.right_u))
    return rows


def _tuple_loop_oracle(x, state, selected_ids, free_camera_ids):
    """Rebuild residual rows from fixture tuples and world geometry only."""
    poses = {ident: keyframe.pose.copy()
             for ident, keyframe in state.keyframes.items()}
    for free_index, ident in enumerate(free_camera_ids):
        offset = 6 * free_index
        poses[ident][:3, :3] = Rotation.from_rotvec(x[offset:offset + 3]).as_matrix()
        poses[ident][:3, 3] = x[offset + 3:offset + 6]

    point_offset = 6 * len(free_camera_ids)
    optimized_points = {
        ident: x[point_offset + 3 * index:point_offset + 3 * index + 3]
        for index, ident in enumerate(selected_ids)
    }
    optimized_rows = []
    for ident in selected_ids:
        landmark = state.landmarks[ident]
        for camera_id, observation in landmark.observations.items():
            if camera_id in poses:
                optimized_rows.extend(_oracle_residual_for_record(
                    optimized_points[ident], poses[camera_id], observation
                ))

    optimized = set(selected_ids)
    held_out_rows = []
    for ident, landmark in state.landmarks.items():
        if ident in optimized or len(landmark.observations) < 2:
            continue
        for camera_id, observation in landmark.observations.items():
            if camera_id in free_camera_ids:
                held_out_rows.extend(_oracle_residual_for_record(
                    landmark.position, poses[camera_id], observation
                ))
    return np.asarray([*optimized_rows, *held_out_rows], dtype=float)


def test_solver_rows_match_independent_tuple_loop_projection_oracle(monkeypatch):
    # The selection/gauge outcome is fixed by this fixture's two components:
    # keyframes 2 and 5 are free, component-2 landmarks are selected first,
    # then the first fourteen landmarks from component 1 are selected.
    state = _disconnected_mixed_state(rotated=True)
    free_camera_ids = (2, 5)
    selected_ids = (*range(26, 46), *range(0, 14))
    # Include a behind-camera point in the optimizer's initial vector; its
    # observations remain the independently generated positive-depth pixels.
    state.landmarks[13].position[2] = -1.0
    point_offset = 6 * len(free_camera_ids)
    expected_initial = np.r_[
        *[
            np.r_[
                Rotation.from_matrix(state.keyframes[ident].pose[:3, :3]).as_rotvec(),
                state.keyframes[ident].pose[:3, 3],
            ]
            for ident in free_camera_ids
        ],
        np.asarray([state.landmarks[ident].position for ident in selected_ids]).ravel(),
    ]

    optimized_dimensions = [
        3 if observation.right_u is not None else 2
        for ident in selected_ids
        for observation in state.landmarks[ident].observations.values()
    ]
    held_out_ids = range(14, 26)
    held_out_dimensions = [
        3 if state.landmarks[ident].observations[2].right_u is not None else 2
        for ident in held_out_ids
    ]
    assert 2 in optimized_dimensions and 3 in optimized_dimensions
    assert 2 in held_out_dimensions and 3 in held_out_dimensions

    captures = {}

    def inspect_solver(residual, initial, **_kwargs):
        np.testing.assert_allclose(initial, expected_initial, atol=1e-12)
        shifted = initial.copy()
        shifted[:6] += np.array([0.004, -0.003, 0.002, 0.015, -0.01, 0.005])
        shifted[-3:] += np.array([0.02, -0.01, 0.03])
        behind = initial.copy()
        behind[point_offset + 3 * selected_ids.index(13) + 2] = -2.5
        clipped = initial.copy()
        offset_to_point = state.landmarks[0].position - state.keyframes[2].pose[:3, 3]
        angle = -float(np.arctan2(offset_to_point[2], offset_to_point[0])) + 1e-4
        clipped[:3] = Rotation.from_euler("y", angle).as_rotvec()
        for name, vector in (("initial", initial), ("shifted", shifted),
                             ("behind", behind), ("clipped", clipped)):
            captures[name] = (vector.copy(), residual(vector))
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True)

    monkeypatch.setattr(bundle, "least_squares", inspect_solver)
    local_bundle_adjustment(
        state,
        MATRIX,
        BASELINE,
        window=5,
        max_landmarks=34,
        disparity_offset=DISPARITY_OFFSET,
    )

    assert set(captures) == {"initial", "shifted", "behind", "clipped"}
    expected_row_count = sum(optimized_dimensions) + sum(held_out_dimensions)
    for name, (vector, actual_rows) in captures.items():
        expected_rows = _tuple_loop_oracle(vector, state, selected_ids, free_camera_ids)
        assert actual_rows.shape == (expected_row_count,)
        np.testing.assert_allclose(actual_rows, expected_rows, atol=1e-9, rtol=1e-12)
    assert np.max(np.abs(captures["shifted"][1])) > 1e-3
    assert np.count_nonzero(captures["behind"][1] == 1e4) > 0
    # The first 120 rows belong to two-pixel component-2 tuples. Then the
    # first selected component-1 landmark contributes camera 0 (3), camera 1
    # (2), and camera 2 (3) rows. The chosen camera-2 orientation makes the
    # positive-depth point's left-u row clip high; its right row stays un-clipped.
    assert captures["clipped"][1][125] == 1e4
    assert captures["clipped"][1][127] > 1e4


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
