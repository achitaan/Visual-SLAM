"""Structural-rank regression tests for the original local image objective."""

from types import SimpleNamespace
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import local_bundle as bundle
from mapping_geometry import project, right_pixel
from pose_observability import _observation_jacobians, original_image_pose_observability
from slam_state import MapState, MappingKeyframe, Observation


K = np.array([[320.0, 3.0, 320.0], [0.0, 315.0, 240.0], [0.0, 0.0, 1.0]])
BASELINE = 0.24
OFFSET = 2.75


def _pose(center, rotvec=(0.0, 0.0, 0.0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    result[:3, 3] = np.asarray(center, dtype=float)
    return result


def _rigidity_state(*, third=None, excluded=False, spread=1.0,
                    geometry_scale=1.0, baseline_scale=1.0):
    """Two anchored views, three free cluster views, and deterministic points."""
    state = MapState(metric=True)
    camera_poses = [
        _pose(np.asarray(center) * geometry_scale, rotvec)
        for center, rotvec in (
            ([-0.30, -0.20, 0.0], [0.0, 0.0, 0.0]),
            ([0.00, -0.20, 0.0], [0.0, 0.003, 0.0]),
            ([0.45, -0.20, 0.0], [0.002, -0.004, 0.001]),
            ([0.80, -0.18, 0.01], [-0.003, 0.006, 0.002]),
            ([1.15, -0.22, -0.01], [0.004, -0.003, -0.002]),
        )
    ]
    axis_xy = np.array([0.38, 0.08]) * geometry_scale
    baseline = BASELINE * baseline_scale
    state._test_baseline = baseline
    points = [
        np.r_[axis_xy, 6.5 * geometry_scale],
        np.r_[axis_xy, 11.5 * geometry_scale],
    ]
    rng = np.random.default_rng(24581)
    for _ in range(20):
        offset = rng.uniform([-0.65, -0.50, 0.0], [0.65, 0.50, 0.0])
        z = rng.uniform(6.0, 12.0) * geometry_scale
        points.append(np.r_[axis_xy + spread * geometry_scale * offset[:2], z])
    if third is not None:
        if third == "collinear":
            points.append(np.r_[axis_xy, 8.5 * geometry_scale])
        elif third == "noncollinear":
            points.append(np.r_[axis_xy + geometry_scale * np.array([0.42, 0.31]), 9.0 * geometry_scale])
        else:
            raise ValueError(third)

    obs_by_id = []
    for index, point in enumerate(points):
        observed = range(5) if index < 2 or (third is not None and index == len(points) - 1) else (2, 3, 4)
        observations = {}
        for keyframe_id in observed:
            pixel, depth = project(point[None], camera_poses[keyframe_id], K)
            right = right_pixel(pixel[0, 0], depth[0], K[0, 0], baseline, OFFSET)
            observations[keyframe_id] = Observation(pixel[0].astype(np.float32), float(right))
        ident = state.add_landmark(
            np.asarray(point, dtype=float),
            np.full(128, index + 1, dtype=np.float32),
            0,
            observations,
        )
        obs_by_id.append((ident, observations))

    excluded_id = None
    if excluded:
        # Only one observation lies inside selected window [1,2,3,4], so this
        # point is fixed-world holdout evidence rather than an optimized point.
        point = np.r_[axis_xy + geometry_scale * np.array([1.2, -0.4]), 9.2 * geometry_scale]
        observations = {}
        for keyframe_id in (0, 2):
            pixel, depth = project(point[None], camera_poses[keyframe_id], K)
            right = right_pixel(pixel[0, 0], depth[0], K[0, 0], baseline, OFFSET)
            observations[keyframe_id] = Observation(pixel[0].astype(np.float32), float(right))
        excluded_id = state.add_landmark(
            point, np.full(128, 90, dtype=np.float32), 0, observations
        )
        obs_by_id.append((excluded_id, observations))

    for keyframe_id, pose in enumerate(camera_poses):
        ids, pixels = [], []
        for ident, observations in obs_by_id:
            if keyframe_id in observations:
                ids.append(ident)
                pixels.append(observations[keyframe_id].pixel)
        state.keyframes[keyframe_id] = MappingKeyframe(
            keyframe_id,
            keyframe_id,
            pose.copy(),
            np.asarray(pixels, dtype=np.float32).reshape(-1, 2),
            np.zeros((len(ids), 128), dtype=np.float32),
            np.asarray(ids, dtype=np.int64),
            image_size=(640, 480),
        )
    for keyframe_id, pose in enumerate(camera_poses):
        state.record(pose.copy(), "tracking", keyframe_id)
    return state, excluded_id


def _snapshot(state):
    return {
        "revision": state.revision,
        "geometry_revision": state.geometry_revision,
        "poses": [p.copy() for p in state.poses],
        "relative_poses": [p.copy() for p in state.relative_poses],
        "anchors": list(state.pose_anchors),
        "statuses": list(state.statuses),
        "keyframes": {k: v.pose.copy() for k, v in state.keyframes.items()},
        "points": {k: v.position.copy() for k, v in state.landmarks.items()},
        "registry": copy.deepcopy(state.retained_source_observations),
    }


def _assert_snapshot_same(state, saved):
    assert (state.revision, state.geometry_revision) == (saved["revision"], saved["geometry_revision"])
    assert state.pose_anchors == saved["anchors"]
    assert state.statuses == saved["statuses"]
    for actual, expected in zip(state.poses, saved["poses"]):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(state.relative_poses, saved["relative_poses"]):
        np.testing.assert_array_equal(actual, expected)
    for ident, expected in saved["keyframes"].items():
        np.testing.assert_array_equal(state.keyframes[ident].pose, expected)
    for ident, expected in saved["points"].items():
        np.testing.assert_array_equal(state.landmarks[ident].position, expected)
    assert state.retained_source_observations == saved["registry"]


def _solve_identity(fun, initial, **kwargs):
    # A no-step result tests solver entry/kwargs without introducing a commit.
    return SimpleNamespace(x=np.asarray(initial).copy(), nfev=1, success=True,
                           status=1, message="identity test result", cost=0.0,
                           optimality=0.0)


def _run(state, *, diagnostic_sink=None, baseline="fixture"):
    if baseline == "fixture":
        baseline = getattr(state, "_test_baseline", BASELINE)
    return bundle.local_bundle_adjustment(
        state, K, baseline, window=4, max_landmarks=200,
        disparity_offset=OFFSET, diagnostic_sink=diagnostic_sink,
    )


def _capture_residual(monkeypatch, state, *, bypass_guard=False):
    captured = {}
    real_guard = bundle.original_image_pose_observability
    if bypass_guard:
        def force_full_rank(*args, **kwargs):
            output = real_guard(*args, **kwargs)
            output = dict(output)
            output.update(status="observable", reason=None, nullity=0,
                          rank=output.get("pose_columns", 0))
            return output
        monkeypatch.setattr(bundle, "original_image_pose_observability", force_full_rank)

    def solver(fun, initial, **kwargs):
        captured["fun"] = fun
        captured["initial"] = np.asarray(initial, dtype=float).copy()
        captured["kwargs"] = kwargs
        return _solve_identity(fun, initial, **kwargs)

    monkeypatch.setattr(bundle, "least_squares", solver)
    diagnostics = {}
    report = _run(state, diagnostic_sink=lambda phase, payload: diagnostics.__setitem__(phase, payload))
    captured["diagnostics"] = diagnostics
    captured["report"] = report
    return captured


def _axis_rotation(point_on_axis, theta):
    Q = np.eye(4)
    Q[:3, :3] = Rotation.from_rotvec([0.0, 0.0, theta]).as_matrix()
    Q[:3, 3] = point_on_axis - Q[:3, :3] @ point_on_axis
    return Q


def _apply_cluster_orbit(captured, axis_xy, theta):
    initial = captured["initial"].copy()
    layout = captured["diagnostics"]["prepared"]["parameter_layout"]
    selected = captured["diagnostics"]["prepared"]["selection"]["selected_landmarks"]
    pose_offsets = {int(k): int(v) for k, v in layout["pose_offsets"].items()}
    point_offset = int(layout["point_offset"])
    Q = _axis_rotation(np.r_[axis_xy, 0.0], theta)
    alternative = initial.copy()
    for keyframe_id in (2, 3, 4):
        offset = pose_offsets[keyframe_id]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(initial[offset:offset + 3]).as_matrix()
        pose[:3, 3] = initial[offset + 3:offset + 6]
        transformed = Q @ pose
        alternative[offset:offset + 3] = Rotation.from_matrix(transformed[:3, :3]).as_rotvec()
        alternative[offset + 3:offset + 6] = transformed[:3, 3]
    for item in selected:
        index = int(item["rank"])
        point_start = point_offset + 3 * index
        point = initial[point_start:point_start + 3]
        alternative[point_start:point_start + 3] = Q[:3, :3] @ point + Q[:3, 3]
    return alternative


def test_two_bridge_rotation_is_an_exact_null_of_the_production_selected_and_held_residual(monkeypatch):
    state, _ = _rigidity_state()
    captured = _capture_residual(monkeypatch, state, bypass_guard=True)
    assert "fun" in captured
    alternate = _apply_cluster_orbit(captured, np.array([0.38, 0.08]), theta=0.71)
    np.testing.assert_allclose(captured["fun"](captured["initial"]),
                               captured["fun"](alternate), atol=2e-7, rtol=0.0)


def test_two_bridge_production_graph_is_rejected_before_solver_and_any_map_mutation(monkeypatch):
    state, _ = _rigidity_state()
    before = _snapshot(state)
    calls = []
    monkeypatch.setattr(bundle, "least_squares", lambda *a, **k: calls.append((a, k)))
    report = _run(state)
    assert report["applied"] is False
    assert report["reason"] == "unobservable_image_pose_graph"
    observability = report["image_pose_observability"]
    assert observability["status"] == "unobservable"
    assert observability["pose_columns"] == 18
    assert observability["rank"] == 17
    assert observability["nullity"] == 1
    assert calls == []
    _assert_snapshot_same(state, before)


@pytest.mark.parametrize("third", ["noncollinear", "collinear"])
def test_third_bridge_only_breaks_the_gauge_when_noncollinear(monkeypatch, third):
    state, _ = _rigidity_state(third=third)
    calls = []
    monkeypatch.setattr(bundle, "least_squares", lambda fun, initial, **kwargs: (
        calls.append(dict(kwargs)) or _solve_identity(fun, initial, **kwargs)
    ))
    report = _run(state)
    if third == "collinear":
        assert report["reason"] == "unobservable_image_pose_graph"
        assert report["image_pose_observability"]["nullity"] == 1
        assert calls == []
    else:
        assert len(calls) == 1
        assert report["image_pose_observability"]["status"] == "observable"
        assert report["image_pose_observability"]["nullity"] == 0
        assert set(calls[0]) == {"jac_sparsity", "loss", "f_scale", "max_nfev", "x_scale", "tr_solver"}


def test_excluded_fixed_worldpoint_observation_breaks_cluster_rotation_gauge(monkeypatch):
    state, excluded_id = _rigidity_state(excluded=True)
    assert excluded_id is not None
    calls = []
    monkeypatch.setattr(bundle, "least_squares", lambda fun, initial, **kwargs: (
        calls.append(dict(kwargs)) or _solve_identity(fun, initial, **kwargs)
    ))
    report = _run(state)
    assert len(calls) == 1
    assert report["image_pose_observability"]["status"] == "observable"
    assert report["image_pose_observability"]["nullity"] == 0


@pytest.mark.parametrize("rotvec", [np.zeros(3), np.array([0.18, -0.09, 0.13])])
def test_observation_jacobian_matches_central_difference_with_nonzero_offset(rotvec):
    pose = _pose([0.25, -0.12, 0.08], rotvec)
    point = np.array([1.1, 0.32, 7.8])
    pixel = np.array([334.0, 251.0])
    right = 321.5
    mask = np.array([True, True, True])
    values, pose_jac, point_jac = _observation_jacobians(
        pose, rotvec, point, pixel, right, mask, K, BASELINE, OFFSET)

    def residual(vector):
        candidate = _pose(vector[3:6], vector[:3])
        uv, depth = project(vector[6:9][None], candidate, K)
        error = np.clip(uv[0] - pixel, -1e4, 1e4)
        r = np.r_[error, right_pixel(uv[0, 0], depth[0], K[0, 0], BASELINE, OFFSET) - right]
        if depth[0] <= 0:
            r[:] = 1e4
        return r

    vector = np.r_[rotvec, pose[:3, 3], point]
    numeric = np.column_stack([
        (residual(vector + np.eye(9)[i] * 1e-6)
         - residual(vector - np.eye(9)[i] * 1e-6)) / 2e-6
        for i in range(9)
    ])
    np.testing.assert_allclose(np.c_[pose_jac, point_jac], numeric, atol=2e-5, rtol=2e-6)
    assert values.shape == (3,)


def test_clip_semantics_preserve_unclipped_right_derivative_and_behind_camera_has_zero_rows():
    pose = _pose([0., 0., 0.], [0.07, -0.11, 0.04])
    point = np.array([1.2, 0.3, 6.0])
    # Far outside the left-u clip while keeping the stereo right row active.
    _values, Jpose, _Jpoint = _observation_jacobians(
        pose, [0.07, -0.11, 0.04], point, [-30000., 240.], 0.,
        [True, True, True], K, BASELINE, OFFSET)
    np.testing.assert_array_equal(Jpose[0], np.zeros(6))
    assert np.linalg.norm(Jpose[2]) > 1e-4

    behind = np.array([1.2, 0.3, -6.0])
    _values, Jpose, Jpoint = _observation_jacobians(
        pose, [0.07, -0.11, 0.04], behind, [100., 240.], 0.,
        [True, True, True], K, BASELINE, OFFSET)
    assert np.isfinite(Jpose).all() and np.isfinite(Jpoint).all()
    np.testing.assert_array_equal(Jpose, np.zeros((3, 6)))
    np.testing.assert_array_equal(Jpoint, np.zeros((3, 3)))
    zero_depth = np.array([1.2, 0.3, 0.0])
    _values, Jpose, Jpoint = _observation_jacobians(
        pose, [0.07, -0.11, 0.04], zero_depth, [100., 240.], 0.,
        [True, True, True], K, BASELINE, OFFSET)
    np.testing.assert_array_equal(Jpose, np.zeros((3, 6)))
    np.testing.assert_array_equal(Jpoint, np.zeros((3, 3)))


def test_inactive_right_measurement_may_be_missing_but_active_right_must_be_finite():
    pose = _pose([0., 0., 0.], [0.03, -0.08, 0.04])
    point = np.array([0.5, 0.2, 7.0])
    values, Jpose, Jpoint = _observation_jacobians(
        pose, [0.03, -0.08, 0.04], point, [330., 250.], np.nan,
        [True, True, False], K, BASELINE, OFFSET)
    assert values.shape == (2,)
    assert np.isfinite(Jpose).all() and np.isfinite(Jpoint).all()
    with pytest.raises(ValueError, match="invalid_active_right_measurement"):
        _observation_jacobians(
            pose, [0.03, -0.08, 0.04], point, [330., 250.], np.nan,
            [True, True, True], K, BASELINE, OFFSET)


def test_provider_augmented_objectives_explicitly_skip_original_only_guard(monkeypatch):
    # A callable provider is a different full objective; original rows alone
    # must never veto it as though they were the complete model.
    from test_free_source_stereo_bundle import _free_source_case, _provider
    from test_bundle_diagnostics import MATRIX as CAMERA_K, BASELINE as CAMERA_B

    state, truth, source, ids = _free_source_case(source_bias=0.16)
    monkeypatch.setattr(bundle, "least_squares", _solve_identity)
    report = bundle.local_bundle_adjustment(
        state, CAMERA_K, CAMERA_B, window=3, max_landmarks=24,
        training_factor_provider=_provider(state, truth, source, ids),
    )
    assert report["image_pose_observability"]["status"] == "skipped"
    assert report["image_pose_observability"]["reason"] == "augmented_model_scope_not_supported"


def test_monocular_objective_keeps_existing_guard_scope_and_solver_path(monkeypatch):
    state, _ = _rigidity_state()
    state.metric = False
    calls = []
    monkeypatch.setattr(bundle, "least_squares", lambda fun, initial, **kwargs: (
        calls.append(dict(kwargs)) or _solve_identity(fun, initial, **kwargs)
    ))
    monkeypatch.setattr(bundle, "original_image_pose_observability",
                        lambda *a, **k: pytest.fail("mono guard must not run"))
    report = _run(state, baseline=None)
    assert len(calls) == 1
    assert report["image_pose_observability"]["status"] == "skipped"
    assert report["image_pose_observability"]["reason"] == "monocular_scope_not_validated"


def test_full_rank_narrow_far_support_is_not_rejected_by_condition_cutoff(monkeypatch):
    state, _ = _rigidity_state(third="noncollinear", geometry_scale=4.0,
                               baseline_scale=1.0, spread=0.12)
    calls = []
    monkeypatch.setattr(bundle, "least_squares", lambda fun, initial, **kwargs: (
        calls.append(dict(kwargs)) or _solve_identity(fun, initial, **kwargs)
    ))
    report = _run(state)
    assert len(calls) == 1
    assert report["image_pose_observability"]["status"] == "observable"
    assert report["image_pose_observability"]["rank"] == 18


def test_consistent_world_and_metric_scale_preserve_geometric_rank(monkeypatch):
    results = []
    for factor in (1.0, 7.0):
        state, _ = _rigidity_state(third="noncollinear", geometry_scale=factor,
                                   baseline_scale=factor)
        monkeypatch.setattr(bundle, "least_squares", _solve_identity)
        report = _run(state)
        results.append(report["image_pose_observability"])
    assert results[0]["rank"] == results[1]["rank"] == 18
    assert results[0]["nullity"] == results[1]["nullity"] == 0
    np.testing.assert_allclose(results[0]["singular_values"],
                               results[1]["singular_values"], rtol=2e-8, atol=2e-8)


def test_huber_sized_measurement_errors_do_not_change_raw_geometric_rank(monkeypatch):
    reports = []
    for perturb in (0.0, 85.0):
        state, _ = _rigidity_state(third="noncollinear")
        if perturb:
            for landmark in state.landmarks.values():
                for observation in landmark.observations.values():
                    observation.pixel = np.asarray(observation.pixel, float) + [perturb, -0.7 * perturb]
        monkeypatch.setattr(bundle, "least_squares", _solve_identity)
        reports.append(_run(state)["image_pose_observability"])
    assert reports[0]["rank"] == reports[1]["rank"] == 18
    np.testing.assert_allclose(reports[0]["singular_values"], reports[1]["singular_values"],
                               rtol=0.0, atol=1e-12)
