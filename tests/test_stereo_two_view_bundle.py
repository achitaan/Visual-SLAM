"""Independent geometry/ownership contracts for the opt-in two-view bundle.

These small fixtures test correctness, not efficacy or convergence of the capped
production solver. No dataset poses or images participate in any fit.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import stereo_two_view_bundle as bundle


K = np.array([[320., 3.25, 320.], [0., 315., 240.], [0., 0., 1.]])
BASELINE = .24
OFFSET = 2.75
SIZE = (640, 480)


def _pose(phi=(.06, -.045, .018), center=(.28, .04, .12)):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(phi).as_matrix()
    out[:3, 3] = center
    return out


def _project6(points, target, matrix=K, baseline=BASELINE, offset=OFFSET):
    """Independent unclipped six-image-component oracle for valid geometry."""
    output = []
    for camera in (np.eye(4), target):
        p = (points - camera[:3, 3]) @ camera[:3, :3]
        h = p @ matrix.T
        uv = h[:, :2] / h[:, 2, None]
        right = uv[:, 0] - matrix[0, 0] * baseline / p[:, 2] - offset
        output.append(np.column_stack((uv, right)))
    return np.column_stack(output)


def _scene(count=24, *, kind="nonplanar", offset=OFFSET):
    pixel_grid = np.array([(u, v) for v in (80., 180., 290., 400.)
                           for u in (80., 170., 270., 370., 470., 560.)])
    pixels = pixel_grid[:count].copy()
    depth = np.linspace(7., 14., count)
    points = np.column_stack((pixels, np.ones(count))) @ np.linalg.inv(K).T
    points *= depth[:, None]
    if kind == "collinear":
        points = np.column_stack((np.linspace(-6., 6., count),
                                  np.zeros(count), np.full(count, 10.)))
    elif kind == "narrow":
        points[:, :2] = np.column_stack((np.linspace(-.1, .1, count),
                                         np.sin(np.arange(count)) * .08))
    elif kind != "nonplanar":
        raise ValueError(kind)
    truth = _pose()
    measured = _project6(points, truth, offset=offset)
    target_points = (points - truth[:3, 3]) @ truth[:3, :3]
    seed = truth.copy()
    seed[:3, :3] = Rotation.from_rotvec([.0006, -.0004, .0003]).as_matrix() @ truth[:3, :3]
    seed[:3, 3] += [.002, -.001, .0015]
    pairs = np.column_stack((np.arange(count), np.arange(count))).astype(np.int64)
    return dict(seed_pose=seed, reverse_pose=np.linalg.inv(truth), fit_pairs=pairs, forward_inlier_pairs=pairs.copy(),
                reverse_inlier_pairs=pairs.copy(), source_points=points.copy(), target_points=target_points.copy(),
                source_pixels=measured[:, :2].copy(), source_right_u=measured[:, 2].copy(),
                target_pixels=measured[:, 3:5].copy(), target_right_u=measured[:, 5].copy(),
                matrix=K.copy(), baseline=BASELINE, disparity_offset=offset,
                image_size=SIZE, min_inliers=15), truth


def _build(data, *, reverse_pose=None):
    arguments = dict(data)
    if reverse_pose is not None:
        arguments["reverse_pose"] = reverse_pose
    return bundle.build_two_view_stereo_problem(**arguments)


def _huber(values):
    r = np.asarray(values)
    a = np.abs(r)
    return float(np.where(a <= 1.5, .5 * r * r, 1.5 * (a - .75)).sum())


def _solver_result(fun, x, *, success=False, status=0):
    return SimpleNamespace(x=np.asarray(x).copy(), fun=np.asarray(fun(x)).copy(), cost=_huber(fun(x)),
                           success=success, status=status, message="controlled test iterate",
                           nfev=15, njev=15, optimality=3.)


def _reduced_rank(jacobian, count):
    """Independently eliminate each six-row/full-XYZ nuisance block."""
    j = jacobian.toarray()
    camera = []
    point_ranks = []
    eps = np.finfo(float).eps
    for i in range(count):
        block = j[6 * i:6 * i + 6]
        b = block[:, 6 + 3 * i:9 + 3 * i]
        u, s, _ = np.linalg.svd(b, full_matrices=True)
        rank = int(np.count_nonzero(s > eps * max(b.shape) * s[0]))
        point_ranks.append(rank)
        camera.append(u[:, rank:].T @ block[:, :6])
    reduced = np.vstack(camera)
    singular = np.linalg.svd(reduced, compute_uv=False)
    tolerance = eps * max(reduced.shape) * singular[0]
    return point_ranks, int(np.count_nonzero(singular > tolerance)), singular, tolerance


def test_six_components_full_matrix_offset_and_native_measured_right():
    data, _ = _scene(offset=3.125)
    # At u~300 a float32 conversion would erase this measurement perturbation.
    data["source_right_u"][0] += 2e-7
    data["target_right_u"][1] -= 3e-7
    problem = _build(data)
    camera, points = problem.decode(problem.x0)
    pairs = problem.selected_pairs
    measured = np.column_stack((data["source_pixels"][pairs[:, 0]],
                                data["source_right_u"][pairs[:, 0]],
                                data["target_pixels"][pairs[:, 1]],
                                data["target_right_u"][pairs[:, 1]]))
    expected = (_project6(points, camera, offset=3.125) - measured).ravel()
    np.testing.assert_allclose(problem.residual(problem.x0), expected, atol=2e-12, rtol=0.)
    assert problem.residual(problem.x0)[2] == pytest.approx(-2e-7, abs=2e-12)
    assert len(expected) == 6 * len(pairs)
    assert len(problem.x0) == 6 + 3 * len(pairs)


@pytest.mark.parametrize("chart", ["principal", "wrapped"])
def test_all_variable_analytic_jacobian_columns_and_exact_sparse_dependencies(chart):
    data, _ = _scene()
    # Use a complete camera-consistent fixture, including the ORIGINAL reverse
    # proof; changing only the seed would test rejection before reaching J.
    camera = _pose(phi=(.09, -.07, .035))
    measured = _project6(data["source_points"], camera)
    data.update(seed_pose=camera, reverse_pose=np.linalg.inv(camera),
                target_points=(data["source_points"] - camera[:3, 3]) @ camera[:3, :3],
                target_pixels=measured[:, 3:5].copy(), target_right_u=measured[:, 5].copy())
    problem = _build(data)
    x = problem.x0.copy()
    if chart == "wrapped":
        phi = x[:3].copy()
        x[:3] = phi * (1. + 2. * np.pi / np.linalg.norm(phi))
        assert np.linalg.norm(x[:3]) > 2. * np.pi
        principal, _ = problem.decode(problem.x0)
        wrapped, _ = problem.decode(x)
        np.testing.assert_allclose(wrapped, principal, atol=2e-14, rtol=0.)
    actual = problem.jacobian(x).toarray()
    numerical = np.empty_like(actual)
    for column in range(len(x)):
        step = 1e-6 * max(1., abs(float(x[column])))
        plus, minus = x.copy(), x.copy()
        plus[column] += step
        minus[column] -= step
        numerical[:, column] = (problem.residual(plus) - problem.residual(minus)) / (2 * step)
    np.testing.assert_allclose(actual, numerical, atol=3e-6, rtol=2e-6)
    expected_sparsity = np.zeros_like(actual, dtype=bool)
    for i in range(len(problem.selected_pairs)):
        # Source camera is fixed. Only the target rows depend on six pose cols.
        np.testing.assert_array_equal(actual[6 * i:6 * i + 3, :6], 0.)
        other = np.ones(actual.shape[1] - 6, bool)
        other[3 * i:3 * i + 3] = False
        np.testing.assert_array_equal(actual[6 * i:6 * i + 6, 6:][:, other], 0.)
        expected_sparsity[6 * i:6 * i + 6, 6 + 3 * i:9 + 3 * i] = True
        expected_sparsity[6 * i + 3:6 * i + 6, :6] = True
    np.testing.assert_array_equal(problem.sparsity().toarray().astype(bool), expected_sparsity)


def test_full_xyz_moves_source_bearing_and_metric_rank_is_six():
    data, _ = _scene()
    problem = _build(data)
    x = problem.x0.copy()
    before = problem.residual(x).reshape(-1, 6)
    x[6] += .01
    after = problem.residual(x).reshape(-1, 6)
    assert abs(after[0, 0] - before[0, 0]) > .1
    assert abs(after[0, 2] - before[0, 2]) > .1
    np.testing.assert_array_equal(after[1:], before[1:])
    point_ranks, pose_rank, singular, tolerance = _reduced_rank(
        problem.jacobian(problem.x0), len(problem.selected_pairs))
    assert set(point_ranks) == {3}
    assert pose_rank == 6 and singular[-1] > tolerance


def test_selection_uses_inlier_union_in_original_fit_order_and_no_other_rows():
    data, _ = _scene()
    original = data["fit_pairs"].copy()
    data["fit_pairs"] = original[::-1].copy()
    data["forward_inlier_pairs"] = original[:18].copy()
    data["reverse_inlier_pairs"] = original[6:21].copy()
    problem = _build(data)
    expected_indices = np.flatnonzero(np.isin(data["fit_pairs"][:, 0], np.arange(21)))
    np.testing.assert_array_equal(problem.selected_pair_indices, expected_indices)
    np.testing.assert_array_equal(problem.selected_pairs, data["fit_pairs"][expected_indices])
    assert not np.isin(problem.selected_pairs[:, 0], [21, 22, 23]).any()


def test_selection_rejects_unknown_inlier_and_conflicting_physical_pair():
    data, _ = _scene()
    data["forward_inlier_pairs"][0] = [1000, 1000]
    with pytest.raises(ValueError):
        _build(data)
    data, _ = _scene()
    data["fit_pairs"][1] = [0, 1]
    with pytest.raises(ValueError):
        _build(data)


def test_problem_owns_input_measurements_and_geometry():
    data, _ = _scene()
    problem = _build(data)
    before = problem.residual(problem.x0).copy()
    for name in ("source_pixels", "source_right_u", "target_pixels", "target_right_u", "source_points", "target_points"):
        data[name][:] = -999
    np.testing.assert_array_equal(problem.residual(problem.x0), before)


def test_clean_nonplanar_small_seed_recovery_under_fixed_production_cap():
    data, truth = _scene()
    problem = _build(data)
    candidate, report = bundle.refine_two_view_stereo_training(problem, np.linalg.inv(truth))
    assert candidate is not None and report["accepted"]
    assert report["solver"]["max_nfev"] == 15
    assert report["solver"]["loss"] == "huber"
    assert report["solver"]["f_scale"] == 1.5
    np.testing.assert_allclose(candidate[:3, 3], truth[:3, 3], atol=2e-4, rtol=0.)
    assert np.linalg.norm(Rotation.from_matrix(candidate[:3, :3].T @ truth[:3, :3]).as_rotvec()) < 2e-5
    # This controlled clean fixture is not a guarantee of stationarity on data.
    assert report["solver"]["cost"] < report["solver"]["initial_cost"]
    assert report["initial_observability"]["rank"] == 6
    assert report["final_observability"]["rank"] == 6


@pytest.mark.parametrize("field", ["source_points", "target_points", "source_pixels", "target_pixels"])
def test_nonfinite_active_measurement_is_rejected(field):
    data, _ = _scene()
    data[field].flat[0] = np.nan
    with pytest.raises(ValueError):
        _build(data)


@pytest.mark.parametrize("baseline", [0., -.24, np.nan])
def test_unknown_or_nonmetric_baseline_is_rejected(baseline):
    data, _ = _scene()
    data["baseline"] = baseline
    with pytest.raises(ValueError):
        _build(data)


@pytest.mark.parametrize("depth", [-1., 0., .09, 100.1])
def test_seed_point_depth_outside_existing_acquisition_domain_is_rejected(monkeypatch, depth):
    data, _ = _scene()
    data["source_points"][0, 2] = depth
    monkeypatch.setattr(bundle, "least_squares", lambda *a, **k: pytest.fail("bad seed depth entered optimizer"))
    try:
        problem = _build(data)
    except ValueError:
        return
    candidate, report = bundle.refine_two_view_stereo_training(problem, data["reverse_pose"])
    assert candidate is None and not report["accepted"]


def test_rejected_no_step_solver_preserves_seed_and_every_input(monkeypatch):
    data, truth = _scene()
    saved = {key: value.copy() for key, value in data.items() if isinstance(value, np.ndarray)}
    problem = _build(data)
    before_x = problem.x0.copy()
    calls = []
    def no_step(fun, initial, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("method", "trf") == "trf" and kwargs["tr_solver"] == "lsmr"
        assert kwargs["max_nfev"] == 15
        assert callable(kwargs["jac"])
        return _solver_result(fun, initial)
    monkeypatch.setattr(bundle, "least_squares", no_step)
    candidate, report = bundle.refine_two_view_stereo_training(problem, np.linalg.inv(truth))
    assert candidate is None and not report["accepted"] and len(calls) == 1
    assert report["solver"]["status"] == 0 and not report["solver"]["success"]
    np.testing.assert_array_equal(problem.x0, before_x)
    for key, value in saved.items():
        np.testing.assert_array_equal(data[key], value)


@pytest.mark.parametrize("kind", ["collinear", "narrow"])
def test_unobservable_or_unscattered_training_abstains_before_solver(monkeypatch, kind):
    data, truth = _scene(kind=kind)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("unobservable/unscattered rows must not enter the optimizer")
    monkeypatch.setattr(bundle, "least_squares", forbidden)
    try:
        problem = _build(data)
    except ValueError:
        assert not calls
        return
    if kind == "collinear":
        point_ranks, rank, _, _ = _reduced_rank(problem.jacobian(problem.x0),
                                               len(problem.selected_pairs))
        assert set(point_ranks) == {3} and rank == 5
    candidate, report = bundle.refine_two_view_stereo_training(problem, np.linalg.inv(truth))
    assert candidate is None and not report["accepted"] and not calls


def test_duplicate_physical_pixels_with_incompatible_depth_abstain():
    data, _ = _scene()
    data["source_pixels"][1] = data["source_pixels"][0]
    with pytest.raises(ValueError):
        _build(data)


def test_analytic_clip_boundary_is_not_given_a_fictitious_derivative():
    data, _ = _scene()
    # Keep the large-x kink point inside BOTH depth domains so this test reaches
    # the nondifferentiable projection branch, rather than an earlier depth veto.
    data["seed_pose"] = np.eye(4)
    data["reverse_pose"] = np.eye(4)
    data["target_points"] = data["source_points"].copy()
    data["target_pixels"] = data["source_pixels"].copy()
    data["target_right_u"] = data["source_right_u"].copy()
    problem = _build(data)
    bad = problem.x0.copy()
    # This arithmetic is exact in binary for first observed u=80: projected
    # u=10080 and error=10000, at the existing left clipping kink.
    bad[6:9] = [244., 0., 8.]
    with pytest.raises(ValueError, match="nondifferentiable_left_clip_boundary"):
        problem.jacobian(bad)


def _retarget_far(data, camera):
    data = {key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in data.items()}
    pixels = data["source_pixels"]
    points = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    points *= np.linspace(72., 88., len(points))[:, None]
    measured = _project6(points, camera)
    data.update(source_points=points, target_points=(points - camera[:3, 3]) @ camera[:3, :3],
                source_pixels=measured[:, :2], source_right_u=measured[:, 2],
                target_pixels=measured[:, 3:5], target_right_u=measured[:, 5])
    return data


def _inject_pose(monkeypatch, camera, points):
    calls = []
    def solver(fun, initial, **kwargs):
        calls.append(True)
        value = np.r_[Rotation.from_matrix(camera[:3, :3]).as_rotvec(),
                      camera[:3, 3], np.asarray(points).ravel()]
        return _solver_result(fun, value)
    monkeypatch.setattr(bundle, "least_squares", solver)
    return calls


@pytest.mark.parametrize("kind", ["translation", "rotation"])
def test_final_candidate_seed_movement_guard_even_with_perfect_training_fit(monkeypatch, kind):
    data, _ = _scene()
    seed = data["seed_pose"].copy()
    candidate = seed.copy()
    if kind == "translation":
        candidate[2, 3] += .51
    else:
        candidate[:3, :3] = Rotation.from_rotvec([0., np.deg2rad(1.6), 0.]).as_matrix() @ seed[:3, :3]
    data = _retarget_far(data, candidate)
    middle = seed.copy()
    if kind == "translation":
        middle[2, 3] += .255
    else:
        middle[:3, :3] = Rotation.from_rotvec([0., np.deg2rad(.8), 0.]).as_matrix() @ seed[:3, :3]
    problem = _build(data, reverse_pose=np.linalg.inv(middle))
    calls = _inject_pose(monkeypatch, candidate, data["source_points"])
    result, report = bundle.refine_two_view_stereo_training(problem, np.linalg.inv(middle))
    assert calls and result is None and not report["accepted"]
    assert report["solver"]["cost"] < report["solver"]["initial_cost"]
    assert report["guards"]["seed_motion_passed"] is False
    assert report["guards"]["reverse_pose_motion_passed"] is True


def test_final_reverse_consistency_guard_even_when_seed_step_is_small(monkeypatch):
    data, _ = _scene()
    seed = data["seed_pose"].copy()
    candidate = seed.copy()
    candidate[2, 3] += .30
    data = _retarget_far(data, candidate)
    reverse_equivalent = seed.copy()
    reverse_equivalent[2, 3] -= .31
    problem = _build(data, reverse_pose=np.linalg.inv(reverse_equivalent))
    calls = _inject_pose(monkeypatch, candidate, data["source_points"])
    result, report = bundle.refine_two_view_stereo_training(problem, np.linalg.inv(reverse_equivalent))
    assert calls and result is None and not report["accepted"]
    assert report["solver"]["cost"] < report["solver"]["initial_cost"]
    assert report["guards"]["reverse_pose_motion_passed"] is False
    assert report["guards"]["seed_motion_passed"] is True


def test_original_reverse_xyz_guard_cannot_be_replaced_by_optimized_training_xyz(monkeypatch):
    data, truth = _scene()
    # A depth bias preserves the target LEFT bearing. It changes the original
    # reverse temporal geometry, which must still be checked after refinement.
    data["target_points"] *= 1.8
    data["target_right_u"] = (data["target_pixels"][:, 0]
                              - K[0, 0] * BASELINE / data["target_points"][:, 2] - OFFSET)
    data["seed_pose"][:3, 3] += [.015, 0., 0.]
    problem = _build(data)
    calls = _inject_pose(monkeypatch, truth, data["source_points"])
    candidate, report = bundle.refine_two_view_stereo_training(problem, np.linalg.inv(truth))
    assert calls and candidate is None and not report["accepted"]
    assert report["guards"]["directions"]["passed"] is False
    assert report["guards"]["directions"]["reverse_original_median_px"] > 1.5


def test_capped_finite_improved_iterate_is_not_falsely_reported_converged(monkeypatch):
    data, truth = _scene()
    problem = _build(data)
    _inject_pose(monkeypatch, truth, data["source_points"])
    candidate, report = bundle.refine_two_view_stereo_training(problem, np.linalg.inv(truth))
    assert candidate is not None and report["accepted"]
    assert report["solver"]["status"] == 0
    assert report["solver"]["success"] is False
    assert report["solver"]["nfev"] == 15
    assert report["solver"]["capped_not_converged"] is True
    assert report["solver"]["cost"] < report["solver"]["initial_cost"]


@pytest.mark.parametrize("which", ["source_right_u", "target_right_u"])
def test_missing_right_drops_only_ineligible_training_point(which):
    data, _ = _scene()
    data[which][2] = np.nan
    problem = _build(data)
    assert len(problem.selected_pairs) == 23
    assert 2 not in problem.selected_pairs[:, 0]
    assert np.isfinite(problem.residual(problem.x0)).all()


def test_unused_unsupported_xyz_and_right_holes_do_not_invalidate_active_problem():
    data, _ = _scene()
    expected = _build(data)
    for field in ("source_points", "target_points"):
        data[field] = np.vstack((data[field], [np.nan, np.nan, np.nan]))
    data["source_pixels"] = np.vstack((data["source_pixels"], [320., 240.]))
    data["target_pixels"] = np.vstack((data["target_pixels"], [310., 230.]))
    for field in ("source_right_u", "target_right_u"):
        data[field] = np.r_[data[field], np.nan]
    actual = _build(data)
    np.testing.assert_array_equal(actual.selected_pairs, expected.selected_pairs)
    np.testing.assert_array_equal(actual.x0, expected.x0)
    np.testing.assert_array_equal(actual.residual(actual.x0), expected.residual(expected.x0))


def test_invalid_competing_endpoint_cannot_disappear_before_physical_uniqueness():
    data, _ = _scene()
    data["fit_pairs"][1] = [0, 1]
    data["forward_inlier_pairs"] = data["fit_pairs"].copy()
    data["reverse_inlier_pairs"] = data["fit_pairs"].copy()
    data["target_right_u"][1] = np.nan
    with pytest.raises(ValueError):
        _build(data)


@pytest.mark.parametrize("excluded_kind", ["source_alias", "target_alias", "source_landmark", "target_landmark"])
def test_builder_excludes_each_held_endpoint_and_landmark_claim(excluded_kind):
    data, _ = _scene()
    original_pairs = data["fit_pairs"].copy()
    train, held = original_pairs[:20], original_pairs[20:]
    data.update(fit_pairs=train.copy(), forward_inlier_pairs=train.copy(),
                reverse_inlier_pairs=train.copy(), held_pairs=held.copy())
    if excluded_kind == "source_alias":
        data["source_pixels"][1] = data["source_pixels"][23]
    elif excluded_kind == "target_alias":
        data["target_pixels"][1] = data["target_pixels"][23]
    else:
        source_ids = np.full(24, -1, dtype=np.int64)
        target_ids = np.full(24, -1, dtype=np.int64)
        (source_ids if excluded_kind == "source_landmark" else target_ids)[1] = 900
        data.update(source_landmark_ids=source_ids, target_landmark_ids=target_ids,
                    excluded_landmark_ids={900})
    saved_held = data["held_pairs"].copy()
    problem = _build(data)
    assert len(problem.selected_pairs) == 19
    assert not np.isin(problem.selected_pairs[:, 0], [1, 20, 21, 22, 23]).any()
    np.testing.assert_array_equal(problem.selected_pairs, np.delete(train, 1, axis=0))
    np.testing.assert_array_equal(data["held_pairs"], saved_held)


def test_original_forward_xyz_guard_cannot_be_replaced_by_optimized_training_xyz(monkeypatch):
    data, truth = _scene()
    optimized_points = data["source_points"].copy()
    data["source_points"] *= 1.8
    data["source_right_u"] = (data["source_pixels"][:, 0]
                              - K[0, 0] * BASELINE / data["source_points"][:, 2] - OFFSET)
    problem = _build(data)
    calls = _inject_pose(monkeypatch, truth, optimized_points)
    candidate, report = bundle.refine_two_view_stereo_training(problem, np.linalg.inv(truth))
    assert calls and candidate is None and not report["accepted"]
    assert report["guards"]["directions"]["passed"] is False
    assert report["guards"]["directions"]["forward_original_median_px"] > 1.5


@pytest.mark.parametrize("cap", [True, 14, 16, 30])
def test_fixed_iteration_cap_cannot_be_overridden(monkeypatch, cap):
    data, truth = _scene()
    problem = _build(data)
    monkeypatch.setattr(bundle, "least_squares", lambda *a, **k: pytest.fail("undeclared solver cap entered optimizer"))
    candidate, report = bundle.refine_two_view_stereo_training(problem, np.linalg.inv(truth), max_nfev=cap)
    assert candidate is None and not report["accepted"]


@pytest.mark.parametrize("outcome", ["accepted", "rejected", "builder_failed"])
def test_actual_prepare_keeps_held_and_raw_geometry_and_rejection_seed(monkeypatch, outcome):
    """Actual caller/partition, controlled solver seam; no accuracy claim."""
    import shared_slam as shared
    from test_stereo_arbitration_tracking import camera, record, install_previous

    seed = np.eye(4)
    seed[0, 3] = .002
    candidate = seed.copy()
    candidate[0, 3] = .001
    capture_options = []
    def original_reference(source, target, matrix, **options):
        pairs = options["matcher"](source.descriptors, target.descriptors)
        capture_options.append(bool(options.get("capture_rows", False)))
        result = dict(measurement=seed.copy(), matches=len(pairs), inliers=len(pairs),
                      median_reprojection_px=0., reverse_checked=True,
                      reverse_translation_error_m=.002, reverse_rotation_error_deg=0.,
                      target_features=pairs[:, 1].tolist(), bidirectional_refinement={"applied": False})
        if options.get("capture_rows", False):
            result["training_rows"] = dict(
                fit_pairs=pairs.copy(), forward_inlier_pairs=pairs.copy(),
                reverse_inlier_pairs=pairs.copy(), reverse_measurement=np.eye(4))
        return result
    monkeypatch.setattr(shared, "estimate_stereo_reference", original_reference)

    default = camera()
    active = camera(stereo_two_view_refinement=True)
    try:
        previous_off, current_off = record(default), record(default, 1)
        previous_on, current_on = record(active), record(active, 1)
        install_previous(default, previous_off)
        install_previous(active, previous_on)
        raw_arrays = [previous_on.points, previous_on.pixels, previous_on.right_u,
                      current_on.points, current_on.pixels, current_on.right_u]
        saved_arrays = [v.copy() for v in raw_arrays]
        saved_poses = [v.copy() for v in active.map.poses]
        calls = []
        real_builder = bundle.build_two_view_stereo_problem
        def inspect_builder(**arguments):
            calls.append(arguments)
            if outcome == "builder_failed":
                raise ValueError("controlled builder rejection")
            return real_builder(**arguments)
        def inspect_refine(problem, reverse_pose, **options):
            assert options == {"max_nfev": 15}
            np.testing.assert_array_equal(reverse_pose, np.eye(4))
            return (candidate.copy() if outcome == "accepted" else None,
                    dict(status=outcome, accepted=outcome == "accepted", reason="controlled seam",
                         guards={"optimized_xyz_written_to_map": False, "heldout_used_in_fit": False,
                                 "directions": {"forward_original_median_px": .0003},
                                 "reverse_pose_translation_m": .001,
                                 "reverse_pose_rotation_deg": 0.}))
        monkeypatch.setattr(shared, "build_two_view_stereo_problem", inspect_builder)
        monkeypatch.setattr(shared, "refine_two_view_stereo_training", inspect_refine)
        control_context, control_report = default._prepare_stereo_arbitration(1, current_off)
        assert control_context is not None and not calls
        assert "two_view_stereo_refinement" not in control_report
        active_context, active_report = active._prepare_stereo_arbitration(1, current_on)
        assert active_context is not None and len(calls) == 1
        assert capture_options == [False, True]
        np.testing.assert_array_equal(active_context["fit"], control_context["fit"])
        held = np.column_stack((control_context["evidence"].source_ids,
                                control_context["evidence"].target_ids))
        np.testing.assert_array_equal(calls[0]["held_pairs"], held)
        for field in ("points", "left", "right_u", "source_ids", "target_ids", "landmark_ids"):
            np.testing.assert_array_equal(getattr(active_context["evidence"], field),
                                          getattr(control_context["evidence"], field))
        assert active_context["excluded_landmarks"] == control_context["excluded_landmarks"]
        assert active_context["excluded_targets"] == control_context["excluded_targets"]
        assert active_context["excluded_target_pixels"] == control_context["excluded_target_pixels"]
        np.testing.assert_array_equal(calls[0]["source_points"], previous_on.points)
        np.testing.assert_array_equal(calls[0]["target_points"], current_on.points)
        np.testing.assert_array_equal(calls[0]["source_right_u"], previous_on.right_u)
        np.testing.assert_array_equal(calls[0]["target_right_u"], current_on.right_u)
        expected = candidate if outcome == "accepted" else seed
        np.testing.assert_array_equal(active_context["verified"]["measurement"], expected)
        if outcome == "accepted":
            assert active_context["verified"]["median_reprojection_px"] == .0003
            assert active_context["verified"]["reverse_translation_error_m"] == .001
            assert active_context["verified"]["reverse_rotation_error_deg"] == 0.
            assert active_context["verified"]["two_view_seed_median_reprojection_px"] == 0.
        else:
            assert active_context["verified"]["median_reprojection_px"] == 0.
        assert active_report["two_view_stereo_refinement"]["seed_pose_preserved_on_rejection"] is (outcome != "accepted")
        for actual, saved in zip(raw_arrays, saved_arrays):
            np.testing.assert_array_equal(actual, saved)
        for actual, saved in zip(active.map.poses, saved_poses):
            np.testing.assert_array_equal(actual, saved)
        assert not active.map.landmarks
    finally:
        default.close()
        active.close()
