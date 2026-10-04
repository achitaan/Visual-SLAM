"""Certified two-bridge finite gauge action for the original stereo BA."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import bundle_gauge as gauge
import local_bundle as bundle
import test_bundle_rigidity as rigidity


def _source_capture(monkeypatch, state):
    return rigidity._capture_residual(monkeypatch, state, bypass_guard=True)


def _observations_in_rank_order(state, prepared):
    selected = prepared["selection"]["selected_landmarks"]
    id_to_point = {int(item.id): item.position for item in state.landmarks.values()}
    id_to_observations = {int(item.id): item.observations for item in state.landmarks.values()}
    cameras, points = [], []
    for item in selected:
        ident = int(item["landmark_id"])
        points.append(np.asarray(id_to_point[ident], dtype=float))
        for camera in sorted(id_to_observations[ident]):
            cameras.append(int(camera))
    point_rows = [int(item["landmark_id"]) for item in selected]
    camera_rows = []
    for ident in point_rows:
        observations = id_to_observations[ident]
        camera_rows.extend(sorted(observations))
    return np.asarray(points), np.asarray(camera_rows), np.repeat(
        np.arange(len(point_rows), dtype=np.int64),
        [len(id_to_observations[i]) for i in point_rows],
    )


def _certificate(capture, state, held_points=None, held_cameras=None):
    prepared = capture["diagnostics"]["prepared"]
    _points, cameras, point_indices = _observations_in_rank_order(state, prepared)
    initial = np.asarray(capture["initial"], dtype=float)
    point_offset = 18
    points = initial[point_offset:].reshape(-1, 3)
    if held_points is None:
        held_points = np.empty((0, 3), dtype=float)
        held_cameras = np.empty(0, dtype=np.int64)
    ids = sorted(state.keyframes)
    observability = {
        "status": "unobservable", "nullity": 1, "pose_columns": 18,
        "per_point_rank_histogram": {"3": len(points)},
    }
    return gauge.certify_two_bridge_rotation(
        camera_ids=ids,
        free_camera_indices=[2, 3, 4],
        selected_record_cameras=cameras,
        selected_record_points=point_indices,
        initial_points=points,
        excluded_points=held_points,
        excluded_cameras=held_cameras,
        initial_observability=observability,
    )


def _camera_poses_from_vector(state, vector):
    poses = [state.keyframes[i].pose.copy() for i in sorted(state.keyframes)]
    for row, camera in enumerate((2, 3, 4)):
        start = 6 * row
        poses[camera][:3, :3] = Rotation.from_rotvec(vector[start:start + 3]).as_matrix()
        poses[camera][:3, 3] = vector[start + 3:start + 6]
    return np.asarray(poses)


def _state_vector(state, points):
    poses = [state.keyframes[i].pose for i in (2, 3, 4)]
    camera = np.concatenate([
        np.r_[Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]]
        for pose in poses
    ])
    return np.r_[camera, np.asarray(points, dtype=float).reshape(-1)]


def test_certificate_and_canonicalizer_preserve_complete_production_residual(monkeypatch):
    state, _ = rigidity._rigidity_state()
    capture = _source_capture(monkeypatch, state)
    certificate, cert_report = _certificate(capture, state)
    assert certificate is not None, cert_report
    assert certificate["cluster_camera_ids"] == (2, 3, 4)
    assert len(certificate["bridge_point_indices"]) == 2
    assert len(certificate["internal_point_indices"]) == 20

    x0 = capture["initial"]
    raw = rigidity._apply_cluster_orbit(capture, np.array([0.38, 0.08]), theta=0.71)
    raw_cameras = _camera_poses_from_vector(state, raw)
    initial_cameras = np.asarray([state.keyframes[i].pose for i in sorted(state.keyframes)])
    canonical, report = gauge.canonicalize_two_bridge_candidate(
        initial_vector=x0,
        raw_candidate_vector=raw,
        camera_ids=sorted(state.keyframes),
        free_camera_indices=[2, 3, 4],
        free_offsets=np.arange(18).reshape(3, 6),
        point_offset=18,
        point_limit=len(x0),
        initial_camera_poses=initial_cameras,
        candidate_camera_poses=raw_cameras,
        certificate=certificate,
        excluded_points=np.empty((0, 3)),
        excluded_cameras=np.empty(0, dtype=np.int64),
        complete_residual=capture["fun"],
    )
    assert canonical is not None, report
    assert report["status"] == "canonicalized"
    assert report["candidate_action_check"]["max_abs_difference"] <= report["candidate_action_check"]["roundoff_tolerance"]
    assert all(item["status"] == "passed" for item in report["initial_action_checks"])
    np.testing.assert_allclose(canonical, x0, rtol=0.0, atol=2e-10)
    assert report["bridge_points_unchanged"] is True


def test_candidate_axis_is_recomputed_and_off_axis_held_row_rejects():
    state, _ = rigidity._rigidity_state()
    points = np.asarray([item.position for item in state.landmarks.values()])
    records_c, records_p = [], []
    for point_index, item in enumerate(state.landmarks.values()):
        for camera in sorted(item.observations):
            records_c.append(camera)
            records_p.append(point_index)
    # Two additional fixed rows lie exactly on the initial bridge axis.
    held = np.array([[0.38, 0.08, 4.0], [0.38, 0.08, 14.0]])
    held_cameras = np.array([2, 4])
    cert, report = gauge.certify_two_bridge_rotation(
        [0, 1, 2, 3, 4], [2, 3, 4], records_c, records_p, points,
        held, held_cameras,
        {"status": "unobservable", "nullity": 1, "pose_columns": 18,
         "per_point_rank_histogram": {"3": len(points)}},
    )
    assert cert is not None, report
    x0 = _state_vector(state, points)
    initial_cameras = np.asarray([state.keyframes[i].pose for i in range(5)])
    raw_cameras = initial_cameras.copy()
    # Maintain a proper candidate layout while moving both bridge points so
    # the new candidate axis no longer contains the held fixed rows.
    candidate_points = points.copy()
    for idx in cert["bridge_point_indices"]:
        candidate_points[idx, 1] += 0.3
    raw = np.r_[x0[:18], candidate_points.reshape(-1)]
    candidate_cameras = raw_cameras.copy()
    for camera in (2, 3, 4):
        start = 6 * (camera - 2)
        raw[start:start + 3] = Rotation.from_matrix(candidate_cameras[camera, :3, :3]).as_rotvec()
        raw[start + 3:start + 6] = candidate_cameras[camera, :3, 3]
    canonical, rejected = gauge.canonicalize_two_bridge_candidate(
        x0, raw, [0, 1, 2, 3, 4], [2, 3, 4], np.arange(18).reshape(3, 6),
        18, len(x0), initial_cameras, candidate_cameras, cert, held,
        held_cameras, lambda x: np.zeros(1),
    )
    assert canonical is None
    assert rejected["reason"] == "candidate_held_point_off_bridge_axis"


def test_perpendicular_pi_rotation_is_rejected_as_nonunique():
    state, _ = rigidity._rigidity_state()
    points = np.asarray([item.position for item in state.landmarks.values()])
    records_c, records_p = [], []
    for point_index, item in enumerate(state.landmarks.values()):
        for camera in sorted(item.observations):
            records_c.append(camera)
            records_p.append(point_index)
    cert, report = gauge.certify_two_bridge_rotation(
        [0, 1, 2, 3, 4], [2, 3, 4], records_c, records_p, points,
        np.empty((0, 3)), np.empty(0, dtype=np.int64),
        {"status": "unobservable", "nullity": 1, "pose_columns": 18,
         "per_point_rank_histogram": {"3": len(points)}},
    )
    assert cert is not None, report
    x0 = _state_vector(state, points)
    initial_cameras = np.asarray([state.keyframes[i].pose for i in range(5)])
    candidate = initial_cameras.copy()
    axis = np.asarray(cert["initial_axis"])
    basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
    perpendicular = np.cross(axis, basis)
    perpendicular /= np.linalg.norm(perpendicular)
    candidate[2, :3, :3] = Rotation.from_rotvec(perpendicular * np.pi).as_matrix() @ candidate[2, :3, :3]
    raw = x0.copy()
    raw[:3] = Rotation.from_matrix(candidate[2, :3, :3]).as_rotvec()
    raw[3:6] = candidate[2, :3, 3]
    canonical, rejected = gauge.canonicalize_two_bridge_candidate(
        x0, raw, [0, 1, 2, 3, 4], [2, 3, 4], np.arange(18).reshape(3, 6),
        18, len(x0), initial_cameras, candidate, cert, np.empty((0, 3)),
        np.empty(0, dtype=np.int64), lambda x: np.zeros(1),
    )
    assert canonical is None
    assert rejected["reason"] == "closest_orientation_nonunique"


@pytest.mark.parametrize("field", ["initial_axis", "initial_pivot", "reference_camera_id"])
def test_malformed_certificate_is_rejected_without_exception(field):
    state, _ = rigidity._rigidity_state()
    points = np.asarray([item.position for item in state.landmarks.values()])
    cameras, point_indices = [], []
    for point_index, item in enumerate(state.landmarks.values()):
        for camera in sorted(item.observations):
            cameras.append(camera)
            point_indices.append(point_index)
    cert, _ = gauge.certify_two_bridge_rotation(
        [0, 1, 2, 3, 4], [2, 3, 4], cameras, point_indices, points,
        np.empty((0, 3)), np.empty(0, dtype=np.int64),
        {"status": "unobservable", "nullity": 1, "pose_columns": 18,
         "per_point_rank_histogram": {"3": len(points)}},
    )
    broken = copy.deepcopy(cert)
    broken.pop(field)
    x = _state_vector(state, points)
    poses = np.asarray([state.keyframes[i].pose for i in range(5)])
    canonical, report = gauge.canonicalize_two_bridge_candidate(
        x, x, [0, 1, 2, 3, 4], [2, 3, 4], np.arange(18).reshape(3, 6),
        18, len(x), poses, poses, broken, np.empty((0, 3)),
        np.empty(0, dtype=np.int64), lambda v: np.zeros(1),
    )
    assert canonical is None
    assert report["reason"] == "certificate_malformed"


def test_opt_in_solver_path_keeps_default_veto_and_full_rank_solver_contract(monkeypatch):
    state, _ = rigidity._rigidity_state()
    calls = []
    monkeypatch.setattr(bundle, "least_squares", lambda *args, **kwargs: calls.append(kwargs))
    default = rigidity._run(state)
    assert default["applied"] is False
    assert default["reason"] == "unobservable_image_pose_graph"
    assert calls == []

    state, _ = rigidity._rigidity_state(third="noncollinear")
    calls.clear()
    monkeypatch.setattr(bundle, "least_squares", lambda fun, initial, **kwargs: (
        calls.append(dict(kwargs)) or rigidity._solve_identity(fun, initial, **kwargs)
    ))
    report = bundle.local_bundle_adjustment(
        state, rigidity.K, rigidity.BASELINE, window=4, max_landmarks=200,
        disparity_offset=rigidity.OFFSET, gauge_mode="canonical_two_bridge",
    )
    assert len(calls) == 1
    assert set(calls[0]) == {"jac_sparsity", "loss", "f_scale", "max_nfev", "x_scale", "tr_solver"}
    assert report["image_pose_observability"]["status"] == "observable"


def test_opt_in_solver_canonicalizes_and_atomically_commits_a_real_improvement():
    state, _ = rigidity._rigidity_state()
    # Introduce a small, deterministic image-only discrepancy in one existing
    # selected row. The real sparse solver must improve the original objective;
    # the gauge action may remove only the unobservable cluster rotation.
    landmark_id = int(state.keyframes[4].landmark_ids[7])
    row = int(np.flatnonzero(state.keyframes[4].landmark_ids == landmark_id)[0])
    state.keyframes[4].pixels[row, 0] += 1.5
    state.landmarks[landmark_id].observations[4].pixel[0] += 1.5
    before = rigidity._snapshot(state)
    diagnostics = {}
    report = bundle.local_bundle_adjustment(
        state, rigidity.K, rigidity.BASELINE, window=4, max_landmarks=200,
        disparity_offset=rigidity.OFFSET, solver_accuracy="precise",
        gauge_mode="canonical_two_bridge",
        diagnostic_sink=lambda phase, payload: diagnostics.__setitem__(phase, payload),
    )
    assert report["applied"] is True, report
    assert report["final_cost"] < report["initial_cost"]
    assert report["gauge_correction"]["status"] == "canonicalized"
    assert report["gauge_correction"]["bridge_points_unchanged"] is True
    assert report["gauge_correction"]["raw_optimizer_vector_retained_separately"] is True
    assert state.revision == before["revision"] + 1
    assert state.geometry_revision == before["geometry_revision"] + 1
    np.testing.assert_array_equal(state.keyframes[0].pose, before["keyframes"][0])
    np.testing.assert_array_equal(state.keyframes[1].pose, before["keyframes"][1])
    assert any(not np.array_equal(state.keyframes[k].pose, before["keyframes"][k])
               for k in (2, 3, 4))
    assert any(not np.array_equal(state.landmarks[k].position, value)
               for k, value in before["points"].items())
    assert diagnostics["solved"]["result"]["x_role"] == "canonical_candidate_vector"
    assert diagnostics["solved"]["result"]["optimizer_metadata_role"] == "raw_solver_vector"


def test_candidate_rank_change_rejects_before_map_mutation(monkeypatch):
    state, _ = rigidity._rigidity_state()
    before = rigidity._snapshot(state)
    real_observability = bundle.original_image_pose_observability
    calls = []

    def changed_candidate_rank(*args, **kwargs):
        value = dict(real_observability(*args, **kwargs))
        calls.append(value)
        if len(calls) == 2:
            value.update(status="observable", reason=None, nullity=0,
                         rank=value["pose_columns"])
        return value

    monkeypatch.setattr(bundle, "original_image_pose_observability", changed_candidate_rank)
    monkeypatch.setattr(bundle, "least_squares", rigidity._solve_identity)
    report = bundle.local_bundle_adjustment(
        state, rigidity.K, rigidity.BASELINE, window=4, max_landmarks=200,
        disparity_offset=rigidity.OFFSET, gauge_mode="canonical_two_bridge",
    )
    assert report["applied"] is False
    assert report["reason"] == "gauge_candidate_rank_changed"
    assert len(calls) == 2
    rigidity._assert_snapshot_same(state, before)


@pytest.mark.parametrize("veto", ["stale_revision", "independent_stereo_motion_inconsistency"])
def test_postsolve_veto_never_commits_canonicalized_candidate(monkeypatch, veto):
    state, _ = rigidity._rigidity_state()
    # Give the real optimizer a small supported image residual to reduce. The
    # later veto is injected independently of that image-objective improvement.
    landmark_id = int(state.keyframes[4].landmark_ids[7])
    row = int(np.flatnonzero(state.keyframes[4].landmark_ids == landmark_id)[0])
    state.keyframes[4].pixels[row, 0] += 1.5
    state.landmarks[landmark_id].observations[4].pixel[0] += 1.5
    if veto == "independent_stereo_motion_inconsistency":
        bad_motion = np.eye(4)
        bad_motion[0, 3] = 2.0
        state.add_stereo_motion(2, 3, bad_motion)
    before = rigidity._snapshot(state)
    real_solver = bundle.least_squares

    def solve_then_veto(fun, initial, **kwargs):
        result = real_solver(fun, initial, **kwargs)
        if veto == "stale_revision":
            # Simulate a concurrent accepted-state update after the immutable
            # solver snapshot was taken; do not alter the geometry itself.
            with state.lock:
                state.revision += 1
        return result

    monkeypatch.setattr(bundle, "least_squares", solve_then_veto)
    report = bundle.local_bundle_adjustment(
        state, rigidity.K, rigidity.BASELINE, window=4, max_landmarks=200,
        disparity_offset=rigidity.OFFSET, solver_accuracy="precise",
        gauge_mode="canonical_two_bridge",
    )
    assert report["applied"] is False
    assert report["final_cost"] < report["initial_cost"]
    assert report["reason"] == veto
    # The stale test intentionally advances only the revision sentinel. The
    # complete geometry, ledger, and registry remain byte-for-byte untouched.
    after = rigidity._snapshot(state)
    expected_revision = before["revision"] + (veto == "stale_revision")
    assert after["revision"] == expected_revision
    assert after["geometry_revision"] == before["geometry_revision"]
    assert after["anchors"] == before["anchors"]
    for actual, expected in zip(after["poses"], before["poses"]):
        np.testing.assert_array_equal(actual, expected)
    for ident, expected in before["points"].items():
        np.testing.assert_array_equal(after["points"][ident], expected)
    assert after["registry"] == before["registry"]
