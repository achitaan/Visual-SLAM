"""Report-only source-history cohort diagnostics stay detached from BA."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

import local_bundle as bundle
from local_bundle import local_bundle_adjustment
from mapping_geometry import project
from test_bundle_diagnostics import BASELINE, MATRIX
from test_free_source_stereo_bundle import OFFSET, _free_source_case, _provider
from test_source_history_bundle import _shared_history_fixture
from test_free_source_stereo_bundle import _assert_snapshot_equal, _snapshot


def _pose_from_parameters(values):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(np.asarray(values[:3], float)).as_matrix()
    pose[:3, 3] = np.asarray(values[3:6], float)
    return pose


def _history_rows(state, source_pose, ids):
    rows = []
    for landmark_id in ids:
        point = state.landmarks[landmark_id].position
        pixel = project(point[None], source_pose, MATRIX)[0][0].astype(np.float32)
        rows.append({"frame_id": 3, "landmark_id": int(landmark_id), "pixel": pixel})
    return rows


def _active_history_case(monkeypatch, *, history_ids=(2, 0, 1), diagnostic_sink=None):
    state, truth, source_pose, selected_ids = _free_source_case(source_bias=0.22)
    provider_rows = _history_rows(state, source_pose, history_ids)
    expected_pixels = {
        row["landmark_id"]: row["pixel"].copy() for row in provider_rows
    }

    def source_history_provider(payload):
        if payload["source_history_phase"] == "validate":
            return [], {"status": "validated"}
        return provider_rows, {"status": "eligible", "captured_rows": len(provider_rows)}

    solver_capture = {}
    real_solver = bundle.least_squares

    def inspect_solver(fun, initial, **kwargs):
        prepared = solver_capture["prepared"]
        table = prepared["source_history_observations"]
        rows = table["rows"]
        initial = np.asarray(initial, float).copy()
        residual = np.asarray(fun(initial), float)
        row_count = len(rows)
        assert row_count == len(history_ids)
        history_slice = slice(len(residual) - 2 * row_count, len(residual))

        layout = prepared["parameter_layout"]
        chart = layout["target_relative_chart"]
        selected_index = {
            int(row["landmark_id"]): int(row["rank"])
            for row in prepared["selection"]["selected_landmarks"]
        }
        target_offset = int(layout["pose_offsets"]["2"])
        source_offset = int(chart["target_relative_camera_offsets"]["3"])
        target_pose = _pose_from_parameters(initial[target_offset:target_offset + 6])
        relative_source = _pose_from_parameters(initial[source_offset:source_offset + 6])
        assert [int(row["landmark_id"]) for row in rows] == list(history_ids)

        for row_index, row in enumerate(rows):
            ident = int(row["landmark_id"])
            point_offset = int(layout["point_offset"]) + 3 * selected_index[ident]
            point = initial[point_offset:point_offset + 3]
            expected = project(point[None], target_pose @ relative_source, MATRIX)[0][0]
            expected = expected - expected_pixels[ident]
            actual = residual[history_slice.start + 2 * row_index:
                              history_slice.start + 2 * row_index + 2]
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-7)
            assert int(row["selected_point_index"]) == selected_index[ident]
            assert row["frame_id"] == 3
            np.testing.assert_array_equal(
                np.asarray(row["pixel_float32"], dtype=np.float32), expected_pixels[ident]
            )

        solver_capture["residual"] = residual.copy()
        return real_solver(fun, initial, **kwargs)

    monkeypatch.setattr(bundle, "least_squares", inspect_solver)

    def capture(phase, payload):
        if phase == "prepared":
            solver_capture["prepared"] = copy.deepcopy(payload)
        if diagnostic_sink is not None:
            diagnostic_sink(phase, payload)

    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET,
        training_factor_provider=_provider(state, truth, source_pose, selected_ids),
        source_history_provider=source_history_provider,
        diagnostic_sink=capture,
    )
    return state, report, solver_capture, provider_rows, expected_pixels


def test_prepared_history_table_matches_installed_rows_order_indices_and_owned_pixels(monkeypatch):
    state, report, capture, provider_rows, expected_pixels = _active_history_case(monkeypatch)
    prepared = capture["prepared"]
    table = prepared["source_history_observations"]
    history = report["owned_stereo_image_bundle"]["source_history_bundle"]

    assert table["schema"] == "source_history_image_rows_v1"
    assert table["status"] == "active"
    assert table["source_frame"] == 3
    assert table["row_count"] == 3
    assert table["component_count"] == 6
    assert table["measurement_role"] == "tracking_fit_consumed"
    assert table["independent_unused_claim"] is False
    assert history["installed_rows"] == table["row_count"]
    assert [row["landmark_id"] for row in table["rows"]] == [2, 0, 1]
    assert [row["selected_point_index"] for row in table["rows"]] == [
        next(int(item["rank"]) for item in prepared["selection"]["selected_landmarks"]
             if int(item["landmark_id"]) == ident)
        for ident in (2, 0, 1)
    ]

    # The report owns float32 measurement copies; later provider mutations must
    # not rewrite the serialized measurements or the objective used by BA.
    expected_residual = capture["residual"][-6:].copy()
    for row in provider_rows:
        row["pixel"][:] = np.nan
    for row in table["rows"]:
        np.testing.assert_array_equal(
            np.asarray(row["pixel_float32"], dtype=np.float32),
            expected_pixels[int(row["landmark_id"])],
        )
    np.testing.assert_array_equal(capture["residual"][-6:], expected_residual)


def test_empty_source_history_has_explicit_skipped_empty_table(monkeypatch):
    state, truth, source_pose, selected_ids = _free_source_case(source_bias=0.12)
    captured = {}

    def source_history_provider(payload):
        if payload["source_history_phase"] == "validate":
            return [], {"status": "validated"}
        return [], {"status": "skipped", "reason": "no_eligible_source_history_rows"}

    def solver(fun, initial, **kwargs):
        residual = np.asarray(fun(initial), float)
        return type("Result", (), {
            "x": np.asarray(initial).copy(), "nfev": 1, "success": True,
            "status": 1, "message": "diagnostic snapshot", "cost": 0.0,
            "optimality": 0.0,
        })()

    monkeypatch.setattr(bundle, "least_squares", solver)

    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET,
        training_factor_provider=_provider(state, truth, source_pose, selected_ids),
        source_history_provider=source_history_provider,
        diagnostic_sink=lambda phase, payload: captured.__setitem__(phase, copy.deepcopy(payload)),
    )
    table = captured["prepared"]["source_history_observations"]
    assert table["schema"] == "source_history_image_rows_v1"
    assert table["status"] == "skipped"
    assert table["reason"] == "no_eligible_source_history_rows"
    assert table["row_count"] == table["component_count"] == 0
    assert table["rows"] == []
    assert report["owned_stereo_image_bundle"]["source_history_bundle"]["installed_rows"] == 0


def test_ba76_cohort_snapshot_keeps_unavailable_ids_and_owns_map_state(tmp_path):
    from bundle_diagnostics import BundleDiagnosticsWriter

    slam, _source_pose, _pixels = _shared_history_fixture()
    original_lock = slam.map.lock

    class CountingLock:
        def __init__(self):
            self.entries = 0

        def __enter__(self):
            self.entries += 1
            return self

        def __exit__(self, *_args):
            return False

    try:
        slam.bundle_diagnostic_writer = BundleDiagnosticsWriter(
            tmp_path, frames=(69, 76)
        )
        live_calibration = slam._live_owned_bundle_calibration()
        slam._source_history_diagnostic_cohort = {
            "cohort_BA_frame": 69,
            "source_history_observations": {
                "status": "active",
                "source_snapshot_valid": True,
                "source_frame": 3,
                "source_calibration_identity": live_calibration,
                "rows": [{"landmark_id": 0}, {"landmark_id": 99999}, {"landmark_id": 2}],
            },
        }
        landmark = slam.map.landmarks[0]
        observation_key = min(landmark.observations)
        landmark.observations[observation_key].right_u = float("nan")
        original_position = landmark.position.copy()
        original_pixel = landmark.observations[observation_key].pixel.copy()

        counting_lock = CountingLock()
        slam.map.lock = counting_lock
        snapshot = slam._source_history_cohort_diagnostic_state(76)

        assert counting_lock.entries == 1
        assert snapshot["schema"] == "source_history_cohort_state_v1"
        assert snapshot["cohort_BA_frame"] == 69
        assert snapshot["current_BA_frame"] == 76
        assert snapshot["source_frame"] == 3
        assert snapshot["cross_phase_comparison_eligible"] is True
        assert [row["landmark_id"] for row in snapshot["rows"]] == [0, 99999, 2]
        assert snapshot["rows"][0]["available"] is True
        assert snapshot["rows"][1] == {
            "landmark_id": 99999,
            "available": False,
            "reason": "culled_or_missing",
            "current_world_position": None,
            "current_anchor_id": None,
            "actual_observations": [],
        }
        saved_observation = next(
            row for row in snapshot["rows"][0]["actual_observations"]
            if row["keyframe_id"] == observation_key
        )
        assert saved_observation["right_u"] is None
        assert saved_observation["right_u_valid"] is False
        assert saved_observation["dimensions"] == 2

        # Returned JSON data is detached from map arrays and later corrections.
        landmark.position[:] += np.array([0.3, -0.2, 0.5])
        landmark.observations[observation_key].pixel[:] += np.array([7.0, -4.0])
        np.testing.assert_array_equal(
            snapshot["rows"][0]["current_world_position"], original_position
        )
        np.testing.assert_array_equal(
            saved_observation["pixel"], np.asarray(original_pixel, dtype=np.float32)
        )
    finally:
        slam.map.lock = original_lock
        slam.close()


def test_disabled_source_history_capture_and_cohort_snapshot_do_no_map_work():
    slam, _source_pose, _pixels = _shared_history_fixture()
    original_lock = slam.map.lock

    class CountingLock:
        def __init__(self):
            self.entries = 0

        def __enter__(self):
            self.entries += 1
            return self

        def __exit__(self, *_args):
            return False

    try:
        slam.config = replace(slam.config, stereo_source_history_bundle=False)
        slam.bundle_diagnostic_writer = None
        slam._source_history_endpoint = lambda _frame: (_ for _ in ()).throw(
            AssertionError("disabled capture inspected map endpoints")
        )
        counting_lock = CountingLock()
        slam.map.lock = counting_lock
        assert slam._capture_source_history_snapshot(target_frame=4) is None
        assert slam._source_history_cohort_diagnostic_state(76) is None
        assert counting_lock.entries == 0
    finally:
        slam.map.lock = original_lock
        slam.close()


def test_stale_cohort_calibration_is_preserved_but_cross_phase_comparison_abstains(tmp_path):
    from bundle_diagnostics import BundleDiagnosticsWriter

    slam, _source_pose, _pixels = _shared_history_fixture()
    try:
        slam.bundle_diagnostic_writer = BundleDiagnosticsWriter(
            tmp_path, frames=(69, 76)
        )
        slam._source_history_diagnostic_cohort = {
            "cohort_BA_frame": 69,
            "source_history_observations": {
                "status": "active",
                "source_snapshot_valid": True,
                "source_frame": 3,
                "source_calibration_identity": slam._live_owned_bundle_calibration(),
                "rows": [{"landmark_id": 0}, {"landmark_id": 99999}],
            },
        }
        # A disparity-offset calibration change after BA69 invalidates a
        # cross-phase cost comparison while retaining the cohort IDs.
        slam.stereo = SimpleNamespace(
            stereo=slam.stereo.stereo,
            Q=np.asarray(slam.stereo.Q, float).copy(),
            baseline=slam.stereo.baseline,
            disparity_offset=slam.stereo.disparity_offset + 1.0,
        )
        snapshot = slam._source_history_cohort_diagnostic_state(76)
        assert snapshot["cross_phase_comparison_eligible"] is False
        assert snapshot["cross_phase_ineligible_reason"] == "calibration_identity_changed"
        assert [row["landmark_id"] for row in snapshot["rows"]] == [0, 99999]
    finally:
        slam.close()


def test_bundle_writer_failure_does_not_change_source_history_solve(tmp_path):
    from bundle_diagnostics import BundleDiagnosticsWriter
    from shared_slam import MappingConfig, SharedSlam, StereoCamera

    baseline, truth, source_pose, selected_ids = _free_source_case(source_bias=0.22)
    failing, _truth, _source_pose, _ids = _free_source_case(source_bias=0.22)
    baseline_report = local_bundle_adjustment(
        baseline, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET,
        training_factor_provider=_provider(baseline, truth, source_pose, selected_ids),
        source_history_provider=lambda payload: (
            ([], {"status": "validated"}) if payload["source_history_phase"] == "validate"
            else (_history_rows(baseline, source_pose, (2, 0, 1)), {"status": "eligible"})
        ),
    )
    expected = _snapshot(baseline)

    slam = SharedSlam(
        MATRIX, StereoCamera(None, np.eye(4), BASELINE),
        config=MappingConfig(stereo_pose_arbitration=True,
                             stereo_owned_image_bundle=True,
                             stereo_source_history_bundle=True),
    )
    original_map = slam.map
    slam.map = failing
    writer = BundleDiagnosticsWriter(tmp_path, frames=(2,))
    writer.emit = lambda **_kwargs: (_ for _ in ()).throw(OSError("diagnostic disk failure"))
    slam.bundle_diagnostic_writer = writer

    try:
        result = local_bundle_adjustment(
            failing, MATRIX, BASELINE, window=3, max_landmarks=40,
            disparity_offset=OFFSET,
            training_factor_provider=_provider(failing, truth, source_pose, selected_ids),
            source_history_provider=lambda payload: (
                ([], {"status": "validated"}) if payload["source_history_phase"] == "validate"
                else (_history_rows(failing, source_pose, (2, 0, 1)), {"status": "eligible"})
            ),
            diagnostic_sink=lambda phase, payload: slam._emit_bundle_diagnostic(
                2, phase, payload, (640, 480)
            ),
        )
        assert result["applied"] == baseline_report["applied"]
        _assert_snapshot_equal(failing, expected)
        assert [row["phase"] for row in slam.bundle_diagnostic_errors] == ["prepared", "solved"]
        assert all(row["error_type"] == "OSError" for row in slam.bundle_diagnostic_errors)
    finally:
        slam.map = original_map
        slam.close()
