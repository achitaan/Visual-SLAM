"""Synthetic coverage for left-only source-history connectivity in local BA."""

import copy
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import local_bundle as bundle
from local_bundle import local_bundle_adjustment
from mapping_geometry import project
from test_bundle_diagnostics import MATRIX, BASELINE
from test_free_source_stereo_bundle import (
    OFFSET,
    _free_source_case,
    _params_to_pose,
    _pose_to_params,
    _provider,
    _snapshot,
    _assert_snapshot_equal,
)
from test_target_relative_stereo_bundle import _selected_shared_factor


def _pose(x, rotvec=(0.0, 0.0, 0.0)):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, float)).as_matrix()
    pose[0, 3] = float(x)
    return pose


def _huber2(values):
    absolute = np.abs(np.asarray(values, float))
    return float(np.sum(np.where(absolute <= 2.0, 0.5 * absolute**2,
                                2.0 * (absolute - 1.0))))


def _source_history_provider(state, true_source, *, landmark_ids=range(3),
                              rows_override=None, calls=None):
    """Build exact float32 left pixels from a known synthetic source camera."""
    calls = [] if calls is None else calls
    rows_override = rows_override

    def provide(payload):
        phase = payload["source_history_phase"]
        calls.append((phase, copy.deepcopy(payload)))
        if phase == "validate":
            return [], {"status": "validated"}
        frame = int(payload["source_history_source_frame"])
        selected = set(map(int, payload["source_history_selected_landmark_ids"]))
        rows = []
        for ident in landmark_ids:
            ident = int(ident)
            if ident not in selected:
                continue
            point = state.landmarks[ident].position
            pixel = project(point[None], true_source, MATRIX)[0][0].astype(np.float32)
            rows.append({"frame_id": frame, "landmark_id": ident, "pixel": pixel})
        if rows_override is not None:
            rows = rows_override(rows, payload)
        return rows, {"status": "eligible", "captured_rows": len(rows)}

    return provide


def _owned_provider_capture(state, truth, true_source, landmark_ids, prepared):
    provider = _provider(state, truth, true_source, landmark_ids)

    def provide(payload):
        if payload.get("training_factor_phase") == "prepare":
            prepared["owned"] = copy.deepcopy(payload)
        return provider(payload)

    return provide


def _diagnostic_capture(prepared):
    def capture(phase, payload):
        if phase == "prepared":
            prepared["solver"] = copy.deepcopy(payload)
    return capture


def _shared_history_fixture():
    from shared_slam import MappingConfig, SharedSlam, StereoCamera
    from stereo_pose_arbitration import SupportedStereoFrame

    state, _truth, true_source, _pair_ids = _free_source_case(source_bias=0.18)
    config = MappingConfig(
        loop_mode="off", bundle_enabled=False, stereo_pose_arbitration=True,
        stereo_owned_image_bundle=True, stereo_source_history_bundle=True,
    )
    slam = SharedSlam(MATRIX, StereoCamera(None, np.eye(4), BASELINE), config=config)
    slam.map = state
    track_ids = (0, 1, 2)
    pixels = np.asarray([
        project(state.landmarks[ident].position[None], true_source, MATRIX)[0][0]
        for ident in track_ids
    ], dtype=np.float32)
    points = np.ones((len(track_ids), 3), dtype=np.float32)
    descriptors = np.zeros((len(track_ids), 128), dtype=np.float32)
    right = np.full(len(track_ids), 300., dtype=np.float32)
    slam.previous_supported_stereo = SupportedStereoFrame(
        pixels, descriptors, points, right, np.asarray(track_ids, np.int64),
        3, (640, 480), slam.stereo_calibration_identity,
    )
    slam.previous_gray = np.zeros((480, 640), np.uint8)
    slam.previous_tracks = [(ident, pixel.copy()) for ident, pixel in zip(track_ids, pixels)]
    # Single-view points are present in the tracker cache but cannot become
    # multiview source-history rows.
    slam.previous_tracks.append((24, np.array([320., 240.], np.float32)))
    slam._previous_tracks_frame = 3
    return slam, true_source, pixels


def _shared_history_provider(slam, capture, *, held_ids=(), held_source_indices=(1,),
                             evidence=True):
    source = slam.previous_supported_stereo
    measurement = np.eye(4)
    arbitration = {
        "verified": {"measurement": measurement.copy()},
        "previous": source,
        "excluded_landmarks": set(held_ids),
        "evidence": (SimpleNamespace(source_ids=np.asarray(held_source_indices, np.int64))
                     if evidence else None),
    }
    return slam._owned_source_history_provider(
        capture,
        arbitration,
        {"choice": "independent"},
        {"pose_source": "reserved_stereo_arbitration"},
        (source.frame, measurement.copy()),
        False,
    )


def _source_history_payload(*, phase="prepare", frame=3, selected=(0, 1, 2),
                            original=(), owned=()):
    return {
        "source_history_phase": phase,
        "source_history_source_frame": frame,
        "source_history_selected_landmark_ids": list(selected),
        "source_history_original_frame_landmark_ids": list(original),
        "source_history_owned_frame_landmark_ids": list(owned),
    }


def test_history_rows_use_source_camera_and_existing_T_C_P_blocks_with_numeric_sparsity(monkeypatch):
    state, truth, true_source, pair_ids = _free_source_case(source_bias=0.22)
    # Noncommuting reference rotation exercises the full projection chain.
    target = _pose(1.08, (0.13, -0.075, 0.045))
    state.keyframes[2].pose = target.copy()
    state.poses[2] = target.copy()
    history_ids = (0, 1, 2)
    seen = {}
    history_calls = []
    real_solver = bundle.least_squares

    def inspect_solver(fun, initial, **kwargs):
        prepared = diagnostics["solver"]
        layout = prepared["parameter_layout"]
        chart = layout["target_relative_chart"]
        initial = np.asarray(initial, float).copy()
        residual0 = fun(initial)
        history_count = len(history_ids)
        history_slice = slice(len(residual0) - 2 * history_count, len(residual0))
        assert residual0[history_slice].shape == (2 * history_count,)
        sparse = kwargs["jac_sparsity"].toarray().astype(bool)
        assert sparse.shape == (len(residual0), len(initial))
        target_offset = layout["pose_offsets"]["2"]
        source_offset = chart["target_relative_camera_offsets"]["3"]
        selected = {int(row["landmark_id"]): int(row["rank"])
                    for row in prepared["selection"]["selected_landmarks"]}
        rows_by_id = {int(row["landmark_id"]): np.asarray(row["pixel"], np.float32)
                      for row in seen["history_rows"]}
        for index, ident in enumerate(history_ids):
            point_offset = layout["point_offset"] + 3 * selected[ident]
            point = initial[point_offset:point_offset + 3]
            target_pose = _params_to_pose(initial[target_offset:target_offset + 6])
            source_relative = _params_to_pose(initial[source_offset:source_offset + 6])
            predicted = project(point[None], target_pose @ source_relative, MATRIX)[0][0]
            expected = predicted - rows_by_id[ident]
            row_slice = slice(history_slice.start + 2 * index,
                              history_slice.start + 2 * index + 2)
            np.testing.assert_allclose(residual0[row_slice], expected, atol=2e-7, rtol=0.0)
            assert sparse[row_slice, target_offset:target_offset + 6].all()
            assert sparse[row_slice, source_offset:source_offset + 6].all()
            assert sparse[row_slice, point_offset:point_offset + 3].all()

            for offset, width in ((target_offset, 6), (source_offset, 6),
                                  (point_offset, 3)):
                derivatives = []
                for column in range(offset, offset + width):
                    plus, minus = initial.copy(), initial.copy()
                    plus[column] += 1e-6
                    minus[column] -= 1e-6
                    derivative = (fun(plus)[row_slice] - fun(minus)[row_slice]) / 2e-6
                    derivatives.append(derivative)
                    assert not np.any((np.abs(derivative) > 1e-5)
                                      & ~sparse[row_slice, column])
                assert np.linalg.norm(np.column_stack(derivatives)) > 1e-5

        # A common world transform of T and P leaves the historical image
        # measurements unchanged while the already-charted C stays fixed.
        common = _pose(-0.31, (0.11, -0.08, 0.06))
        transformed = initial.copy()
        transformed[target_offset:target_offset + 6] = _pose_to_params(
            common @ _params_to_pose(initial[target_offset:target_offset + 6])
        )
        for ident in history_ids:
            point_offset = layout["point_offset"] + 3 * selected[ident]
            transformed[point_offset:point_offset + 3] = (
                common[:3, :3] @ initial[point_offset:point_offset + 3]
                + common[:3, 3]
            )
        np.testing.assert_allclose(fun(transformed)[history_slice],
                                   residual0[history_slice], rtol=1e-9, atol=2e-7)

        seen["residual_initial"] = residual0.copy()
        seen["history_slice"] = history_slice
        result = real_solver(fun, initial, **kwargs)
        seen["result_x"] = result.x.copy()
        seen["residual_final"] = fun(result.x).copy()
        return result

    def history(rows, payload):
        if payload["source_history_phase"] == "prepare":
            seen["history_rows"] = copy.deepcopy(rows)
        return rows

    # Wrap only row selection so the same exact pixels reach the residual.
    source_provider = _source_history_provider(
        state, true_source, landmark_ids=history_ids,
        rows_override=lambda rows, payload: history(rows, payload),
        calls=history_calls,
    )
    diagnostics = {}
    monkeypatch.setattr(bundle, "least_squares", inspect_solver)
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET,
        training_factor_provider=_owned_provider_capture(
            state, truth, true_source, pair_ids, diagnostics
        ),
        source_history_provider=source_provider,
        diagnostic_sink=_diagnostic_capture(diagnostics),
    )
    assert report["owned_stereo_image_bundle"]["source_history_bundle"]["installed_rows"] == 3
    assert report["owned_stereo_image_bundle"]["source_history_bundle"]["components_added"] == 6
    assert [phase for phase, _ in history_calls] == ["prepare", "validate"]
    history = report["owned_stereo_image_bundle"]["source_history_bundle"]
    assert history["cost_before"] == pytest.approx(
        _huber2(seen["residual_initial"][seen["history_slice"]]), abs=1e-10
    )
    assert history["cost_after"] == pytest.approx(
        _huber2(seen["residual_final"][seen["history_slice"]]), abs=1e-10
    )
    assert report["augmented_initial_cost"] == pytest.approx(
        _huber2(seen["residual_initial"]), abs=1e-9
    )
    assert report["augmented_final_cost"] == pytest.approx(
        _huber2(seen["residual_final"]), abs=1e-9
    )


def test_source_history_recovery_keeps_original_truth_and_dual_objective_gates():
    state, truth, true_source, pair_ids = _free_source_case(source_bias=0.30)
    initial_source = state.poses[3].copy()
    history_ids = tuple(range(24))
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET,
        training_factor_provider=_provider(state, truth, true_source, pair_ids),
        source_history_provider=_source_history_provider(
            state, true_source, landmark_ids=history_ids
        ),
    )
    assert report["applied"] is True, report
    summary = report["owned_stereo_image_bundle"]
    history = summary["source_history_bundle"]
    assert history["status"] == "accepted"
    assert history["installed_rows"] == 24
    assert history["components_added"] == 48
    assert history["tracking_fit_evidence_reused"] is True
    assert history["covariance_claim"] is False
    assert history["heldout_validation_claim"] is False
    initial_error = np.linalg.norm(initial_source[:3, 3] - true_source[:3, 3])
    final_error = np.linalg.norm(state.poses[3][:3, 3] - true_source[:3, 3])
    assert final_error < 0.5 * initial_error, (
        f"source translation error {initial_error:.6f}m -> {final_error:.6f}m"
    )
    true_relative = np.linalg.inv(truth[2]) @ true_source
    final_relative = np.linalg.inv(state.keyframes[2].pose) @ state.poses[3]
    relative_error = np.linalg.inv(true_relative) @ final_relative
    assert np.linalg.norm(relative_error[:3, 3]) < 1e-3
    assert np.degrees(Rotation.from_matrix(relative_error[:3, :3]).magnitude()) < 0.01
    assert summary["affected_objective_reduced"] is True
    assert summary["augmented_objective_reduced"] is True
    assert len(state.landmarks) == 48


def test_zero_eligible_history_rows_preserve_owned_solver_inputs_exactly(monkeypatch):
    captures = {}

    def run(name, enable_empty_history):
        state, truth, true_source, pair_ids = _free_source_case(source_bias=0.12)
        owned = _provider(state, truth, true_source, pair_ids)
        captured = {}

        def solver(fun, initial, **kwargs):
            captured["initial"] = np.asarray(initial).copy()
            captured["residual"] = np.asarray(fun(initial)).copy()
            captured["options"] = dict(kwargs)
            captured["sparsity"] = kwargs["jac_sparsity"].toarray().copy()
            return SimpleNamespace(x=np.asarray(initial).copy(), nfev=1, success=True,
                                   status=1, message="zero-row equivalence",
                                   cost=0.0, optimality=0.0)

        monkeypatch.setattr(bundle, "least_squares", solver)
        options = {"training_factor_provider": owned}
        if enable_empty_history:
            options["source_history_provider"] = lambda payload: (
                ([], {"status": "empty"}) if payload["source_history_phase"] == "prepare"
                else ([], {"status": "validated"})
            )
        report = local_bundle_adjustment(
            state, MATRIX, BASELINE, window=3, max_landmarks=40,
            disparity_offset=OFFSET, **options,
        )
        captures[name] = captured
        return state, report

    legacy_state, legacy_report = run("legacy", False)
    empty_state, empty_report = run("empty", True)
    left, right = captures["legacy"], captures["empty"]
    np.testing.assert_array_equal(left["initial"], right["initial"])
    np.testing.assert_array_equal(left["residual"], right["residual"])
    np.testing.assert_array_equal(left["sparsity"], right["sparsity"])
    assert left["options"].keys() == right["options"].keys()
    for name in left["options"]:
        if name == "jac_sparsity":
            continue
        left_value, right_value = left["options"][name], right["options"][name]
        if isinstance(left_value, np.ndarray):
            np.testing.assert_array_equal(left_value, right_value)
        else:
            assert left_value == right_value
    assert legacy_report["applied"] == empty_report["applied"]
    _assert_snapshot_equal(legacy_state, _snapshot(empty_state))
    assert empty_report["owned_stereo_image_bundle"]["source_history_bundle"][
        "installed_rows"
    ] == 0


def test_history_validation_rejection_rolls_back_every_geometry_write():
    state, truth, true_source, pair_ids = _free_source_case(source_bias=0.25)
    before = _snapshot(state)
    base = _source_history_provider(state, true_source, landmark_ids=(0, 1, 2))

    def reject_on_validate(payload):
        if payload["source_history_phase"] == "validate":
            return [], {"status": "rejected", "reason": "stale_source_cache"}
        return base(payload)

    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET,
        training_factor_provider=_provider(state, truth, true_source, pair_ids),
        source_history_provider=reject_on_validate,
    )
    assert report["applied"] is False
    _assert_snapshot_equal(state, before)
    history = report["owned_stereo_image_bundle"]["source_history_bundle"]
    assert history["status"] == "rejected"
    assert history["reason"] == "stale_source_cache"


@pytest.mark.parametrize(
    ("report_status", "row_change", "expected_reason"),
    [
        ("rejected", "none", "source_cache_unbound"),
        ("eligible", "nonfinite", "no_eligible_history_rows"),
    ],
)
def test_untrusted_or_malformed_history_capture_fails_closed(
        report_status, row_change, expected_reason):
    state, truth, true_source, pair_ids = _free_source_case(source_bias=0.15)
    base = _source_history_provider(state, true_source, landmark_ids=(0,))

    def provider(payload):
        rows, _ = base(payload)
        if payload["source_history_phase"] == "prepare":
            if row_change == "nonfinite":
                rows[0]["pixel"] = np.array([np.nan, 200.0], np.float32)
            return rows, {"status": report_status, "reason": expected_reason}
        return [], {"status": "validated"}

    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET,
        training_factor_provider=_provider(state, truth, true_source, pair_ids),
        source_history_provider=provider,
    )
    assert report["applied"] is True
    history = report["owned_stereo_image_bundle"]["source_history_bundle"]
    assert history["installed_rows"] == 0
    assert history["status"] == "skipped"
    assert history["reason"] == expected_reason
    if row_change == "nonfinite":
        assert history["exclusions"]["invalid_or_mismatched_row"] == 1


def test_exact_duplicate_history_rows_collapse_and_conflicts_fail_closed():
    def run_case(name, row_transform):
        state, truth, true_source, pair_ids = _free_source_case(source_bias=0.12)
        base = _source_history_provider(state, true_source, landmark_ids=(0, 1))

        def provider(payload):
            rows, report = base(payload)
            if payload["source_history_phase"] == "prepare":
                rows = row_transform(rows)
            return rows, report

        result = local_bundle_adjustment(
            state, MATRIX, BASELINE, window=3, max_landmarks=40,
            disparity_offset=OFFSET,
            training_factor_provider=_provider(state, truth, true_source, pair_ids),
            source_history_provider=provider,
        )
        return result["owned_stereo_image_bundle"]["source_history_bundle"]

    duplicate = run_case("duplicate", lambda rows: [rows[0], copy.deepcopy(rows[0])])
    assert duplicate["installed_rows"] == 1
    assert duplicate["duplicate_rows_collapsed"] == 1
    assert duplicate["components_added"] == 2

    conflicting_id = run_case(
        "id-conflict",
        lambda rows: [rows[0], {**rows[0], "pixel": rows[0]["pixel"] + [5.0, 0.0]}],
    )
    assert conflicting_id["installed_rows"] == 0
    assert conflicting_id["exclusions"]["landmark_multiple_source_pixels"] == 1

    def same_pixel_different_ids(rows):
        return [rows[0], {**rows[1], "pixel": rows[0]["pixel"].copy()}]

    conflicting_pixel = run_case("pixel-conflict", same_pixel_different_ids)
    assert conflicting_pixel["installed_rows"] == 0
    assert conflicting_pixel["exclusions"]["physical_pixel_multiple_landmarks"] == 1


def test_existing_source_owner_excludes_same_landmark_and_physical_pixel_aliases():
    def run_case(ident, pixel_from_ident, change_pixel=False):
        state, truth, true_source, pair_ids = _free_source_case(source_bias=0.12)
        base = _provider(state, truth, true_source, pair_ids)

        def owned_with_shared_original(payload):
            factors, report = base(payload)
            if payload.get("training_factor_phase") == "prepare":
                shared = _selected_shared_factor(state, true_source, payload)
                return (*factors, shared), {**report, "selected_factors": len(factors) + 1}
            return factors, report

        base_history = _source_history_provider(state, true_source, landmark_ids=(ident,))

        def colliding_history(payload):
            rows, report = base_history(payload)
            if payload["source_history_phase"] == "prepare":
                owned_pixel = project(
                    state.landmarks[pixel_from_ident].position[None], true_source, MATRIX
                )[0][0].astype(np.float32)
                if change_pixel:
                    owned_pixel = owned_pixel + np.array([7.0, 0.0], np.float32)
                rows[0]["pixel"] = owned_pixel
            return rows, report

        result = local_bundle_adjustment(
            state, MATRIX, BASELINE, window=3, max_landmarks=40,
            disparity_offset=OFFSET,
            training_factor_provider=owned_with_shared_original,
            source_history_provider=colliding_history,
        )
        return result["owned_stereo_image_bundle"]["source_history_bundle"]

    # The same (frame, LM) is already an owned source factor. A different
    # measured pixel cannot be installed as a second view of that pair.
    same_pair = run_case(ident=0, pixel_from_ident=0, change_pixel=True)
    assert same_pair["installed_rows"] == 0
    assert same_pair["exclusions"]["same_frame_landmark_already_owned"] == 1

    # A distinct LM may not claim the exact physical source pixel of another
    # selected LM's already-owned factor.
    pixel_alias = run_case(ident=1, pixel_from_ident=0)
    assert pixel_alias["installed_rows"] == 0
    assert pixel_alias["exclusions"]["physical_pixel_conflicts_with_existing_owner"] == 1


def test_source_history_config_requires_owned_metric_stereo():
    from shared_slam import MappingConfig, SharedSlam, StereoCamera

    with pytest.raises(ValueError, match="requires stereo_owned_image_bundle"):
        MappingConfig(stereo_source_history_bundle=True)
    enabled = MappingConfig(
        stereo_owned_image_bundle=True,
        stereo_pose_arbitration=True,
        stereo_source_history_bundle=True,
    )
    with pytest.raises(ValueError, match="calibrated cameras"):
        SharedSlam(MATRIX, config=enabled)
    slam = SharedSlam(
        MATRIX,
        stereo=StereoCamera(None, np.eye(4), BASELINE),
        config=enabled,
    )
    assert slam.config.stereo_source_history_bundle is True


def test_evaluator_rejects_source_history_flag_without_owned_bundle_before_dataset_access(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_shared_slam.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(tmp_path),
         "--stereo-source-history-bundle"],
        cwd=script.parents[1], text=True, capture_output=True, timeout=15,
        check=False,
    )
    assert result.returncode == 2
    assert "--stereo-source-history-bundle requires --stereo-owned-image-bundle" in result.stderr
    assert "FileNotFoundError" not in result.stderr


def test_shared_cache_is_detached_uses_exact_source_pixels_and_applies_held_filters():
    slam, true_source, source_pixels = _shared_history_fixture()
    try:
        capture = slam._capture_source_history_snapshot(target_frame=4)
        assert capture["status"] == "eligible"
        assert capture["frame"] == 3
        captured = {row["landmark_id"]: row["pixel"].copy()
                    for row in capture["tracks"]}
        assert set(captured) == {0, 1, 2, 24}
        for index, ident in enumerate((0, 1, 2)):
            assert captured[ident].dtype == np.float32
            np.testing.assert_array_equal(captured[ident], source_pixels[index])

        # A geometry correction after source acceptance may change P, but it
        # must not synthesize a new measurement from the corrected geometry.
        slam.map.landmarks[2].position += np.array([0.17, -0.04, 0.21])
        provider = _shared_history_provider(slam, capture, held_ids=(0,))
        rows, report = provider(_source_history_payload())
        assert report["status"] == "eligible"
        assert [row["landmark_id"] for row in rows] == [2]
        np.testing.assert_array_equal(rows[0]["pixel"], captured[2])
        assert not np.allclose(
            rows[0]["pixel"],
            project(slam.map.landmarks[2].position[None], true_source, MATRIX)[0][0],
        )
        assert report["exclusions"]["held_landmarks"] == 1
        assert report["exclusions"]["held_source_pixels"] == 1

        # The capture owns its rows and is unchanged by target-frame cache writes.
        slam.previous_tracks = [(999, np.array([10., 10.], np.float32))]
        np.testing.assert_array_equal(capture["tracks"][2]["pixel"], captured[2])
    finally:
        slam.close()


def test_one_observation_track_is_captured_before_target_makes_it_multiview():
    from slam_state import Observation

    slam, _true_source, _source_pixels = _shared_history_fixture()
    try:
        old_pixel = slam.previous_tracks[-1][1].copy()
        capture = slam._capture_source_history_snapshot(target_frame=4)
        row = next(item for item in capture["tracks"] if item["landmark_id"] == 24)
        np.testing.assert_array_equal(row["pixel"], old_pixel)

        # This is the target-keyframe observation; it appears after the source
        # image capture. The exact old left pixel is now eligible for the
        # selected multiview variable, without reprojecting it from the update.
        slam.map.landmarks[24].observations[3] = Observation(
            np.array([319.5, 239.5], np.float32), 300.0
        )
        provider = _shared_history_provider(slam, capture)
        rows, report = provider(_source_history_payload(selected=(24,)))
        assert report["status"] == "eligible"
        assert [item["landmark_id"] for item in rows] == [24]
        np.testing.assert_array_equal(rows[0]["pixel"], old_pixel)
    finally:
        slam.close()


@pytest.mark.parametrize("invalid", ["frame", "lost", "calibration"])
def test_shared_cache_abstains_on_wrong_source_binding_or_lost_endpoint(invalid):
    slam, _true_source, _source_pixels = _shared_history_fixture()
    try:
        if invalid == "frame":
            slam._previous_tracks_frame = 2
        elif invalid == "lost":
            slam.map.statuses[3] = "lost"
        else:
            # Supported extraction still claims the old calibration after a
            # live intrinsics change.
            slam.K[0, 0] += 1.0
        capture = slam._capture_source_history_snapshot(target_frame=4)
        assert capture["status"] != "eligible"
    finally:
        slam.close()


def test_shared_provider_requires_holdout_evidence_and_rolls_back_late_source_change():
    slam, _true_source, _source_pixels = _shared_history_fixture()
    try:
        capture = slam._capture_source_history_snapshot(target_frame=4)
        no_evidence = _shared_history_provider(slam, capture, evidence=False)
        rows, report = no_evidence(_source_history_payload())
        assert rows == ()
        assert report["status"] == "rejected"

        provider = _shared_history_provider(slam, capture, held_ids=(0,))
        rows, report = provider(_source_history_payload())
        assert report["status"] == "eligible"
        assert [row["landmark_id"] for row in rows] == [2]
        # Prepare binds the then-current world point. A later correction cannot
        # commit an image row against a point the optimizer did not see.
        slam.map.landmarks[2].position[0] += 0.01
        late_rows, late_report = provider(_source_history_payload(phase="validate"))
        assert late_rows == ()
        assert late_report["status"] == "rejected"
        assert late_report["reason"] == "source_history_landmark_changed_before_apply"
    finally:
        slam.close()


def test_source_track_ambiguity_and_complex_pixels_do_not_become_history_rows():
    slam, _true_source, source_pixels = _shared_history_fixture()
    try:
        # A/B/A for one stable ID must remain ambiguous instead of resurrecting
        # the first measurement after the conflicting middle row is removed.
        slam.previous_tracks = [
            (0, source_pixels[0].copy()),
            (0, source_pixels[0] + np.array([2., 0.], np.float32)),
            (0, source_pixels[0].copy()),
            (1, np.array([complex(source_pixels[1, 0], 1.0), source_pixels[1, 1]])),
        ]
        capture = slam._capture_source_history_snapshot(target_frame=4)
        identifiers = {row["landmark_id"] for row in capture.get("tracks", [])}
        assert 0 not in identifiers
        assert 1 not in identifiers
    finally:
        slam.close()
