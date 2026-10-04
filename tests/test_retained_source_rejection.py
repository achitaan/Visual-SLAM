"""Fail-closed tests for retained-source context and nested registry records."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

import local_bundle as bundle
from local_bundle import local_bundle_adjustment
from mapping_geometry import project
from slam_state import MapState
from test_bundle_diagnostics import MATRIX, BASELINE
from test_free_source_stereo_bundle import OFFSET
from test_retained_source_observations import (
    CALIBRATION_ID,
    _assert_map_snapshot,
    _clone_state,
    _history_provider,
    _long_frame_scene,
    _map_snapshot,
    _owned_factor_provider,
    _retained_record,
)
from test_source_history_bundle import _shared_history_fixture


def _replace_fields(value, **changes):
    fields = vars(value).copy()
    fields.update(changes)
    return SimpleNamespace(**fields)


def _shared_exclusion_context(slam):
    previous = slam.previous_supported_stereo
    current = SimpleNamespace(
        frame=4,
        image_size=previous.image_size,
        calibration_identity=previous.calibration_identity,
    )
    evidence = SimpleNamespace(
        source_frame=3,
        source_ids=np.asarray([0, 2], dtype=np.int64),
        landmark_ids=np.asarray([0, 2], dtype=np.int64),
        calibration_identity=previous.calibration_identity,
    )
    context = {
        "previous": previous,
        "current": current,
        "evidence": evidence,
        "excluded_landmarks": {0, 2, 24},
    }
    slam._arbitration_context = context
    return context, previous, current, evidence


def test_shared_retained_exclusion_snapshot_owns_exact_reserved_rows():
    slam, _true_source, _pixels = _shared_history_fixture()
    context, previous, _current, evidence = _shared_exclusion_context(slam)
    previous_pixels = previous.pixels.copy()

    result = slam._retained_source_exclusion_snapshot(4, (640, 480))

    assert result["valid"] is True
    assert result["reason"] is None
    assert result["source_frame"] == 3
    assert result["landmark_ids"] == [0, 2, 24]
    assert result["source_pixels"].dtype == np.float32
    np.testing.assert_array_equal(result["source_pixels"], previous_pixels[[0, 2]])

    # Neither output list/array aliases the arbitration state or the readonly
    # supported-frame observation arrays.
    result["landmark_ids"].append(999)
    result["source_pixels"][:] = -1
    assert context["excluded_landmarks"] == {0, 2, 24}
    np.testing.assert_array_equal(previous.pixels, previous_pixels)
    np.testing.assert_array_equal(evidence.source_ids, [0, 2])


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("bool_previous_frame", "reserved_frame_binding_invalid"),
        ("fractional_previous_frame", "reserved_frame_binding_invalid"),
        ("negative_previous_frame", "reserved_frame_binding_invalid"),
        ("mismatched_current_frame", "reserved_frame_binding_invalid"),
        ("mismatched_evidence_frame", "reserved_frame_binding_invalid"),
        ("calibration_identity", "reserved_calibration_invalid"),
        ("complex_baseline", "reserved_calibration_invalid"),
        ("wrong_image_size", "reserved_image_domain_invalid"),
        ("nonpositive_image_domain", "reserved_image_domain_invalid"),
        ("outside_pixel", "reserved_source_pixels_invalid"),
        ("nonfinite_pixel", "reserved_source_pixels_invalid"),
        ("complex_pixels", "reserved_source_ids_invalid"),
        ("fractional_source_ids", "reserved_source_ids_invalid"),
        ("bool_source_ids", "reserved_source_ids_invalid"),
        ("negative_source_ids", "reserved_source_ids_invalid"),
        ("out_of_range_source_ids", "reserved_source_ids_invalid"),
        ("duplicate_source_ids", "reserved_source_ids_invalid"),
        ("evidence_ids_bool", "reserved_landmark_binding_invalid"),
        ("evidence_ids_not_held", "reserved_landmark_binding_invalid"),
        ("held_ids_bool", "reserved_landmark_ids_invalid"),
        ("held_ids_fractional", "reserved_landmark_ids_invalid"),
        ("held_ids_negative", "reserved_landmark_ids_invalid"),
    ],
)
def test_shared_retained_exclusion_snapshot_rejects_bad_bindings(mutation, reason):
    slam, _true_source, _pixels = _shared_history_fixture()
    context, previous, current, evidence = _shared_exclusion_context(slam)

    if mutation == "bool_previous_frame":
        previous = _replace_fields(previous, frame=True)
        context["previous"] = previous
    elif mutation == "fractional_previous_frame":
        context["previous"] = _replace_fields(previous, frame=3.5)
    elif mutation == "negative_previous_frame":
        context["previous"] = _replace_fields(previous, frame=-1)
    elif mutation == "mismatched_current_frame":
        context["current"] = _replace_fields(current, frame=5)
    elif mutation == "mismatched_evidence_frame":
        context["evidence"] = _replace_fields(evidence, source_frame=2)
    elif mutation == "calibration_identity":
        context["previous"] = _replace_fields(
            previous, calibration_identity="stale-calibration"
        )
    elif mutation == "complex_baseline":
        object.__setattr__(slam.stereo, "baseline", complex(BASELINE, 0.1))
    elif mutation == "wrong_image_size":
        context["current"] = _replace_fields(current, image_size=(641, 480))
    elif mutation == "nonpositive_image_domain":
        current.image_size = (640, 0)
    elif mutation == "outside_pixel":
        raw = previous.pixels.copy()
        raw[0] = [640., 100.]
        context["previous"] = _replace_fields(previous, pixels=raw)
    elif mutation == "nonfinite_pixel":
        raw = previous.pixels.copy()
        raw[0, 0] = np.nan
        context["previous"] = _replace_fields(previous, pixels=raw)
    elif mutation == "complex_pixels":
        raw = previous.pixels.astype(np.complex64) + 1j
        context["previous"] = _replace_fields(previous, pixels=raw)
    elif mutation == "fractional_source_ids":
        context["evidence"] = _replace_fields(
            evidence, source_ids=np.asarray([0., 2.])
        )
    elif mutation == "bool_source_ids":
        context["evidence"] = _replace_fields(
            evidence, source_ids=np.asarray([True, False])
        )
    elif mutation == "negative_source_ids":
        context["evidence"] = _replace_fields(
            evidence, source_ids=np.asarray([-1, 2], dtype=np.int64)
        )
    elif mutation == "out_of_range_source_ids":
        context["evidence"] = _replace_fields(
            evidence, source_ids=np.asarray([0, 99], dtype=np.int64)
        )
    elif mutation == "duplicate_source_ids":
        context["evidence"] = _replace_fields(
            evidence, source_ids=np.asarray([0, 0], dtype=np.int64)
        )
    elif mutation == "evidence_ids_bool":
        context["evidence"] = _replace_fields(
            evidence, landmark_ids=np.asarray([True, False])
        )
    elif mutation == "evidence_ids_not_held":
        context["evidence"] = _replace_fields(
            evidence, landmark_ids=np.asarray([0, 7], dtype=np.int64)
        )
    elif mutation == "held_ids_bool":
        context["excluded_landmarks"] = [0, True]
    elif mutation == "held_ids_fractional":
        context["excluded_landmarks"] = {0, 1.5}
    elif mutation == "held_ids_negative":
        context["excluded_landmarks"] = {0, -1}

    result = slam._retained_source_exclusion_snapshot(4, (640, 480))
    assert result["valid"] is False
    assert result["reason"] == reason
    assert "source_pixels" not in result
    assert "landmark_ids" not in result


def test_shared_retained_exclusion_snapshot_distinguishes_missing_from_malformed_context():
    slam, _true_source, _pixels = _shared_history_fixture()
    slam._arbitration_context = None
    missing = slam._retained_source_exclusion_snapshot(4, (640, 480))
    assert missing == {"valid": False, "reason": "reserved_context_unavailable"}

    slam._arbitration_context = "not-a-context"
    malformed = slam._retained_source_exclusion_snapshot(4, (640, 480))
    assert malformed == {"valid": False, "reason": "reserved_context_invalid"}


def _retained_invalid_state():
    state, source_true, truth_points, anchor_id, _target_id = _long_frame_scene()
    pixel = project(truth_points[1][None], source_true, MATRIX)[0][0].astype(np.float32)
    record = _retained_record(state, 68, 1, pixel)
    corrected = {key: value.pose.copy() for key, value in state.keyframes.items()}
    assert state.apply_corrections(
        state.revision, corrected,
        retained_source_updates={68: record},
        retained_source_limits={"max_frames": 3, "max_rows_per_frame": 40},
    )
    # Malform a nested field after acceptance to exercise local_bundle's deep
    # parser, rather than MapState's atomic write validation.
    state.retained_source_observations[68]["rows"][0]["pixel_float32"] = np.array(
        [100.0, 200.0, 300.0], dtype=np.float64
    )
    return state, source_true, truth_points, anchor_id


def _retained_inner_binding_invalid_state():
    state, source_true, truth_points, anchor_id, _target_id = _long_frame_scene()
    pixel = project(truth_points[1][None], source_true, MATRIX)[0][0].astype(np.float32)
    record = _retained_record(state, 68, 1, pixel)
    corrected = {key: value.pose.copy() for key, value in state.keyframes.items()}
    assert state.apply_corrections(
        state.revision, corrected,
        retained_source_updates={68: record},
        retained_source_limits={"max_frames": 3, "max_rows_per_frame": 40},
    )
    # This is a valid registry record and each transform remains an SE(3),
    # but the live frame endpoint no longer matches anchor @ its authoritative
    # relative pose. The early registry metadata validator cannot see this
    # cross-object binding error; the inner retained-row parser must reject it.
    bundle._validate_retained_registry_metadata(
        state.retained_source_observations, state.revision, state.geometry_revision,
        len(state.poses), len(state.statuses),
    )
    stale_relative = state.relative_poses[68].copy()
    stale_relative[0, 3] += 0.01
    state.relative_poses[68] = stale_relative
    assert not np.allclose(
        state.keyframes[anchor_id].pose @ state.relative_poses[68], state.poses[68],
        rtol=0., atol=1e-7,
    )
    return state, source_true, truth_points


def _empty_history_provider(payload):
    if payload["source_history_phase"] == "validate":
        return [], {"status": "validated", "reason": None}
    return [], {"status": "eligible", "captured_rows": 0}


def _call_owned_retention(state, source_true, truth_points, *, retain, monkeypatch=None):
    owned = _owned_factor_provider(
        state, source_frame=68, target_frame=76,
        source_true=source_true, truth_points=truth_points,
    )
    return local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET,
        training_factor_provider=owned,
        source_history_provider=_empty_history_provider,
        retain_source_observations=retain,
        retained_source_calibration_identity=CALIBRATION_ID,
        retained_source_exclusions=(
            {"valid": True, "landmark_ids": [], "source_frame": 68,
             "source_pixels": np.empty((0, 2), dtype=np.float32)}
            if retain else None
        ),
    )


def _install_solver_tripwire(monkeypatch):
    calls = []

    def forbidden_solver(*_args, **_kwargs):
        calls.append("called")
        raise AssertionError("solver must not run after retention context rejection")

    monkeypatch.setattr(bundle, "least_squares", forbidden_solver)
    return calls


def test_nested_retained_record_parser_rejection_prevents_solver_and_any_commit(monkeypatch):
    state, source_true, truth_points, _anchor_id = _retained_invalid_state()
    before = _map_snapshot(state)
    solver_calls = _install_solver_tripwire(monkeypatch)

    report = _call_owned_retention(
        state, source_true, truth_points, retain=True
    )

    assert report["applied"] is False
    assert report["reason"] == "retained_source_context_invalid"
    retained = report["retained_source_observations"]
    assert retained["status"] == "rejected"
    assert retained["reason"] == "retained_source_registry_snapshot_invalid"
    assert retained["installed_rows"] == 0
    assert retained["components_added"] == 0
    assert solver_calls == []
    _assert_map_snapshot(state, before)


def test_initial_registry_rejection_survives_later_owned_chart_failure_and_off_falls_back(
    monkeypatch,
):
    state_on, source_true, truth_points, _anchor_id = _retained_invalid_state()
    before_on = _map_snapshot(state_on)
    real_lil = bundle.lil_matrix
    real_solver = bundle.least_squares
    augmented_shapes = []

    def fail_on_augmented_pattern(shape, *args, **kwargs):
        augmented_shapes.append(tuple(shape))
        if len(augmented_shapes) == 2:
            raise RuntimeError("synthetic-later-owned-chart-failure")
        return real_lil(shape, *args, **kwargs)

    monkeypatch.setattr(bundle, "lil_matrix", fail_on_augmented_pattern)
    solver_calls_on = _install_solver_tripwire(monkeypatch)
    report_on = _call_owned_retention(
        state_on, source_true, truth_points, retain=True
    )
    assert len(augmented_shapes) == 2
    assert report_on["applied"] is False
    assert report_on["reason"] == "retained_source_context_invalid"
    assert report_on["retained_source_observations"]["reason"] == (
        "retained_source_registry_snapshot_invalid"
    )
    assert solver_calls_on == []
    _assert_map_snapshot(state_on, before_on)

    # With retention OFF, neither the nested registry record nor its rejection
    # context participates. The same later owned-chart failure falls back to
    # the original BA inputs and invokes the legacy solver once.
    state_off_bad = _clone_state(state_on)
    state_off_clean = _clone_state(state_on)
    state_off_clean.retained_source_observations = {}
    calls = {}

    def run_off(state, label):
        shapes = []

        def fail_second_pattern(shape, *args, **kwargs):
            shapes.append(tuple(shape))
            if len(shapes) == 2:
                raise RuntimeError("synthetic-later-owned-chart-failure")
            return real_lil(shape, *args, **kwargs)

        captured = {}

        def capture_legacy_solver(fun, initial, **kwargs):
            initial = np.asarray(initial, float).copy()
            captured["initial"] = initial
            captured["residual"] = np.asarray(fun(initial), float)
            captured["sparsity"] = kwargs["jac_sparsity"].toarray().copy()
            captured["options"] = {key: value for key, value in kwargs.items()
                                    if key != "jac_sparsity"}
            return real_solver(fun, initial, **kwargs)

        monkeypatch.setattr(bundle, "lil_matrix", fail_second_pattern)
        monkeypatch.setattr(bundle, "least_squares", capture_legacy_solver)
        report = _call_owned_retention(
            state, source_true, truth_points, retain=False
        )
        calls[label] = (report, captured, shapes)
        return report, captured, shapes

    report_bad, captured_bad, shapes_bad = run_off(state_off_bad, "bad")
    report_clean, captured_clean, shapes_clean = run_off(state_off_clean, "clean")
    assert len(shapes_bad) == len(shapes_clean) == 2
    assert report_bad.get("reason") != "retained_source_context_invalid"
    assert report_clean.get("reason") != "retained_source_context_invalid"
    np.testing.assert_array_equal(captured_bad["initial"], captured_clean["initial"])
    np.testing.assert_array_equal(captured_bad["residual"], captured_clean["residual"])
    np.testing.assert_array_equal(captured_bad["sparsity"], captured_clean["sparsity"])
    assert captured_bad["options"].keys() == captured_clean["options"].keys()
    assert state_off_bad.retained_source_observations.keys() == {68}
    assert state_off_clean.retained_source_observations == {}


def test_inner_retained_binding_rejection_survives_later_owned_chart_failure(
    monkeypatch,
):
    state, source_true, truth_points = _retained_inner_binding_invalid_state()
    before = _map_snapshot(state)
    metadata_validator = bundle._validate_retained_registry_metadata
    metadata_checks = []

    def record_metadata_validation(*args, **kwargs):
        metadata_checks.append(True)
        return metadata_validator(*args, **kwargs)

    monkeypatch.setattr(
        bundle, "_validate_retained_registry_metadata", record_metadata_validation
    )
    real_lil = bundle.lil_matrix
    pattern_shapes = []

    def fail_after_inner_parser(shape, *args, **kwargs):
        pattern_shapes.append(tuple(shape))
        if len(pattern_shapes) == 2:
            raise RuntimeError("synthetic-later-owned-chart-failure")
        return real_lil(shape, *args, **kwargs)

    monkeypatch.setattr(bundle, "lil_matrix", fail_after_inner_parser)
    solver_calls = _install_solver_tripwire(monkeypatch)
    report = _call_owned_retention(state, source_true, truth_points, retain=True)

    assert metadata_checks == [True]
    assert len(pattern_shapes) == 2
    assert solver_calls == []
    assert report["applied"] is False
    assert report["reason"] == "retained_source_context_invalid"
    # The valid metadata reaches the inner frame/anchor/relative-pose binding
    # check, which marks a sticky veto before augmented chart construction
    # raises. The later exception must not replace this more specific reason.
    assert report["retained_source_observations"]["status"] == "rejected"
    assert report["retained_source_observations"]["reason"] == (
        "retained_source_snapshot_invalid"
    )
    assert report["owned_stereo_image_bundle"]["reason"] == (
        "synthetic-later-owned-chart-failure"
    )
    _assert_map_snapshot(state, before)
