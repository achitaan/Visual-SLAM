"""Numerical-policy controls for the local bundle's inner LSMR solve."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import local_bundle as bundle
import shared_slam as shared_module
from local_bundle import local_bundle_adjustment
from shared_slam import MappingConfig, SharedSlam
from slam_state import MapState
from test_bundle_diagnostics import BASELINE, MATRIX
from test_free_source_stereo_bundle import (
    OFFSET,
    _assert_snapshot_equal,
    _free_source_case,
    _provider,
    _snapshot,
)


def _copy_solver_options(options):
    copied = {}
    for key, value in options.items():
        if key == "jac_sparsity":
            copied[key] = value.toarray().astype(bool)
        elif isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, dict):
            copied[key] = deepcopy(value)
        else:
            copied[key] = value
    return copied


def _capture_bundle(monkeypatch, *, accuracy="default", provider_mode="off",
                    count=24, max_landmarks=40):
    state, truth, true_source, landmark_ids = _free_source_case(count=count)
    before = _snapshot(state)
    captured = {}

    def inspect_solver(residual, initial, **options):
        initial = np.asarray(initial, dtype=float).copy()
        captured.update(
            initial=initial,
            initial_residual=np.asarray(residual(initial), dtype=float).copy(),
            options=_copy_solver_options(options),
        )
        return SimpleNamespace(
            x=initial.copy(), nfev=1, success=True, status=1,
            message="solver-policy capture", cost=0.0, optimality=0.0,
        )

    monkeypatch.setattr(bundle, "least_squares", inspect_solver)
    kwargs = {"solver_accuracy": accuracy}
    if provider_mode == "active":
        kwargs["training_factor_provider"] = _provider(
            state, truth, true_source, landmark_ids
        )
    elif provider_mode == "empty":
        kwargs["training_factor_provider"] = lambda _payload: (
            (), {"status": "rejected", "reason": "empty_test_pool"}
        )
    elif provider_mode == "rejected":
        kwargs["training_factor_provider"] = lambda _payload: (
            (object(),), {"status": "accepted", "reason": None}
        )
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=max_landmarks,
        disparity_offset=OFFSET, **kwargs,
    )
    return state, before, captured, report


def _precise_options(options, variable_count):
    assert options["tr_options"] == {
        "atol": 1e-12,
        "btol": 1e-12,
        "maxiter": max(500, variable_count),
    }
    assert options["tr_solver"] == "lsmr"
    assert options["max_nfev"] == 30
    assert options["loss"] == "huber"
    assert options["f_scale"] == 2.0


def test_default_fallback_kwargs_are_legacy_and_precise_off_matches_active_inner_policy(monkeypatch):
    captures = {}
    reports = {}
    for mode in ("off", "empty", "rejected", "active"):
        state, before, capture, report = _capture_bundle(
            monkeypatch,
            accuracy="default",
            provider_mode=mode,
        )
        captures[("default", mode)] = capture
        reports[("default", mode)] = report
        if mode != "active":
            _assert_snapshot_equal(state, before)

    original = captures[("default", "off")]
    legacy = original["options"]
    assert set(legacy) == {
        "jac_sparsity", "loss", "f_scale", "max_nfev", "x_scale", "tr_solver"
    }
    assert legacy["max_nfev"] == 30
    assert legacy["loss"] == "huber" and legacy["f_scale"] == 2.0
    assert legacy["tr_solver"] == "lsmr"
    for mode in ("empty", "rejected"):
        current = captures[("default", mode)]
        assert current["options"].keys() == legacy.keys()
        for key in legacy:
            np.testing.assert_array_equal(current["options"][key], legacy[key])
        np.testing.assert_array_equal(current["initial"], original["initial"])
        np.testing.assert_array_equal(current["initial_residual"], original["initial_residual"])
        assert reports[("default", mode)]["applied"] is False
        assert reports[("default", mode)]["solver_accuracy_effective"] == "legacy_defaults"

    active_default = captures[("default", "active")]
    _precise_options(active_default["options"], len(active_default["initial"]))
    assert reports[("default", "off")]["solver_accuracy_requested"] == "default"
    assert reports[("default", "off")]["solver_accuracy_effective"] == "legacy_defaults"
    assert reports[("default", "off")]["solver_inner_options"] is None
    assert reports[("default", "active")]["solver_accuracy_effective"] == "precise_lsmr"
    assert reports[("default", "active")]["solver_inner_options"] == active_default["options"]["tr_options"]

    for mode in ("off", "empty", "rejected", "active"):
        state, before, capture, report = _capture_bundle(
            monkeypatch,
            accuracy="precise",
            provider_mode=mode,
        )
        captures[("precise", mode)] = capture
        reports[("precise", mode)] = report
        _precise_options(capture["options"], len(capture["initial"]))
        assert report["solver_accuracy_requested"] == "precise"
        assert report["solver_accuracy_effective"] == "precise_lsmr"
        assert report["solver_inner_options"] == capture["options"]["tr_options"]
        if mode != "active":
            _assert_snapshot_equal(state, before)

    precise_off = captures[("precise", "off")]
    assert precise_off["options"]["tr_options"] == active_default["options"]["tr_options"]
    assert set(precise_off["options"]) == set(original["options"]) | {"tr_options"}
    # Precision changes only LSMR stopping. The original-only graph keeps the
    # same variables, residual rows, gauge, scaling, and outer robust solve.
    np.testing.assert_array_equal(precise_off["initial"], original["initial"])
    np.testing.assert_array_equal(precise_off["initial_residual"], original["initial_residual"])
    for key in original["options"]:
        if key == "tr_options":
            assert key not in precise_off["options"]
            continue
        left, right = original["options"][key], precise_off["options"][key]
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    assert reports[("default", "off")].get("reason") == reports[("precise", "off")].get("reason")
    assert reports[("default", "off")]["applied"] == reports[("precise", "off")]["applied"]
    for mode in ("empty", "rejected"):
        current = captures[("precise", mode)]
        np.testing.assert_array_equal(current["initial"], precise_off["initial"])
        np.testing.assert_array_equal(current["initial_residual"], precise_off["initial_residual"])
        np.testing.assert_array_equal(current["options"]["jac_sparsity"],
                                      precise_off["options"]["jac_sparsity"])
        np.testing.assert_array_equal(current["options"]["x_scale"],
                                      precise_off["options"]["x_scale"])

    precise_active = captures[("precise", "active")]
    for key in ("initial", "initial_residual"):
        np.testing.assert_array_equal(precise_active[key], active_default[key])
    for key in active_default["options"]:
        left, right = active_default["options"][key], precise_active["options"][key]
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def test_size_aware_precise_inner_iteration_cap_exceeds_500_for_large_chart(monkeypatch):
    _state, _before, capture, report = _capture_bundle(
        monkeypatch,
        accuracy="precise",
        provider_mode="active",
        count=190,
        max_landmarks=240,
    )
    assert len(capture["initial"]) > 500
    _precise_options(capture["options"], len(capture["initial"]))
    assert capture["options"]["tr_options"]["maxiter"] == len(capture["initial"])
    assert report["solver_accuracy_effective"] == "precise_lsmr"


def test_default_augmented_and_precise_augmented_are_same_recovered_synthetic_solve():
    outcomes = {}
    for accuracy in ("default", "precise"):
        state, truth, true_source, landmark_ids = _free_source_case(source_bias=0.30)
        initial_source_error = float(np.linalg.norm(state.poses[3][:3, 3] - true_source[:3, 3]))
        report = local_bundle_adjustment(
            state, MATRIX, BASELINE, window=3, max_landmarks=40,
            disparity_offset=OFFSET,
            training_factor_provider=_provider(state, truth, true_source, landmark_ids),
            solver_accuracy=accuracy,
        )
        assert report["applied"] is True
        assert report["solver_accuracy_requested"] == accuracy
        assert report["solver_accuracy_effective"] == "precise_lsmr"
        final_source_error = float(np.linalg.norm(state.poses[3][:3, 3] - true_source[:3, 3]))
        assert final_source_error < 0.5 * initial_source_error
        outcomes[accuracy] = (state, report)

    default_state, default_report = outcomes["default"]
    precise_state, precise_report = outcomes["precise"]
    for key in default_state.keyframes:
        np.testing.assert_allclose(default_state.keyframes[key].pose,
                                   precise_state.keyframes[key].pose,
                                   rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(default_state.poses, precise_state.poses,
                               rtol=0.0, atol=1e-11)
    for ident in default_state.landmarks:
        np.testing.assert_allclose(default_state.landmarks[ident].position,
                                   precise_state.landmarks[ident].position,
                                   rtol=0.0, atol=1e-11)
    for key in ("initial_cost", "final_cost", "affected_final_cost", "augmented_final_cost"):
        if key in default_report:
            assert default_report[key] == pytest.approx(precise_report[key], abs=1e-12)


def test_invalid_accuracy_policy_fails_before_solver_or_map_mutation(monkeypatch):
    state, _truth, _source, _ids = _free_source_case()
    before = _snapshot(state)

    def must_not_solve(*_args, **_kwargs):
        pytest.fail("invalid policy reached least_squares")

    monkeypatch.setattr(bundle, "least_squares", must_not_solve)
    with pytest.raises(ValueError):
        local_bundle_adjustment(
            state, MATRIX, BASELINE, disparity_offset=OFFSET,
            solver_accuracy="fast_but_loose",
        )
    _assert_snapshot_equal(state, before)

    with pytest.raises(ValueError):
        SharedSlam(MATRIX, config=MappingConfig(bundle_solver_accuracy="unknown"))


def test_mapping_config_default_and_skipped_bundle_report_requested_policy():
    assert MappingConfig().bundle_solver_accuracy == "default"
    assert MappingConfig(bundle_solver_accuracy="precise").bundle_solver_accuracy == "precise"
    report = local_bundle_adjustment(
        MapState(), MATRIX, BASELINE, solver_accuracy="precise"
    )
    assert report["reason"] == "insufficient_keyframes"
    assert report["solver_accuracy_requested"] == "precise"
    assert report["solver_accuracy_effective"] == "not_run"
    assert report["solver_inner_options"] is None


@pytest.mark.parametrize("accuracy", ["default", "precise"])
def test_shared_slam_forwards_configured_accuracy_to_local_bundle(monkeypatch, accuracy):
    state, _truth, _source, _ids = _free_source_case()
    slam = SharedSlam(
        MATRIX,
        config=MappingConfig(
            loop_mode="off", keyframe_interval=1, bundle_window=3,
            bundle_solver_accuracy=accuracy,
        ),
    )
    slam.map = state
    slam.last_keyframe = 2
    forwarded = {}

    def empty_features(_image, _right):
        return (np.empty((0, 2), dtype=np.float32),
                np.empty((0, 128), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    def tracked_pose(_pixels, _descriptors, _size, **_kwargs):
        return (state.poses[-1].copy(), {}), {
            "tracking_ok": True, "num_matches": 100, "num_inliers": 100,
            "inlier_ratio": 1.0, "valid_3d": 100,
        }

    def capture_ba(*args, **kwargs):
        forwarded.update(kwargs)
        return {"applied": False, "reason": "test_capture"}

    slam._extract = empty_features
    slam._track = tracked_pose
    monkeypatch.setattr(shared_module, "local_bundle_adjustment", capture_ba)
    try:
        # The existing scene gives a bounded, dataset-free path through process()
        # to the same BA call used by tracking.
        slam.process(len(state.poses), np.zeros((480, 640), dtype=np.uint8))
        assert forwarded["solver_accuracy"] == accuracy
        assert forwarded["window"] == 3
    finally:
        slam.close(finish=False)


@pytest.mark.parametrize(
    "script",
    [
        "src/main.py",
        "scripts/evaluate_shared_slam.py",
        "scripts/run_development_tests.py",
    ],
)
def test_accuracy_option_is_exposed_by_cli_without_loading_a_dataset(script):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / script), "--bundle-solver-accuracy", "precise", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--bundle-solver-accuracy" in completed.stdout
    assert "default" in completed.stdout and "precise" in completed.stdout


def _load_development_runner(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "accuracy_policy_history_under_test", root / "scripts" / "run_development_tests.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_default_history_is_normalized_only_for_cost_estimation(tmp_path, monkeypatch):
    from test_development_timing_history import (
        evaluation_report, expected_case, expected_configuration,
    )

    module = _load_development_runner(monkeypatch)
    history_path = tmp_path / "legacy-default-evaluation.json"
    history_path.write_text(json.dumps(evaluation_report()), encoding="utf-8")
    estimate = module.estimate_case_runtime(
        350, [], [history_path], expected_case(), "partial", expected_configuration(),
        "current-source-fingerprint", fallback_rate=4.0,
    )

    accepted = estimate["timing_history"]["accepted"]
    assert len(accepted) == 1
    assert accepted[0]["use"] == "cost_estimate_only"
    assert accepted[0]["bundle_solver_accuracy_legacy_default_normalized"] is True
    assert "metrics" not in accepted[0]


def test_precise_history_requires_explicit_mode_in_identity_report_and_configuration(
        tmp_path, monkeypatch):
    from test_development_timing_history import (
        evaluation_report, expected_case, expected_configuration,
    )

    module = _load_development_runner(monkeypatch)
    precise_identity = expected_case()
    precise_identity["bundle_solver_accuracy"] = "precise"
    precise_configuration = {
        **expected_configuration(), "bundle_solver_accuracy": "precise",
    }
    complete = evaluation_report()
    complete["development_identity"]["bundle_solver_accuracy"] = "precise"
    complete["bundle_solver_accuracy"] = "precise"
    complete["configuration"]["bundle_solver_accuracy"] = "precise"

    for missing in (None, "identity", "report", "configuration"):
        report = deepcopy(complete)
        if missing == "identity":
            report["development_identity"].pop("bundle_solver_accuracy")
        elif missing == "report":
            report.pop("bundle_solver_accuracy")
        elif missing == "configuration":
            report["configuration"].pop("bundle_solver_accuracy")
        history_path = tmp_path / f"precise-{missing or 'complete'}.json"
        history_path.write_text(json.dumps(report), encoding="utf-8")
        estimate = module.estimate_case_runtime(
            350, [], [history_path], precise_identity, "partial", precise_configuration,
            "current-source-fingerprint", fallback_rate=4.0,
        )
        if missing is None:
            accepted = estimate["timing_history"]["accepted"]
            assert len(accepted) == 1
            assert accepted[0]["use"] == "cost_estimate_only"
            assert "metrics" not in accepted[0]
        else:
            assert not estimate["timing_history"]["accepted"]
            rejected = estimate["timing_history"]["rejected"]
            assert len(rejected) == 1
            assert "missing or mismatched precise bundle solver accuracy mode" in rejected[0]["reason"]


def test_export_reuse_rejects_legacy_identity_without_accuracy_policy(tmp_path, monkeypatch):
    from test_development_runner import _write_completed_bundle

    module = _load_development_runner(monkeypatch)
    _output, report_path, legacy_identity = _write_completed_bundle(
        module, tmp_path, monkeypatch,
    )
    for policy in ("default", "precise"):
        current_identity = {**legacy_identity, "bundle_solver_accuracy": policy}
        assert not module.reusable_export(report_path, current_identity)


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        (("--bundle-solver-accuracy", "precise"),
         "--bundle-solver-accuracy precise requires --slam"),
    ],
)
def test_main_cli_parses_accuracy_and_applies_mode_guard_before_data_loading(
        arguments, expected_error):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "src" / "main.py"), *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 2
    assert expected_error in completed.stderr
    assert "invalid choice" not in completed.stderr


def test_development_cli_rejects_precise_baseline_before_reading_dataset(tmp_path):
    root = Path(__file__).resolve().parents[1]
    missing_data = tmp_path / "no-kitti-data"
    missing_poses = tmp_path / "no-ground-truth"
    output = tmp_path / "should-not-be-created"
    completed = subprocess.run(
        [
            sys.executable, str(root / "scripts" / "run_development_tests.py"),
            "--data-root", str(missing_data), "--poses-root", str(missing_poses),
            "--variants", "baseline", "--bundle-solver-accuracy", "precise",
            "--output", str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 2
    assert "--bundle-solver-accuracy precise cannot be combined with the preserved baseline variant" in completed.stderr
    assert not output.exists()
