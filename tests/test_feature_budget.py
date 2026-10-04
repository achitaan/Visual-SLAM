import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for folder in (REPO / "src", REPO / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from feature_cache import extraction_signature
from shared_slam import MappingConfig, SharedSlam
import evaluate_shared_slam
import main as app_main
import run_development_tests


MATRIX = np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]])


@pytest.mark.parametrize("value", [0, -1, 10001, 1.5, "1500", True, np.bool_(False)])
def test_feature_budget_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="features"):
        MappingConfig(features=value)


def test_default_and_requested_budget_are_forwarded_to_sift_constructor(monkeypatch):
    original = cv.SIFT_create
    calls = []

    def spy(**kwargs):
        calls.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr("shared_slam.cv.SIFT_create", spy)
    default = SharedSlam(MATRIX, config=MappingConfig())
    requested = SharedSlam(MATRIX, config=MappingConfig(features=3000))
    try:
        assert [call["nfeatures"] for call in calls] == [1500, 3000]
        assert default.detector.getNFeatures() == 1500
        assert requested.detector.getNFeatures() == 3000
    finally:
        default.close()
        requested.close()


def test_known_image_acquisition_reaches_larger_requested_budget():
    image = np.random.default_rng(20261004).integers(0, 256, (1024, 1024), dtype=np.uint8)
    unbounded = cv.SIFT_create(nfeatures=0, contrastThreshold=0.04)
    available = unbounded.detectAndCompute(image, None)[1]
    assert available is not None and len(available) > 3000
    default = SharedSlam(MATRIX, config=MappingConfig(features=1500))
    expanded = SharedSlam(MATRIX, config=MappingConfig(features=3000))
    try:
        default_rows = default._extract(image, None)[0]
        expanded_rows = expanded._extract(image, None)[0]
        assert len(default_rows) >= 1500
        assert len(expanded_rows) > len(default_rows)
    finally:
        default.close()
        expanded.close()


def test_feature_budget_changes_feature_cache_signature():
    default = SharedSlam(MATRIX, config=MappingConfig(features=1500))
    expanded = SharedSlam(MATRIX, config=MappingConfig(features=3000))
    try:
        assert extraction_signature(default, cv) != extraction_signature(expanded, cv)
    finally:
        default.close()
        expanded.close()


def _args(**overrides):
    values = dict(features=1500, disable_bundle=False, loop_mode="off",
                  stereo_depth_policy="supported", stereo_pose_arbitration=False,
                  stereo_physical_match_pool=False, stereo_raw_reference_retry=False,
                  stereo_owned_image_bundle=False, stereo_source_history_bundle=False,
                  stereo_retained_source_observations=False, bundle_solver_accuracy="default")
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("budget", [1500, 3000])
def test_cli_config_builders_forward_default_and_requested_budget(budget):
    args = _args(features=budget)
    assert evaluate_shared_slam._mapping_config_from_args(args).features == budget
    assert app_main._mapping_config_from_args(args).features == budget


def test_development_identity_and_reuse_separate_feature_budgets():
    default = run_development_tests.current_mapping_configuration("bundle", "supported")
    expanded = run_development_tests.current_mapping_configuration(
        "bundle", "supported", features=3000
    )
    assert default["features"] == 1500
    assert expanded["features"] == 3000
    identity = {"features": 1500, "frames": 80}
    report = {"development_identity": dict(identity), "status": "completed", "frames": 80}
    assert run_development_tests.reusable(report, identity)
    assert not run_development_tests.reusable(report, {"features": 3000, "frames": 80})
    with pytest.raises(ValueError, match="baseline"):
        run_development_tests.current_mapping_configuration(
            "baseline", "supported", features=3000
        )


@pytest.mark.parametrize("module,argv", [
    (app_main, ["main.py", "--features", "0"]),
    (evaluate_shared_slam, ["evaluate_shared_slam.py", "--features", "10001", "--output", "unused"]),
])
def test_cli_rejects_invalid_budget_before_processing(monkeypatch, module, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
