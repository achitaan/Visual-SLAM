"""Guarded raw-supported retry for failed ordinary stereo references."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import cv2 as cv
import numpy as np
import pytest

import mapping_geometry as geometry
import shared_slam as module
from feature_cache import extraction_signature
from loop_geometry import StereoLoopFrame
from shared_slam import MappingConfig, SharedSlam
from stereo_pose_arbitration import SupportedStereoFrame
from test_stereo_arbitration_tracking import (
    BASELINE, K, OFFSET, camera, record, scene,
)


REPO = Path(__file__).resolve().parents[1]


def _known_motion_case(slam, map_z=1.2, configured_target_scale=None):
    pixels, world, descriptors = scene()
    truth = np.eye(4)
    truth[2, 3] = .5
    target_pixels, _ = geometry.project(world, truth, K)
    target_pixels = np.asarray(target_pixels, np.float32)
    target_points = (world-truth[:3, 3]) @ truth[:3, :3]
    current_pixels = [pixels.copy()]

    def extract(image, right):
        index = len(slam.map.poses)
        if index == 0:
            observed, supported_points = pixels.copy(), world.copy()
            configured_points = supported_points.copy()
        else:
            observed, supported_points = target_pixels.copy(), target_points.copy()
            configured_points = (np.full_like(supported_points, np.nan)
                                 if configured_target_scale is None else
                                 supported_points * float(configured_target_scale))
        disparity = K[0, 0] * BASELINE / supported_points[:, 2] + OFFSET
        slam.current_disparity = np.full(
            image.shape[:2], float(disparity[0]), np.float32)
        right_u = observed[:, 0] - disparity
        slam._supported_extraction = (supported_points.copy(), right_u.copy())
        current_pixels[0] = observed.copy()
        return observed.copy(), descriptors.copy(), configured_points, right_u.copy()

    def track(*args):
        keyframe = slam.map.keyframes[slam.last_keyframe]
        identifiers = [int(value) for value in keyframe.landmark_ids if value >= 0]
        misses = {}
        for ident in identifiers:
            landmark = slam.map.landmarks[ident]
            misses[ident] = int(landmark.misses)
            landmark.misses = 0
        slam._last_map_track_inlier_misses = misses
        slam.accepted_tracks = [
            (ident, current_pixels[0][row].copy())
            for row, ident in enumerate(keyframe.landmark_ids) if ident >= 0
        ]
        pose = np.eye(4)
        pose[2, 3] = map_z
        return (pose, {row: int(ident) for row, ident in enumerate(keyframe.landmark_ids)
                       if ident >= 0}), {
            'num_matches': len(identifiers), 'num_inliers': len(identifiers),
            'valid_3d': len(identifiers), 'inlier_ratio': 1., 'tracking_ok': True,
        }

    pairs = np.arange(len(pixels), dtype=np.int32)
    slam._extract = extract
    slam._track = track
    slam._match = lambda first, second: np.column_stack((pairs, pairs)).astype(np.int32)
    if slam.config.stereo_pose_arbitration:
        slam._prepare_stereo_arbitration = lambda index, current: (
            None, {'choice': 'map', 'reason': 'independent_training_failed'})
    return pixels, world, descriptors, target_pixels, truth


def _start_then_break_configured_reference(slam, case, break_source=True):
    image = np.zeros((376, 1241), np.uint8)
    slam.process(0, image, image)
    if break_source:
        source, index = slam.previous_stereo_geometry
        slam.previous_stereo_geometry = (
            StereoLoopFrame(source.pixels, np.full_like(source.points, np.nan),
                            source.descriptors, source.image_size), index)
    return image


def test_retry_only_after_configured_none_and_keeps_raw_pose_connected(monkeypatch):
    cv.setRNGSeed(0)
    slam = camera(enabled=True, stereo_raw_reference_retry=True)
    # The ordinary fit sees finite but inconsistent target depth: its forward
    # left-camera fit is plausible while reverse metric verification fails.
    # The immutable supported rows retain the actual target depth.
    case = _known_motion_case(slam, configured_target_scale=3.0)
    image = _start_then_break_configured_reference(slam, case, break_source=False)
    real_estimator = module.estimate_stereo_reference
    calls = []

    def observe_estimator(source, target, matrix, **kwargs):
        result = real_estimator(source, target, matrix, **kwargs)
        calls.append({
            'source_finite': int(np.isfinite(source.points).all(axis=1).sum()),
            'target_finite': int(np.isfinite(target.points).all(axis=1).sum()),
            'reverse_checked': bool(result and result.get('reverse_checked')),
            'returned_none': result is None,
        })
        return result

    monkeypatch.setattr(module, 'estimate_stereo_reference', observe_estimator)
    try:
        pose, info = slam.process(1, image, image)
        assert len(calls) == 2, (calls, info)
        np.testing.assert_allclose(pose, case[-1], atol=2e-3, err_msg=str(info))
        assert calls[0]['source_finite'] == len(case[0])
        assert calls[0]['target_finite'] == len(case[0])
        assert calls[0]['returned_none'] or not calls[0]['reverse_checked']
        assert calls[1]['source_finite'] == len(case[0])
        assert calls[1]['target_finite'] == len(case[0])
        assert calls[1]['reverse_checked']
        retry = info['stereo_raw_reference_retry']
        assert retry['attempted'] and retry['eligible'] and retry['installed']
        assert retry['pose_selected']
        assert retry['configured_reference_rejection'] == 'configured_reference_returned_none'
        assert retry['fit_source'] == 'raw_supported_reference_retry'
        assert retry['fit_depth_policy'] == 'supported_raw'
        assert retry['held_out_arbitration_used'] is False
        assert retry['reservation_context_available'] is False
        assert retry['physical_identity'] == 'exact_float32_pixels_canonicalized'
        assert retry['association_validation']['eligible']
        assert info['reference_association_validation']['independent_fit_source'] == \
            'raw_supported_reference_retry'
        assert info['reference_association_validation']['prediction_seed_supplied'] is False
        assert info['map_pose_rejected_for_stereo_conflict']
        assert (0, 1) in slam.map.stereo_motion
        np.testing.assert_allclose(slam.map.stereo_motion[(0, 1)], case[-1], atol=2e-3)
        assert slam._arbitration_context is None
        assert 'stereo_pose_arbitration_before_bundle' not in info
        assert 'stereo_pose_arbitration_after_bundle' not in info
    finally:
        slam.close()


def test_raw_retry_preserves_orientation_aliases_but_counts_one_physical_edge(monkeypatch):
    cv.setRNGSeed(0)
    slam = camera(enabled=False, stereo_raw_reference_retry=True)
    try:
        previous = record(slam, 0, duplicate=True)
        current = record(slam, 1, duplicate=True)
        source_points = previous.points.copy()
        target_points = current.points.copy()
        source_points[1] = source_points[0]
        target_points[1] = target_points[0]
        previous = SupportedStereoFrame(
            previous.pixels, previous.descriptors, source_points, previous.right_u,
            np.zeros(len(source_points), int), 0, previous.image_size,
            previous.calibration_identity)
        current = SupportedStereoFrame(
            current.pixels, current.descriptors, target_points, current.right_u,
            np.full(len(target_points), -1, int), 1, current.image_size,
            current.calibration_identity)
        slam.previous_supported_stereo = previous
        slam.previous_stereo_geometry = (
            StereoLoopFrame(previous.pixels, previous.points,
                            previous.descriptors, previous.image_size), 0)
        slam.map.record(np.eye(4), 'tracking')
        slam.current_supported_stereo = current
        disparity = K[0, 0] * BASELINE / 20. + OFFSET
        slam.current_disparity = np.full((376, 1241), disparity, np.float32)
        target = StereoLoopFrame(current.pixels, current.points,
                                 current.descriptors, current.image_size)
        pairs, report = slam._full_supported_match_pairs(
            previous, current, 0, 1, slam.previous_stereo_geometry[0], target,
            (1241, 376), physical_canonical=True)
        assert len(previous.descriptors) == 48  # keep all appearance orientations
        assert len(pairs) == 47  # exact-pixel alias rows contribute one edge
        assert report['physical_identity'] == 'exact_float32_pixels_canonicalized'
        assert report['held_out_arbitration_used'] is False
        verified, fit_report, _, _, _ = slam._estimate_raw_supported_reference_retry(
            previous, current, 0, 1,
            StereoLoopFrame(previous.pixels, previous.points,
                            previous.descriptors, previous.image_size),
            target, (1241, 376))
        assert verified is not None, fit_report
        assert verified['reverse_checked'] is True
        assert fit_report['reference_matches'] == 47
        assert fit_report['reference_inliers'] == 47
        np.testing.assert_allclose(verified['measurement'], np.eye(4), atol=2e-3)

        complex_pose = np.eye(4, dtype=np.complex128)
        complex_pose[0, 3] = 1j
        slam.map.poses[0] = complex_pose
        rejected, malformed, *_ = slam._estimate_raw_supported_reference_retry(
            previous, current, 0, 1,
            StereoLoopFrame(previous.pixels, previous.points,
                            previous.descriptors, previous.image_size),
            target, (1241, 376))
        assert rejected is None
        assert malformed['reason'] == 'invalid_source_map_epoch'

        slam.map.poses[0] = np.eye(4)
        real_estimator = module.estimate_stereo_reference

        def mutate_source_during_fit(*args, **kwargs):
            result = real_estimator(*args, **kwargs)
            changed = np.eye(4, dtype=np.complex128)
            changed[0, 3] = 1j
            slam.map.poses[0] = changed
            return result

        monkeypatch.setattr(module, 'estimate_stereo_reference', mutate_source_during_fit)
        rejected, changed_epoch, *_ = slam._estimate_raw_supported_reference_retry(
            previous, current, 0, 1,
            StereoLoopFrame(previous.pixels, previous.points,
                            previous.descriptors, previous.image_size),
            target, (1241, 376))
        assert rejected is None
        assert changed_epoch['reason'] == 'source_pose_epoch_changed'
    finally:
        slam.close()


def test_forward_only_raw_retry_does_not_install_motion_or_replace_map(monkeypatch):
    slam = camera(enabled=False, stereo_raw_reference_retry=True)
    case = _known_motion_case(slam)
    image = _start_then_break_configured_reference(slam, case)
    real_estimator = module.estimate_stereo_reference

    def forward_only(source, target, matrix, **kwargs):
        result = real_estimator(source, target, matrix, **kwargs)
        if result is not None and np.isfinite(source.points).all():
            return {**result, 'reverse_checked': False}
        return result

    monkeypatch.setattr(module, 'estimate_stereo_reference', forward_only)
    old_predictor = np.eye(4)
    old_predictor[0, 3] = .125
    slam.verified_stereo_motion = (old_predictor.copy(), 0)
    try:
        pose, info = slam.process(1, image, image)
        assert pose[2, 3] == pytest.approx(1.2)
        assert not slam.map.stereo_motion
        np.testing.assert_array_equal(slam.verified_stereo_motion[0], old_predictor)
        assert slam.verified_stereo_motion[1] == 0
        assert not info['stereo_raw_reference_retry']['eligible']
        assert info['stereo_raw_reference_retry']['reason'] == \
            'full_supported_reference_not_reverse_checked'
    finally:
        slam.close()


def test_late_calibration_change_abandons_raw_retry_without_aging_map(monkeypatch):
    slam = camera(enabled=True, stereo_raw_reference_retry=True)
    case = _known_motion_case(slam)
    image = _start_then_break_configured_reference(slam, case)
    original_guard = slam._hard_reference_retention_guard
    guard_calls = []

    def mutate_on_final_consumption(*args, **kwargs):
        guard_calls.append(True)
        if len(guard_calls) == 3:
            slam.stereo.Q[3, 3] += .01
        return original_guard(*args, **kwargs)

    slam._hard_reference_retention_guard = mutate_on_final_consumption
    old_predictor = np.eye(4)
    old_predictor[0, 3] = .125
    slam.verified_stereo_motion = (old_predictor.copy(), 0)
    keyframe_ids = slam.map.keyframes[slam.last_keyframe].landmark_ids.copy()
    try:
        pose, info = slam.process(1, image, image)
        assert pose[2, 3] == pytest.approx(1.2)
        assert len(guard_calls) == 3
        assert info['tracking_ok']
        assert info['stereo_raw_reference_retry']['reason'] == \
            'raw_reference_final_source_guard_failed'
        assert info['stereo_raw_reference_retry']['installed'] is False
        assert not slam.map.stereo_motion
        np.testing.assert_array_equal(slam.verified_stereo_motion[0], old_predictor)
        assert slam.verified_stereo_motion[1] == 0
        assert all(slam.map.landmarks[int(lid)].misses == 0 for lid in keyframe_ids if lid >= 0)
        np.testing.assert_array_equal(slam.map.keyframes[slam.last_keyframe].landmark_ids,
                                      keyframe_ids)
        if 'stereo_pose_arbitration' in info:
            assert info['stereo_pose_arbitration']['choice'] == 'map'
            assert info['stereo_pose_arbitration']['reason'] == \
                'raw_reference_final_source_guard_failed'
    finally:
        slam.close()


def test_post_helper_calibration_change_is_rejected_before_candidate_selection(monkeypatch):
    slam = camera(enabled=True, stereo_raw_reference_retry=True)
    case = _known_motion_case(slam)
    image = _start_then_break_configured_reference(slam, case)
    original_guard = slam._hard_reference_retention_guard
    guard_calls = []

    def mutate_after_fit(*args, **kwargs):
        guard_calls.append(True)
        if len(guard_calls) == 2:
            slam.stereo.Q[3, 3] += .01
        return original_guard(*args, **kwargs)

    slam._hard_reference_retention_guard = mutate_after_fit
    prior = np.eye(4); prior[0, 3] = .125
    slam.verified_stereo_motion = (prior.copy(), 0)
    keyframe_ids = slam.map.keyframes[slam.last_keyframe].landmark_ids.copy()
    try:
        pose, info = slam.process(1, image, image)
        assert pose[2, 3] == pytest.approx(1.2)
        assert len(guard_calls) == 2
        assert not slam.map.stereo_motion
        np.testing.assert_array_equal(slam.verified_stereo_motion[0], prior)
        assert slam.verified_stereo_motion[1] == 0
        assert all(slam.map.landmarks[int(lid)].misses == 0
                   for lid in keyframe_ids if lid >= 0)
        retry = info['stereo_raw_reference_retry']
        assert not retry['eligible'] and not retry.get('installed', False)
        assert retry['reason'] == 'stale_calibration_identity'
        # This is before the late retention transaction, so no final-guard
        # arbitration rollback claim is appropriate.
        if 'stereo_pose_arbitration' in info:
            assert info['stereo_pose_arbitration']['choice'] != 'independent'
    finally:
        slam.close()


def test_disabled_retry_keeps_configured_failure_path_unchanged(monkeypatch):
    slam = camera(enabled=False, stereo_raw_reference_retry=False)
    case = _known_motion_case(slam)
    image = _start_then_break_configured_reference(slam, case)
    monkeypatch.setattr(module, 'estimate_stereo_reference',
                        lambda *args, **kwargs: None)
    try:
        pose, info = slam.process(1, image, image)
        assert pose[2, 3] == pytest.approx(1.2)
        assert 'stereo_raw_reference_retry' not in info
        assert slam.previous_supported_stereo is None
        assert not slam.map.stereo_motion
    finally:
        slam.close()


def test_raw_retry_config_and_feature_cache_mode_are_explicit():
    assert MappingConfig().stereo_raw_reference_retry is False
    with pytest.raises(ValueError, match='two calibrated cameras'):
        SharedSlam(K, config=MappingConfig(stereo_raw_reference_retry=True))
    q = np.array([[1, 0, 0, -K[0, 2]], [0, 1, 0, -K[1, 2]],
                  [0, 0, 0, K[0, 0]], [0, 0, 1/BASELINE, -OFFSET/BASELINE]])
    stereo = module.StereoCamera(None, q, BASELINE)
    disabled = SharedSlam(K, stereo=stereo)
    enabled = SharedSlam(K, stereo=stereo,
                         config=MappingConfig(stereo_raw_reference_retry=True))
    try:
        assert extraction_signature(disabled, cv) != extraction_signature(enabled, cv)
    finally:
        disabled.close()
        enabled.close()


def test_raw_retry_cli_requirements_and_runner_mode_identity(tmp_path):
    def run_cli(script, *arguments):
        return subprocess.run(
            [sys.executable, str(REPO / script), *map(str, arguments)],
            cwd=REPO, capture_output=True, text=True, timeout=30)

    result = run_cli('src/main.py', '--stereo-raw-reference-retry')
    assert result.returncode == 2
    assert '--stereo-raw-reference-retry requires --slam --stereo' in result.stderr
    result = run_cli('src/main.py', '--slam', '--stereo', '--stereo-raw-reference-retry')
    assert result.returncode == 2
    assert '--stereo-raw-reference-retry requires --slam --stereo' not in result.stderr
    assert '--stereo requires --data-root' in result.stderr
    result = run_cli('scripts/evaluate_shared_slam.py', '--stereo-raw-reference-retry',
                     '--output', tmp_path / 'evaluation')
    assert result.returncode == 2
    assert '--stereo-raw-reference-retry requires --stereo' in result.stderr
    result = run_cli('scripts/evaluate_shared_slam.py', '--stereo',
                     '--stereo-raw-reference-retry', '--output', tmp_path / 'evaluation')
    assert result.returncode == 2
    assert '--stereo-raw-reference-retry requires --stereo' not in result.stderr

    scripts = str(REPO / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    path = REPO / 'scripts' / 'run_development_tests.py'
    spec = importlib.util.spec_from_file_location('raw_retry_dev_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    disabled = runner.current_mapping_configuration('bundle', 'supported')
    enabled = runner.current_mapping_configuration(
        'bundle', 'supported', stereo_raw_reference_retry=True)
    assert disabled['stereo_raw_reference_retry'] is False
    assert enabled['stereo_raw_reference_retry'] is True

    identity = {
        'runtime_identity': {'version': 1}, 'variant': 'bundle',
        'sequence': '04', 'frames': 80, 'stereo_pose_arbitration': False,
        'stereo_raw_reference_retry': True, 'cached': False,
        'performance': {'matching_backend': 'cpu', 'retrieval': 'current',
                        'cpu_optimizations': True, 'opencv_threads': 1},
        'stereo_depth_policy': 'supported', 'input': 'input-hash',
        'reference': 'reference-hash',
    }
    report = {
        'status': 'completed', 'sequence': '04', 'frames': 80, 'stereo': True,
        'coverage': 'partial', 'development_identity': {
            **identity, 'stereo_raw_reference_retry': False,
        },
        'feature_cache': {'enabled': False},
        'configuration': disabled,
        'performance_configuration': {
            'matching_backend': 'cpu', 'retrieval': 'current',
            'cpu_optimizations': True, 'profile': False,
        },
        'matching_backend': {'requested': 'cpu'}, 'opencv_threads': 1,
        'stereo_depth_policy': 'supported', 'elapsed_s': 8.0,
    }
    evidence = tmp_path / 'old-off-evaluation.json'
    evidence.write_text(json.dumps(report), encoding='utf-8')
    history = runner.inspect_timing_history(
        [evidence], identity, 'partial', enabled, 'revision')
    assert not history['accepted']
    assert 'stereo-raw-reference-retry mode' in history['rejected'][0]['reason']
    off_identity = {**identity, 'stereo_raw_reference_retry': False}
    assert not runner.reusable({'development_identity': identity,
                                'status': 'completed', 'frames': 80}, off_identity)
