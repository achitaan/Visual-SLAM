"""Reserved fits that conflict with the map are retried on the full raw pool."""
import numpy as np
import pytest

import shared_slam as module
import mapping_geometry as geometry
from loop_geometry import StereoLoopFrame
from stereo_pose_arbitration import SupportedStereoFrame
from test_stereo_arbitration_tracking import (
    BASELINE, K, OFFSET, camera, record, scene, verified,
)


def _pose_with_depth(z):
    pose = np.eye(4)
    pose[2, 3] = z
    return pose


def _fit_result(pose, pairs):
    return verified(pose.copy(), pairs)


def _install_process_case(slam, monkeypatch, *, map_z, half_z, full_result):
    pixels, world, descriptors = scene()
    truth = _pose_with_depth(.5)
    calls = []
    current_pixels = [pixels.copy()]

    def extract(image, right):
        frame = len(slam.map.poses)
        if frame == 0:
            observed, points, depth = pixels.copy(), world.copy(), np.full(len(world), 20.)
        else:
            observed, depth = geometry.project(world, truth, K)
            points = (world-truth[:3, 3]) @ truth[:3, :3]
        current_pixels[0] = observed.copy()
        disparity = K[0, 0] * BASELINE / depth + OFFSET
        slam.current_disparity = np.broadcast_to(
            disparity[0], image.shape[:2]).copy().astype(np.float32)
        return observed.copy(), descriptors.copy(), points.copy(), observed[:, 0]-disparity

    def track(*args):
        identifiers = slam.map.keyframes[slam.last_keyframe].landmark_ids.tolist()
        identifiers = [int(value) for value in identifiers if value >= 0]
        misses = {}
        for ident in identifiers:
            landmark = slam.map.landmarks[ident]
            misses[ident] = int(landmark.misses)
            landmark.misses = 0  # _track's provisional successful-inlier reset.
        slam._last_map_track_inlier_misses = misses
        slam.accepted_tracks = [(ident, current_pixels[0][i].copy())
                                for i, ident in enumerate(identifiers)]
        pose = _pose_with_depth(map_z)
        return (pose, {i: ident for i, ident in enumerate(identifiers)}), {
            'num_matches': len(identifiers), 'num_inliers': len(identifiers),
            'valid_3d': len(identifiers), 'inlier_ratio': 1., 'tracking_ok': True,
        }

    def match(first, second):
        return np.column_stack((np.arange(min(len(first), len(second))),
                                np.arange(min(len(first), len(second))))).astype(np.int32)

    def estimate(source, target, matrix, **kwargs):
        pairs = kwargs['matcher'](source.descriptors, target.descriptors)
        calls.append({'count': len(pairs), 'initial_pose': kwargs.get('initial_pose')})
        if len(pairs) == 24:
            return _fit_result(_pose_with_depth(half_z), pairs)
        assert len(pairs) == 48
        if full_result is None:
            return None
        return _fit_result(full_result, pairs)

    slam._extract = extract
    slam._track = track
    slam._match = match
    monkeypatch.setattr(module, 'estimate_stereo_reference', estimate)
    return truth, calls


def _run_two_frames(slam):
    image = np.zeros((376, 1241), np.uint8)
    slam.process(0, image, image)
    return slam.process(1, image, image)


def test_wrong_half_pool_is_replaced_when_full_pool_agrees_with_map(monkeypatch):
    slam = camera()
    truth, calls = _install_process_case(
        slam, monkeypatch, map_z=.5, half_z=1.25, full_result=_pose_with_depth(.5))
    slam._arbitrate_supported_pose = lambda *args: pytest.fail(
        'full-pool retry must not run held-out arbitration')
    try:
        pose, info = _run_two_frames(slam)
        np.testing.assert_allclose(pose, truth, atol=1e-12)
        assert [call['count'] for call in calls] == [24, 48]
        assert all(call['initial_pose'] is None for call in calls)
        report = info['stereo_pose_arbitration']
        assert report['reason'] == 'full_supported_reference_agrees_with_map'
        assert report['fit_source'] == 'full_supported_reference'
        assert report['fit_depth_policy'] == 'supported_raw'
        assert report['held_out_arbitration_used'] is False
        assert info['full_supported_reference_fallback']['half_pool_conflicted_with_map']
        assert 'stereo_pose_arbitration_before_bundle' not in info
        assert 'stereo_pose_arbitration_after_bundle' not in info
        np.testing.assert_allclose(slam.map.stereo_motion[(0, 1)], truth, atol=1e-12)
        np.testing.assert_allclose(slam.verified_stereo_motion[0], truth, atol=1e-12)
    finally:
        slam.close()


def test_conflicting_full_pool_pose_uses_fixed_pose_connection_guard(monkeypatch):
    slam = camera()
    truth, calls = _install_process_case(
        slam, monkeypatch, map_z=1.2, half_z=1.9, full_result=_pose_with_depth(.5))
    slam._arbitrate_supported_pose = lambda *args: pytest.fail(
        'consumed holdout rows cannot score the full-pool result')
    try:
        pose, info = _run_two_frames(slam)
        np.testing.assert_allclose(pose, truth, atol=1e-12)
        assert [call['count'] for call in calls] == [24, 48]
        assert info['map_pose_rejected_for_stereo_conflict']
        validation = info['reference_association_validation']
        assert validation['eligible'], validation['association_validation']
        assert validation['independent_fit_source'] == 'full_supported_reference'
        assert validation['independent_fit_depth_policy'] == 'supported_raw'
        assert validation['fit_depth_policy'] == 'supported_raw'
        assert validation['prediction_seed_supplied'] is False
        assert validation['held_out_arbitration_used'] is False
        assert validation['reservation_context_available'] is True
        assert info['stereo_pose_arbitration']['held_out_arbitration_used'] is False
        assert slam.map.keyframes[slam.last_keyframe].frame == 0
        assert len(slam.accepted_tracks) >= 15
        np.testing.assert_allclose(slam.map.stereo_motion[(0, 1)], truth, atol=1e-12)
    finally:
        slam.close()


def test_full_pool_failure_abstains_ages_map_once_and_installs_no_half_edge(monkeypatch):
    slam = camera()
    _, calls = _install_process_case(
        slam, monkeypatch, map_z=1.2, half_z=.5, full_result=None)
    slam._keyframe_stereo_reference = lambda *args, **kwargs: (None, {})
    slam._relocalize = lambda *args, **kwargs: (None, {})
    try:
        image = np.zeros((376, 1241), np.uint8)
        slam.process(0, image, image)
        identifiers = [int(value) for value in
                       slam.map.keyframes[slam.last_keyframe].landmark_ids if value >= 0]
        prior_prediction = _pose_with_depth(.2)
        slam.verified_stereo_motion = (prior_prediction.copy(), 0)
        _, info = slam.process(1, image, image)
        assert [call['count'] for call in calls] == [24, 48]
        assert not info['tracking_ok'], info
        assert info['stereo_pose_arbitration']['choice'] == 'abstain'
        assert info['stereo_pose_arbitration']['reason'] == 'full_supported_reference_failed'
        assert info['full_supported_reference_fallback']['reason'] == \
            'full_supported_reference_failed'
        assert not slam.map.stereo_motion
        np.testing.assert_array_equal(slam.verified_stereo_motion[0], prior_prediction)
        assert slam.verified_stereo_motion[1] == 0
        assert slam.previous_supported_stereo.frame == 0
        assert all(slam.map.landmarks[ident].misses == 1 for ident in identifiers)
    finally:
        slam.close()


def test_full_pool_failure_can_recover_geometrically_without_half_motion(monkeypatch):
    slam = camera()
    truth, calls = _install_process_case(
        slam, monkeypatch, map_z=1.2, half_z=.5, full_result=None)
    image = np.zeros((376, 1241), np.uint8)
    slam._keyframe_stereo_reference = lambda *args, **kwargs: (
        (truth.copy(), {}), {'pose_source': 'keyframe_stereo_reference'})
    try:
        slam.process(0, image, image)
        identifiers = [int(value) for value in
                       slam.map.keyframes[slam.last_keyframe].landmark_ids if value >= 0]
        prior_prediction = _pose_with_depth(.2)
        slam.verified_stereo_motion = (prior_prediction.copy(), 0)
        pose, info = slam.process(1, image, image)
        np.testing.assert_allclose(pose, truth, atol=1e-12)
        assert [call['count'] for call in calls] == [24, 48]
        assert info['tracking_ok']
        assert slam.map.statuses[-1] == 'relocalized'
        assert not slam.map.stereo_motion
        np.testing.assert_array_equal(slam.verified_stereo_motion[0], prior_prediction)
        assert slam.verified_stereo_motion[1] == 0
        assert all(slam.map.landmarks[ident].misses == 1 for ident in identifiers)
    finally:
        slam.close()


def test_missing_map_hypothesis_uses_reserved_reference_without_scoring(monkeypatch):
    slam = camera()
    truth, calls = _install_process_case(
        slam, monkeypatch, map_z=.5, half_z=.5, full_result=_pose_with_depth(.5))
    image = np.zeros((376, 1241), np.uint8)
    try:
        slam.process(0, image, image)
        slam._track = lambda *args, **kwargs: (None, {
            'num_matches': 0, 'num_inliers': 0, 'valid_3d': 0,
            'inlier_ratio': 0., 'tracking_ok': False,
        })
        slam._arbitrate_supported_pose = lambda *args: pytest.fail(
            'missing-map recovery has no map candidate to score')
        pose, info = slam.process(1, image, image)
        np.testing.assert_allclose(pose, truth, atol=1e-12)
        assert [call['count'] for call in calls] == [24]
        assert info['stereo_pose_arbitration']['reason'] == 'missing_map_hypothesis'
        assert 'full_supported_reference_fallback' not in info
        assert 'stereo_pose_arbitration_before_bundle' not in info
        assert 'stereo_pose_arbitration_after_bundle' not in info
    finally:
        slam.close()


def _full_pair_case(source_points=None, target_points=None,
                    source_pixels=None, target_pixels=None):
    slam = camera()
    previous = record(slam, 0)
    current = record(slam, 1)
    source_points = previous.points.copy() if source_points is None else source_points
    target_points = current.points.copy() if target_points is None else target_points
    source_pixels = previous.pixels.copy() if source_pixels is None else source_pixels
    target_pixels = current.pixels.copy() if target_pixels is None else target_pixels
    previous = SupportedStereoFrame(
        source_pixels, previous.descriptors, source_points, previous.right_u,
        previous.landmark_ids, 0, previous.image_size, previous.calibration_identity)
    current = SupportedStereoFrame(
        target_pixels, current.descriptors, target_points, current.right_u,
        current.landmark_ids, 1, current.image_size, current.calibration_identity)
    slam.previous_supported_stereo = previous
    slam.previous_stereo_geometry = (
        StereoLoopFrame(previous.pixels, previous.points, previous.descriptors,
                        previous.image_size), 0)
    slam.map.record(np.eye(4), 'tracking')
    slam.current_disparity = np.full((376, 1241), 20., np.float32)
    source_frame = StereoLoopFrame(previous.pixels, previous.points,
                                   previous.descriptors, previous.image_size)
    target_frame = StereoLoopFrame(current.pixels, current.points,
                                   current.descriptors, current.image_size)
    slam._match = lambda first, second: np.column_stack((np.arange(48), np.arange(48)))
    return slam, previous, current, source_frame, target_frame


def test_full_pool_keeps_asymmetric_depth_rows_for_estimator(monkeypatch):
    temporary = camera()
    previous = record(temporary, 0)
    current = record(temporary, 1)
    temporary.close()
    source_points = previous.points.copy()
    target_points = current.points.copy()
    source_points[0] = np.nan
    target_points[1] = np.nan
    slam, source, target, source_frame, target_frame = _full_pair_case(
        source_points=source_points, target_points=target_points)
    seen = {}

    def estimate(source_arg, target_arg, matrix, **kwargs):
        pairs = kwargs['matcher'](source_arg.descriptors, target_arg.descriptors)
        seen['count'] = len(pairs)
        seen['source_nan'] = not np.isfinite(source_arg.points[0]).all()
        seen['target_nan'] = not np.isfinite(target_arg.points[1]).all()
        return _fit_result(np.eye(4), pairs)

    monkeypatch.setattr(module, 'estimate_stereo_reference', estimate)
    try:
        fitted, report = slam._estimate_full_supported_reference(
            source, target, 0, 1, source_frame, target_frame, (1241, 376))
        assert fitted is not None and report['eligible']
        assert seen == {'count': 48, 'source_nan': True, 'target_nan': True}
    finally:
        slam.close()


def test_full_pool_drops_physical_duplicates_and_fails_closed_on_bad_endpoints():
    temporary = camera()
    source = record(temporary, 0)
    target = record(temporary, 1)
    temporary.close()
    source_pixels = source.pixels.copy(); source_pixels[1] = source_pixels[0]
    target_pixels = target.pixels.copy(); target_pixels[1] = target_pixels[0]
    slam, source, target, source_frame, target_frame = _full_pair_case(
        source_pixels=source_pixels, target_pixels=target_pixels)
    try:
        pairs, report = slam._full_supported_match_pairs(
            source, target, 0, 1, source_frame, target_frame, (1241, 376))
        assert report['eligible']
        assert len(pairs) == 46
        assert not {0, 1} & set(pairs[:, 0])
        assert not {0, 1} & set(pairs[:, 1])

        malformed = StereoLoopFrame(source_frame.pixels, source_frame.points,
                                    source_frame.descriptors, None)
        pairs, invalid = slam._full_supported_match_pairs(
            source, target, 0, 1, malformed, target_frame, (1241, 376))
        assert pairs is None
        assert not invalid['eligible']
        assert invalid['reason'] == 'missing_live_supported_endpoint'
    finally:
        slam.close()


def test_full_pool_matcher_must_be_one_to_one():
    slam, source, target, source_frame, target_frame = _full_pair_case()
    slam._match = lambda first, second: np.array([[0, 0], [0, 1]], np.int32)
    try:
        pairs, report = slam._full_supported_match_pairs(
            source, target, 0, 1, source_frame, target_frame, (1241, 376))
        assert pairs is None
        assert not report['eligible']
        assert report['reason'] == 'no_unique_full_supported_matches'
    finally:
        slam.close()


@pytest.mark.parametrize(('mutation', 'reason'), [
    ('forward_only', 'full_supported_reference_not_reverse_checked'),
    ('invalid_feature', 'malformed_full_supported_reference_result'),
])
def test_full_pool_fit_metadata_must_be_valid_and_reverse_checked(
    monkeypatch, mutation, reason,
):
    slam, source, target, source_frame, target_frame = _full_pair_case()

    def estimate(first, second, matrix, **kwargs):
        pairs = kwargs['matcher'](first.descriptors, second.descriptors)
        result = _fit_result(np.eye(4), pairs)
        if mutation == 'forward_only':
            result['reverse_checked'] = False
        else:
            result['target_features'][0] = len(second.pixels)
        return result

    monkeypatch.setattr(module, 'estimate_stereo_reference', estimate)
    try:
        fitted, report = slam._estimate_full_supported_reference(
            source, target, 0, 1, source_frame, target_frame, (1241, 376))
        assert fitted is None
        assert not report['eligible']
        assert report['reason'] == reason
    finally:
        slam.close()
