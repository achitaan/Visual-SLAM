"""Guard reverse-verified hard-conflict map reconnection at a fixed pose."""
import numpy as np
import pytest

import shared_slam as module
from loop_geometry import StereoLoopFrame
from stereo_pose_arbitration import SupportedStereoFrame
from test_stereo_arbitration_tracking import (
    BASELINE, K, OFFSET, camera, record, scene, verified,
)


def _guard_case(enabled=True):
    slam = camera(enabled=enabled)
    previous = record(slam, 0)
    slam.previous_supported_stereo = previous
    slam.previous_stereo_geometry = (
        StereoLoopFrame(previous.pixels, previous.points,
                        previous.descriptors, previous.image_size),
        0,
    )
    slam.map.record(np.eye(4), 'tracking')
    current = record(slam, 1)
    slam.current_supported_stereo = current
    disparity = K[0, 0] * BASELINE / 20. + OFFSET
    slam.current_disparity = np.full((376, 1241), disparity, np.float32)
    measured = StereoLoopFrame(current.pixels, current.points,
                               current.descriptors, current.image_size)
    measurement = np.eye(4)
    source_pose = slam.map.poses[0].copy()
    reference_pose = source_pose @ measurement
    return (slam, current, measured, verified(measurement, np.c_[np.arange(24),
                                                                  np.arange(24)]),
            source_pose, reference_pose, slam.map.revision)


def _check(case, *, index=1, previous_index=0, previous_frame=None,
           measured=None, verified_result=None, source_pose=None,
           reference_pose=None, source_revision=None):
    slam, current, measured_default, verified_default, source_default, reference_default, rev_default = case
    if previous_frame is None:
        previous_frame = slam.previous_stereo_geometry[0]
    with slam.map.lock:
        return slam._hard_reference_retention_guard(
            index, previous_index, previous_frame,
            measured_default if measured is None else measured,
            verified_default if verified_result is None else verified_result,
            source_default if source_pose is None else source_pose,
            reference_default if reference_pose is None else reference_pose,
            rev_default if source_revision is None else source_revision,
            (1241, 376),
        )


def test_context_absent_reverse_reference_retains_map_links_and_normal_cadence(monkeypatch):
    slam = camera()
    pixels, points, descriptors = scene()
    disparity_value = K[0, 0] * BASELINE / 20. + OFFSET
    wrong = np.eye(4)
    angle = np.radians(2.0)
    wrong[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]

    def extract(image, right):
        slam.current_disparity = np.full(image.shape[:2], disparity_value, np.float32)
        return pixels.copy(), descriptors.copy(), points.copy(), pixels[:, 0] - disparity_value

    def no_reservation(index, current):
        return None, {'choice': 'map', 'reason': 'insufficient_reserved_support'}

    def track(*args):
        identifiers = slam.map.keyframes[slam.last_keyframe].landmark_ids.tolist()
        identifiers = [int(value) for value in identifiers if value >= 0]
        slam.accepted_tracks = [(ident, pixels[i].copy()) for i, ident in enumerate(identifiers)]
        slam._last_map_track_inlier_misses = {
            ident: int(slam.map.landmarks[ident].misses) for ident in identifiers
        }
        return (wrong.copy(), {i: ident for i, ident in enumerate(identifiers)}), {
            'num_matches': len(identifiers), 'num_inliers': len(identifiers),
            'valid_3d': len(identifiers), 'tracking_ok': True,
        }

    slam._extract = extract
    slam._prepare_stereo_arbitration = no_reservation
    slam._track = track
    monkeypatch.setattr(module, 'estimate_stereo_reference',
                        lambda *args, **kwargs: verified(np.eye(4), np.c_[np.arange(24), np.arange(24)]))
    image = np.zeros((376, 1241), np.uint8)
    try:
        slam.process(0, image, image)
        original_ids = slam.map.keyframes[slam.last_keyframe].landmark_ids.copy()
        pose, info = slam.process(1, image, image)
        np.testing.assert_allclose(pose, np.eye(4), atol=1e-12)
        assert info['map_pose_rejected_for_stereo_conflict']
        assert info['reference_association_validation']['eligible']
        assert not info['reference_association_validation']['reservation_context_available']
        assert info['reference_association_validation']['independent_fit_source'] == \
            'map_coordinate_independent_stereo_reference'
        assert info['reference_association_validation']['independent_fit_depth_policy'] == 'supported'
        assert info['reference_association_validation']['prediction_seed_supplied']
        assert slam.map.keyframes[slam.last_keyframe].frame == 0
        assert len(slam.accepted_tracks) >= 15
        assert set(lid for lid, _ in slam.accepted_tracks) == set(original_ids.tolist())

        # Low support activates the ordinary two-frame keyframe cadence. The
        # very same old map IDs must be linked in that later keyframe.
        _, next_info = slam.process(2, image, image)
        assert next_info['reference_association_validation']['eligible']
        assert slam.map.keyframes[slam.last_keyframe].frame == 2
        np.testing.assert_array_equal(
            slam.map.keyframes[slam.last_keyframe].landmark_ids[:len(original_ids)],
            original_ids,
        )
    finally:
        slam.close()


def test_reference_connection_support_failure_forces_kf_and_ages_once(monkeypatch):
    slam = camera()
    pixels, points, descriptors = scene()
    disparity_value = K[0, 0] * BASELINE / 20. + OFFSET
    wrong = np.eye(4)
    angle = np.radians(2.0)
    wrong[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]

    def extract(image, right):
        # The map remains geometrically plausible, but there is no fresh raw
        # right-image measurement with which to retain its old associations.
        value = disparity_value if len(slam.map.poses) == 0 else np.nan
        slam.current_disparity = np.full(image.shape[:2], value, np.float32)
        return pixels.copy(), descriptors.copy(), points.copy(), pixels[:, 0] - disparity_value

    slam._extract = extract
    slam._prepare_stereo_arbitration = lambda index, current: (
        None, {'choice': 'map', 'reason': 'insufficient_reserved_support'})

    def track(*args):
        identifiers = [int(value) for value in
                       slam.map.keyframes[slam.last_keyframe].landmark_ids.tolist() if value >= 0]
        previous = {}
        for ident in identifiers:
            landmark = slam.map.landmarks[ident]
            landmark.misses = 3
            previous[ident] = landmark.misses
            landmark.misses = 0  # mimic _track's provisional inlier reset
        slam._last_map_track_inlier_misses = previous
        slam.accepted_tracks = [(ident, pixels[i].copy()) for i, ident in enumerate(identifiers)]
        return (wrong.copy(), {i: ident for i, ident in enumerate(identifiers)}), {
            'num_matches': len(identifiers), 'num_inliers': len(identifiers),
            'valid_3d': len(identifiers), 'tracking_ok': True,
        }

    slam._track = track
    monkeypatch.setattr(module, 'estimate_stereo_reference',
                        lambda *args, **kwargs: verified(np.eye(4), np.c_[np.arange(24), np.arange(24)]))
    image = np.zeros((376, 1241), np.uint8)
    try:
        slam.process(0, image, image)
        original_ids = slam.map.keyframes[slam.last_keyframe].landmark_ids.copy()
        _, info = slam.process(1, image, image)
        report = info['reference_association_validation']
        assert not report['eligible']
        assert report['association_validation']['retained_landmarks'] == 0
        assert report['current_right_measurement'] == \
            'supported_only_raw_disparity_at_actual_observation'
        assert slam.map.keyframes[slam.last_keyframe].frame == 1
        assert slam.accepted_tracks == []
        assert all(slam.map.landmarks[int(ident)].misses == 4 for ident in original_ids)
    finally:
        slam.close()


@pytest.mark.parametrize('mutation,reason', [
    ('endpoint', 'endpoint_pixels_or_descriptors_mismatch'),
    ('calibration', 'stale_calibration_identity'),
    ('measurement_pose', 'invalid_se3_pose'),
    ('complex_measurement_pose', 'invalid_se3_pose'),
    ('source_pose', 'invalid_se3_pose'),
    ('held_source', 'source_frame_not_accepted'),
    ('gap_four', 'endpoint_or_frame_gap_mismatch'),
    ('disparity_shape', 'invalid_current_disparity_shape'),
    ('out_of_domain', 'malformed_target_record'),
    ('malformed_record', 'malformed_target_record'),
    ('complex_record', 'malformed_target_record'),
    ('forward_only', 'reverse_verification_or_feature_flag_missing'),
    ('bad_endpoint_type', 'malformed_endpoint_or_revision_type'),
    ('bad_revision_type', 'malformed_endpoint_or_revision_type'),
    ('revision_changed', 'map_revision_changed'),
    ('source_pose_changed', 'source_pose_epoch_changed'),
    ('source_geometry_size', 'endpoint_image_size_mismatch'),
])
def test_hard_reference_guard_fails_closed(mutation, reason):
    case = _guard_case()
    slam, current, measured, result, source_pose, reference_pose, revision = case
    index = 1
    previous_index = 0
    if mutation == 'endpoint':
        changed = measured.pixels.copy(); changed[0, 0] += .25
        measured = StereoLoopFrame(changed, measured.points, measured.descriptors, measured.image_size)
    elif mutation == 'calibration':
        slam.K[0, 0] += 1.
    elif mutation == 'measurement_pose':
        bad = result['measurement'].copy(); bad[0, 0] = 2.
        result = {**result, 'measurement': bad}
    elif mutation == 'complex_measurement_pose':
        bad = result['measurement'].astype(complex); bad[0, 0] += 1j
        result = {**result, 'measurement': bad}
    elif mutation == 'source_pose':
        source_pose = source_pose.copy(); source_pose[0, 0] = 2.
    elif mutation == 'held_source':
        slam.map.statuses[0] = 'lost'
    elif mutation == 'gap_four':
        for _ in range(3):
            slam.map.record(np.eye(4), 'lost')
        index = 4
        current = SupportedStereoFrame(
            current.pixels, current.descriptors, current.points, current.right_u,
            current.landmark_ids, index, current.image_size, current.calibration_identity)
        slam.current_supported_stereo = current
    elif mutation == 'disparity_shape':
        slam.current_disparity = np.zeros((375, 1241), np.float32)
    elif mutation == 'out_of_domain':
        changed = current.pixels.copy(); changed[0, 0] = 1241.
        current = SupportedStereoFrame(
            changed, current.descriptors, current.points, current.right_u,
            current.landmark_ids, current.frame, current.image_size, current.calibration_identity)
        slam.current_supported_stereo = current
    elif mutation == 'malformed_record':
        current = SupportedStereoFrame(
            current.pixels, current.descriptors, current.points[:, :2], current.right_u,
            current.landmark_ids, current.frame, current.image_size, current.calibration_identity)
        slam.current_supported_stereo = current
    elif mutation == 'complex_record':
        bad_points = current.points.astype(complex); bad_points[0, 0] += 1j
        current = SupportedStereoFrame(
            current.pixels, current.descriptors, bad_points, current.right_u,
            current.landmark_ids, current.frame, current.image_size, current.calibration_identity)
        slam.current_supported_stereo = current
    elif mutation == 'forward_only':
        result = {**result, 'reverse_checked': False}
    elif mutation == 'bad_endpoint_type':
        index = True
    elif mutation == 'bad_revision_type':
        revision = '0'
    elif mutation == 'revision_changed':
        revision += 1
    elif mutation == 'source_pose_changed':
        slam.map.poses[0][0, 3] += .01
    elif mutation == 'source_geometry_size':
        old, old_index = slam.previous_stereo_geometry
        slam.previous_stereo_geometry = (
            StereoLoopFrame(old.pixels, old.points, old.descriptors, (640, 480)), old_index)
    try:
        report = _check(
            case, index=index, previous_index=previous_index, measured=measured,
            verified_result=result, source_pose=source_pose,
            reference_pose=reference_pose, source_revision=revision)
        assert not report['eligible']
        assert report['reason'] == reason
    finally:
        slam.close()


def test_supported_depth_nan_is_allowed_but_nonproper_composed_pose_is_rejected():
    case = _guard_case()
    slam, current, _, result, source_pose, _, revision = case
    points = current.points.copy(); points[0] = np.nan
    slam.current_supported_stereo = SupportedStereoFrame(
        current.pixels, current.descriptors, points, current.right_u,
        current.landmark_ids, current.frame, current.image_size, current.calibration_identity)
    reference_pose = source_pose @ result['measurement']
    bad_reference = reference_pose.copy(); bad_reference[:3, :3] *= 1.01
    try:
        report = _check(case, reference_pose=bad_reference, source_revision=revision)
        assert not report['eligible'] and report['reason'] == 'invalid_se3_pose'
        # NaN supported depth itself is expected on unsupported pixels. With a
        # proper pose it does not invalidate endpoint provenance.
        report = _check(case, source_revision=revision)
        assert report['eligible']
    finally:
        slam.close()


def test_default_off_guard_is_inert():
    case = _guard_case(enabled=False)
    slam = case[0]
    try:
        report = _check(case)
        assert not report['eligible']
        assert report['reason'] == 'reverse_verification_or_feature_flag_missing'
    finally:
        slam.close()
