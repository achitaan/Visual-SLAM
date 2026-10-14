"""Reserved evidence may beat a hard map conflict before full-pool retry."""
import numpy as np

import mapping_geometry as geometry
from test_stereo_arbitration_tracking import (
    BASELINE, K, OFFSET, camera, scene, verified,
)


def _pose(z):
    value = np.eye(4)
    value[2, 3] = z
    return value


def test_heldout_half_fit_can_win_hard_conflict_and_keep_map_connected(monkeypatch):
    slam = camera()
    pixels, world, descriptors = scene()
    fit_calls = []
    full_calls = []

    def extract(image, right):
        index = len(slam.map.poses)
        truth = _pose(.5 * index)
        observed, depth = geometry.project(world, truth, K)
        camera_points = (world - truth[:3, 3]) @ truth[:3, :3]
        disparity = K[0, 0] * BASELINE / depth + OFFSET
        slam.current_disparity = np.full(image.shape[:2], disparity[0], np.float32)
        return observed, descriptors.copy(), camera_points, observed[:, 0] - disparity

    def match(first, second):
        count = min(len(first), len(second))
        return np.column_stack((np.arange(count), np.arange(count))).astype(np.int32)

    def fit(source, target, matrix, **kwargs):
        pairs = kwargs['matcher'](source.descriptors, target.descriptors)
        fit_calls.append(len(pairs))
        return verified(_pose(.5), pairs)

    def map_track(image_pixels, image_descriptors, size):
        frame = len(slam.map.poses)
        context = slam._arbitration_context
        associations = {}
        tracks = []
        for source_id, target_id in context['fit']:
            landmark_id = int(context['previous'].landmark_ids[source_id])
            assert landmark_id >= 0
            associations[int(target_id)] = landmark_id
            tracks.append((landmark_id, image_pixels[target_id].copy()))
            context['map_fit_landmarks'].add(landmark_id)
            context['map_fit_targets'].add(int(target_id))
        slam.accepted_tracks = tracks
        map_pose = _pose(1.2 if frame == 1 else .5 * frame)
        return (map_pose, associations), {
            'num_matches': len(associations), 'num_inliers': len(associations),
            'valid_3d': len(associations), 'inlier_ratio': 1.0, 'tracking_ok': True,
        }

    slam._extract = extract
    slam._match = match
    slam._track = map_track
    slam._estimate_full_supported_reference = lambda *args, **kwargs: (
        full_calls.append(args) or (_ for _ in ()).throw(
            AssertionError('a held-out win must bypass full-pool retry')))
    monkeypatch.setattr('shared_slam.estimate_stereo_reference', fit)
    image = np.zeros((376, 1241), np.uint8)
    try:
        slam.process(0, image, image)
        first_pose, first = slam.process(1, image, image)

        expected_first = _pose(.5)
        np.testing.assert_allclose(first_pose, expected_first, atol=1e-10)
        assert first['map_pose_rejected_for_stereo_conflict']
        report = first['stereo_pose_arbitration']
        assert report['choice'] == 'independent', report
        assert report['reason'] == 'reserved_stereo_evidence_improved'
        assert report['fit_source'] == 'reserved_supported_training_rows'
        assert report['held_out_arbitration_used'] is True
        assert report['independent']['eligible']
        assert report['independent']['cost'] < report['map']['cost']
        assert report['association_validation']['eligible']
        assert report['association_validation']['retained_landmarks'] >= 15
        assert first['stereo_pose_arbitration_before_bundle']['eligible']
        assert first['stereo_pose_arbitration_after_bundle']['eligible']
        assert len(slam.map.keyframes) == 1
        first_ids = {landmark_id for landmark_id, _ in slam.accepted_tracks}
        assert len(first_ids) == report['association_validation']['retained_landmarks']
        assert not full_calls
        assert len(fit_calls) == 1
        np.testing.assert_allclose(slam.map.stereo_motion[(0, 1)], _pose(.5), atol=1e-10)
        np.testing.assert_allclose(slam.verified_stereo_motion[0], _pose(.5), atol=1e-10)
        assert np.isfinite(slam.map.stereo_motion[(0, 1)]).all()

        # The next ordinary low-support cadence keyframe must reconnect the
        # same preexisting map landmarks, rather than creating duplicates.
        second_pose, second = slam.process(2, image, image)
        np.testing.assert_allclose(second_pose, _pose(1.0), atol=1e-10)
        assert second['tracking_ok']
        assert slam.map.keyframes[slam.last_keyframe].frame == 2
        keyframe = slam.map.keyframes[slam.last_keyframe]
        connected_ids = {int(value) for value in keyframe.landmark_ids if value >= 0}
        assert first_ids <= connected_ids
        expected_links = {
            int(target_id): int(slam._arbitration_context['previous'].landmark_ids[source_id])
            for source_id, target_id in slam._arbitration_context['fit']
            if slam._arbitration_context['previous'].landmark_ids[source_id] >= 0
        }
        assert expected_links
        for feature, landmark_id in expected_links.items():
            assert keyframe.landmark_ids[feature] == landmark_id
        np.testing.assert_allclose(slam.map.stereo_motion[(1, 2)], _pose(.5), atol=1e-10)
        assert not full_calls
    finally:
        slam.close()


def test_eligible_but_higher_cost_half_candidate_still_runs_full_retry(monkeypatch):
    slam = camera()
    xs = [450., 520., 590., 660., 730.]
    ys = [50., 100., 150., 200., 260., 320.]
    central = np.array([(x, y) for y in ys for x in xs], np.float32)
    anchors = np.array([(100., 50.), (1100., 50.),
                        (100., 330.), (1100., 330.)], np.float32)
    pixels = np.vstack((central, anchors))
    world = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T * 80.
    descriptors = np.eye(len(pixels), 128, dtype=np.float32)
    fit_calls = []
    full_calls = []

    def extract(image, right):
        index = len(slam.map.poses)
        selected_pixels = pixels if index == 0 else central
        selected_world = world if index == 0 else world[:len(central)]
        depth = selected_world[:, 2]
        disparity = K[0, 0] * BASELINE / depth + OFFSET
        slam.current_disparity = np.full(image.shape[:2], disparity[0], np.float32)
        return selected_pixels.copy(), descriptors[:len(selected_pixels)].copy(), \
            selected_world.copy(), selected_pixels[:, 0] - disparity

    def match(first, second):
        count = min(len(first), len(second))
        return np.column_stack((np.arange(count), np.arange(count))).astype(np.int32)

    def fit(source, target, matrix, **kwargs):
        pairs = kwargs['matcher'](source.descriptors, target.descriptors)
        fit_calls.append(len(pairs))
        return verified(_pose(.6), pairs)

    def map_track(image_pixels, image_descriptors, size):
        context = slam._arbitration_context
        associations = {}
        tracks = []
        for source_id, target_id in context['fit']:
            landmark_id = int(context['previous'].landmark_ids[source_id])
            assert landmark_id >= 0
            associations[int(target_id)] = landmark_id
            tracks.append((landmark_id, image_pixels[target_id].copy()))
            context['map_fit_landmarks'].add(landmark_id)
            context['map_fit_targets'].add(int(target_id))
        slam.accepted_tracks = tracks
        return (np.eye(4), associations), {
            'num_matches': len(associations), 'num_inliers': len(associations),
            'valid_3d': len(associations), 'inlier_ratio': 1.0, 'tracking_ok': True,
        }

    def full_fit(source, target, previous_index, index,
                 previous_frame, measured_frame, size):
        full_calls.append((previous_index, index))
        pairs = np.column_stack((np.arange(len(central)), np.arange(len(central))))
        return verified(np.eye(4), pairs), {
            'eligible': True, 'reason': 'full_supported_reference_verified',
            'fit_source': 'full_supported_reference',
            'fit_depth_policy': 'supported_raw',
        }

    slam._extract = extract
    slam._match = match
    slam._track = map_track
    slam._estimate_full_supported_reference = full_fit
    monkeypatch.setattr('shared_slam.estimate_stereo_reference', fit)
    image = np.zeros((376, 1241), np.uint8)
    try:
        slam.process(0, image, image)
        pose, info = slam.process(1, image, image)
        np.testing.assert_allclose(pose, np.eye(4), atol=1e-10)
        assert fit_calls == [15]
        assert full_calls == [(0, 1)]
        pre_score = info['full_supported_reference_fallback']['pre_full_reserved_arbitration']
        assert pre_score['choice'] == 'map'
        assert pre_score['reason'] == 'no_strict_cost_improvement'
        assert pre_score['map']['eligible'] and pre_score['independent']['eligible']
        assert pre_score['independent']['inliers'] >= 15
        assert pre_score['independent']['cost'] > pre_score['map']['cost']
        assert info['stereo_pose_arbitration']['fit_source'] == 'full_supported_reference'
        assert info['stereo_pose_arbitration']['held_out_arbitration_used'] is False
        np.testing.assert_allclose(slam.map.stereo_motion[(0, 1)], np.eye(4), atol=1e-10)
        np.testing.assert_allclose(slam.verified_stereo_motion[0], np.eye(4), atol=1e-10)
        assert slam.previous_supported_stereo.frame == 1
    finally:
        slam.close()
