"""Reserved raw stereo must remain independent through the tracking pipeline."""
import numpy as np
import pytest
import shared_slam as module
import mapping_geometry as geometry
from shared_slam import SharedSlam, MappingConfig, StereoCamera
from stereo_pose_arbitration import SupportedStereoFrame
from loop_geometry import StereoLoopFrame

K = np.array([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1.]])
BASELINE = .54
OFFSET = .35


def camera(enabled=True, **options):
    q = np.array([[1, 0, 0, -K[0, 2]], [0, 1, 0, -K[1, 2]],
                  [0, 0, 0, K[0, 0]], [0, 0, 1/BASELINE, -OFFSET/BASELINE]])
    return SharedSlam(K, StereoCamera(None, q, BASELINE),
                      config=MappingConfig(loop_mode='off', bundle_enabled=False,
                                           stereo_pose_arbitration=enabled, **options))


def scene():
    pixels = np.array([(x, y) for y in [50., 100., 150., 200., 260., 320.]
                       for x in [140., 260., 390., 520., 650., 780., 910., 1080.]], np.float32)
    points = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T*20.
    desc = np.eye(len(pixels), 128, dtype=np.float32)
    return pixels, points, desc


def record(slam, frame=0, duplicate=False):
    pixels, points, desc = scene()
    if duplicate:
        pixels = pixels.copy(); pixels[1] = pixels[0]
    return SupportedStereoFrame(pixels, desc, points,
        pixels[:, 0]-K[0, 0]*BASELINE/20.-OFFSET, np.arange(len(pixels)), frame,
        (1241, 376), slam.stereo_calibration_identity)


def install_previous(slam, previous):
    slam.previous_supported_stereo = previous
    slam.previous_stereo_geometry = (StereoLoopFrame(previous.pixels, previous.points,
                                                    previous.descriptors, previous.image_size), previous.frame)
    slam.map.record(np.eye(4), 'tracking')


def verified(measurement, pairs):
    return dict(measurement=measurement, matches=len(pairs), inliers=len(pairs),
                median_reprojection_px=0., reverse_checked=True,
                target_features=pairs[:, 1].tolist(), bidirectional_refinement={'applied': True})


def test_enabled_monocular_rejected_and_default_off():
    assert not MappingConfig().stereo_pose_arbitration
    with pytest.raises(ValueError, match='two calibrated cameras'):
        SharedSlam(K, config=MappingConfig(stereo_pose_arbitration=True))


def test_cache_hit_reconstructs_supported_only_and_owns_snapshot():
    slam = camera(stereo_depth_policy='verified_fallback')
    try:
        pixels, _, desc = scene()
        slam.current_disparity = np.full((376, 1241), 20., np.float32)
        pixels = pixels.copy(); pixels[0] += .25
        x, y = pixels[0].astype(int)
        slam.current_disparity[y+1, x+1] = -1
        # Cached restored points are intentionally not passed to this producer.
        raw = slam._capture_supported_stereo(0, pixels, desc, (1241, 376))
        assert np.isnan(raw.points[0]).all()
        assert np.isnan(raw.right_u[0])
        assert np.isfinite(raw.points[1:]).all()
        pixels[:] = 0; slam.current_disparity[:] = 80
        assert raw.pixels[1, 0] > 0 and not raw.points.flags.writeable
    finally:
        slam.close()


@pytest.mark.parametrize('duplicate_source,duplicate_target', [(True, False), (False, True), (True, True)])
def test_duplicate_physical_observations_drop_before_split(monkeypatch, duplicate_source, duplicate_target):
    slam = camera()
    try:
        previous = record(slam, duplicate=duplicate_source); current = record(slam, 1, duplicate=duplicate_target)
        install_previous(slam, previous)
        calls = []
        def fit(source, target, matrix, **kwargs):
            pairs = kwargs['matcher'](source.descriptors, target.descriptors)
            calls.append(pairs)
            return verified(np.eye(4), pairs)
        monkeypatch.setattr(module, 'estimate_stereo_reference', fit)
        context, report = slam._prepare_stereo_arbitration(1, current)
        assert context is not None
        assert report['dropped_duplicate_matches'] == 2
        assert not {0, 1} & set(context['fit'][:, 0])
        assert not {0, 1} & set(context['evidence'].source_ids)
        assert not set(context['fit'][:, 0]) & set(context['evidence'].source_ids)
        assert np.array_equal(calls[0], context['fit'])
    finally:
        slam.close()


def test_forward_reverse_and_refinement_receive_only_fit_rows(monkeypatch):
    slam = camera()
    try:
        previous, current = record(slam), record(slam, 1)
        install_previous(slam, previous)
        seen = []
        def pose(points, pixels, *args, **kwargs):
            seen.append((points.copy(), pixels.copy()))
            return np.eye(4), np.arange(len(points)), 0.
        def refinement(measurement, source, target_pixels, target, source_pixels, matrix):
            seen.append((source.copy(), source_pixels.copy()))
            seen.append((target.copy(), target_pixels.copy()))
            return measurement, {'applied': True}
        monkeypatch.setattr(geometry, 'estimate_pose', pose)
        monkeypatch.setattr(geometry, 'refine_bidirectional_stereo', refinement)
        context, _ = slam._prepare_stereo_arbitration(1, current)
        assert context is not None and len(seen) == 4
        expected = previous.points[context['fit'][:, 0]]
        for points, pixels in seen:
            np.testing.assert_array_equal(points, expected)
            assert not any(tuple(p) in {tuple(v) for v in current.pixels[context['evidence'].target_ids]}
                           for p in pixels)
    finally:
        slam.close()


def test_failed_training_or_stale_frame_abstains_before_map_exclusions(monkeypatch):
    slam = camera()
    try:
        install_previous(slam, record(slam))
        monkeypatch.setattr(module, 'estimate_stereo_reference', lambda *args, **kwargs: None)
        context, report = slam._prepare_stereo_arbitration(1, record(slam, 1))
        assert context is None and report['reason'] == 'independent_training_failed'
        slam.previous_stereo_geometry = (slam.previous_stereo_geometry[0], 1)
        assert slam._prepare_stereo_arbitration(2, record(slam, 2))[0] is None
    finally:
        slam.close()


def test_track_excludes_descriptor_and_linked_flow_landmarks(monkeypatch):
    slam = camera()
    try:
        previous = record(slam); current = record(slam, 1)
        install_previous(slam, previous)
        monkeypatch.setattr(module, 'estimate_stereo_reference',
                            lambda source, target, matrix, **kw: verified(np.eye(4), kw['matcher'](None, None)))
        context, _ = slam._prepare_stereo_arbitration(1, current)
        slam._arbitration_context = context
        for j in range(len(previous.pixels)):
            slam.map.add_landmark(previous.points[j], previous.descriptors[j], 0, {})
        # A linked source holdout flow has no detector identity in the map fit.
        slam.previous_tracks = [(1, np.array([10., 10.], np.float32))]
        slam.previous_gray = slam.current_gray = np.zeros((376, 1241), np.uint8)
        seen = []
        def pose(points, pixels, *args, **kwargs):
            seen.append(points.copy())
            return np.eye(4), np.arange(len(points)), 0.
        monkeypatch.setattr(module, 'estimate_pose', pose)
        monkeypatch.setattr(module, 'refine_stereo_map_pose', lambda answer, *args: (answer, {}))
        slam.current_disparity = np.full((376, 1241), K[0, 0]*BASELINE/20.+OFFSET)
        answer, _ = slam._track(current.pixels, current.descriptors, current.image_size)
        assert answer is not None and seen
        assert context['map_fit_landmarks'] == set(context['fit'][:, 0])
        assert not context['map_fit_landmarks'] & set(context['evidence'].landmark_ids)
        assert len(seen[0]) == len(context['fit'])
    finally:
        slam.close()


def test_process_selects_independent_roll_clears_tracks_and_records_provenance(monkeypatch):
    slam = camera()
    pixels, points, desc = scene()
    correct = np.eye(4); correct[2, 3] = 2.1
    current_pixels, depth = geometry.project(points, correct, K)
    current_points = (points-correct[:3, 3]) @ correct[:3, :3]
    wrong = correct.copy(); angle = np.radians(1.1)
    wrong[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    calls = []
    def extract(image, right):
        index = len(slam.map.poses)
        p, xyz, z = (pixels, points, 20.) if index == 0 else (current_pixels, current_points, 17.9)
        disparity = K[0, 0]*BASELINE/z+OFFSET
        slam.current_disparity = np.full(image.shape, disparity)
        return p.copy(), desc.copy(), xyz.copy(), p[:, 0]-disparity
    def fit(source, target, matrix, **kw):
        pairs = kw['matcher'](source.descriptors, target.descriptors)
        calls.append(pairs)
        return verified(correct.copy(), pairs)
    def track(*args):
        context = slam._arbitration_context
        context['map_fit_targets'].update(context['fit'][:, 1])
        context['map_fit_landmarks'].update(context['fit'][:, 0])
        slam.accepted_tracks = [(0, current_pixels[0].copy())]
        return (wrong.copy(), {0: 0}), {'num_matches': 24, 'num_inliers': 24, 'tracking_ok': True}
    slam._extract = extract
    monkeypatch.setattr(module, 'estimate_stereo_reference', fit)
    try:
        image = np.zeros((376, 1241), np.uint8)
        slam.process(0, image, image)
        slam._track = track
        output, info = slam.process(1, image, image)
        np.testing.assert_allclose(output, correct)
        assert info['pose_source'] == 'reserved_stereo_arbitration'
        assert slam.accepted_tracks == []
        assert slam.map.keyframes[slam.last_keyframe].frame == 1
        assert slam.map.keyframes[slam.last_keyframe].landmark_ids[0] != 0
        assert len(calls) == 1
        np.testing.assert_array_equal(slam.map.stereo_motion[(0, 1)], correct)
        assert info['stereo_pose_arbitration']['measurement_provenance'] == 'immutable_supported_extraction'
        assert not info['stereo_pose_arbitration']['association_validation']['eligible']
        assert info['stereo_pose_arbitration']['association_validation']['reason'] == 'insufficient_inliers'
        assert info['stereo_pose_arbitration_before_bundle']['cost'] < 1e-10
        assert info['stereo_pose_arbitration_after_bundle']['cost'] < 1e-10
        assert slam.previous_supported_stereo.frame == 1
    finally:
        slam.close()


def validation_scene(slam, pose=None):
    pixels, points, _ = scene()
    pose = np.eye(4) if pose is None else pose
    projected, depth = geometry.project(points, pose, K)
    disparity = K[0, 0] * BASELINE / depth + OFFSET
    slam.current_disparity = np.full((376, 1241), disparity[0], np.float32)
    identifiers = [slam.map.add_landmark(
        point, np.zeros(128, np.float32), 0, {}) for point in points]
    tracks = [(ident, pixel.copy()) for ident, pixel in zip(identifiers, projected)]
    associations = {feature: ident for feature, ident in enumerate(identifiers)}
    return points, projected, identifiers, tracks, associations


def test_selected_pose_keeps_only_stereo_consistent_map_links_and_flow_pixels():
    slam = camera()
    pose = np.eye(4)
    pose[2, 3] = 2.1
    try:
        world, observed, identifiers, tracks, associations = validation_scene(slam, pose)
        # Simulate old map depths that project to the same left ray but disagree
        # with current metric stereo. A left-only check would incorrectly keep them.
        wrong = set(identifiers[::5])
        for ident in wrong:
            camera_point = (slam.map.landmarks[ident].position - pose[:3, 3]) @ pose[:3, :3]
            slam.map.landmarks[ident].position = (camera_point * 1.5) @ pose[:3, :3].T + pose[:3, 3]
        # One accepted flow point has no detector feature and must retain its
        # original subpixel coordinate through keyframe materialization.
        flow_id = identifiers[3]
        flow_pixel = observed[3] + np.array([0.17, -0.12])
        tracks = [(ident, flow_pixel.copy() if ident == flow_id else pixel)
                  for ident, pixel in tracks]
        associations.pop(3)
        selected_pose = pose.copy()
        before = selected_pose.tobytes()
        kept_associations, kept_tracks, diagnostic = slam._validate_stereo_associations_at_pose(
            selected_pose, associations, tracks, observed, (1241, 376), len(identifiers))
        assert diagnostic['eligible']
        assert diagnostic['retained_landmarks'] >= 15
        assert diagnostic['retained_ratio'] >= .25
        assert diagnostic['spatial_coverage'] >= 3
        assert diagnostic['median_left_residual_px'] <= 1.5
        # This checks the nonzero principal-point offset sign in right projection.
        assert diagnostic['median_right_residual_px'] < 1e-5
        assert diagnostic['rejected']['right_residual'] == len(wrong)
        assert wrong.isdisjoint({lid for _, lid in kept_associations.items()})
        assert selected_pose.tobytes() == before
        assert any(lid == flow_id and np.array_equal(pixel, flow_pixel)
                   for lid, pixel in kept_tracks)

        slam.accepted_tracks = kept_tracks
        slam.current_gray = np.zeros((376, 1241), np.uint8)
        blank_points = np.full((len(observed), 3), np.nan)
        blank_right = np.full(len(observed), np.nan)
        keyframe_id = slam._keyframe(1, selected_pose, observed, np.zeros((len(observed), 128), np.float32),
                                     blank_points, blank_right, kept_associations)
        keyframe = slam.map.keyframes[keyframe_id]
        flow_row = int(np.flatnonzero(keyframe.landmark_ids == flow_id)[0])
        np.testing.assert_array_equal(keyframe.pixels[flow_row], flow_pixel)
        assert keyframe_id in slam.map.landmarks[flow_id].observations
        np.testing.assert_array_equal(slam.map.landmarks[flow_id].observations[keyframe_id].pixel, flow_pixel)
    finally:
        slam.close()


def test_fixed_pose_association_gate_rejects_bad_left_domain_and_missing_stereo():
    slam = camera()
    try:
        _, observed, identifiers, tracks, associations = validation_scene(slam)
        # Deliberately make exactly three rows invalid in different ways.
        tracks = list(tracks)
        tracks[0] = (identifiers[0], observed[0] + [5., 0.])
        tracks[1] = (identifiers[1], np.array([-1., observed[1, 1]]))
        x, y = np.rint(observed[2]).astype(int)
        slam.current_disparity[y, x] = np.nan
        kept_associations, kept_tracks, diagnostic = slam._validate_stereo_associations_at_pose(
            np.eye(4), associations, tracks, observed, (1241, 376), len(identifiers))
        assert diagnostic['eligible']
        assert identifiers[0] not in {lid for lid, _ in kept_tracks}
        assert identifiers[1] not in {lid for lid, _ in kept_tracks}
        assert identifiers[2] not in {lid for lid, _ in kept_tracks}
        assert diagnostic['rejected']['left_residual'] == 1
        assert diagnostic['rejected']['invalid_observation'] == 1
        assert diagnostic['rejected']['invalid_stereo_measurement'] == 1
        assert identifiers[0] not in kept_associations.values()
        assert identifiers[1] not in kept_associations.values()
        assert identifiers[2] not in kept_associations.values()
    finally:
        slam.close()


def test_fixed_pose_gate_clears_all_links_when_surviving_support_is_too_small():
    slam = camera()
    try:
        _, observed, identifiers, tracks, associations = validation_scene(slam)
        # Make all but twelve current stereo measurements unavailable. The
        # independent pose remains valid, but old links cannot carry a KF.
        for pixel in observed[12:]:
            x, y = np.rint(pixel).astype(int)
            slam.current_disparity[y, x] = np.nan
        kept_associations, kept_tracks, diagnostic = slam._validate_stereo_associations_at_pose(
            np.eye(4), associations, tracks, observed, (1241, 376), len(identifiers))
        assert not diagnostic['eligible']
        assert diagnostic['reason'] == 'insufficient_inliers'
        assert diagnostic['retained_landmarks'] < 15
        assert not kept_associations and not kept_tracks
        assert all(slam.map.landmarks[lid].misses >= 1 for lid in identifiers)
    finally:
        slam.close()


def test_selected_pose_keeps_detector_landmarks_until_normal_keyframe_cadence(monkeypatch):
    slam = camera()
    pixels, world_points, desc = scene()
    increment = np.eye(4)
    increment[2, 3] = .5
    associations = {feature: feature for feature in range(0, len(pixels), 2)}
    even_features = np.asarray(list(associations), int)

    def extract(image, right):
        frame = len(slam.map.poses)
        truth = np.eye(4)
        truth[2, 3] = increment[2, 3] * frame
        current_pixels, depth = geometry.project(world_points, truth, K)
        disparity = K[0, 0] * BASELINE / depth + OFFSET
        slam.current_disparity = np.full(image.shape, disparity[0], np.float32)
        camera_points = (world_points - truth[:3, 3]) @ truth[:3, :3]
        return current_pixels, desc.copy(), camera_points, current_pixels[:, 0]-disparity

    def fit(source, target, matrix, **kwargs):
        pairs = kwargs['matcher'](source.descriptors, target.descriptors)
        return verified(increment.copy(), pairs)

    def track(image_pixels, image_desc, size):
        frame = len(slam.map.poses)
        truth = np.eye(4)
        truth[2, 3] = increment[2, 3] * frame
        wrong_map_pose = truth.copy()
        angle = np.radians(.05)
        wrong_map_pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        tracks = [(int(feature), image_pixels[feature].copy()) for feature in even_features]
        context = slam._arbitration_context
        context['map_fit_landmarks'].update(associations.values())
        context['map_fit_targets'].update(associations.keys())
        slam.accepted_tracks = tracks
        return (wrong_map_pose, associations.copy()), {
            'num_matches': len(even_features), 'valid_3d': len(even_features),
            'num_inliers': len(even_features), 'inlier_ratio': 1.0,
            'tracking_ok': True,
        }

    slam._extract = extract
    slam._track = track
    slam._arbitrate_supported_pose = lambda context, map_pose, measurement: {
        'choice': 'independent', 'reason': 'synthetic_selected',
        'measurement_provenance': 'immutable_supported_extraction', 'cost': 0.0,
        'map': {'cost': 0.0},
    }
    monkeypatch.setattr(module, 'estimate_stereo_reference', fit)
    image = np.zeros((376, 1241), np.uint8)
    try:
        slam.process(0, image, image)
        first_pose, first_info = slam.process(1, image, image)
        expected_first = np.eye(4)
        expected_first[2, 3] = .5
        np.testing.assert_allclose(first_pose, expected_first, atol=1e-12)
        assert first_info['stereo_pose_arbitration']['association_validation']['eligible']
        assert first_info['tracked_landmarks'] == len(even_features)
        assert len(slam.map.keyframes) == 1  # no forced KF on an eligible arbitration win
        for feature in even_features:
            assert slam.previous_supported_stereo.landmark_ids[feature] == feature

        second_pose, second_info = slam.process(2, image, image)
        expected_second = np.eye(4)
        expected_second[2, 3] = 1.0
        np.testing.assert_allclose(second_pose, expected_second, atol=1e-12)
        assert second_info['stereo_pose_arbitration']['association_validation']['eligible']
        assert len(slam.map.keyframes) == 2  # ordinary <80-track cadence creates it
        current_kf = slam.map.keyframes[slam.last_keyframe]
        assert current_kf.frame == 2
        for feature in even_features:
            assert current_kf.landmark_ids[feature] == feature
            assert slam.last_keyframe in slam.map.landmarks[feature].observations
    finally:
        slam.close()


def install_miss_tracking_pipeline(slam, monkeypatch, map_rotation_degrees=0.05):
    pixels, world_points, desc = scene()
    def extract(image, right):
        frame = len(slam.map.poses)
        disparity = K[0, 0] * BASELINE / 20. + OFFSET
        slam.current_disparity = np.full(image.shape, disparity, np.float32)
        # Current raw stereo support is recaptured from disparity. Leaving map
        # depth empty after bootstrap prevents unassociated rows becoming new LMs.
        points = world_points.copy() if frame == 0 else np.full_like(world_points, np.nan)
        return pixels.copy(), desc.copy(), points, pixels[:, 0]-disparity

    def map_pose(points, observations, matrix, size, min_inliers,
                 initial_pose=None, diagnostics=None):
        angle = np.radians(map_rotation_degrees)
        pose = np.eye(4)
        pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        return pose, np.arange(len(points)), 0.

    def reference_fit(source, target, matrix, **kwargs):
        pairs = kwargs['matcher'](source.descriptors, target.descriptors)
        return verified(np.eye(4), pairs)

    slam._extract = extract
    slam._arbitrate_supported_pose = lambda context, map_pose, measurement: {
        'choice': 'independent', 'reason': 'synthetic_selected',
        'measurement_provenance': 'immutable_supported_extraction', 'cost': 0.0,
        'map': {'cost': 0.0},
    }
    monkeypatch.setattr(module, 'estimate_pose', map_pose)
    monkeypatch.setattr(module, 'refine_stereo_map_pose', lambda solution, *args: (solution, {}))
    monkeypatch.setattr(module, 'estimate_stereo_reference', reference_fit)
    return pixels, world_points, desc


def test_selected_pose_miss_counts_accumulate_and_cull_rejected_map_landmark(monkeypatch):
    slam = camera()
    _, world_points, desc = install_miss_tracking_pipeline(slam, monkeypatch)
    image = np.zeros((376, 1241), np.uint8)
    try:
        slam.process(0, image, image)
        rejected_id, retained_id, withheld_id = 0, 2, 1
        # Same left ray but wrong depth: each independent pose must reject this
        # old map link on its right-image residual.
        slam.map.landmarks[rejected_id].position *= 1.5
        slam.map.landmarks[rejected_id].misses = 0
        slam.map.landmarks[retained_id].misses = 4
        slam.map.landmarks[withheld_id].misses = 4
        unmatched_descriptor = np.zeros(128, np.float32)
        unmatched_descriptor[48] = 1.
        untouched_id = slam.map.add_landmark(
            world_points[0], unmatched_descriptor, 0, {})
        slam.map.landmarks[untouched_id].misses = 3

        for frame in range(1, 6):
            _, info = slam.process(frame, image, image)
            if frame < 5:
                assert slam.map.landmarks[rejected_id].misses == frame
            assert info['stereo_pose_arbitration']['association_validation']['eligible']
            assert slam.map.landmarks[retained_id].misses == 0
            # Odd detector rows are reserved for holdout and never offered to
            # map tracking; an unmatched map row is also never attempted.
            assert slam.map.landmarks[withheld_id].misses == 4
            assert slam.map.landmarks[untouched_id].misses == 3

        assert rejected_id not in slam.map.landmarks
        assert slam.map.keyframes[0].landmark_ids[rejected_id] == -1
    finally:
        slam.close()


def test_hard_stereo_conflict_ages_provisional_map_inliers_once(monkeypatch):
    slam = camera()
    install_miss_tracking_pipeline(slam, monkeypatch, map_rotation_degrees=2.0)
    image = np.zeros((376, 1241), np.uint8)
    try:
        slam.process(0, image, image)
        slam.map.landmarks[0].misses = 3
        slam.map.landmarks[2].misses = 4
        slam.map.landmarks[1].misses = 4  # held out of the map solve
        _, info = slam.process(1, image, image)
        assert info['map_pose_rejected_for_stereo_conflict']
        assert slam.map.landmarks[0].misses == 4
        assert 2 not in slam.map.landmarks  # one age reaches the existing cull limit
        assert slam.map.landmarks[1].misses == 4
    finally:
        slam.close()


def test_context_absent_hard_conflict_uses_enabled_tracking_snapshot(monkeypatch):
    slam = camera()
    install_miss_tracking_pipeline(slam, monkeypatch, map_rotation_degrees=2.0)
    image = np.zeros((376, 1241), np.uint8)
    try:
        slam.process(0, image, image)
        slam.map.landmarks[0].misses = 2
        slam._prepare_stereo_arbitration = lambda index, current: (
            None, {'choice': 'map', 'reason': 'reserved_support_unavailable'})
        _, info = slam.process(1, image, image)
        assert info['map_pose_rejected_for_stereo_conflict']
        assert slam._arbitration_context is None
        assert slam.map.landmarks[0].misses == 3
    finally:
        slam.close()


def test_failed_map_tracking_attempt_does_not_age_misses_or_geometry():
    slam = camera()
    pixels, points, desc = scene()
    identifiers = [slam.map.add_landmark(point, desc[i], 0, {})
                   for i, point in enumerate(points)]
    slam.map.record(np.eye(4), 'tracking')
    for i, ident in enumerate(identifiers):
        slam.map.landmarks[ident].misses = (i % 4) + 1
    before_misses = {ident: lm.misses for ident, lm in slam.map.landmarks.items()}
    before_points = {ident: lm.position.copy() for ident, lm in slam.map.landmarks.items()}
    slam.current_disparity = np.full((376, 1241), 20., np.float32)
    slam._arbitration_context = {
        'excluded_landmarks': set(), 'excluded_targets': set(),
        'excluded_target_pixels': set(), 'map_fit_landmarks': set(),
        'map_fit_targets': set(),
    }
    slam._match = lambda first, second: np.empty((0, 2), int)
    try:
        result, _ = slam._track(pixels, desc, (1241, 376))
        assert result is None
        assert slam._last_map_track_inlier_misses is None
        assert {ident: lm.misses for ident, lm in slam.map.landmarks.items()} == before_misses
        for ident, lm in slam.map.landmarks.items():
            np.testing.assert_array_equal(lm.position, before_points[ident])
    finally:
        slam.close()


def test_native_capture_precedes_restoration(monkeypatch):
    slam = camera(stereo_depth_policy='verified_fallback')
    try:
        pixel = np.array([[140.25, 50.25]], np.float32)
        slam.current_disparity = np.full((376, 1241), 20., np.float32)
        slam.current_disparity[51, 141] = -1
        monkeypatch.setattr(module, 'verify_stereo_depth_candidates',
                            lambda left, right, pixels, *args: (pixels[:, 0]-20., {'verified': len(pixels)}))
        points, right = slam._measure_stereo_pixels(pixel, capture_supported=True)
        assert np.isfinite(points).all() and np.isfinite(right).all()
        assert np.isnan(slam._supported_extraction[0]).all()
        assert np.isnan(slam._supported_extraction[1]).all()
    finally:
        slam.close()


def test_held_frame_never_replaces_accepted_supported_source(monkeypatch):
    slam = camera()
    pixels, points, desc = scene()
    def extract(image, right):
        slam.current_disparity = np.full(image.shape, K[0, 0]*BASELINE/20.+OFFSET)
        return pixels.copy(), desc.copy(), points.copy(), pixels[:, 0]-K[0, 0]*BASELINE/20.-OFFSET
    slam._extract = extract
    try:
        image = np.zeros((376, 1241), np.uint8)
        slam.process(0, image, image)
        previous = slam.previous_supported_stereo
        monkeypatch.setattr(module, 'estimate_stereo_reference', lambda *args, **kwargs: None)
        slam._track = lambda *args: (None, {})
        slam._keyframe_stereo_reference = lambda *args, **kwargs: (None, {})
        slam._relocalize = lambda *args: (None, {})
        _, info = slam.process(1, image, image)
        assert info['state'] == 'lost'
        assert slam.previous_supported_stereo is previous
        assert slam.previous_stereo_geometry[1] == previous.frame == 0
        assert slam.current_supported_stereo.frame == 1
    finally:
        slam.close()


def test_flag_off_never_captures_or_reserves_new_evidence():
    slam = camera(enabled=False)
    pixels, points, desc = scene()
    def extract(image, right):
        return pixels.copy(), desc.copy(), points.copy(), pixels[:, 0]-20.
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled arbitration called a new producer')
    slam._extract = extract
    slam._capture_supported_stereo = slam._prepare_stereo_arbitration = forbidden
    try:
        image = np.zeros((376, 1241), np.uint8)
        _, info = slam.process(0, image, image)
        assert info['state'] == 'tracking'
        assert 'stereo_pose_arbitration' not in info
        assert slam.previous_supported_stereo is None
    finally:
        slam.close()


def test_source_pixel_aliases_linked_flow_id_excluded(monkeypatch):
    slam = camera()
    try:
        previous, current = record(slam), record(slam, 1)
        install_previous(slam, previous)
        slam.previous_tracks = [(999, previous.pixels[1].copy())]
        monkeypatch.setattr(module, 'estimate_stereo_reference',
                            lambda source, target, matrix, **kw: verified(np.eye(4), kw['matcher'](None, None)))
        context, _ = slam._prepare_stereo_arbitration(1, current)
        assert 999 in context['excluded_landmarks']
    finally:
        slam.close()


def test_current_descriptor_alias_excludes_different_flow_landmark(monkeypatch):
    slam = camera()
    try:
        raw = record(slam)
        previous = SupportedStereoFrame(raw.pixels, raw.descriptors, raw.points, raw.right_u,
            np.full(len(raw.pixels), -1, int), 0, raw.image_size, raw.calibration_identity)
        current = record(slam, 1)
        install_previous(slam, previous)
        monkeypatch.setattr(module, 'estimate_stereo_reference',
                            lambda source, target, matrix, **kw: verified(np.eye(4), kw['matcher'](None, None)))
        context, _ = slam._prepare_stereo_arbitration(1, current)
        slam._arbitration_context = context
        # Different map IDs own the current descriptors. Source snapshot IDs
        # alone cannot exclude these landmarks or their detector-free flow.
        for j in range(len(previous.pixels)):
            feature = (j+1) % len(previous.pixels)
            slam.map.add_landmark(previous.points[feature], previous.descriptors[feature], 0, {})
        slam.previous_tracks = [(0, np.array([10., 10.], np.float32))]
        slam.previous_gray = slam.current_gray = np.zeros((376, 1241), np.uint8)
        monkeypatch.setattr(module, 'estimate_pose',
                            lambda points, pixels, *args, **kwargs: (np.eye(4), np.arange(len(points)), 0.))
        monkeypatch.setattr(module, 'refine_stereo_map_pose', lambda answer, *args: (answer, {}))
        slam.current_disparity = np.full((376, 1241), K[0, 0]*BASELINE/20.+OFFSET)
        answer, _ = slam._track(current.pixels, current.descriptors, current.image_size)
        assert answer is not None
        assert context['excluded_landmarks'] == set(range(0, 48, 2))
        assert context['map_fit_landmarks'] == set(range(1, 48, 2))
        assert context['map_fit_targets'] == set(range(0, 48, 2))
    finally:
        slam.close()
