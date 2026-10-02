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
        assert info['stereo_pose_arbitration_before_bundle']['cost'] < 1e-10
        assert info['stereo_pose_arbitration_after_bundle']['cost'] < 1e-10
        assert slam.previous_supported_stereo.frame == 1
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
