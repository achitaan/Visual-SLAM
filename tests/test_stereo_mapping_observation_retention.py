"""Mapping observations cannot substitute for spatially supported pose acceptance."""
import numpy as np
import pytest

import mapping_geometry as geometry
import shared_slam as module
from loop_geometry import StereoLoopFrame
from stereo_pose_arbitration import SupportedStereoFrame
from test_stereo_arbitration_tracking import BASELINE, K, OFFSET, camera, verified


SIZE = (1241, 376)


def _scene():
    narrow = [(x, y) for y in (30., 50., 70., 90.)
              for x in (140., 200., 260., 520., 580., 640.)]
    broad = [(x, y) for y in (170., 240., 310.)
             for x in (140., 280., 420., 560., 700., 840., 980., 1120.)]
    pixels = np.asarray(narrow + broad, np.float32)
    points = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T * 20.
    return pixels, points, np.eye(len(pixels), 128, dtype=np.float32)


@pytest.fixture
def fixed_case():
    slam = camera(stereo_mapping_observation_retention=True)
    pixels, points, descriptors = _scene()
    disparity = float(K[0, 0] * BASELINE / 20. + OFFSET)
    rights = pixels[:, 0] - disparity

    def extract(image, right):
        slam.current_disparity = np.full(image.shape, disparity, np.float32)
        return pixels.copy(), descriptors.copy(), points.copy(), rights.copy()

    slam._extract = extract
    image = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    slam.process(0, image, image)
    ids = slam.map.keyframes[0].landmark_ids.copy()
    source = slam.previous_supported_stereo
    assert source.frame == 0
    current = SupportedStereoFrame(pixels, descriptors, points, rights,
        ids, 1, SIZE, slam.stereo_calibration_identity)
    slam.current_supported_stereo = current
    measured = StereoLoopFrame(pixels, points, descriptors, SIZE)
    result = geometry.estimate_stereo_reference(slam.previous_stereo_geometry[0],
        measured, K, matcher=lambda *_: np.c_[np.arange(48), np.arange(48)])
    assert result is not None and result['reverse_checked']
    source_pose = slam.map.poses[0].copy()
    reference_pose = source_pose @ result['measurement']
    revision = slam.map.revision
    guard = slam._hard_reference_retention_guard(1, 0,
        slam.previous_stereo_geometry[0], measured, result,
        source_pose, reference_pose, revision, SIZE)
    assert guard['eligible'], guard
    context = dict(guard=guard, verified=result, source_pose=source_pose,
        reference_pose=reference_pose, source_map_revision=revision,
        source_geometry_revision=slam.map.geometry_revision,
        source_frame=0, target_frame=1,
        previous_frame=slam.previous_stereo_geometry[0], measured_frame=measured)
    case = dict(slam=slam, pixels=pixels, points=points, descriptors=descriptors,
        rights=rights, ids=ids, context=context, image=image)
    try:
        yield case
    finally:
        slam.close()


def _validate(case, *, rows=None, tracks=None, associations=None,
              denominator=24, context=True):
    rows = list(range(24)) if rows is None else list(rows)
    slam = case['slam']
    ids, pixels = case['ids'], case['pixels']
    associations = ({i: int(ids[i]) for i in rows}
                    if associations is None else associations)
    tracks = ([(int(ids[i]), pixels[i].copy()) for i in rows]
              if tracks is None else tracks)
    misses = {int(ids[i]): 3 for i in rows}
    for ident in misses:
        slam.map.landmarks[ident].misses = 0  # provisional map inlier reset
    pose = case['context']['reference_pose'].copy()
    before = pose.copy()
    answer = slam._validate_stereo_associations_at_pose(pose, associations,
        tracks, pixels, SIZE, denominator, misses,
        retention_context=case['context'] if context else None)
    np.testing.assert_array_equal(pose, before)
    return answer


def test_only_mapping_links_are_retained_and_real_keyframe_rows_are_written(fixed_case):
    case = fixed_case
    associations, tracks, report = _validate(case)
    assert not report['eligible']
    assert report['mapping_observation_retention_allowed']
    assert report['spatial_coverage'] == 2
    assert set(associations.values()) == set(map(int, case['ids'][:24]))
    assert len(tracks) == 24
    slam = case['slam']
    slam.accepted_tracks = tracks
    slam._keyframe(1, np.eye(4), case['pixels'], case['descriptors'],
                   case['points'], case['rights'], associations)
    for row in range(24):
        ident = int(case['ids'][row])
        observation = slam.map.landmarks[ident].observations[1]
        np.testing.assert_array_equal(observation.pixel, case['pixels'][row])
        np.testing.assert_allclose(observation.right_u, case['rights'][row], atol=1e-4)
        assert slam.map.keyframes[1].landmark_ids[row] == ident
        assert slam.map.landmarks[ident].misses == 0


@pytest.mark.parametrize('mode', ['default', 'missing_certificate', 'forward_only'])
def test_coverage_failure_still_clears_without_selected_independent_certificate(fixed_case, mode):
    case = fixed_case
    if mode == 'default':
        from dataclasses import replace
        case['slam'].config = replace(case['slam'].config,
            stereo_mapping_observation_retention=False)
    elif mode == 'forward_only':
        case['context']['verified']['reverse_checked'] = False
    associations, tracks, report = _validate(case, context=mode != 'missing_certificate')
    assert not report['eligible']
    assert not report['mapping_observation_retention_allowed']
    assert associations == {} and tracks == []
    assert all(case['slam'].map.landmarks[int(i)].misses == 4 for i in case['ids'][:24])


@pytest.mark.parametrize('failure', ['inliers', 'ratio', 'left_median', 'right_median'])
def test_other_aggregate_failure_never_retains_mapping_links(fixed_case, failure):
    case = fixed_case
    rows = range(14) if failure == 'inliers' else range(24)
    denominator = 200 if failure == 'ratio' else 24
    if failure == 'left_median':
        for ident in case['ids'][:24]:
            case['slam'].map.landmarks[int(ident)].position[1] += 1.75 * 20 / K[1, 1]
    elif failure == 'right_median':
        case['slam'].current_disparity += 1.75
    associations, tracks, report = _validate(case, rows=rows, denominator=denominator)
    assert not report['eligible'] and not report['mapping_observation_retention_allowed']
    assert associations == {} and tracks == []


def test_bad_individual_measurements_are_excluded_and_misses_aged_once(fixed_case):
    case = fixed_case
    slam = case['slam']
    # Independent raw reference geometry stays intact; these are map-link failures.
    slam.map.landmarks[int(case['ids'][0])].position[0] += 3. * 20 / K[0, 0]
    slam.map.landmarks[int(case['ids'][1])].position[2] = -20.
    for row, value in ((2, np.nan), (3, float(slam.current_disparity[0, 0]) + 3.)):
        x, y = case['pixels'][row].astype(int)
        slam.current_disparity[y:y+2, x:x+2] = value
    associations, tracks, report = _validate(case)
    assert report['mapping_observation_retention_allowed']
    expected = set(map(int, case['ids'][4:24]))
    assert set(associations.values()) == expected
    assert {ident for ident, _ in tracks} == expected
    for row in range(24):
        assert slam.map.landmarks[int(case['ids'][row])].misses == (4 if row < 4 else 0)


@pytest.mark.parametrize('mutation', ['revision', 'geometry', 'calibration', 'source_pose', 'status', 'endpoint'])
def test_sampler_epoch_mutation_vetoes_retention_and_ages_once(fixed_case, monkeypatch, mutation):
    case = fixed_case
    slam = case['slam']
    original = slam._measure_supported_stereo_pixels
    def measure(pixels):
        answer = original(pixels)
        if mutation == 'revision':
            slam.map.revision += 1
        elif mutation == 'geometry':
            slam.map.geometry_revision += 1
        elif mutation == 'calibration':
            slam.K[0, 0] += .5
        elif mutation == 'source_pose':
            slam.map.poses[0][0, 3] += .01
        elif mutation == 'status':
            slam.map.statuses[0] = 'lost'
        else:
            old = slam.current_supported_stereo
            changed = old.pixels.copy()
            changed[0, 0] += .25
            slam.current_supported_stereo = SupportedStereoFrame(changed,
                old.descriptors, old.points, old.right_u, old.landmark_ids,
                old.frame, old.image_size, old.calibration_identity)
        return answer
    monkeypatch.setattr(slam, '_measure_supported_stereo_pixels', measure)
    associations, tracks, report = _validate(case)
    assert not report['mapping_observation_retention_allowed']
    assert associations == {} and tracks == []
    assert all(slam.map.landmarks[int(i)].misses == 4 for i in case['ids'][:24])


@pytest.mark.parametrize('ambiguity', ['lm_two_pixels', 'pixel_two_lms', 'detector_alias_conflict'])
def test_physical_ownership_conflicts_drop_all_claimants(fixed_case, ambiguity):
    case = fixed_case
    ids, pixels = case['ids'], case['pixels']
    tracks = [(int(ids[i]), pixels[i].copy()) for i in range(24)]
    associations = {i: int(ids[i]) for i in range(24)}
    if ambiguity == 'lm_two_pixels':
        tracks.append((int(ids[0]), pixels[0] + np.array([.25, 0.])))
        rejected = {int(ids[0])}
    elif ambiguity == 'pixel_two_lms':
        tracks[1] = (int(ids[1]), pixels[0].copy())
        rejected = {int(ids[0]), int(ids[1])}
    else:
        case['pixels'] = pixels.copy()
        case['pixels'][1] = case['pixels'][0]
        # Rebuild an internally consistent endpoint and actual verification;
        # this is an ownership conflict, not a stale endpoint certificate.
        current_pixels = case['pixels']
        current_points = np.c_[current_pixels, np.ones(len(current_pixels))] @ np.linalg.inv(K).T * 20.
        current_right = current_pixels[:, 0] - K[0, 0] * BASELINE / 20. - OFFSET
        slam = case['slam']
        slam.current_supported_stereo = SupportedStereoFrame(current_pixels,
            case['descriptors'], current_points, current_right, ids, 1,
            SIZE, slam.stereo_calibration_identity)
        measured = StereoLoopFrame(current_pixels, current_points,
                                   case['descriptors'], SIZE)
        paired_rows = np.r_[0, np.arange(2, 48)]
        result = geometry.estimate_stereo_reference(slam.previous_stereo_geometry[0],
            measured, K, matcher=lambda *_: np.c_[paired_rows, paired_rows])
        assert result is not None and result['reverse_checked']
        context = case['context']
        context['verified'] = result
        context['measured_frame'] = measured
        context['reference_pose'] = context['source_pose'] @ result['measurement']
        context['guard'] = slam._hard_reference_retention_guard(1, 0,
            context['previous_frame'], measured, result, context['source_pose'],
            context['reference_pose'], context['source_map_revision'], SIZE)
        assert context['guard']['eligible'], context['guard']
        rejected = {int(ids[0]), int(ids[1])}
    associations, tracks, report = _validate(case, tracks=tracks, associations=associations)
    assert report['mapping_observation_retention_allowed']
    assert not (set(associations.values()) & rejected)
    assert not ({ident for ident, _ in tracks} & rejected)


def test_reserved_landmark_and_exact_pixel_exclusions_remain_reserved(fixed_case):
    case = fixed_case
    case['context']['excluded_landmark_ids'] = {int(case['ids'][0])}
    case['context']['excluded_target_pixels'] = [case['pixels'][1].copy()]
    tracks = [(int(case['ids'][i]), case['pixels'][i].copy()) for i in range(24)]
    # A held detector claim remains reserved even when its associated flow
    # pixel is nearby rather than exactly the SIFT coordinate.
    tracks[1] = (int(case['ids'][1]), case['pixels'][1] + np.array([.25, 0.]))
    associations, tracks, report = _validate(case, tracks=tracks)
    assert report['mapping_observation_retention_allowed']
    excluded = set(map(int, case['ids'][:2]))
    assert not set(associations.values()) & excluded
    assert not {ident for ident, _ in tracks} & excluded


@pytest.mark.parametrize('malformed', ['bool_frame', 'fractional_revision',
                                     'complex_pose', 'bool_excluded_id',
                                     'wrong_shape_excluded_pixel', 'complex_excluded_pixel'])
def test_malformed_certificate_or_reserved_ownership_fails_closed(fixed_case, malformed):
    case = fixed_case
    context = case['context']
    if malformed == 'bool_frame':
        context['target_frame'] = True
    elif malformed == 'fractional_revision':
        context['source_map_revision'] += .25
    elif malformed == 'complex_pose':
        context['source_pose'] = context['source_pose'].astype(complex)
        context['source_pose'][0, 0] += 1j
    elif malformed == 'bool_excluded_id':
        context['excluded_landmark_ids'] = [True]
    elif malformed == 'wrong_shape_excluded_pixel':
        context['excluded_target_pixels'] = [[10., 20., 30.]]
    else:
        context['excluded_target_pixels'] = [np.asarray([10. + 1j, 20.])]
    associations, tracks, report = _validate(case)
    assert not report['mapping_observation_retention_allowed']
    assert associations == {} and tracks == []


def test_repeated_same_flow_is_one_observation_and_actual_subpixel_is_authoritative(fixed_case):
    case = fixed_case
    tracks = [(int(case['ids'][i]), case['pixels'][i].copy()) for i in range(24)]
    actual = case['pixels'][0] + np.asarray([.25, .25], np.float32)
    tracks[0] = (int(case['ids'][0]), actual.copy())
    tracks.append((int(case['ids'][0]), actual.copy()))
    associations, kept, report = _validate(case, tracks=tracks)
    assert report['mapping_observation_retention_allowed']
    assert len(kept) == 24
    slam = case['slam']
    slam.accepted_tracks = kept
    slam._keyframe(1, case['context']['reference_pose'], case['pixels'],
        case['descriptors'], case['points'], case['rights'], associations)
    observation = slam.map.landmarks[int(case['ids'][0])].observations[1]
    np.testing.assert_array_equal(observation.pixel, actual)
    np.testing.assert_allclose(observation.right_u, case['rights'][0] + .25, atol=1e-4)


@pytest.mark.parametrize('enabled,mutation', [(False, None), (True, None),
                                           (True, 'geometry'), (True, 'reverse')])
def test_process_independent_reference_preserves_only_mapping_links(enabled, mutation, monkeypatch):
    slam = camera(stereo_mapping_observation_retention=enabled,
                  keyframe_interval=1)
    pixels, points, descriptors = _scene()
    disparity = float(K[0, 0] * BASELINE / 20. + OFFSET)
    rights = pixels[:, 0] - disparity
    image = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    wrong = np.eye(4)
    angle = np.radians(2.)
    wrong[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    def extract(image, right):
        slam.current_disparity = np.full(image.shape, disparity, np.float32)
        return pixels.copy(), descriptors.copy(), points.copy(), rights.copy()
    slam._extract = extract
    slam._prepare_stereo_arbitration = lambda *_: (
        None, {'choice': 'map', 'reason': 'insufficient_reserved_support'})
    # Keep the actual reference estimator and its forward/reverse checks. Only
    # descriptor lookup is fixed here; map tracking is a controlled candidate.
    original_reference = module.estimate_stereo_reference
    def reference(source, target, matrix, **kwargs):
        kwargs['matcher'] = lambda *_: np.c_[np.arange(48), np.arange(48)]
        kwargs['initial_pose'] = None
        result = original_reference(source, target, matrix, **kwargs)
        if mutation == 'reverse':
            result = {**result, 'reverse_checked': False}
        return result
    monkeypatch.setattr(module, 'estimate_stereo_reference', reference)
    original_ids = None
    def track(*args):
        ids = original_ids[:24]
        slam._last_map_track_inlier_misses = {int(i): 3 for i in ids}
        for ident in ids:
            slam.map.landmarks[int(ident)].misses = 0
        slam.accepted_tracks = [(int(i), pixels[row].copy())
                                for row, i in enumerate(ids)]
        if mutation == 'geometry':
            original_measure = slam._measure_supported_stereo_pixels
            def measure(observed):
                answer = original_measure(observed)
                slam.map.geometry_revision += 1
                return answer
            slam._measure_supported_stereo_pixels = measure
        return (wrong.copy(), {row: int(i) for row, i in enumerate(ids)}), {
            'num_matches': 24, 'num_inliers': 24, 'valid_3d': 24,
            'tracking_ok': True}
    slam._track = track
    try:
        slam.process(0, image, image)
        original_ids = slam.map.keyframes[0].landmark_ids.copy()
        pose, info = slam.process(1, image, image)
        np.testing.assert_allclose(pose, np.eye(4), atol=1e-5)
        assert info['map_pose_rejected_for_stereo_conflict']
        outer = info['reference_association_validation']
        connection = outer['association_validation']
        retained = enabled and mutation is None
        if connection is not None:
            assert not connection['eligible']
            assert connection['mapping_observation_retention_allowed'] == retained
        else:
            assert not outer['eligible'] and not retained
        assert slam.map.keyframes[slam.last_keyframe].frame == 1
        new_frame = slam.last_keyframe
        for row, ident in enumerate(original_ids[:24]):
            landmark = slam.map.landmarks[int(ident)]
            assert landmark.misses == (0 if retained else 4)
            if retained:
                actual = landmark.observations[new_frame]
                np.testing.assert_array_equal(actual.pixel, pixels[row])
                np.testing.assert_allclose(actual.right_u, rights[row], atol=1e-4)
                assert slam.map.keyframes[new_frame].landmark_ids[row] == ident
            else:
                assert new_frame not in landmark.observations
        assert len(slam.accepted_tracks) == (24 if retained else 0)
    finally:
        slam.close()


def test_real_tracking_pose_acceptance_still_requires_three_cells():
    pixels, _, _ = _scene()
    pixels = pixels[:24]
    depths = np.linspace(12., 28., len(pixels))
    points = (np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T) * depths[:, None]
    diagnostics = {}
    result = geometry.estimate_pose(points, pixels, K, SIZE,
                                   initial_pose=np.eye(4), diagnostics=diagnostics)
    assert result is None
    assert diagnostics['pose_rejection_reason'] == 'insufficient_spatial_support'


@pytest.mark.parametrize('path', ['arbitration_selected', 'full_supported_retry'])
def test_reserved_process_paths_keep_real_observations_and_account_for_consumed_holdout(path):
    slam = camera(stereo_mapping_observation_retention=True, keyframe_interval=1)
    broad, _, _ = _scene()
    narrow = [(x, y) for y in (20., 40., 60., 80., 100., 120.)
              for x in (140., 200., 260., 520., 580., 640.)]
    pixels = np.vstack([np.asarray(narrow, np.float32), broad[24:]])
    points = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T * 20.
    descriptors = np.eye(len(pixels), 128, dtype=np.float32)
    disparity = float(K[0, 0] * BASELINE / 20. + OFFSET)
    rights = pixels[:, 0] - disparity
    image = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    def extract(image, right):
        slam.current_disparity = np.full(image.shape, disparity, np.float32)
        return pixels.copy(), descriptors.copy(), points.copy(), rights.copy()
    slam._extract = extract
    slam._match = lambda a, b: np.c_[np.arange(min(len(a), len(b))),
                                    np.arange(min(len(a), len(b)))]
    wrong = np.eye(4)
    angle = np.radians(1. if path == 'arbitration_selected' else 2.)
    wrong[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    original_ids = None
    rows_seen = []
    if path == 'full_supported_retry':
        original_arbitrate = slam._arbitrate_supported_pose
        def controlled_abstention(*args):
            actual = original_arbitrate(*args)
            # Force the full-pool caller branch after real held scoring. This
            # test validates row ownership/consumption, not the arbiter's choice.
            return {**actual, 'choice': 'abstain',
                    'reason': 'controlled_abstention_for_full_pool_call_path'}
        slam._arbitrate_supported_pose = controlled_abstention
    def track(*args):
        # Respect the actual reservation exclusions. The controlled map
        # candidate does not consume any held detector or landmark row.
        context = slam._arbitration_context
        assert context is not None and context['verified']['reverse_checked']
        excluded = context['excluded_targets']
        rows = [i for i in range(36) if i not in excluded]
        rows_seen[:] = rows
        assert len(rows) == 18
        ids = [int(original_ids[row]) for row in rows]
        slam._last_map_track_inlier_misses = {ident: 3 for ident in ids}
        for ident in ids:
            slam.map.landmarks[ident].misses = 0
        slam.accepted_tracks = [(int(original_ids[row]), pixels[row].copy()) for row in rows]
        return (wrong.copy(), {row: int(original_ids[row]) for row in rows}), {
            'num_matches': len(rows), 'num_inliers': len(rows),
            'valid_3d': len(rows), 'tracking_ok': True}
    slam._track = track
    try:
        slam.process(0, image, image)
        original_ids = slam.map.keyframes[0].landmark_ids.copy()
        pose, info = slam.process(1, image, image)
        np.testing.assert_allclose(pose, np.eye(4), atol=1e-5)
        arbitration = info['stereo_pose_arbitration']
        if path == 'arbitration_selected':
            assert arbitration['choice'] == 'independent'
            connection = arbitration['association_validation']
            assert arbitration['held_out_arbitration_used']
        else:
            assert arbitration['choice'] == 'existing_reference'
            assert info['full_supported_reference_fallback']['eligible']
            assert not arbitration['held_out_arbitration_used']
            assert info['full_supported_reference_fallback'][
                'pre_full_reserved_arbitration']['held_out_arbitration_used']
            assert slam._arbitration_context is None
            connection = info['reference_association_validation']['association_validation']
        assert not connection['eligible']
        assert not connection['pose_support_eligible']
        assert connection['mapping_observation_retention_allowed']
        assert not connection['held_out_rows_reused']
        keyframe_id = slam.last_keyframe
        assert slam.map.keyframes[keyframe_id].frame == 1
        assert {ident for ident, _ in slam.accepted_tracks} == {
            int(original_ids[row]) for row in rows_seen}
        for row in rows_seen:
            ident = int(original_ids[row])
            actual = slam.map.landmarks[ident].observations[keyframe_id]
            np.testing.assert_array_equal(actual.pixel, pixels[row])
            np.testing.assert_allclose(actual.right_u, rights[row], atol=1e-4)
            assert slam.map.landmarks[ident].misses == 0
        for row in range(1, 36, 2):
            assert keyframe_id not in slam.map.landmarks[int(original_ids[row])].observations
    finally:
        slam.close()


def test_consistent_same_pixel_detector_aliases_write_one_landmark_observation(fixed_case):
    case = fixed_case
    slam = case['slam']
    pixels = np.vstack([case['pixels'], case['pixels'][0]])
    points = np.vstack([case['points'], case['points'][0]])
    descriptors = np.vstack([case['descriptors'], case['descriptors'][0]])
    rights = np.r_[case['rights'], case['rights'][0]]
    ids = np.r_[case['ids'], case['ids'][0]]
    slam.current_supported_stereo = SupportedStereoFrame(pixels, descriptors,
        points, rights, ids, 1, SIZE, slam.stereo_calibration_identity)
    measured = StereoLoopFrame(pixels, points, descriptors, SIZE)
    result = geometry.estimate_stereo_reference(slam.previous_stereo_geometry[0],
        measured, K, matcher=lambda *_: np.c_[np.arange(48), np.arange(48)])
    assert result is not None and result['reverse_checked']
    context = case['context']
    context['verified'] = result
    context['measured_frame'] = measured
    context['reference_pose'] = context['source_pose'] @ result['measurement']
    context['guard'] = slam._hard_reference_retention_guard(1, 0,
        context['previous_frame'], measured, result, context['source_pose'],
        context['reference_pose'], context['source_map_revision'], SIZE)
    assert context['guard']['eligible'], context['guard']
    case.update(pixels=pixels, points=points, descriptors=descriptors, rights=rights)
    associations = {i: int(ids[i]) for i in range(24)}
    associations[48] = int(ids[0])
    kept, tracks, report = _validate(case, associations=associations)
    assert report['mapping_observation_retention_allowed']
    assert len(tracks) == 24 and kept[0] == kept[48] == int(ids[0])
    slam.accepted_tracks = tracks
    slam._keyframe(1, context['reference_pose'], pixels, descriptors, points, rights, kept)
    assert slam.map.keyframes[1].landmark_ids[0] == slam.map.keyframes[1].landmark_ids[48]
    landmark = slam.map.landmarks[int(ids[0])]
    assert set(landmark.observations) == {0, 1}
    np.testing.assert_array_equal(landmark.observations[1].pixel, pixels[0])
    np.testing.assert_allclose(landmark.observations[1].right_u, rights[0], atol=1e-4)
