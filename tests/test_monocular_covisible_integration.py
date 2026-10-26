"""Small real-triangulation contracts for older covisible keyframe mapping.

Only descriptor pairing is analytic. The real keyframe, physical-endpoint
collapse, reciprocal observation ownership and triangulation paths run here.
These fixtures do not measure trajectory quality or long-horizon retention.
"""
import cv2 as cv
import numpy as np
import pytest

from mapping_geometry import project
from owned_observation import OwnedObservation
from shared_slam import MappingConfig, SharedSlam
from slam_state import MappingKeyframe, Observation


K = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
WORLD = np.array([[0., 0., 5.], [.3, .2, 4.], [-.2, -.3, 6.], [.3, .2, 4.]])


def _pose(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


@pytest.fixture
def scene(monkeypatch):
    """Three older useful views, two recent views below the parallax gate."""
    previous_threads = cv.getNumThreads()
    cv.setNumThreads(1)
    slam = SharedSlam(K, config=MappingConfig(
        loop_mode='off', bundle_enabled=False, bundle_window=2,
        monocular_covisible_triangulation=True,
        monocular_scale_safe_triangulation=True,
    ))
    descriptors = np.arange(4*128, dtype=np.float32).reshape(4, 128)
    shared = slam.map.add_landmark(WORLD[0], descriptors[0], 0, {})
    for ident, x in enumerate([-.7, -1., -.8, -.002, -.001]):
        pixels, _ = project(WORLD, _pose(x), K)
        frame = MappingKeyframe(
            ident, ident, _pose(x), pixels.astype(np.float32),
            descriptors.copy(), np.array([shared, -1, -1, -1]),
            depth_points=np.full((4, 3), np.nan),
        )
        slam.map.keyframes[ident] = frame
        slam.map.landmarks[shared].observations[ident] = Observation(frame.pixels[0].copy())
    slam.last_keyframe = 4
    calls = []

    def identity_pairs(first, second):
        # Distinct descriptor arrays identify the actual reference call order;
        # row 3 is a second orientation of row 1's physical image point.
        ident = next(ident for ident, frame in slam.map.keyframes.items()
                     if first is frame.descriptors)
        calls.append(ident)
        assert len(first) == len(second) == 4
        return np.column_stack((np.arange(4), np.arange(4)))

    monkeypatch.setattr(slam, '_match', identity_pairs)
    try:
        yield slam, descriptors, shared, calls
    finally:
        slam.close(finish=False)
        cv.setNumThreads(previous_threads)


def _insert(slam, descriptors, associations, index=5, x=0.):
    pixels, _ = project(WORLD, _pose(x), K)
    ident = slam._keyframe(
        index, _pose(x), pixels.astype(np.float32), descriptors,
        np.full((4, 3), np.nan), np.full(4, np.nan), associations,
    )
    return slam.map.keyframes[ident]


def _assert_reciprocal_unique_observations(slam):
    for lid, landmark in slam.map.landmarks.items():
        for ident, observation in landmark.observations.items():
            frame = slam.map.keyframes[ident]
            rows = np.flatnonzero(frame.landmark_ids == lid)
            assert len(rows) > 0
            assert all(np.array_equal(frame.pixels[row], observation.pixel) for row in rows)
        for ident, frame in slam.map.keyframes.items():
            if np.any(frame.landmark_ids == lid):
                assert ident in landmark.observations
    for frame in slam.map.keyframes.values():
        owners = {}
        for pixel, lid in zip(frame.pixels, frame.landmark_ids):
            if lid < 0:
                continue
            key = tuple(pixel.tolist())
            assert key not in owners or owners[key] == lid
            owners[key] = lid


@pytest.mark.parametrize('enabled', [False, True])
def test_keyframe_keeps_recent_order_then_bounded_older_views(scene, enabled):
    slam, descriptors, shared, calls = scene
    slam.config = MappingConfig(
        loop_mode='off', bundle_enabled=False, bundle_window=2,
        monocular_covisible_triangulation=enabled,
        monocular_scale_safe_triangulation=True,
    )
    current = _insert(slam, descriptors, {0: shared})
    assert current.id == 5 and slam.last_keyframe == 5
    # The initial most-recent call only populates independent depth geometry;
    # the following calls are the landmark-creation reference sequence.
    assert calls[:3] == [4, 4, 3]
    # An existing shared landmark does not manufacture independent two-view
    # depth when the most-recent baseline is below the gate.
    assert np.isnan(current.depth_points[0]).all()
    if enabled:
        assert np.isfinite(current.depth_points[1:]).all()
    else:
        assert np.isnan(current.depth_points).all()
    if not enabled:
        assert calls == [4, 4, 3]
        assert slam._monocular_mapping_report is None
        assert len(slam.map.landmarks) == 1
        assert current.landmark_ids.tolist() == [shared, -1, -1, -1]
        return
    report = slam._monocular_mapping_report
    assert report['recent_reference_ids'] == [4, 3]
    assert report['extra_reference_ids'] == [1, 2]
    assert len(report['extra_reference_ids']) == slam.config.bundle_window
    assert all(ident < min(report['recent_reference_ids'])
               for ident in report['extra_reference_ids'])
    assert calls == [4, 4, 3, 1, 2]
    assert [attempt['keyframe'] for attempt in report['attempts']] == [4, 3, 1, 2]
    assert [attempt['unowned_correspondences'] for attempt in report['attempts']] == [2, 2, 2, 0]
    assert [attempt['triangulated_landmarks'] for attempt in report['attempts']] == [0, 0, 2, 0]
    assert len(slam.map.landmarks) == 3  # alias row did not create a fourth point
    assert current.landmark_ids[1] == current.landmark_ids[3] >= 0
    assert current.landmark_ids[2] != current.landmark_ids[1]
    for row in (1, 2):
        lid = int(current.landmark_ids[row])
        landmark = slam.map.landmarks[lid]
        assert landmark.anchor == 1
        assert set(landmark.observations) == {1, current.id}
        assert all(type(obs) is OwnedObservation for obs in landmark.observations.values())
        # Pixels are stored as float32: tolerate their few-micrometre depth
        # quantization at this fixed, metre-scale analytic geometry.
        assert np.allclose(landmark.position, WORLD[row], atol=1e-5, rtol=0.)
    _assert_reciprocal_unique_observations(slam)


def test_older_created_alias_landmarks_survive_next_keyframe_without_duplication(scene):
    slam, descriptors, shared, _ = scene
    first = _insert(slam, descriptors, {0: shared})
    first_ids = first.landmark_ids.copy()
    assert len(slam.map.landmarks) == 3
    second = _insert(slam, descriptors, dict(enumerate(first_ids)), index=6, x=.005)
    assert second.id == 6 and len(slam.map.landmarks) == 3
    assert np.array_equal(second.landmark_ids, first_ids)
    assert all(attempt['unowned_correspondences'] == 0
               for attempt in slam._monocular_mapping_report['attempts'])
    for lid in set(first_ids):
        assert second.id in slam.map.landmarks[int(lid)].observations
        assert type(slam.map.landmarks[int(lid)].observations[second.id]) is OwnedObservation
    _assert_reciprocal_unique_observations(slam)


@pytest.mark.parametrize('stale_kind', ['deleted_landmark', 'missing_forward_claim', 'missing_reverse_claim'])
def test_keyframe_does_not_reach_older_views_through_stale_support(scene, stale_kind):
    slam, descriptors, shared, calls = scene
    if stale_kind == 'deleted_landmark':
        del slam.map.landmarks[shared]
    elif stale_kind == 'missing_forward_claim':
        for ident in (0, 1, 2):
            slam.map.keyframes[ident].landmark_ids[0] = -1
    else:
        for ident in (0, 1, 2):
            del slam.map.landmarks[shared].observations[ident]
    current = _insert(slam, descriptors, {0: shared})
    assert slam._monocular_mapping_report['extra_reference_ids'] == []
    assert calls == [4, 4, 3]
    assert len(slam.map.landmarks) == (0 if stale_kind == 'deleted_landmark' else 1)
    assert np.all(current.landmark_ids[1:] == -1)
