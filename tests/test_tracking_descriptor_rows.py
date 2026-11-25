"""Check descriptor catalog invalidation on normal map edits."""

import numpy as np

from shared_slam import SharedSlam
from slam_state import MapState, MappingKeyframe, Observation
from tracking_descriptor_rows import TrackingDescriptorRows


def test_tracking_descriptor_rows_follow_observation_and_keyframe_edits():
    state = MapState(metric=True, defer_bulk_landmark_positions=True)
    first = np.broadcast_to(np.arange(1, 7, dtype=np.float32)[:, None], (6, 128)).copy()
    second = np.full((1, 128), 9., dtype=np.float32)
    first.setflags(write=False)
    second.setflags(write=False)
    pixel = np.array([20., 15.], dtype=np.float32)
    old = MappingKeyframe(0, 0, np.eye(4), np.tile(pixel, (6, 1)), first, np.zeros(6, dtype=int))
    recent = MappingKeyframe(1, 1, np.eye(4), pixel[None, :], second, np.array([-1]))
    state.keyframes.update({0: old, 1: recent})
    identifier = state.add_landmark(np.array([0., 0., 10.]), first[0], 0, {0: Observation(pixel)})
    landmark = state.landmarks[identifier]
    cache = TrackingDescriptorRows(
        state, SharedSlam._landmark_descriptor_bank, SharedSlam._invalidate_kf_descriptor_cache,
    )
    state.index._tracking_descriptor_rows = cache

    def check(values):
        descs, ids = cache.bank([landmark])
        np.testing.assert_array_equal(descs, values)
        np.testing.assert_array_equal(ids, np.full(len(values), identifier, dtype=int))
        assert ids.dtype == np.dtype(int)

    check(first)
    check(first)
    recent.landmark_ids[0] = identifier
    state.add_observation(identifier, 1, Observation(pixel))
    check(np.vstack((first, second)))

    old.landmark_ids[0] = -1
    old.landmark_ids_version += 1
    state.landmark_ids_version += 1
    check(np.vstack((first[1:], second)))

    state.remove_observation(identifier, 1)
    check(first[1:])
    state.index.rebuild_from_map(state)
    check(first[1:])
    state.remove_landmark(identifier)
    assert identifier not in cache._lm_desc_cache
