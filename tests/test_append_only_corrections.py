"""Corrections retain the map gauge and may only cross verified append-only progress."""
import numpy as np
import pytest
from slam_state import Landmark, MapState, MappingKeyframe, Observation


def pose(x):
    p=np.eye(4);p[0,3]=x;return p


def keyframe(i):
    return MappingKeyframe(i,i*10,pose(i),np.zeros((1,2)),np.zeros((1,128)),np.array([-1]))


@pytest.mark.parametrize('metric',[True,False])
def test_append_only_correction_updates_new_geometry_and_current_pose(metric):
    state=MapState(metric)
    state.keyframes={i:keyframe(i) for i in range(2)}
    source={i:k.pose.copy() for i,k in state.keyframes.items()}
    state.keyframes[2]=keyframe(2);state.revision+=1
    state.record(pose(2.2),'tracking',2)
    state.add_landmark(np.array([2,0,5]),np.ones(128),2,{2:Observation(np.ones(2))})
    scale=1.0 if metric else 1.5
    assert state.apply_snapshot_corrections(0,0,source,{0:pose(0),1:pose(.5)},np.array([1.,scale]))
    assert state.poses[-1][0,3]==pytest.approx(.5+scale*1.2)
    assert state.keyframes[2].pose[0,3]==pytest.approx(.5+scale)
    assert state.landmarks[0].position[0]==pytest.approx(.5+scale)
    assert state.geometry_revision==1


def test_conflicting_bundle_update_is_rejected_without_mutation():
    state=MapState(True);state.keyframes={i:keyframe(i) for i in range(3)}
    source={i:k.pose.copy() for i,k in state.keyframes.items()}
    assert state.apply_corrections(0,{0:pose(0),1:pose(.8),2:pose(1.8)})
    state.keyframes[3]=keyframe(3);state.revision+=1
    before=state.keyframes[2].pose.copy()
    assert not state.apply_snapshot_corrections(0,0,source,source)
    assert np.array_equal(state.keyframes[2].pose,before)


@pytest.mark.parametrize('count',[300,1000])
def test_large_correction_propagation_is_finite_and_consistent(count):
    # Validates map application, not the large-graph optimizer or its production guard.
    state=MapState(True);state.keyframes={i:keyframe(i) for i in range(count)}
    for i in range(count):state.record(pose(i),'tracking',i)
    source={i:k.pose.copy() for i,k in state.keyframes.items()}
    corrected={i:pose(i*.9) for i in range(count)}
    assert state.apply_snapshot_corrections(0,0,source,corrected)
    assert all(np.isfinite(p).all() for p in state.poses)
    assert state.poses[-1][0,3]==pytest.approx((count-1)*.9)


def test_deferred_landmark_positions_survive_swap_growth_and_rebuild():
    state = MapState(metric=True, defer_bulk_landmark_positions=True)
    state.keyframes = {i: keyframe(i) for i in range(2)}
    points = np.random.default_rng(12).normal(size=(13_000, 3))
    count = len(points)

    def add(anchor):
        return state.add_landmarks_batch(points, np.zeros((count, 8), dtype=np.uint8), anchor,
                                         np.zeros((count, 2), dtype=np.float32), np.ones(count),
                                         np.zeros(count, dtype=np.int16))

    ids = add(1)
    old_position = state.landmarks[ids[1]].position
    assert state.apply_corrections(state.revision, {0: state.keyframes[0].pose, 1: pose(1.5)})
    assert state.index._position_epoch == 1
    dropped = state.remove_landmark(ids[1])
    # Changing the anchor from x=1 to x=1.5 translates only its landmarks by 0.5 m.
    translation = np.array([.5, 0., 0.])
    np.testing.assert_array_equal(old_position, points[1])
    np.testing.assert_array_equal(dropped.position, points[1] + translation)
    np.testing.assert_array_equal(state.landmarks[ids[-1]].position, points[-1] + translation)

    updated = points[5] + 2.
    assert state.apply_corrections(state.revision, {i: frame.pose for i, frame in state.keyframes.items()},
                                   propagate_landmarks=False, landmark_updates={ids[5]: updated})
    added = add(0)
    state.index.rebuild_from_map(state)
    np.testing.assert_array_equal(state.landmarks[ids[5]].position, updated)
    np.testing.assert_array_equal(state.landmarks[ids[6]].position, points[6] + translation)
    np.testing.assert_array_equal(state.landmarks[added[-1]].position, points[-1])

    assert state.apply_corrections(state.revision, {0: state.keyframes[0].pose, 1: pose(1.75)})
    assert state.index._position_epoch == 2
    # The second anchor move adds 0.25 m even to an explicitly updated landmark.
    np.testing.assert_array_equal(state.landmarks[ids[5]].position, updated + [.25, 0., 0.])
    np.testing.assert_array_equal(state.landmarks[added[-1]].position, points[-1])

    # A legacy landmark installed by rebuilding the index retains its ordinary position field.
    detached = state.landmarks[ids[8]]
    old_value = points[8] + [.75, 0., 0.]
    state.landmarks[ids[8]] = Landmark(ids[8], old_value.copy(), detached.descriptor,
                                      detached.anchor, detached.observations)
    state.index.rebuild_from_map(state)
    assert state.apply_corrections(state.revision, {0: state.keyframes[0].pose, 1: pose(2.)})
    assert state.index._position_epoch == 2
    np.testing.assert_allclose(detached.position, old_value, rtol=0., atol=1e-12)
    np.testing.assert_allclose(state.landmarks[ids[8]].position, old_value + [.25, 0., 0.],
                               rtol=0., atol=1e-12)
