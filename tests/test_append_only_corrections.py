"""Corrections retain the map gauge and may only cross verified append-only progress."""
import numpy as np
import pytest
from slam_state import MapState, MappingKeyframe, Observation


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
