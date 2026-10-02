"""A local camera refinement must not rigidly move landmarks excluded from its solve."""
import numpy as np
from local_bundle import local_bundle_adjustment
from mapping_geometry import project
from slam_state import MapState,MappingKeyframe,Observation


def test_unoptimized_world_landmark_survives_local_camera_refinement():
    matrix=np.array([[250.,0,320],[0,250,240],[0,0,1.]])
    world=np.random.default_rng(22).uniform([-2,-2,5],[2,2,12],(24,3))
    state=MapState(metric=True)
    for i,x in enumerate([0,.5,1.]):
        truth=np.eye(4);truth[0,3]=x
        pixels,z=project(world,truth,matrix)
        estimated=truth.copy()
        if i==2:estimated[0,3]+=.1
        state.keyframes[i]=MappingKeyframe(i,i,estimated,pixels,np.zeros((24,128)),np.arange(24))
        state.record(estimated,'tracking',i)
    for j,point in enumerate(world):
        observations={}
        for i,x in enumerate([0,.5,1.]):
            truth=np.eye(4);truth[0,3]=x
            pixel,z=project(point[None],truth,matrix)
            observations[i]=Observation(pixel[0],float(pixel[0,0]-250*.2/z[0]))
        state.add_landmark(point,np.ones(128),2,observations)
    excluded=state.landmarks[23].position.copy()
    result=local_bundle_adjustment(state,matrix,.2,window=3,max_landmarks=20)
    assert result['applied'] and result['final_cost']<result['initial_cost']
    assert abs(state.keyframes[2].pose[0,3]-1.1)>1e-3
    assert np.allclose(state.landmarks[23].position,excluded,atol=1e-8)
