"""A local camera refinement must not rigidly move landmarks excluded from its solve."""
import numpy as np
from types import SimpleNamespace
import pytest
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
    singleton=world[0]+np.array([.1,0,0])
    singleton_pixel,z=project(singleton[None],state.keyframes[2].pose,matrix)
    singleton_id=state.add_landmark(singleton,np.ones(128),2,{2:Observation(singleton_pixel[0],float(singleton_pixel[0,0]-250*.2/z[0]))})
    result=local_bundle_adjustment(state,matrix,.2,window=3,max_landmarks=20)
    assert result['applied'] and result['final_cost']<result['initial_cost']
    assert abs(state.keyframes[2].pose[0,3]-1.1)>1e-3
    assert np.linalg.norm(state.keyframes[2].pose[:3, 3] - [1., 0., 0.]) < 1e-3
    assert np.linalg.norm(state.keyframes[1].pose[:3, 3] - [.5, 0., 0.]) < 1e-3
    assert result['affected_final_cost'] < result['affected_initial_cost'] * 1e-4
    assert np.allclose(state.landmarks[23].position,excluded,atol=1e-8)
    assert result['anchor_propagated_single_view_landmarks']==1
    pixel_after,_=project(state.landmarks[singleton_id].position[None],state.keyframes[2].pose,matrix)
    assert np.allclose(pixel_after,singleton_pixel,atol=1e-8)


def test_subset_improvement_cannot_override_worse_map_observations(monkeypatch):
    import local_bundle as module
    matrix=np.array([[250.,0,320],[0,250,240],[0,0,1.]])
    world=np.random.default_rng(23).uniform([-2,-2,5],[2,2,12],(120,3))
    state=MapState(metric=True)
    for i,x in enumerate([0,.5,1.1]):
        pose=np.eye(4);pose[0,3]=x
        state.keyframes[i]=MappingKeyframe(i,i,pose,np.empty((0,2)),np.empty((0,128)),np.empty(0,int))
    for j,point in enumerate(world):
        observations={}
        for i,x in enumerate([0,.5,1. if j<20 else 1.1]):
            pose=np.eye(4);pose[0,3]=x
            pixel,z=project(point[None],pose,matrix)
            observations[i]=Observation(pixel[0],float(pixel[0,0]-250*.2/z[0]))
        state.add_landmark(point,np.ones(128),0,observations)

    def proposed(_residual,initial,**kwargs):
        value=initial.copy();value[9]=1.0
        return SimpleNamespace(x=value,nfev=1,success=True)

    monkeypatch.setattr(module,'least_squares',proposed)
    revision=state.revision
    report=local_bundle_adjustment(state,matrix,.2,window=3,max_landmarks=20)
    assert report['final_cost']<report['initial_cost']
    assert report['affected_final_cost']>report['affected_initial_cost']
    assert report['reason']=='affected_observations_worsened'
    assert not report['applied'] and state.revision==revision
    assert state.keyframes[2].pose[0,3]==1.1


def test_nonfinite_explicit_landmark_update_rejects_entire_commit():
    state=MapState(metric=True)
    for i,x in enumerate([0.,1.]):
        pose=np.eye(4);pose[0,3]=x
        state.keyframes[i]=MappingKeyframe(i,i,pose,np.empty((0,2)),np.empty((0,128)),np.empty(0,int))
    ident=state.add_landmark(np.array([1.,0,5]),np.ones(128),1,{1:Observation(np.array([320.,240.]))})
    before=state.landmarks[ident].position.copy()
    shifted=np.eye(4);shifted[0,3]=2.
    with pytest.raises(ValueError,match='Nonfinite'):
        state.apply_corrections(0,{0:np.eye(4),1:shifted},propagate_landmarks=False,
                                landmark_updates={ident:np.array([np.nan,0,5])})
    assert state.revision==0 and state.geometry_revision==0
    assert state.keyframes[1].pose[0,3]==1.
    assert np.array_equal(state.landmarks[ident].position,before)


@pytest.mark.parametrize('mixed_right', [False, True])
def test_held_out_jacobian_pattern_matches_numeric_derivatives(monkeypatch, mixed_right):
    """Every residual must declare every camera variable that affects it."""
    import local_bundle as module
    matrix = np.array([[250., 0, 320], [0, 250, 240], [0, 0, 1.]])
    world = np.random.default_rng(732).uniform([-2, -2, 5], [2, 2, 12], (24, 3))
    state = MapState(metric=True)
    for ident, x in enumerate([0., .5, 1.]):
        pose = np.eye(4); pose[0, 3] = x
        state.keyframes[ident] = MappingKeyframe(ident, ident, pose,
            np.empty((0, 2)), np.empty((0, 128)), np.empty(0, int))
    for ident, point in enumerate(world):
        observations = {}
        for keyframe, x in enumerate([0., .5, 1.]):
            pose = np.eye(4); pose[0, 3] = x
            pixel, z = project(point[None], pose, matrix)
            right = float(pixel[0, 0] - 250*.2/z[0])
            if mixed_right and (ident + keyframe) % 2:
                right = None
            observations[keyframe] = Observation(pixel[0], right)
        state.add_landmark(point, np.ones(128), 0, observations)

    inspected = []
    def inspect(residual, initial, **kwargs):
        pattern = kwargs['jac_sparsity'].toarray().astype(bool)
        for column in range(12):  # All six variables in each free camera.
            left, right = initial.copy(), initial.copy()
            left[column] -= 1e-6; right[column] += 1e-6
            derivative = (residual(right) - residual(left)) / 2e-6
            undeclared = np.abs(derivative[~pattern[:, column]])
            assert np.max(undeclared, initial=0.) < 1e-5, (
                f'Undeclared residual dependence on camera variable {column}')
        inspected.append(True)
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True)
    monkeypatch.setattr(module, 'least_squares', inspect)
    local_bundle_adjustment(state, matrix, .2, window=3, max_landmarks=20)
    assert inspected == [True]


@pytest.mark.parametrize('metric', [True, False])
@pytest.mark.parametrize('units', [.01, 1., 100.])
def test_bundle_refinement_recovers_known_pose_across_world_units(metric, units):
    matrix = np.array([[250., 0, 320], [0, 250, 240], [0, 0, 1.]])
    world = np.random.default_rng(411).uniform([-2, -2, 5], [2, 2, 12], (32, 3)) * units
    state = MapState(metric=metric)
    truths = []
    for ident, x in enumerate([0., .5, 1., 1.5]):
        truth = np.eye(4); truth[0, 3] = x * units
        truths.append(truth)
        estimate = truth.copy()
        if ident >= 2:
            estimate[0, 3] += .1 * units
        state.keyframes[ident] = MappingKeyframe(ident, ident, estimate,
            np.empty((0, 2)), np.empty((0, 128)), np.empty(0, int))
    for point in world:
        observations = {}
        for ident, truth in enumerate(truths):
            pixel, z = project(point[None], truth, matrix)
            right = float(pixel[0, 0] - 250*.2*units/z[0]) if metric else None
            observations[ident] = Observation(pixel[0], right)
        state.add_landmark(point, np.ones(128), 0, observations)
    result = local_bundle_adjustment(state, matrix, .2*units if metric else 0.,
        window=4, max_landmarks=20)
    assert result['applied']
    assert result['affected_final_cost'] < result['affected_initial_cost'] * 1e-4
    for ident in [2, 3]:
        assert np.linalg.norm(state.keyframes[ident].pose[:3, 3]-truths[ident][:3, 3])/units < 1e-3
