"""Relative stereo motion must account for verified depth in both directions."""
import numpy as np
from mapping_geometry import project, estimate_stereo_reference
from loop_geometry import StereoLoopFrame


def test_two_depth_views_reduce_one_direction_scale_bias():
    rng=np.random.default_rng(803)
    points=rng.uniform([-20.,-12.,20.],[20.,12.,60.],(180,3))
    matrix=np.array([[250.,0,320],[0,250,240],[0,0,1.]])
    truth=np.eye(4);truth[0,3]=1.5
    first,_=project(points,np.eye(4),matrix)
    second,_=project(points,truth,matrix)
    descriptors=rng.normal(size=(len(points),128)).astype(np.float32)
    source=StereoLoopFrame(first,points*.97,descriptors,(640,480))
    target=StereoLoopFrame(second,(points-truth[:3,3])*1.03,descriptors.copy(),(640,480))
    result=estimate_stereo_reference(source,target,matrix)
    assert result is not None and result['reverse_checked']
    refinement=result['bidirectional_refinement']
    assert refinement['applied'] and refinement['final_cost']<refinement['initial_cost']
    assert abs(result['measurement'][0,3]-1.5)<abs(1.5*.97-1.5)*.5


def test_refinement_cannot_turn_false_descriptors_into_verified_motion():
    rng=np.random.default_rng(804)
    matrix=np.array([[250.,0,320],[0,250,240],[0,0,1.]])
    points=rng.uniform([-20.,-12.,20.],[20.,12.,60.],(180,3))
    pixels,_=project(points,np.eye(4),matrix)
    source=StereoLoopFrame(pixels,points,rng.normal(size=(180,128)).astype(np.float32),(640,480))
    target=StereoLoopFrame(pixels,points,rng.normal(size=(180,128)).astype(np.float32),(640,480))
    assert estimate_stereo_reference(source,target,matrix) is None
