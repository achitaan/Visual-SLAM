"""Manufactured contracts for the exact supplied October23 consensus overlay."""
import cv2 as cv
import numpy as np
import pytest
import mapping_geometry as geometry

K=np.array([[250.,0.,320.],[0.,250.,240.],[0.,0.,1.]])
SIZE=(640,480)


def scene(n=24):
    pixels=np.array([[80.+160.*(i%4),80.+160.*((i//4)%3)] for i in range(n)])
    depths=np.full(n,10.)
    points=np.column_stack(((pixels[:,0]-320.)/250.*depths,
                            (pixels[:,1]-240.)/250.*depths,depths))
    return points,pixels


def validated(points,pixels,ids=None,tv=None,minimum=10,rv=None):
    diagnostic={}
    result=geometry._validated_refined_consensus(points,pixels,K,SIZE,minimum,
        np.arange(len(points)) if ids is None else np.array(ids),
        np.zeros((3,1)) if rv is None else rv,
        np.zeros((3,1)) if tv is None else tv,diagnostic)
    return result,diagnostic


def test_pruning_is_final_and_extra_fit_is_bounded(monkeypatch):
    points,pixels=scene()
    points[:2]*=np.array([.1,.2])[:,None]
    calls=[]
    def refit(*args):
        calls.append(args)
        return np.zeros((3,1)),np.array([[.02],[0.],[0.]])
    monkeypatch.setattr(cv,'solvePnPRefineLM',refit)
    result,report=validated(points,pixels,tv=np.array([[.01],[0.],[0.]]))
    pose,ids,_=result
    assert ids.tolist()==list(range(2,24))
    assert len(calls)==1 and report['consensus_refits']==1
    predicted,depth=geometry.project(points[ids],pose,K)
    assert np.all(np.isfinite(predicted)) and np.all(depth>0)
    assert np.max(np.linalg.norm(predicted-pixels[ids],axis=1))<=2.
    assert report['refined_consensus_rejected']==2


def test_original_denominator_not_surviving_mask(monkeypatch):
    points,pixels=scene(48)
    points[0]*=.1
    monkeypatch.setattr(cv,'solvePnPRefineLM',lambda *a:pytest.fail('Count gate should reject before refit'))
    result,report=validated(points,pixels,ids=range(12),tv=np.array([[.02],[0.],[0.]]))
    assert result is None
    assert report['candidate_inliers']==11
    assert report['candidate_inlier_ratio']==11/48
    assert report['pose_rejection_reason']=='insufficient_refined_inlier_support'


def test_never_recruits_non_ransac_rows():
    points,pixels=scene()
    result,report=validated(points,pixels,ids=range(12))
    assert result[1].tolist()==list(range(12))
    assert report['consensus_refits']==0


def test_exact_two_pixel_boundary_is_inclusive():
    points,pixels=scene()
    pixels[0,0]+=2.
    result,_=validated(points,pixels)
    assert 0 in result[1]


@pytest.mark.parametrize('field',['rv','tv'])
def test_nonfinite_pose_is_rejected(field):
    points,pixels=scene()
    result,report=validated(points,pixels,**{field:np.full((3,1),np.nan)})
    assert result is None and report['pose_rejection_reason']=='nonfinite_refined_pose'


def test_median_gate_remains_unchanged():
    points,pixels=scene();pixels[:,0]+=1.6
    result,report=validated(points,pixels)
    assert result is None and report['pose_rejection_reason']=='refined_reprojection_error'


def test_finite_positive_depth_and_count_gates(monkeypatch):
    points,pixels=scene()
    points[0]*=-1
    points[1]=[np.nan,0.,1.]
    monkeypatch.setattr(cv,'solvePnPRefineLM',lambda *a:(np.zeros((3,1)),np.zeros((3,1))))
    result,report=validated(points,pixels)
    assert result[1].tolist()==list(range(2,24))
    result,report=validated(points,pixels,minimum=23)
    assert result is None and report['candidate_inliers']==22


@pytest.mark.parametrize('scale',[1e-150,1e-12,1.,1e100])
def test_filter_uses_actual_depth_under_positive_scaling(scale):
    points,pixels=scene()
    result,_=validated(points*scale,pixels)
    assert result[1].tolist()==list(range(24))
    assert result[2]<1e-10


def test_seed_failure_still_calls_independent_hypothesis(monkeypatch):
    points,pixels=scene();calls=[]
    def hypothesis(*args,**kwargs):
        calls.append(kwargs)
        return None if len(calls)==1 else (np.eye(4),np.arange(24),0.)
    monkeypatch.setattr(geometry,'_estimate_pose_hypothesis',hypothesis)
    report={}
    result=geometry.estimate_pose(points,pixels,K,SIZE,initial_pose=np.eye(4),
                                   diagnostics=report,revalidate_inliers=True)
    assert result is not None and len(calls)==2
    assert all(call['revalidate_inliers'] for call in calls)
    assert report['pnp_method']=='epnp_ransac'


def test_disabled_wrapper_does_not_add_private_keyword(monkeypatch):
    points,pixels=scene()
    def legacy_solver(points,pixels,matrix,size,min_inliers=15,initial_pose=None,diagnostics=None):
        return np.eye(4),np.arange(len(points)),0.
    monkeypatch.setattr(geometry,'_estimate_pose_hypothesis',legacy_solver)
    assert geometry.estimate_pose(points,pixels,K,SIZE) is not None
