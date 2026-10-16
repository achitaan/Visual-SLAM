from pathlib import Path
import sys,ast,copy,pickle,hashlib
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'tests/fixtures/owned_observation_control.py'
CANDIDATE=ROOT/'src'
sys.path.insert(0,str(CANDIDATE))
import shared_slam
from shared_slam import SharedSlam
from slam_state import Observation
import owned_observation
from owned_observation import OwnedObservation
assert Path(owned_observation.__file__).resolve().parent==(ROOT/'src').resolve()
assert Path(shared_slam.__file__).resolve().parent==CANDIDATE.resolve()


def original_method():
    # Expected behavior is the independently frozen pre-optimization algorithm.
    namespace={}
    exec(compile(CONTROL.read_text(),str(CONTROL),'exec'),namespace)
    return namespace['_landmark_pixel_identity_conflicts']


GOLDEN=original_method()


def tracker(keyframes):
    slam=SharedSlam.__new__(SharedSlam);slam.map=SimpleNamespace(keyframes=keyframes);slam._identity_conflict_cache={}
    return slam


def equal(slam,landmarks):
    expected=GOLDEN(slam,landmarks)
    assert slam._landmark_pixel_identity_conflicts(landmarks)==expected
    return expected


@pytest.mark.parametrize('value',[np.array([1.,-0.],np.float32),np.array([[1.],[2.]],np.float64),
                                 np.array([1.,np.nan]),np.array([np.inf,2.]),np.arange(4,dtype=float)[::2],
                                 np.array([np.nextafter(np.float32(0),np.float32(1)),0.],np.float32)])
def test_owned_pixel_values_immutable_and_assignment(value):
    o=OwnedObservation(value,3.)
    assert isinstance(o,Observation)
    assert o.pixel.shape==value.shape and o.pixel.dtype==value.dtype
    assert np.array_equal(o.pixel,value,equal_nan=True)
    with pytest.raises(ValueError):o.pixel.setflags(write=True)
    with pytest.raises(ValueError):np.asarray(o.pixel).reshape(-1)[0]=99
    expected=tuple(np.asarray(value,np.float32).reshape(-1).tolist()) if np.isfinite(value).all() and value.size==2 else None
    assert o._identity_key==expected
    o.pixel=np.array([7.,8.]);assert o._identity_key==(7.,8.)


@pytest.mark.parametrize('operation',[copy.copy,copy.deepcopy,lambda x:pickle.loads(pickle.dumps(x))])
def test_copies_cannot_reenable_mutation(operation):
    o=operation(OwnedObservation(np.array([2.,3.]),5.))
    assert o._identity_key==(2.,3.) and o.right_u==5.
    with pytest.raises(ValueError):np.asarray(o.pixel).setflags(write=True)
    o.pixel=[4.,5.];assert o._identity_key==(4.,5.)


def test_legacy_mutations_missing_frames_and_hook_changes(monkeypatch):
    keyframe=SimpleNamespace(pixels=np.empty((0,2)),landmark_ids=np.empty(0,int))
    slam=tracker({0:keyframe})
    legacy=Observation(np.array([4.,5.]))
    a=SimpleNamespace(id=1,observations={0:OwnedObservation([1.,2.]),5:OwnedObservation([9.,9.])})
    b=SimpleNamespace(id=2,observations={0:legacy})
    assert equal(slam,[a,b])==set()
    legacy.pixel[:]=[1,2];assert equal(slam,[a,b])=={1,2}
    a.observations[0].pixel=[7,8];assert equal(slam,[a,b])==set()
    monkeypatch.setattr(SharedSlam,'_physical_pixel_key',staticmethod(lambda p:(0.,0.)))
    assert equal(slam,[a,b])=={1,2}


