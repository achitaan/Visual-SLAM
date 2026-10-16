from pathlib import Path
import sys,os,copy,pickle,warnings,ast,hashlib
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'tests/fixtures/owned_observation_control.py'
SOURCE=ROOT/'src'
sys.path.insert(0,str(SOURCE))
import owned_observation
import shared_slam
from owned_observation import OwnedObservation
assert Path(owned_observation.__file__).resolve().parent==SOURCE.resolve()
assert Path(shared_slam.__file__).resolve().parent==SOURCE.resolve()


@pytest.mark.parametrize('edit',['strides','dtype','shape','asarray','base','slice'])
def test_snapshot_metadata_cannot_mutate_committed_value(edit):
    o=OwnedObservation(np.array([1.,2.],np.float64));snapshot=o.pixel
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning)
        if edit=='strides':snapshot.strides=(0,)
        elif edit=='dtype':snapshot.dtype=np.float32
        elif edit=='shape':snapshot.shape=(1,2)
        elif edit=='asarray':np.asarray(snapshot).strides=(0,)
        elif edit=='base':snapshot.base.strides=(0,)
        elif edit=='slice':snapshot[::-1].dtype=np.int64
    assert o.pixel.shape==(2,) and o.pixel.dtype==np.dtype(np.float64)
    assert np.array_equal(o.pixel,[1.,2.])
    assert o._identity_key==(1.,2.)
    with pytest.raises(ValueError):o.pixel[0]=4


@pytest.mark.parametrize('clone',[copy.copy,copy.deepcopy,lambda o:pickle.loads(pickle.dumps(o))])
def test_copied_snapshot_edits_are_local_and_explicit_assignment_rekeys(clone):
    o=clone(OwnedObservation(np.array([3.,4.])))
    snapshot=o.pixel
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning);snapshot.strides=(0,)
    assert o._identity_key==tuple(np.asarray(o.pixel,np.float32))==(3.,4.)
    o.pixel=snapshot
    assert o._identity_key==tuple(np.asarray(o.pixel,np.float32))==(3.,3.)


def test_source_alias_and_new_snapshot_identity():
    external=np.array([5.,6.]);o=OwnedObservation(external);external[:]=100
    assert np.array_equal(o.pixel,[5.,6.])
    assert o.pixel is not o.pixel
    with pytest.raises(ValueError):o.pixel.setflags(write=True)


def test_actual_original_conflicts_after_metadata_edit_and_explicit_commit():
    from shared_slam import SharedSlam
    # Same frozen original oracle used by the value-ownership regression tests.
    ns={}
    exec(compile(CONTROL.read_text(),str(CONTROL),'exec'),ns)
    golden=ns['_landmark_pixel_identity_conflicts']
    slam=SharedSlam.__new__(SharedSlam)
    slam.map=SimpleNamespace(keyframes={0:SimpleNamespace(pixels=np.empty((0,2)),landmark_ids=np.empty(0,int))})
    slam._identity_conflict_cache={}
    a=OwnedObservation(np.array([1.,2.]));b=OwnedObservation(np.array([1.,1.]))
    landmarks=[SimpleNamespace(id=1,observations={0:a}),SimpleNamespace(id=2,observations={0:b})]
    snapshot=a.pixel
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning);snapshot.strides=(0,)
    assert slam._landmark_pixel_identity_conflicts(landmarks)==golden(slam,landmarks)==set()
    a.pixel=snapshot
    assert slam._landmark_pixel_identity_conflicts(landmarks)==golden(slam,landmarks)=={1,2}
