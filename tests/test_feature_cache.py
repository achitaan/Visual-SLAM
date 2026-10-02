import numpy as np
from feature_cache import FeatureCache, extraction_signature
import cv2 as cv
from shared_slam import SharedSlam, MappingConfig, StereoCamera


def test_cache_identity_arrays_and_extraction_revision(tmp_path):
    cache=FeatureCache(tmp_path,'revision-one')
    image=np.zeros((20,30),np.uint8);other=image.copy();other[0,0]=1
    key=cache.key(image,image)
    assert key!=cache.key(other,image) and key!=cache.key(image,other)
    assert cache.get(key) is None
    values=(np.zeros((2,2)),np.ones((2,128)),np.ones((2,3)),np.ones(2),np.ones(image.shape))
    cache.put(key,values)
    assert all(np.array_equal(a,b) for a,b in zip(values,cache.get(key)))
    assert FeatureCache(tmp_path,'revision-two').get(key) is None
    assert cache.metadata()['runtime_is_cached_diagnostic']


def test_cache_evicts_only_owned_files_under_limit(tmp_path):
    cache=FeatureCache(tmp_path,'owned',max_bytes=1)
    values=tuple(np.ones(10) for _ in range(5))
    sentinel=tmp_path/'original-input.png';sentinel.write_bytes(b'preserve')
    cache.put('key',values)
    assert not list(cache.folder.glob('*.npz')) and sentinel.read_bytes()==b'preserve'


def test_extraction_identity_reuses_tracking_changes_but_rejects_parameter_changes():
    matrix=np.array([[250.,0,320],[0,250,240],[0,0,1.]])
    camera=StereoCamera(cv.StereoSGBM_create(numDisparities=96,blockSize=5),np.eye(4),.54)
    first=SharedSlam(matrix,stereo=camera)
    second=SharedSlam(matrix,stereo=camera,config=MappingConfig(bundle_enabled=False,keyframe_interval=20))
    expected=extraction_signature(first,cv)
    assert extraction_signature(second,cv)==expected
    first.detector.setContrastThreshold(.08)
    assert extraction_signature(first,cv)!=expected
    camera.stereo.setBlockSize(7)
    assert extraction_signature(second,cv)!=expected
    first.close();second.close()


def test_extraction_identity_includes_calibration_and_extractor_source(monkeypatch):
    matrix=np.array([[250.,0,320],[0,250,240],[0,0,1.]])
    slam=SharedSlam(matrix)
    expected=extraction_signature(slam,cv)
    slam.K[0,0]+=1
    assert extraction_signature(slam,cv)!=expected
    slam.K[0,0]-=1
    assert extraction_signature(slam,cv)==expected

    def changed_extract(self,image,right):
        raise RuntimeError('Changed extraction implementation')

    monkeypatch.setattr(SharedSlam,'_extract',changed_extract)
    assert extraction_signature(slam,cv)!=expected
    slam.close()
