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


def test_pose_arbitration_policy_and_stereo_calibration_change_cache_identity():
    matrix=np.array([[250.,0,320],[0,250,240],[0,0,1.]])
    q=np.eye(4);q[2,3]=100.;q[3,2]=1.;q[3,3]=.5
    stereo=cv.StereoSGBM_create(numDisparities=96,blockSize=5)
    camera=StereoCamera(stereo,q,.54)
    default=SharedSlam(matrix,stereo=camera)
    assert default.config.stereo_pose_arbitration is False
    default_signature=extraction_signature(default,cv)
    enabled=SharedSlam(matrix,stereo=camera,
                       config=MappingConfig(stereo_pose_arbitration=True))
    assert extraction_signature(enabled,cv)!=default_signature
    changed_baseline=SharedSlam(matrix,stereo=StereoCamera(stereo,q,.55))
    assert extraction_signature(changed_baseline,cv)!=default_signature
    default.close();enabled.close();changed_baseline.close()


def test_arbitration_producer_changes_invalidate_cache_signature(monkeypatch):
    matrix=np.array([[250.,0,320],[0,250,240],[0,0,1.]])
    slam=SharedSlam(matrix,stereo=StereoCamera(
        cv.StereoSGBM_create(numDisparities=96,blockSize=5),np.eye(4),.54),
        config=MappingConfig(stereo_pose_arbitration=True))
    expected=extraction_signature(slam,cv)
    producer=SharedSlam._prepare_stereo_arbitration

    def changed_producer(self,index,current):
        return producer(self,index,current)

    monkeypatch.setattr(SharedSlam,'_prepare_stereo_arbitration',changed_producer)
    assert extraction_signature(slam,cv)!=expected
    slam.close()


def test_cache_hit_holdout_uses_fresh_supported_measurement_not_cached_geometry(tmp_path):
    matrix=np.array([[100.,0,32],[0,100,24],[0,0,1.]])
    q=np.eye(4);q[2,3]=100.;q[3,2]=1.;q[3,3]=.5
    slam=SharedSlam(matrix,stereo=StereoCamera(
        cv.StereoSGBM_create(numDisparities=96,blockSize=5),q,.54),
        config=MappingConfig(stereo_pose_arbitration=True))
    cache=FeatureCache(tmp_path,extraction_signature(slam,cv))
    left=np.zeros((48,64),np.uint8);right=left.copy()
    pixels=np.array([[20.25,15.5],[30.5,25.25]],np.float32)
    descriptors=np.ones((2,128),np.float32)
    cached_points=np.full((2,3),999.,np.float32)
    cached_right=np.full(2,999.,np.float32)
    cached_disparity=np.full((48,64),10.,np.float32)
    key=cache.key(left,right)
    cache.put(key,(pixels,descriptors,cached_points,cached_right,cached_disparity))
    restored=cache.get(key)
    slam.current_disparity=restored[4]
    slam._supported_extraction=None

    evidence=slam._capture_supported_stereo(3,restored[0],restored[1],(64,48))
    expected_points,expected_right=slam._measure_supported_stereo_pixels(restored[0])

    assert np.allclose(evidence.points,expected_points,equal_nan=True)
    assert np.allclose(evidence.right_u,expected_right,equal_nan=True)
    assert not np.any(evidence.points==999.)
    assert not np.any(evidence.right_u==999.)
    slam.close()
