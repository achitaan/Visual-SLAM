import numpy as np
from feature_cache import FeatureCache


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
