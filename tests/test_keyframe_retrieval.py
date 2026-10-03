from types import SimpleNamespace
import numpy as np
from keyframe_retrieval import KeyframeRetrieval


def test_retrieval_is_bounded_deterministic_and_keeps_known_view():
    rng=np.random.default_rng(4)
    frames={i:SimpleNamespace(descriptors=rng.normal(size=(80,128)).astype(np.float32)) for i in range(40)}
    index=KeyframeRetrieval();index.update(frames)
    query=frames[17].descriptors+rng.normal(0,.01,frames[17].descriptors.shape)
    first=index.query(query,count=5)
    assert len(first)==5 and 17 in first
    assert first==index.query(query,count=5)
    assert 17 not in index.query(query,count=5,allowed=[0,1,2])
    del frames[17];index.update(frames)
    assert 17 not in index.query(query,count=40)


def test_appearance_retrieval_cannot_claim_tracking_recovery():
    from shared_slam import SharedSlam,MappingConfig
    matrix=np.array([[250.,0,320],[0,250,240],[0,0,1]])
    slam=SharedSlam(matrix,config=MappingConfig(loop_mode='off'))
    desc=np.ones((40,128),np.float32)
    # Identical appearance with no usable map geometry must fail PnP verification.
    slam.map.keyframes[0]=SimpleNamespace(id=0,descriptors=desc,landmark_ids=np.full(40,-1))
    result,_=slam._relocalize(np.zeros((40,2)),desc,(640,480))
    assert result is None
    slam.close(finish=False)
