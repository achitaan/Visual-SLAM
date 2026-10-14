"""Verified stereo velocity can seed matching without fabricating accepted poses."""
import numpy as np
from shared_slam import SharedSlam, StereoCamera


def pose(x):
    value=np.eye(4);value[0,3]=x
    return value


def test_held_frames_do_not_erase_verified_velocity_or_become_estimated_motion():
    slam=SharedSlam(np.array([[250.,0,320],[0,250,240],[0,0,1.]]),
                    stereo=StereoCamera(None,np.eye(4),.54))
    for p,status in [(pose(0),'tracking'),(pose(2),'tracking'),(pose(2),'lost'),(pose(2),'lost')]:
        slam.map.record(p,status)
    slam.verified_stereo_motion=(pose(2),1)
    before=[p.copy() for p in slam.map.poses]
    assert np.allclose(slam._motion_prediction(),pose(8))
    assert slam.motion_prediction_source=='verified_stereo_increment'
    assert all(np.array_equal(a,b) for a,b in zip(before,slam.map.poses))
    assert slam.map.statuses[-2:]==['lost','lost'] and not slam.map.landmarks
    for _ in range(3):slam.map.record(pose(2),'lost')
    assert np.array_equal(slam._motion_prediction(),pose(2))
    assert slam.motion_prediction_source=='held_pose'
    slam.close()


def test_stereo_prior_does_not_enter_monocular_prediction():
    slam=SharedSlam(np.array([[250.,0,320],[0,250,240],[0,0,1.]]))
    slam.map.record(pose(0),'tracking');slam.map.record(pose(.5),'tracking')
    slam.verified_stereo_motion=(pose(10),1)
    assert np.allclose(slam._motion_prediction(),pose(1))
    assert slam.motion_prediction_source=='consecutive_accepted_poses'
    slam.close()
