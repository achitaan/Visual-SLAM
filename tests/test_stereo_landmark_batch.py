import numpy as np
import pytest

from owned_observation import OwnedObservation
from shared_slam import SharedSlam
from slam_state import MappingKeyframe


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_verified_stereo_batch_preserves_landmark_and_pixel_provenance(dtype):
    slam = SharedSlam(np.eye(3))
    try:
        pixels = np.array([[12.5, 4.25], [10.125 + 1e-8, 8.375], [10.125, 8.375],
                           [3.0, 8.0], [3.0, 8.0], [4.0, 5.0], [8.0, 11.0]], dtype=dtype)
        desc = np.arange(len(pixels) * 128, dtype=np.float32).reshape(-1, 128)
        frame = MappingKeyframe(2, 25, np.eye(4), pixels, desc, np.full(len(pixels), -1),
                                octaves=np.arange(len(pixels), dtype=np.int16))
        slam.map.keyframes[frame.id] = frame
        points = np.array([[1.0, 2.0, 8.0], [3.0, 1.0, 9.0], [3.0, 1.0, 9.0],
                           [2.0, 5.0, 7.0], [2.0, 6.0, 7.0], [1.0, 0.0, 4.0],
                           [np.nan, 0.0, 7.0]])
        rights = np.array([9.0, 8.0, 8.0, 1.0, 1.0, 2.0, np.nan])
        groups, group_ids = slam._physical_pixel_groups(pixels)
        pose = np.eye(4)
        pose[:3, 3] = [0.25, -0.5, 1.0]
        slam._insert_verified_stereo_landmarks(frame, pose, points, rights, groups, group_ids,
                                               {int(group_ids[5])})

        # The first point is a singleton, the second has two SIFT orientations,
        # and the remaining groups disagree, are blocked, or lack valid depth.
        np.testing.assert_array_equal(frame.landmark_ids, [0, 1, 1, -1, -1, -1, -1])
        assert list(slam.map.landmarks) == [0, 1]
        assert slam.map.index.landmarks_for_keyframes([frame.id]) == {0, 1}
        for lid in (0, 1):
            landmark = slam.map.landmarks[lid]
            selected_dtype = np.float32 if lid == 0 else dtype
            observation = landmark.observations[frame.id]
            scalar_reference = OwnedObservation(np.asarray(pixels[lid], dtype=selected_dtype), rights[lid])
            np.testing.assert_array_equal(landmark.position, points[lid] + pose[:3, 3])
            np.testing.assert_array_equal(landmark.descriptor, desc[lid])
            assert landmark.octave == lid
            assert observation.pixel.dtype == np.dtype(selected_dtype)
            assert observation.pixel.tobytes() == scalar_reference.pixel.tobytes()
            assert observation.right_u == scalar_reference.right_u
            assert observation._pixel_id_i32 == scalar_reference._pixel_id_i32
    finally:
        slam.close(finish=False)
