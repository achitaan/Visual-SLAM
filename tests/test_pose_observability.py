import numpy as np
import pytest

import native_accel
from mapping_geometry import project, right_pixel
from pose_observability import original_image_pose_observability


def test_native_nullspace_preserves_camera_information_and_defers_rank_deficiency():
    library = native_accel._get_ba_lib()
    if library is None or not hasattr(library, "project_rank3_camera_rows"):
        pytest.skip("native point projection is unavailable")
    rng = np.random.default_rng(857)
    points = rng.normal(size=(3, 9, 3))
    poses = rng.normal(size=(3, 9, 6))
    slots = np.broadcast_to(np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int32), (3, 9))
    actual = native_accel.project_rank3_camera_rows(points, poses, slots, 12)
    assert actual is not None and actual.shape == (3, 6, 12)

    camera = np.zeros((3, 9, 12))
    camera[:, 3:6, :6] = poses[:, 3:6, :]
    camera[:, 6:, 6:] = poses[:, 6:, :]
    u, _, _ = np.linalg.svd(points, full_matrices=True)
    reference = u[..., 3:].swapaxes(-1, -2) @ camera
    # The complement's rows may use a different orthonormal basis. Its
    # camera information matrix is invariant under that change of basis.
    np.testing.assert_allclose(actual.swapaxes(-1, -2) @ actual,
                               reference.swapaxes(-1, -2) @ reference, rtol=1e-12, atol=1e-12)

    points[:, :, 2] = points[:, :, 0] + 1e-13 * points[:, :, 2]
    assert native_accel.project_rank3_camera_rows(points, poses, slots, 12) is None


def _stereo_rank_input():
    rng = np.random.default_rng(803)
    points = rng.uniform((-1.6, -1.5, 4.0), (1.6, 1.5, 12.0), size=(24, 3))
    cameras = np.stack([np.eye(4), np.eye(4)])
    cameras[1, 0, 3] = 0.6
    matrix = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    camera_ids = np.tile([0, 1], len(points))
    point_ids = np.repeat(np.arange(len(points)), 2)
    pixels_depths = [project(points, pose, matrix) for pose in cameras]
    pixels = np.stack([item[0] for item in pixels_depths], axis=1).reshape(-1, 2)
    depth = np.stack([item[1] for item in pixels_depths], axis=1).ravel()
    return (
        cameras, np.array([1]), np.zeros((1, 3)), points, camera_ids, point_ids,
        pixels, right_pixel(pixels[:, 0], depth, 500., 0.2, 0.),
        np.ones((len(camera_ids), 3), dtype=bool), np.empty((0, 3)), np.empty(0, dtype=int),
        np.empty((0, 2)), np.empty(0), np.empty((0, 3), dtype=bool), matrix, 0.2, 0.,
        np.ones((1, 6)), np.ones((len(points), 3)),
    )


@pytest.mark.parametrize("fallback", ["missing", "weak"])
def test_observability_keeps_the_reference_report_on_fallback(monkeypatch, fallback):
    args = _stereo_rank_input()
    reference = original_image_pose_observability(*args)
    assert reference["status"] == "observable" and reference["rank"] == 6

    def project_block(points, _camera, _slots, columns):
        if fallback == "missing":
            return None
        return np.zeros((len(points), points.shape[1] - 3, columns))

    monkeypatch.setattr(native_accel, "project_rank3_camera_rows", project_block)
    actual = original_image_pose_observability(*args, use_native_projection=True)
    del reference["elapsed_s"], actual["elapsed_s"]
    assert actual == reference
