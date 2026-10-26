"""Analytic numerical prerequisite; distinct from the missing October candidate."""
import numpy as np
import pytest

from mapping_geometry import triangulate, ray_angles_degrees, project


@pytest.mark.parametrize("scale", [1e-300, 1e-150, 1e-7, 1., 1e150, 1e300])
def test_ray_angles_do_not_depend_on_world_scale(scale):
    first = np.array([[0., 0., 1.], [0., 0., 1.]]) * scale
    angle = np.deg2rad(0.1)
    second = np.array([[0., 0., 1.], [np.sin(angle), 0., np.cos(angle)]]) * scale
    assert np.allclose(ray_angles_degrees(first, second), [0., .1], atol=1e-9)


def test_degenerate_and_nonfinite_rays_cannot_pass_angle_gate():
    rays = np.array([[0., 0., 0.], [np.nan, 0., 1.], [np.inf, 0., 1.]])
    assert np.isnan(ray_angles_degrees(rays, np.ones((3, 3)))).all()


@pytest.mark.parametrize("scale", [1., 1e-7])
@pytest.mark.parametrize("angle, accepted", [(.1, False), (2., True)])
def test_manufactured_parallax(scale, angle, accepted):
    first, second = np.eye(4), np.eye(4)
    second[0, 3] = scale * np.tan(np.deg2rad(angle))
    pixels1 = np.array([[0., 0.]])
    pixels2 = np.array([[-np.tan(np.deg2rad(angle)), 0.]])
    _, ids = triangulate(pixels1, pixels2, first, second, np.eye(3),
                         scale_safe_angles=True)
    assert bool(len(ids)) is accepted


def test_disabled_option_preserves_legacy_counterexample():
    first, second = np.eye(4), np.eye(4)
    second[0, 3] = 1e-7 * np.tan(np.deg2rad(.1))
    args = (np.array([[0., 0.]]), np.array([[-np.tan(np.deg2rad(.1)), 0.]]),
            first, second, np.eye(3))
    default = triangulate(*args)
    disabled = triangulate(*args, scale_safe_angles=False)
    assert default[1].tolist() == [0]
    assert np.array_equal(default[0], disabled[0])
    assert np.array_equal(default[1], disabled[1])


@pytest.mark.parametrize("angle, accepted", [(.999, False), (1.001, True)])
def test_near_gate_and_rigid_world_frame_change(angle, accepted):
    # Apply a known proper rotation and translation to both cameras. Pixels do
    # not change. Stay clear of the exact threshold's floating-point ambiguity.
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    first, second = np.eye(4), np.eye(4)
    first[:3, :3] = second[:3, :3] = rotation
    first[:3, 3] = [2., -3., 4.]
    second[:3, 3] = first[:3, 3] + rotation @ [np.tan(np.deg2rad(angle)), 0., 0.]
    _, ids = triangulate(np.array([[0., 0.]]),
                         np.array([[-np.tan(np.deg2rad(angle)), 0.]]),
                         first, second, np.eye(3), scale_safe_angles=True)
    assert bool(len(ids)) is accepted


def test_zero_baseline_abstains():
    pixels = np.array([[0., 0.], [.1, .2]])
    with np.errstate(invalid="ignore", divide="ignore"):
        _, ids = triangulate(pixels, pixels, np.eye(4), np.eye(4), np.eye(3),
                             scale_safe_angles=True)
    assert not len(ids)


def test_pure_rotation_cannot_supply_parallax():
    angle = np.deg2rad(10.)
    second = np.eye(4)
    second[:3, :3] = [[np.cos(angle), 0., np.sin(angle)],
                      [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
    points = np.array([[0., 0., 5.], [1., 1., 7.], [-1., -.5, 10.]])
    first_pixels, _ = project(points, np.eye(4), np.eye(3))
    second_pixels, _ = project(points, second, np.eye(3))
    with np.errstate(invalid="ignore", divide="ignore"):
        _, ids = triangulate(first_pixels, second_pixels, np.eye(4), second,
                             np.eye(3), scale_safe_angles=True)
    assert not len(ids)


def test_negative_depth_abstains():
    second = np.eye(4)
    second[0, 3] = 1.
    _, ids = triangulate(np.array([[0., 0.]]), np.array([[.2, 0.]]),
                         np.eye(4), second, np.eye(3), scale_safe_angles=True)
    assert not len(ids)


@pytest.mark.parametrize('enabled', [False, True])
def test_initializer_preserves_disabled_call_shape_and_forwards_enabled(enabled, monkeypatch):
    import mapping_geometry as geometry
    import cv2 as cv
    cv.setRNGSeed(0)
    world = np.random.default_rng(22).uniform([-4., -3., 4.], [4., 3., 14.], (200, 3))
    matrix = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
    second = np.eye(4); second[0, 3] = 1.
    first_pixels, _ = project(world, np.eye(4), matrix)
    second_pixels, _ = project(world, second, matrix)
    original = geometry.triangulate
    calls = []
    def wrapped(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(geometry, 'triangulate', wrapped)
    result = geometry.initialize_monocular(first_pixels, second_pixels, matrix, (640, 480),
                                          scale_safe_angles=enabled)
    assert result is not None and len(calls) == 2
    assert all(call['min_angle'] == 1.5 for call in calls)
    assert all(call.get('scale_safe_angles') is True for call in calls) if enabled else all('scale_safe_angles' not in call for call in calls)


@pytest.mark.parametrize('enabled', [False, True])
def test_monocular_mapping_forwards_only_explicit_enabled_keyword(enabled, monkeypatch):
    import shared_slam as module
    matrix = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
    world = np.random.default_rng(23).uniform([-4., -3., 4.], [4., 3., 14.], (100, 3))
    descriptors = np.random.default_rng(24).normal(size=(100, 128)).astype(np.float32)
    first_pixels, _ = project(world, np.eye(4), matrix)
    second = np.eye(4); second[0, 3] = 1.
    second_pixels, _ = project(world, second, matrix)
    config = module.MappingConfig(bundle_enabled=False, loop_mode='off',
                                  monocular_scale_safe_triangulation=enabled)
    slam = module.SharedSlam(matrix, config=config)
    original, calls = module.triangulate, []
    def wrapped(*args, **kwargs):
        calls.append(kwargs.copy()); return original(*args, **kwargs)
    monkeypatch.setattr(module, 'triangulate', wrapped)
    try:
        slam._keyframe(0, np.eye(4), first_pixels, descriptors,
                       np.full((100, 3), np.nan), np.full(100, np.nan), {})
        slam._keyframe(1, second, second_pixels, descriptors,
                       np.full((100, 3), np.nan), np.full(100, np.nan), {})
        assert calls and len(slam.map.landmarks) > 50
        assert all(call.get('scale_safe_angles') is True for call in calls) if enabled else all('scale_safe_angles' not in call for call in calls)
    finally:
        slam.close(finish=False)


@pytest.mark.parametrize('enabled', [False, True])
def test_shared_slam_forwards_initializer_flag_without_disabled_keyword(enabled, monkeypatch):
    import shared_slam as module
    matrix = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
    world = np.random.default_rng(25).uniform([-4., -3., 4.], [4., 3., 14.], (100, 3))
    pixels, _ = project(world, np.eye(4), matrix)
    descriptors = np.random.default_rng(26).normal(size=(100, 128)).astype(np.float32)
    slam = module.SharedSlam(matrix, config=module.MappingConfig(bundle_enabled=False,
        loop_mode='off', monocular_scale_safe_triangulation=enabled))
    monkeypatch.setattr(slam, '_extract', lambda *args: (pixels, descriptors,
        np.full((100, 3), np.nan), np.full(100, np.nan)))
    calls = []
    def initialize(*args, **kwargs):
        calls.append(kwargs); return None
    monkeypatch.setattr(module, 'initialize_monocular', initialize)
    try:
        image = np.zeros((480, 640), np.uint8)
        slam.process(0, image); slam.process(1, image)
        assert calls == [{'scale_safe_angles': True}] if enabled else calls == [{}]
    finally:
        slam.close(finish=False)
