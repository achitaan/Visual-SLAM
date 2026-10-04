"""Map-only verified depth changes installed observations, not frontend evidence."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_shared_slam
import main as app_main
import shared_slam as module
from shared_slam import MappingConfig, SharedSlam, StereoCamera

K = np.array([[100., 0., 120.], [0., 100., 48.], [0., 0., 1.]])
B = .2


def camera(policy='verified', offset=2., frontend='supported'):
    q = np.array([[1., 0., 0., -120.], [0., 1., 0., -48.],
                  [0., 0., 0., 100.], [0., 0., 5., -5.*offset]])
    return SharedSlam(K, StereoCamera(None, q, B), config=MappingConfig(
        loop_mode='off', bundle_enabled=False, stereo_depth_policy=frontend,
        stereo_map_depth_policy=policy))


@pytest.fixture
def slam():
    value = camera()
    try:
        yield value
    finally:
        value.close()


def images(disparity=10.25):
    left = cv.GaussianBlur(np.random.default_rng(42).integers(
        0, 256, (96, 240), dtype=np.uint8), (3, 3), .6)
    right = cv.warpAffine(left, np.float32([[1., 0., -disparity], [0., 1., 0.]]), (240, 96))
    return left, right


def inputs(slam, pixels=None):
    pixels = np.array([[170.25, 50.25], [151.2, 38.3], [195.5, 65.5]], np.float32) if pixels is None else np.asarray(pixels)
    slam.current_disparity = np.full((96, 240), 9., np.float32)
    points, rights = slam._measure_stereo_pixels(pixels)
    desc = np.eye(len(pixels), 128, dtype=np.float32)
    return pixels, desc, points, rights


def test_config_default_and_invalid_enum():
    assert MappingConfig().stereo_map_depth_policy == 'inherit'
    for invalid in ('all', '', None, True):
        with pytest.raises(ValueError, match='stereo_map_depth_policy'):
            MappingConfig(stereo_map_depth_policy=invalid)
    with pytest.raises(ValueError, match='stereo'):
        SharedSlam(K, config=MappingConfig(stereo_map_depth_policy='verified'))


def test_real_fullrange_map_ncc_uses_q_offset_and_frontend_stays_supported(slam):
    left, right = images()
    slam._prepare_frame_images(left, right)
    pix, _, front_xyz, front_right = inputs(slam)
    config_before = asdict(slam.config)
    supported_before = tuple(a.copy() for a in (front_xyz, front_right))
    xyz, ru = slam._measure_map_stereo_pixels(pix)
    assert np.isfinite(xyz).all() and np.isfinite(ru).all()
    d = pix[:, 0] - ru
    np.testing.assert_allclose(d, 10.25, atol=.3)
    np.testing.assert_allclose(xyz, np.c_[pix, np.ones(len(pix))] @ slam.inverse_K.T * (20./(d-2.))[:, None])
    assert np.max(np.abs(ru-front_right)) > .8
    again = slam._measure_stereo_pixels(pix)
    for actual, expected in zip(again, supported_before):
        np.testing.assert_array_equal(actual, expected)
    assert asdict(slam.config) == config_before


def test_keyframe_installs_verified_map_points_but_raw_reference_geometry_is_unchanged(slam):
    left, right = images(); slam._prepare_frame_images(left, right)
    pix, desc, front_xyz, front_right = inputs(slam)
    pix_before, xyz_before, rights_before = pix.copy(), front_xyz.copy(), front_right.copy()
    expected_xyz, expected_ru = slam._measure_map_stereo_pixels(pix)
    assert np.isfinite(expected_xyz).all()
    pose = np.eye(4); pose[:3, 3] = [1., .2, 3.]
    slam._keyframe(0, pose, pix, desc, front_xyz, front_right, {})
    frame = slam.map.keyframes[0]
    np.testing.assert_array_equal(frame.depth_points, xyz_before)
    np.testing.assert_array_equal(pix, pix_before)
    np.testing.assert_array_equal(front_xyz, xyz_before)
    np.testing.assert_array_equal(front_right, rights_before)
    for row, ident in enumerate(frame.landmark_ids):
        assert ident >= 0
        lm = slam.map.landmarks[int(ident)]; obs = lm.observations[0]
        np.testing.assert_allclose(lm.position, expected_xyz[row] + pose[:3, 3])
        np.testing.assert_array_equal(obs.pixel, pix[row])
        assert obs.right_u == expected_ru[row]


def test_inherit_default_keyframe_admission_does_not_call_ncc(monkeypatch):
    value = camera('inherit')
    try:
        pix, desc, xyz, ru = inputs(value)
        def forbidden(*args, **kwargs):
            pytest.fail('inherit mapping unexpectedly invoked NCC')
        monkeypatch.setattr(module, 'verify_stereo_depth_candidates', forbidden)
        value._keyframe(0, np.eye(4), pix, desc, xyz, ru, {})
        for row, ident in enumerate(value.map.keyframes[0].landmark_ids):
            np.testing.assert_array_equal(value.map.landmarks[int(ident)].position, xyz[row])
            assert value.map.landmarks[int(ident)].observations[0].right_u == ru[row]
    finally:
        value.close()


def test_actual_subpixel_flow_is_remeasured_and_installed_not_detector_pixel(slam, monkeypatch):
    left, right = images(); slam._prepare_frame_images(left, right)
    pix, desc, xyz, ru = inputs(slam)
    calls = []
    def measured(left_img, right_img, query, config, bounds):
        query = np.asarray(query); calls.append(query.copy())
        return query[:, 0] - (10. + .001*query[:, 0]), {'verified': len(query)}
    monkeypatch.setattr(module, 'verify_stereo_depth_candidates', measured)
    slam._keyframe(0, np.eye(4), pix, desc, xyz, ru, {})
    ident = int(slam.map.keyframes[0].landmark_ids[0]); original = slam.map.landmarks[ident].position.copy()
    flow = np.array([170.37, 50.43], np.float64)
    slam.accepted_tracks = [(ident, flow.copy())]
    slam._keyframe(1, np.eye(4), pix, desc, xyz, ru, {0: ident})
    obs = slam.map.landmarks[ident].observations[1]
    np.testing.assert_array_equal(obs.pixel, flow)
    assert obs.right_u == flow[0] - (10. + .001*flow[0])
    assert any(np.array_equal(row, flow) for call in calls for row in call)
    np.testing.assert_array_equal(slam.map.landmarks[ident].position, original)


def test_same_pixel_orientation_aliases_create_one_physical_landmark(slam, monkeypatch):
    left, right = images(); slam._prepare_frame_images(left, right)
    pix, desc, xyz, ru = inputs(slam, [[170.25, 50.25], [170.25, 50.25], [151.2, 38.3]])
    monkeypatch.setattr(module, 'verify_stereo_depth_candidates', lambda a,b,p,c,d: (np.asarray(p)[:,0]-10.25, {'verified':len(p)}))
    slam._keyframe(0, np.eye(4), pix, desc, xyz, ru, {})
    ids = slam.map.keyframes[0].landmark_ids
    assert ids[0] == ids[1] >= 0 and ids[2] != ids[0]
    assert len(slam.map.landmarks) == 2
    assert sum(len(a.observations) for a in slam.map.landmarks.values()) == 2


def test_failed_ncc_never_uses_frontend_or_restored_xyz_for_new_landmarks(slam, monkeypatch):
    left, right = images(); slam._prepare_frame_images(left, right)
    pix, desc, xyz, ru = inputs(slam)
    monkeypatch.setattr(module, 'verify_stereo_depth_candidates', lambda a,b,p,c,d: (np.full(len(p), np.nan), {'verified':0}))
    slam._keyframe(0, np.eye(4), pix, desc, xyz, ru, {})
    assert not slam.map.landmarks and np.all(slam.map.keyframes[0].landmark_ids == -1)
    np.testing.assert_array_equal(slam.map.keyframes[0].depth_points, xyz)


def test_failed_remeasurement_keeps_actual_existing_left_observation_without_right(slam, monkeypatch):
    left, right = images(); slam._prepare_frame_images(left, right)
    pix, desc, xyz, ru = inputs(slam)
    monkeypatch.setattr(module, 'verify_stereo_depth_candidates', lambda a,b,p,c,d: (np.asarray(p)[:,0]-10.25, {'verified':len(p)}))
    slam._keyframe(0, np.eye(4), pix, desc, xyz, ru, {})
    ident = int(slam.map.keyframes[0].landmark_ids[0]); flow = np.array([170.37,50.43])
    old = slam.map.landmarks[ident].position.copy()
    monkeypatch.setattr(module, 'verify_stereo_depth_candidates', lambda a,b,p,c,d: (np.full(len(p), np.nan), {'verified':0}))
    slam.accepted_tracks=[(ident,flow.copy())]
    slam._keyframe(1,np.eye(4),pix,desc,xyz,ru,{0:ident})
    obs=slam.map.landmarks[ident].observations[1]
    np.testing.assert_array_equal(obs.pixel,flow); assert obs.right_u is None
    np.testing.assert_array_equal(slam.map.landmarks[ident].position,old)


@pytest.mark.parametrize('case', ['missing_left','missing_right','low_texture','border','nonfinite','complex'])
def test_unavailable_images_or_queries_cannot_admit_map_depth(slam, case):
    left,right=images();slam._prepare_frame_images(left,right)
    pix=np.array([[170.25,50.25]])
    if case=='missing_left': slam.current_left_gray=None
    elif case=='missing_right': slam.current_right_gray=None
    elif case=='low_texture': slam.current_left_gray[:]=80;slam.current_right_gray[:]=80
    elif case=='border': pix[:]=[2.,50.]
    elif case=='nonfinite': pix[:]=[np.nan,50.]
    elif case=='complex': pix=pix.astype(complex)+1j
    points,ru=slam._measure_map_stereo_pixels(pix)
    assert not np.isfinite(points).any() and not np.isfinite(ru).any()


def test_calibrated_full_search_range_and_fixed_configuration(slam, monkeypatch):
    left,right=images();slam._prepare_frame_images(left,right)
    pix,_,_,_=inputs(slam)
    calls=[]
    def measured(a,b,p,config,bounds):
        calls.append((config,bounds));return np.asarray(p)[:,0]-10.25,{'verified':len(p)}
    monkeypatch.setattr(module,'verify_stereo_depth_candidates',measured)
    slam._measure_map_stereo_pixels(pix)
    assert calls and calls[0][0] is slam.stereo_search_config
    np.testing.assert_allclose(calls[0][1],(2.+20./100.,96.))


def test_old_namespace_entrypoints_default_to_inherit_and_forward_explicit_policy():
    values=dict(features=1500,disable_bundle=False,loop_mode='off',stereo_depth_policy='supported',
        stereo_pose_arbitration=False,stereo_physical_match_pool=False,stereo_raw_reference_retry=False,
        stereo_owned_image_bundle=False,stereo_source_history_bundle=False,stereo_retained_source_observations=False,
        bundle_solver_accuracy='default')
    args=argparse.Namespace(**values)
    for builder in (app_main._mapping_config_from_args,evaluate_shared_slam._mapping_config_from_args):
        assert builder(args).stereo_map_depth_policy=='inherit'
        args.stereo_map_depth_policy='verified'
        assert builder(args).stereo_map_depth_policy=='verified'
        del args.stereo_map_depth_policy
