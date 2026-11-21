"""Unit tests for projective local map matching, neighbor fusion, and far-point triangulation (F10/F11)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_map_tracker import (
    fuse_neighbor_landmarks,
    match_landmarks_by_projection,
    triangulate_far_stereo_landmarks,
)
from map_index import FrustumQueryResult
from slam_state import MapState, MappingKeyframe, Observation


def test_match_landmarks_by_projection_ignores_distant_repetitive_distractors():
    rng = np.random.default_rng(10)
    n_lm = 40
    lm_ids = np.arange(n_lm, dtype=np.int64)
    lm_uv = np.column_stack((rng.uniform(80, 1100, n_lm), rng.uniform(50, 320, n_lm))).astype(np.float32)
    lm_descs = rng.uniform(0.0, 50.0, size=(n_lm, 128)).astype(np.float32)
    lm_octs = np.zeros(n_lm, dtype=np.int16)

    # Frame contains true features (+5 px jitter) PLUS identical repetitive distractors 300 px away!
    true_px = lm_uv + rng.uniform(-5.0, 5.0, size=lm_uv.shape).astype(np.float32)
    distractor_px = (lm_uv + np.array([300.0, 0.0], dtype=np.float32)) % np.array([1240.0, 376.0], dtype=np.float32)
    f_pixels = np.vstack((true_px, distractor_px))
    f_descs = np.vstack((lm_descs + rng.normal(0.0, 1.0, lm_descs.shape).astype(np.float32), lm_descs.copy()))
    f_octs = np.zeros(len(f_pixels), dtype=np.int16)

    frustum = FrustumQueryResult(lm_ids, lm_uv.astype(np.float64), np.full(n_lm, 15.0))
    matches = match_landmarks_by_projection(
        frustum,
        {"pixels": f_pixels, "descriptors": f_descs, "octaves": f_octs},
        lm_descs,
        lm_octs,
        radius_px=30.0,
        ratio=0.75,
        max_octave_diff=1,
    )
    # Global BFMatcher would reject 100% of these due to identical distractors; projective matching finds nearly all!
    assert len(matches) >= 35
    for lid, fidx in matches.items():
        assert fidx == lid


def test_octave_gating_rejects_mismatched_scale():
    rng = np.random.default_rng(2)
    lm_desc = rng.uniform(0.0, 50.0, size=(1, 128)).astype(np.float32)
    frustum = FrustumQueryResult(np.array([7]), np.array([[500.0, 200.0]]), np.array([15.0]))
    matches_bad_oct = match_landmarks_by_projection(
        frustum,
        {"pixels": np.array([[502.0, 201.0]], np.float32), "descriptors": lm_desc, "octaves": np.array([3], np.int16)},
        lm_desc,
        np.array([0], np.int16),
        radius_px=30.0,
        max_octave_diff=1,
    )
    assert matches_bad_oct == {}


def test_triangulate_far_stereo_landmarks_recovers_distant_points():
    rng = np.random.default_rng(55)
    K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[:3, 3] = [1.5, 0.0, 5.0]  # forward + slight lateral motion -> clear parallax even at 80-150 m

    n_pts = 25
    pts_w = np.column_stack((
        rng.uniform(-25.0, 25.0, n_pts),
        rng.uniform(-6.0, 6.0, n_pts),
        rng.uniform(40.0, 140.0, n_pts),
    ))
    T_cw1 = np.linalg.inv(T1)
    T_cw2 = np.linalg.inv(T2)
    pc1 = pts_w @ T_cw1[:3, :3].T + T_cw1[:3, 3]
    pc2 = pts_w @ T_cw2[:3, :3].T + T_cw2[:3, 3]
    uv1 = ((pc1 @ K.T)[:, :2] / pc1[:, 2:]).astype(np.float32)
    uv2 = ((pc2 @ K.T)[:, :2] / pc2[:, 2:]).astype(np.float32)
    descs = rng.uniform(0.0, 50.0, size=(n_pts, 128)).astype(np.float32)

    kf1 = MappingKeyframe(0, 0, T1, uv1, descs, np.full(n_pts, -1, dtype=int), octaves=np.zeros(n_pts, np.int16))
    kf2 = MappingKeyframe(1, 10, T2, uv2, descs + rng.normal(0.0, 0.5, descs.shape).astype(np.float32),
                          np.full(n_pts, -1, dtype=int), octaves=np.zeros(n_pts, np.int16))

    tri = triangulate_far_stereo_landmarks(kf1, kf2, K, baseline=0.54, min_parallax_deg=0.35, min_depth_m=20.0, max_depth_m=400.0)
    assert len(tri) >= 15
    for item in tri:
        err = float(np.linalg.norm(item.position_w - pts_w[item.prev_feat_idx]))
        assert err < 1.0
        assert item.is_far is True


def test_fuse_neighbor_landmarks_attaches_unseen_observations():
    K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])
    st = MapState(metric=True)
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:3, 3] = [0.0, 0.0, 1.5]
    pw = np.array([1.0, 0.2, 15.0])
    desc = np.ones(128, dtype=np.float32)

    h0 = K @ pw
    uv0 = (h0[:2] / h0[2]).astype(np.float32)
    pc1 = pw - T1[:3, 3]
    h1 = K @ pc1
    uv1 = (h1[:2] / h1[2]).astype(np.float32)

    kf0 = MappingKeyframe(0, 0, T0, uv0[None], desc[None], np.array([0], int), image_size=(1241, 376))
    kf1 = MappingKeyframe(1, 10, T1, (uv1 + np.array([1.0, -0.5], np.float32))[None], desc[None], np.array([-1], int), image_size=(1241, 376))
    st.keyframes[0] = kf0
    st.keyframes[1] = kf1
    lm_id = st.add_landmark(pw, desc, anchor=0, observations={0: Observation(uv0, float(uv0[0] - 25.0))})

    fused = fuse_neighbor_landmarks(st, kf1, [0], K, baseline=0.54, radius_px=4.0)
    assert fused == 1
    assert kf1.landmark_ids[0] == lm_id
    assert 1 in st.landmarks[lm_id].observations
