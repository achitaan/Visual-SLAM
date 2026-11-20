"""Unit tests for CovisibilityGraph, chi2 outlier culling, >300-KF weighted PGO, and loop landmark fusion (F12/F13)."""

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

from covisibility import CovisibilityGraph
from live_loops import LiveLoopWorker, fuse_loop_landmarks
from local_bundle import local_bundle_adjustment
from pose_graph import optimize_weighted_pose_graph
from slam_state import MapState, MappingKeyframe, Observation


def test_covisibility_graph_ranks_shared_keyframes():
    st = MapState(metric=True)
    desc = np.zeros(128, dtype=np.float32)
    for k in range(6):
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 0.0, float(k)]
        st.keyframes[k] = MappingKeyframe(
            k, k * 10, pose, np.empty((0, 2), np.float32),
            np.empty((0, 128), np.float32), np.empty(0, int),
        )
    # Keyframe 5 shares 20 landmarks with KF 2, 12 landmarks with KF 4, and 3 landmarks with KF 0
    for i in range(20):
        st.add_landmark([0.1 * i, 0.0, 10.0], desc, anchor=2, observations={
            2: Observation(np.array([100.0 + i, 100.0], np.float32), 80.0 + i),
            5: Observation(np.array([100.0 + i, 100.0], np.float32), 80.0 + i),
        })
    for i in range(12):
        st.add_landmark([0.1 * i, 1.0, 12.0], desc, anchor=4, observations={
            4: Observation(np.array([200.0 + i, 100.0], np.float32), 180.0 + i),
            5: Observation(np.array([200.0 + i, 100.0], np.float32), 180.0 + i),
        })
    for i in range(3):
        st.add_landmark([0.1 * i, -1.0, 14.0], desc, anchor=0, observations={
            0: Observation(np.array([300.0 + i, 100.0], np.float32), 280.0 + i),
            5: Observation(np.array([300.0 + i, 100.0], np.float32), 280.0 + i),
        })

    cg = CovisibilityGraph()
    cg.update_from_map(st, 5)
    top = cg.top_covisible_keyframes(5, max_keyframes=8, min_shared=8)
    assert top == [2, 4]
    assert cg.shared_count(5, 2) == 20
    assert cg.shared_count(5, 4) == 12
    assert cg.shared_count(5, 0) == 3

    win = cg.select_ba_window(st, 5, max_keyframes=4, min_shared=8)
    assert 5 in win and 2 in win and 4 in win
    assert win == sorted(win)


def test_weighted_pose_graph_optimizes_over_300_keyframes():
    n_kf = 350
    angles = np.linspace(0.0, 2.0 * np.pi, n_kf, endpoint=False)
    true_poses = []
    for a in angles:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("y", a).as_matrix()
        T[:3, 3] = [80.0 * np.sin(a), 0.0, 80.0 * (1.0 - np.cos(a))]
        true_poses.append(T)

    drifted = [true_poses[0].copy()]
    odom_edges = []
    for i in range(n_kf - 1):
        rel = np.linalg.inv(true_poses[i]) @ true_poses[i + 1]
        noisy = rel.copy()
        noisy[:3, :3] = rel[:3, :3] @ Rotation.from_euler("y", 0.0005).as_matrix()
        noisy[:3, 3] += [0.004, 0.0, 0.008]
        drifted.append(drifted[-1] @ noisy)
        odom_edges.append((i, i + 1, noisy))

    loop_meas = np.linalg.inv(true_poses[0]) @ true_poses[n_kf - 1]
    loop_edges = [(0, n_kf - 1, loop_meas, {"inliers": 100, "median_reprojection_px": 0.8})]

    err_before = float(np.sqrt(np.mean([
        np.sum((drifted[i][:3, 3] - true_poses[i][:3, 3]) ** 2) for i in range(n_kf)
    ])))
    diag = {}
    opt = optimize_weighted_pose_graph(
        {i: drifted[i] for i in range(n_kf)}, odom_edges, loop_edges, diagnostics=diag
    )
    err_after = float(np.sqrt(np.mean([
        np.sum((opt[i][:3, 3] - true_poses[i][:3, 3]) ** 2) for i in range(n_kf)
    ])))
    assert diag["success"] is True
    assert err_after < err_before * 0.25, f"Expected >4x ATE reduction, got {err_before:.2f} -> {err_after:.2f}"


def test_live_loop_worker_uncapped_allows_over_300_keyframes():
    K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])
    worker = LiveLoopWorker(K, metric=True, mode="live", max_keyframes=0, use_weighted_pgo=True, use_vlad_retrieval=True)
    assert worker.max_keyframes == 0
    assert worker.retrieval_mode == "indexed"
    assert worker.retrieval_index is not None
    worker.close(MapState(metric=True), finish=False)


def test_post_ba_chi2_outlier_culling_drops_bad_observation():
    rng = np.random.default_rng(17)
    K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])
    baseline = 0.54
    st = MapState(metric=True)
    true_poses = []
    for k in range(4):
        p = np.eye(4)
        p[:3, 3] = [0.0, 0.0, 1.5 * k]
        true_poses.append(p)
        noisy = p.copy()
        if k > 0:
            noisy[:3, 3] += rng.normal(0.0, 0.02, 3)
        st.keyframes[k] = MappingKeyframe(
            k, k * 10, noisy, np.empty((0, 2), np.float32),
            np.empty((0, 128), np.float32), np.empty(0, int),
        )
        st.record(noisy, "tracking", anchor=k)

    desc = np.zeros(128, dtype=np.float32)
    outlier_lm_id = None
    for idx in range(25):
        pw = np.array([rng.uniform(-4, 4), rng.uniform(-1.5, 1.5), rng.uniform(12, 30)])
        obs = {}
        for k in range(4):
            pc = np.linalg.inv(true_poses[k])[:3, :3] @ pw + np.linalg.inv(true_poses[k])[:3, 3]
            h = K @ pc
            uv = (h[:2] / h[2]).astype(np.float32)
            uR = float(uv[0] - K[0, 0] * baseline / pc[2])
            if idx == 0 and k == 3:
                # Inject a gross 25-pixel outlier observation on KF 3
                uv = uv + np.array([25.0, -18.0], dtype=np.float32)
            obs[k] = Observation(uv, uR)
        lid = st.add_landmark(pw, desc, anchor=0, observations=obs)
        if idx == 0:
            outlier_lm_id = lid

    rep = local_bundle_adjustment(
        st, K, baseline, solver_accuracy="precise",
        use_analytic_jacobian=True, use_map_index=True, cull_outliers_chi2=True,
    )
    assert rep["applied"] is True
    assert rep.get("culled_outlier_observations", 0) >= 1
    assert 3 not in st.landmarks[outlier_lm_id].observations
    assert len(st.landmarks[outlier_lm_id].observations) == 3
