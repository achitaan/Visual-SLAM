"""Loop closure public interface re-exporting live loop detection, verification, fusion, and PGO."""

from live_loops import LiveLoopWorker, fuse_loop_landmarks
from loop_geometry import StereoLoopFrame, verify_loop
from pose_graph import optimize, optimize_sparse_se3, optimize_weighted_pose_graph

__all__ = [
    "LiveLoopWorker",
    "StereoLoopFrame",
    "fuse_loop_landmarks",
    "optimize",
    "optimize_sparse_se3",
    "optimize_weighted_pose_graph",
    "verify_loop",
]
