from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2 as cv
import numpy as np
import warnings

from config import (
    ENABLE_KEYFRAMES,
    ENABLE_LOCAL_MAP,
    ENABLE_LOOP_CLOSURE,
    ENABLE_POSE_GRAPH,
    LOOP_CLOSURE_THRESHOLD,
    LOST_CONSEC_FRAMES,
    LOST_INLIER_RATIO,
    LOST_MIN_INLIERS,
    POSE_GRAPH_OPT_EVERY,
    RELOCALIZE_MIN_MATCHES,
    RELOCALIZE_PNP_MIN_INLIERS,
    RELOCALIZE_PNP_REPROJ,
    ENABLE_RELOCALIZATION,
    VOCAB_BUILD_MIN_FRAMES,
    VOCAB_NUM_CLUSTERS,
    LOCAL_MAP_MAX_POINTS,
)
from map import Keyframe, LocalMap


@dataclass
class SlamEvent:
    timestamp: float
    type: str
    message: str
    severity: str = "info"


class SlamBackend:
    def __init__(self, camera_matrix: Optional[np.ndarray] = None) -> None:
        self.local_map = LocalMap() if ENABLE_KEYFRAMES or ENABLE_LOCAL_MAP else None
        self._last_keyframe_index: Optional[int] = None
        self._last_keyframe_pose: Optional[np.ndarray] = None
        self._slam = self._try_init_slam() if (ENABLE_POSE_GRAPH or ENABLE_LOOP_CLOSURE) else None
        self._keyframe_poses: List[np.ndarray] = []
        self._keyframe_desc: List[np.ndarray] = []
        self._lost_counter = 0
        self._optimized_poses: Optional[List[np.ndarray]] = None
        self._camera_matrix = camera_matrix

    def _try_init_slam(self):
        try:
            from SLAM import SLAM  # noqa: N812
        except ImportError as exc:
            warnings.warn(f"Pose graph and loop closure unavailable: {exc}", RuntimeWarning)
            return None
        return SLAM()

    def map_state(self) -> dict:
        if self.local_map is None:
            return {"keyframes": 0, "map_points": 0}
        return self.local_map.stats()

    def map_points_sample(self, max_points: int) -> list:
        if self.local_map is None:
            return []
        return [p.tolist() for p in self.local_map.sample_points(max_points)]

    def pose_graph_state(self, current_keyframe_index: Optional[int]) -> Optional[dict]:
        if self._optimized_poses is None or current_keyframe_index is None:
            return None
        frame_indices = [keyframe.index for keyframe in self.local_map.keyframes]
        if current_keyframe_index not in frame_indices:
            return None
        graph_index = frame_indices.index(current_keyframe_index)
        if graph_index >= len(self._optimized_poses):
            return None
        return {
            "optimized_pose_T_wc": self._optimized_poses[graph_index].tolist(),
            "optimized_poses_count": len(self._optimized_poses),
            "optimized_poses": [pose.tolist() for pose in self._optimized_poses],
        }

    def update_tracking_state(self, num_inliers: Optional[int], inlier_ratio: Optional[float]) -> bool:
        if not ENABLE_RELOCALIZATION:
            return False
        low_inliers = num_inliers is None or num_inliers < LOST_MIN_INLIERS
        low_ratio = inlier_ratio is None or inlier_ratio < LOST_INLIER_RATIO
        if low_inliers or low_ratio:
            self._lost_counter += 1
        else:
            self._lost_counter = 0
        return self._lost_counter >= LOST_CONSEC_FRAMES

    def try_relocalize(
        self,
        keypoints: Optional[list],
        descriptors: Optional[np.ndarray],
        timestamp: float,
    ) -> tuple[Optional[np.ndarray], List[SlamEvent]]:
        events: List[SlamEvent] = []
        if not ENABLE_RELOCALIZATION:
            return None, events
        if descriptors is None or len(self._keyframe_desc) == 0:
            return None, events

        events.append(SlamEvent(timestamp, "tracking_lost", "Tracking lost, attempting relocalization.", "warn"))

        norm_type = cv.NORM_L2 if descriptors.dtype != np.uint8 else cv.NORM_HAMMING
        matcher = cv.BFMatcher(norm_type)
        best_idx = None
        best_matches = 0
        for idx, key_desc in enumerate(self._keyframe_desc):
            if key_desc is None or len(key_desc) == 0:
                continue
            matches = matcher.knnMatch(descriptors, key_desc, k=2)
            good = 0
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good += 1
            if good > best_matches:
                best_matches = good
                best_idx = idx

        if best_idx is not None and best_matches >= RELOCALIZE_MIN_MATCHES:
            if self._camera_matrix is not None and self.local_map is not None and keypoints is not None:
                map_desc = []
                map_pts = []
                for point in self.local_map.map_points:
                    if point.descriptor is not None:
                        map_desc.append(point.descriptor)
                        map_pts.append(point.position)
                if map_desc:
                    map_desc = np.asarray(map_desc)
                    if map_desc.dtype != descriptors.dtype:
                        map_desc = map_desc.astype(descriptors.dtype, copy=False)
                    matches = matcher.knnMatch(descriptors, map_desc, k=2)
                    pts_2d = []
                    pts_3d = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            pts_2d.append(keypoints[m.queryIdx].pt)
                            pts_3d.append(map_pts[m.trainIdx])
                    if len(pts_2d) >= RELOCALIZE_PNP_MIN_INLIERS:
                        pts_2d = np.asarray(pts_2d, dtype=np.float32)
                        pts_3d = np.asarray(pts_3d, dtype=np.float32)
                        success, rvec, tvec, inliers = cv.solvePnPRansac(
                            pts_3d,
                            pts_2d,
                            self._camera_matrix,
                            None,
                            reprojectionError=RELOCALIZE_PNP_REPROJ,
                            iterationsCount=100,
                            confidence=0.99,
                        )
                        if success and inliers is not None and len(inliers) >= RELOCALIZE_PNP_MIN_INLIERS:
                            R, _ = cv.Rodrigues(rvec)
                            T_cw = np.eye(4, dtype=np.float64)
                            T_cw[:3, :3] = R
                            T_cw[:3, 3] = tvec.reshape(-1)
                            pose = np.linalg.inv(T_cw)
                            events.append(
                                SlamEvent(
                                    timestamp,
                                    "relocalized",
                                    f"Relocalized via PnP ({len(inliers)} inliers).",
                                )
                            )
                            self._lost_counter = 0
                            return pose, events

            # Appearance retrieval alone does not estimate the current camera pose.

        events.append(SlamEvent(timestamp, "relocalization_failed", "Relocalization failed.", "warn"))
        return None, events

    def maybe_add_keyframe(
        self,
        *,
        index: int,
        pose_T_wc: np.ndarray,
        image: Optional[np.ndarray],
        descriptors: Optional[np.ndarray],
        timestamp: float,
        should_add: bool,
        map_points: Optional[List[np.ndarray]] = None,
        map_descriptors: Optional[List[np.ndarray]] = None,
    ) -> List[SlamEvent]:
        events: List[SlamEvent] = []
        if self.local_map is None or not should_add:
            return events

        keyframe = Keyframe(index=index, pose_T_wc=pose_T_wc, image=image, features=None)
        self.local_map.add_keyframe(keyframe)
        self._keyframe_poses.append(pose_T_wc)
        self._keyframe_desc.append(descriptors)
        if map_points and map_descriptors and len(map_points) == len(map_descriptors):
            self.local_map.add_points_with_desc(map_points, map_descriptors)
        elif map_points:
            self.local_map.add_points(map_points)
        self.local_map.map_points = self.local_map.map_points[-LOCAL_MAP_MAX_POINTS:]

        events.append(SlamEvent(timestamp, "keyframe_added", f"Keyframe {index} added."))
        self._maybe_update_slam(index=index, pose_T_wc=pose_T_wc, descriptors=descriptors, timestamp=timestamp, events=events)
        return events

    def _maybe_update_slam(
        self,
        *,
        index: int,
        pose_T_wc: np.ndarray,
        descriptors: Optional[np.ndarray],
        timestamp: float,
        events: List[SlamEvent],
    ) -> None:
        if self._slam is None:
            return

        # Every keyframe is a graph vertex, including those before BoW startup.
        self._slam.initial_poses = [pose.copy() for pose in self._keyframe_poses]

        # Build vocabulary when enough keyframes collected.
        valid_descriptors = [desc for desc in self._keyframe_desc if desc is not None and len(desc)]
        if len(valid_descriptors) >= VOCAB_BUILD_MIN_FRAMES and self._slam.kmeans is None:
            try:
                self._slam.build_vocabulary(valid_descriptors[:VOCAB_BUILD_MIN_FRAMES], num_clusters=VOCAB_NUM_CLUSTERS)
                self._slam.histograms = [self._slam.compute_bow_histogram(desc) if desc is not None and len(desc) else np.zeros(VOCAB_NUM_CLUSTERS) for desc in self._keyframe_desc[:-1]]
                events.append(SlamEvent(timestamp, "vocab_built", "BoW vocabulary built."))
            except Exception:
                events.append(SlamEvent(timestamp, "vocab_failed", "BoW vocabulary build failed.", "warn"))

        loop_idx = None
        if ENABLE_LOOP_CLOSURE and self._slam.kmeans is not None:
            try:
                hist = self._slam.compute_bow_histogram(descriptors) if descriptors is not None and len(descriptors) else np.zeros(VOCAB_NUM_CLUSTERS)
                loop_idx = self._slam.detect_loop_closure(hist, threshold=LOOP_CLOSURE_THRESHOLD)
                self._slam.histograms.append(hist)
            except Exception:
                loop_idx = None

        if ENABLE_POSE_GRAPH:
            if len(self._keyframe_poses) > 1:
                prev_pose = self._keyframe_poses[-2]
                rel = np.linalg.inv(prev_pose) @ pose_T_wc
                info = np.identity(6) * 10.0
                self._slam.add_odometry_edge(len(self._keyframe_poses) - 2, len(self._keyframe_poses) - 1, rel, information=info)

        if ENABLE_LOOP_CLOSURE and loop_idx is not None and abs(loop_idx - (len(self._keyframe_poses) - 1)) > 1:
            current_idx = len(self._keyframe_poses) - 1
            # Never manufacture a loop measurement from the drifted trajectory.
            # An independently verified PnP/Sim(3) measurement is still required.
            events.append(SlamEvent(timestamp, "loop_candidate", f"Candidate keyframe {loop_idx}; geometric verification required."))

        if ENABLE_POSE_GRAPH and POSE_GRAPH_OPT_EVERY > 0:
            if len(self._keyframe_poses) % POSE_GRAPH_OPT_EVERY == 0:
                try:
                    optimized = self._slam.optimize_pose_graph(num_iterations=100)
                    self._optimized_poses = optimized
                    events.append(SlamEvent(timestamp, "pose_graph_opt", "Pose graph optimized."))
                except Exception:
                    events.append(SlamEvent(timestamp, "pose_graph_opt_failed", "Pose graph optimize failed.", "warn"))
