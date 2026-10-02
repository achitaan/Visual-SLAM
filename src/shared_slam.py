"""Persistent stereo/monocular image-only tracking and local mapping."""

from dataclasses import dataclass
import cv2 as cv
import numpy as np
from slam_state import MapState, MappingKeyframe, Observation
from mapping_geometry import (
    match_descriptors,
    initialize_monocular,
    triangulate,
    estimate_pose,
    project,
    coverage,
    estimate_stereo_reference,
    refine_stereo_map_pose,
)
from local_bundle import local_bundle_adjustment
from live_loops import LiveLoopWorker
from loop_geometry import StereoLoopFrame, verify_loop
from stage_profile import StageProfile
from keyframe_retrieval import KeyframeRetrieval


@dataclass(frozen=True)
class StereoCamera:
    """Calibration and disparity computation, without dataset/reference access."""

    stereo: object
    Q: np.ndarray
    baseline: float

    @property
    def disparity_offset(self):
        return float(-self.Q[3, 3]/self.Q[3, 2]) if self.Q[3, 2] else 0.0

    def __post_init__(self):
        if (
            np.asarray(self.Q).shape != (4, 4)
            or not np.isfinite(self.Q).all()
            or not np.isfinite(self.baseline)
            or self.baseline <= 0
        ):
            raise ValueError("Invalid stereo geometry")


@dataclass(frozen=True)
class MappingConfig:
    """One configuration per sensor mode; max_landmarks bounds active tracking."""

    features: int = 1500
    min_inliers: int = 15
    keyframe_interval: int = 10
    max_landmarks: int = 2000
    bundle_window: int = 5
    bundle_enabled: bool = True
    loop_mode: str = "live"
    retrieval_candidates: int = 8
    stereo_feature_contrast_threshold: float = 0.02


class SharedSlam:
    def __init__(self, matrix, stereo=None, config=None):
        self.K = np.asarray(matrix, float).copy()
        if (
            self.K.shape != (3, 3)
            or not np.isfinite(self.K).all()
            or self.K[0, 0] <= 0
            or self.K[1, 1] <= 0
            or not np.allclose(self.K[2], [0, 0, 1.0])
        ):
            raise ValueError("Invalid intrinsics")
        self.stereo = stereo
        self.inverse_K = np.linalg.inv(self.K)
        if stereo is not None and not isinstance(stereo, StereoCamera):
            raise TypeError(
                "Stereo input must contain calibration and disparity computation only"
            )
        self.config = config or MappingConfig()
        self.map = MapState(metric=stereo is not None)
        # Low-contrast road and surface texture can supply spatial support that
        # high-contrast repeated edges lack. Pose acceptance remains unchanged.
        contrast = self.config.stereo_feature_contrast_threshold if stereo is not None else 0.04
        if not np.isfinite(contrast) or contrast <= 0:
            raise ValueError("Feature contrast threshold must be finite and positive")
        self.detector = cv.SIFT_create(nfeatures=self.config.features, contrastThreshold=contrast)
        self.initial = None
        self.last_keyframe = None
        self.diagnostics = []
        self.bundle_reports = []
        self.loop_worker = LiveLoopWorker(self.K, self.map.metric, mode=self.config.loop_mode)
        self.profile = StageProfile()
        self.retrieval = KeyframeRetrieval()
        for name in ("_extract", "_track", "_keyframe", "_relocalize", "_keyframe_stereo_reference"):
            setattr(self, name, self.profile.wrap(name, getattr(self, name)))
        self.previous_gray = None
        self.previous_tracks = []
        self.accepted_tracks = []
        self.previous_stereo_geometry = None
        self.verified_stereo_motion = None
        self.motion_prediction_source = "held_pose"
        self.current_disparity = None

    def _measure_stereo_pixels(self, pixels):
        """Sample metric depth at the actual left observations, including flow tracks."""
        points = np.full((len(pixels), 3), np.nan)
        right_u = np.full(len(pixels), np.nan)
        if self.stereo is None or self.current_disparity is None or not len(pixels):
            return points, right_u
        finite = np.isfinite(pixels).all(axis=1)
        height, width = self.current_disparity.shape
        inside = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] <= width-1)
        inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] <= height-1)
        ids = np.flatnonzero(inside)
        if not len(ids):
            return points, right_u
        coordinates = np.floor(pixels[ids]).astype(int)
        x, y = coordinates.T
        next_x, next_y = np.minimum(x+1, width-1), np.minimum(y+1, height-1)
        fraction = pixels[ids] - coordinates
        dx, dy = fraction.T
        weights = np.c_[(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy]
        neighbors = np.c_[self.current_disparity[y, x], self.current_disparity[y, next_x],
                          self.current_disparity[next_y, x], self.current_disparity[next_y, next_x]]
        active = weights > 0
        valid_neighbors = np.isfinite(neighbors) & (neighbors > 0) & (neighbors < 96)
        # Interpolate a supported surface at the feature's actual coordinate.
        # A disparity jump larger than the two-pixel stereo residual budget does
        # not identify one surface; do not interpolate through that depth edge.
        spread = (np.max(np.where(active, neighbors, -np.inf), axis=1)
                  - np.min(np.where(active, neighbors, np.inf), axis=1))
        supported = np.all(~active | valid_neighbors, axis=1) & (spread <= 2.)
        disparity = np.sum(np.where(active, neighbors, 0.) * weights, axis=1)
        denominator = self.stereo.Q[3, 2] * disparity + self.stereo.Q[3, 3]
        depth = np.divide(
            self.stereo.Q[2, 3],
            denominator,
            out=np.full(len(ids), np.nan),
            where=denominator > 0,
        )
        valid = supported & (disparity > 0) & (disparity < 96) & np.isfinite(depth)
        valid &= (depth > 0.1) & (depth < 100)
        ids, disparity, depth = ids[valid], disparity[valid], depth[valid]
        rays = np.c_[pixels[ids], np.ones(len(ids))] @ self.inverse_K.T
        points[ids] = rays * depth[:, None]
        right_u[ids] = pixels[ids, 0] - disparity
        return points, right_u

    def _extract(self, image, right):
        if image is None:
            raise ValueError("Missing image")
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
        keypoints, desc = self.detector.detectAndCompute(gray, None)
        pixels = np.array([k.pt for k in keypoints], np.float32).reshape(-1, 2)
        desc = desc if desc is not None else np.empty((0, 128), np.float32)
        points = np.full((len(pixels), 3), np.nan)
        right_u = np.full(len(pixels), np.nan)
        if self.stereo is not None:
            if right is None or right.shape[:2] != image.shape[:2]:
                raise ValueError("Stereo requires matching right image")
            right_gray = (
                cv.cvtColor(right, cv.COLOR_BGR2GRAY) if right.ndim == 3 else right
            )
            disparity = (
                self.stereo.stereo.compute(gray, right_gray).astype(np.float32) / 16.0
            )
            self.current_disparity = disparity
            points, right_u = self._measure_stereo_pixels(pixels)
        return pixels, desc, points, right_u

    def _keyframe(self, index, pose, pixels, desc, points, right_u, associations):
        pixels, desc, points, right_u = (
            pixels.copy(),
            desc.copy(),
            points.copy(),
            right_u.copy(),
        )
        # A valid tracked landmark need not coincide with a freshly detected SIFT keypoint.
        # Preserve its actual image observation rather than dropping it from bundle adjustment.
        observed = set(associations.values())
        for lid, pixel in self.accepted_tracks:
            if lid in self.map.landmarks and lid not in observed:
                feature = len(pixels)
                pixels = np.vstack([pixels, pixel])
                desc = np.vstack([desc, self.map.landmarks[lid].descriptor])
                points = np.vstack([points, np.full(3, np.nan)])
                right_u = np.r_[right_u, np.nan]
                associations[feature] = lid
                observed.add(lid)
        ident = len(self.map.keyframes)
        frame = MappingKeyframe(
            ident,
            index,
            pose.copy(),
            pixels.copy(),
            desc.copy(),
            np.full(len(pixels), -1, int),
            depth_points=points.copy(),
        )
        frame.image_size = (
            (self.current_gray.shape[1], self.current_gray.shape[0])
            if hasattr(self, "current_gray")
            else None
        )
        self.map.keyframes[ident] = frame
        tracked_pixels = {lid: pixel for lid, pixel in self.accepted_tracks}
        for feature, lid in associations.items():
            if lid in self.map.landmarks:
                frame.landmark_ids[feature] = lid
                observed_pixel = tracked_pixels.get(lid, pixels[feature])
                same_pixel = np.allclose(observed_pixel, pixels[feature], atol=1e-5)
                measured_right = (
                    float(right_u[feature])
                    if same_pixel and np.isfinite(right_u[feature])
                    else None
                )
                if self.stereo is not None and self.current_disparity is not None:
                    # The disparity at a nearby detector feature is a different
                    # observation. Re-measure at the accepted flow coordinate.
                    _, measured = self._measure_stereo_pixels(observed_pixel[None])
                    measured_right = (
                        float(measured[0]) if np.isfinite(measured[0]) else None
                    )
                self.map.landmarks[lid].observations[ident] = Observation(
                    observed_pixel.copy(), measured_right
                )
        if self.stereo is not None:
            for feature in np.flatnonzero(np.isfinite(points).all(axis=1)):
                if frame.landmark_ids[feature] >= 0:
                    continue
                world = pose[:3, :3] @ points[feature] + pose[:3, 3]
                lid = self.map.add_landmark(
                    world,
                    desc[feature],
                    ident,
                    {
                        ident: Observation(
                            pixels[feature].copy(), float(right_u[feature])
                        )
                    },
                )
                frame.landmark_ids[feature] = lid
        elif self.last_keyframe is not None:
            previous = self.map.keyframes[self.last_keyframe]
            measured_pairs = match_descriptors(previous.descriptors, desc)
            if len(measured_pairs):
                a, b = measured_pairs.T
                geometry, valid = triangulate(
                    previous.pixels[a], pixels[b], previous.pose, pose, self.K
                )
                # Independent two-view geometry also covers already tracked features.
                # Reusing global landmark coordinates here would create circular loops.
                for position, j in zip(geometry, valid):
                    ia, ib = a[j], b[j]
                    if not np.isfinite(previous.depth_points[ia]).all():
                        previous.depth_points[ia] = previous.pose[:3, :3].T @ (
                            position - previous.pose[:3, 3]
                        )
                    frame.depth_points[ib] = pose[:3, :3].T @ (position - pose[:3, 3])
            # Try older co-visible views as well as the latest keyframe. Small
            # successive baselines often provide no reliable triangulation.
            for previous_id in reversed(
                list(self.map.keyframes)[:-1][-self.config.bundle_window :]
            ):
                previous = self.map.keyframes[previous_id]
                pairs = match_descriptors(previous.descriptors, desc)
                pairs = np.array(
                    [
                        (a, b)
                        for a, b in pairs
                        if previous.landmark_ids[a] < 0 and frame.landmark_ids[b] < 0
                    ],
                    int,
                ).reshape(-1, 2)
                if not len(pairs):
                    continue
                a, b = pairs.T
                positions, valid = triangulate(
                    previous.pixels[a], pixels[b], previous.pose, pose, self.K
                )
                for position, j in zip(positions, valid):
                    ia, ib = a[j], b[j]
                    lid = self.map.add_landmark(
                        position,
                        desc[ib],
                        previous.id,
                        {
                            previous.id: Observation(previous.pixels[ia].copy()),
                            ident: Observation(pixels[ib].copy()),
                        },
                    )
                    previous.landmark_ids[ia] = lid
                    frame.landmark_ids[ib] = lid
                    if previous.depth_points is None:
                        previous.depth_points = np.full(
                            (len(previous.pixels), 3), np.nan
                        )
                    previous.depth_points[ia] = previous.pose[:3, :3].T @ (
                        position - previous.pose[:3, 3]
                    )
                    frame.depth_points[ib] = pose[:3, :3].T @ (position - pose[:3, 3])
        self.last_keyframe = ident
        self.map.revision += 1
        return ident

    def _motion_prediction(self):
        predicted = self.map.poses[-1].copy()
        self.motion_prediction_source = "held_pose"
        # A held output is not a zero-velocity measurement. Keep a short-lived
        # independently verified stereo increment for matching after brief loss.
        if self.stereo is not None and self.verified_stereo_motion is not None:
            increment, frame = self.verified_stereo_motion
            count = len(self.map.poses)
            accepted = [i for i in range(max(0, count-self.config.bundle_window), count)
                        if self.map.statuses[i] in ("tracking", "relocalized")]
            if accepted and count-frame <= self.config.bundle_window:
                gap = count-accepted[-1]
                if gap <= 3:
                    self.motion_prediction_source = "verified_stereo_increment"
                    return self.map.poses[accepted[-1]] @ np.linalg.matrix_power(increment, gap)
        if (
            len(self.map.poses) >= 2
            and self.map.statuses[-1] in ("tracking", "relocalized")
            and self.map.statuses[-2] == "tracking"
        ):
            predicted = (
                predicted @ np.linalg.inv(self.map.poses[-2]) @ self.map.poses[-1]
            )
            self.motion_prediction_source = "consecutive_accepted_poses"
        return predicted

    def _track(self, pixels, desc, size, relocalize=False, candidate_ids=None):
        landmarks = list(self.map.landmarks.values())
        if candidate_ids is not None:
            landmarks = [l for l in landmarks if l.id in candidate_ids]
        if not landmarks:
            return None, {}
        if not relocalize:
            predicted = self._motion_prediction()
            projected, z = project(
                np.array([l.position for l in landmarks]), predicted, self.K
            )
            visible = (
                (z > 0)
                & (projected[:, 0] >= -40)
                & (projected[:, 0] < size[0] + 40)
                & (projected[:, 1] >= -40)
                & (projected[:, 1] < size[1] + 40)
            )
            landmarks = [l for l, ok in zip(landmarks, visible) if ok]
            if len(landmarks) > self.config.max_landmarks:
                landmarks.sort(
                    key=lambda l: (max(l.observations), len(l.observations)),
                    reverse=True,
                )
                landmarks = landmarks[: self.config.max_landmarks]
        if not landmarks:
            return None, {}
        pairs = match_descriptors(np.array([l.descriptor for l in landmarks]), desc)
        candidates = {landmarks[a].id: (pixels[b], int(b)) for a, b in pairs}
        descriptor_candidates = candidates.copy()
        flow_conflicts = 0
        flow_visibility_rejections = 0
        if not relocalize and self.previous_gray is not None and self.previous_tracks:
            old_pixels = np.array(
                [p for _, p in self.previous_tracks], np.float32
            ).reshape(-1, 1, 2)
            # Seed flow using map geometry and the same motion prediction as PnP.
            # Starting at the previous pixel can lock onto an adjacent repeat at
            # high image velocities even when forward/backward flow agrees.
            predicted_pixels = old_pixels.copy()
            flow_visible = np.zeros(len(old_pixels), bool)
            tracked_ids = [
                j
                for j, (lid, _) in enumerate(self.previous_tracks)
                if lid in self.map.landmarks
            ]
            if tracked_ids:
                world = np.array(
                    [
                        self.map.landmarks[self.previous_tracks[j][0]].position
                        for j in tracked_ids
                    ]
                )
                projected, depth = project(world, self._motion_prediction(), self.K)
                valid = (depth > 0) & np.isfinite(projected).all(axis=1)
                valid &= (projected[:, 0] >= 0) & (projected[:, 0] < size[0])
                valid &= (projected[:, 1] >= 0) & (projected[:, 1] < size[1])
                predicted_pixels[np.array(tracked_ids)[valid], 0] = projected[valid]
                flow_visible[np.array(tracked_ids)[valid]] = True
            flow_visibility_rejections = int((~flow_visible).sum())
            flowed, ok, _ = cv.calcOpticalFlowPyrLK(
                self.previous_gray,
                self.current_gray,
                old_pixels,
                predicted_pixels,
                winSize=(21, 21),
                maxLevel=3,
                flags=cv.OPTFLOW_USE_INITIAL_FLOW,
            )
            if flowed is not None:
                reversed_pixels, back_ok, _ = cv.calcOpticalFlowPyrLK(
                    self.current_gray,
                    self.previous_gray,
                    flowed,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                )
                for j, (lid, _) in enumerate(self.previous_tracks):
                    p = flowed[j, 0]
                    if (
                        lid in self.map.landmarks
                        and flow_visible[j]
                        and ok[j]
                        and back_ok[j]
                        and np.linalg.norm(reversed_pixels[j, 0] - old_pixels[j, 0])
                        < 1.0
                        and 0 <= p[0] < size[0]
                        and 0 <= p[1] < size[1]
                    ):
                        descriptor_match = candidates.get(lid)
                        if (
                            descriptor_match is not None
                            and np.linalg.norm(p - descriptor_match[0]) > 3.0
                        ):
                            # Forward/backward flow can consistently follow the
                            # wrong repeat on railings or lane markings. It must
                            # not overwrite an independent mutual descriptor match.
                            flow_conflicts += 1
                            continue
                        distances = np.linalg.norm(pixels - p, axis=1)
                        feature = int(np.argmin(distances)) if len(distances) else -1
                        if feature >= 0 and distances[feature] > 3:
                            feature = -1
                        candidates[lid] = (p, feature)
        if len(candidates) < self.config.min_inliers:
            return None, {
                "num_matches": len(candidates),
                "flow_visibility_rejections": flow_visibility_rejections,
                "flow_descriptor_conflicts": flow_conflicts,
            }
        seed = None
        if not relocalize:
            seed = self._motion_prediction()

        def solve(measurements):
            identifiers = list(measurements)
            positions = np.array([self.map.landmarks[i].position for i in identifiers])
            observations = np.array([measurements[i][0] for i in identifiers], float)
            diagnostics = {}
            solution = estimate_pose(
                positions,
                observations,
                self.K,
                size,
                self.config.min_inliers if not relocalize else 20,
                initial_pose=seed,
                diagnostics=diagnostics,
            )
            if solution is not None and self.stereo is not None:
                _, measured_right = self._measure_stereo_pixels(observations)
                solution, stereo_diagnostics = refine_stereo_map_pose(
                    solution, positions, observations, measured_right, self.K,
                    self.stereo.baseline, size,
                    self.config.min_inliers if not relocalize else 20,
                    self.stereo.disparity_offset,
                )
                diagnostics.update(stereo_diagnostics)
            return solution, diagnostics, identifiers, positions, observations

        result, pose_diagnostics, ids, world, observed = solve(candidates)
        source = "descriptor_map" if relocalize else "flow_assisted_map"
        augmented_correspondences = len(candidates)
        flow_rejection = None
        if result is None and not relocalize and len(descriptor_candidates) >= self.config.min_inliers:
            # A large, coherent cluster of repeated-texture flow can dominate
            # RANSAC while failing spatial or stereo checks. Independently matched
            # map descriptors must still get a solve with the same acceptance gates.
            fallback, fallback_diagnostics, fallback_ids, fallback_world, fallback_observed = solve(descriptor_candidates)
            if fallback is not None:
                flow_rejection = pose_diagnostics.get("pose_rejection_reason")
                result, pose_diagnostics = fallback, fallback_diagnostics
                candidates = descriptor_candidates
                ids, world, observed = fallback_ids, fallback_world, fallback_observed
                source = "descriptor_map_fallback"
        info = {
            "num_matches": len(ids),
            "valid_3d": len(world),
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            **pose_diagnostics,
            "flow_descriptor_conflicts": flow_conflicts,
            "flow_visibility_rejections": flow_visibility_rejections,
            "descriptor_correspondences": len(descriptor_candidates),
        }
        if source == "descriptor_map_fallback":
            info.update(flow_pose_rejection_reason=flow_rejection,
                        augmented_correspondences=augmented_correspondences)
        if result is None:
            return None, info
        pose, valid, error = result
        inlier_set = set(valid.tolist())
        for j, lid in enumerate(ids):
            landmark = self.map.landmarks[lid]
            landmark.misses = 0 if j in inlier_set else landmark.misses + 1
        info.update(
            num_inliers=len(valid),
            inlier_ratio=len(valid) / len(ids),
            reprojection_error=error,
            feature_cells=coverage(observed[valid], size),
            pose_source=source,
        )
        associations = {
            candidates[ids[j]][1]: ids[j] for j in valid if candidates[ids[j]][1] >= 0
        }
        self.accepted_tracks = [(ids[j], observed[j].copy()) for j in valid]
        return (pose, associations), info

    def _keyframe_stereo_reference(self, pixels, desc, points, size, ranked=None):
        """Verify camera motion against raw keyframe stereo, without map points."""
        if self.stereo is None:
            return None, {}
        if ranked is None:
            local = list(self.map.keyframes.values())[-self.config.bundle_window :]
            ranked = sorted(
                ((len(match_descriptors(k.descriptors, desc)), k.id) for k in local),
                reverse=True,
            )
        query = StereoLoopFrame(pixels, points, desc, size)
        best, stats = None, {}
        for support, ident in ranked[:5]:
            if support < 20:
                continue
            keyframe = self.map.keyframes[ident]
            if keyframe.depth_points is None:
                continue
            source = StereoLoopFrame(
                keyframe.pixels,
                keyframe.depth_points,
                keyframe.descriptors,
                keyframe.image_size or size,
            )
            verified = verify_loop(source, query, self.K, min_inliers=20)
            if verified is not None and verified["inliers"] > stats.get(
                "num_inliers", 0
            ):
                best = (keyframe.pose @ verified["measurement"], {})
                stats = {
                    "num_matches": verified["matches"],
                    "num_inliers": verified["inliers"],
                    "inlier_ratio": verified["inliers"] / verified["matches"],
                    "reprojection_error": verified["median_reprojection_px"],
                    "pose_source": "keyframe_stereo_reference",
                    "relocalization_keyframe": ident,
                    "stereo_reference_verified": True,
                }
        return best, stats

    def _relocalize(self, pixels, desc, size, points=None):
        # Retrieval proposes views; only image-to-landmark PnP can recover tracking.
        self.retrieval.update(self.map.keyframes)
        candidate_ids = (self.retrieval.query(desc, self.config.retrieval_candidates)
                         if self.config.retrieval_candidates > 0 else list(self.map.keyframes))
        ranked = sorted(
            (
                (len(match_descriptors(k.descriptors, desc)), k.id)
                for k in (self.map.keyframes[i] for i in candidate_ids)
            ),
            reverse=True,
        )
        if self.stereo is not None and points is not None:
            verified, stats = self._keyframe_stereo_reference(
                pixels, desc, points, size, ranked
            )
            if verified is not None:
                self.accepted_tracks = []
                return verified, stats
        best = None
        best_stats = {}
        best_tracks = []
        for support, ident in ranked[:5]:
            if support < 20:
                continue
            ids = set(self.map.keyframes[ident].landmark_ids.tolist()) - {-1}
            result, stats = self._track(
                pixels, desc, size, relocalize=True, candidate_ids=ids
            )
            if result is not None and stats["num_inliers"] > best_stats.get(
                "num_inliers", 0
            ):
                best = result
                best_stats = stats
                best_tracks = list(self.accepted_tracks)
                best_stats["relocalization_keyframe"] = ident
        if best is not None:
            self.accepted_tracks = best_tracks
        return best, best_stats

    def process(self, index, image, right=None):
        if index != len(self.map.poses):
            raise ValueError(
                "Frames must arrive in consecutive order, starting at zero"
            )
        self.loop_worker.poll(self.map)
        self.current_gray = (
            cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
        )
        pixels, desc, points, right_u = self._extract(image, right)
        size = (image.shape[1], image.shape[0])
        info = {
            "frame": index,
            "features": len(pixels),
            "valid_stereo_depth": int(np.isfinite(points).all(axis=1).sum()),
            "num_matches": 0,
            "num_inliers": 0,
            "tracking_ok": False,
        }
        pose = self.map.poses[-1].copy() if self.map.poses else np.eye(4)
        anchor = self.last_keyframe
        inlier_features = set()
        verified_motion = None
        status = "initializing" if not self.map.landmarks else "lost"
        if not self.map.keyframes:
            if self.stereo is not None:
                valid = np.flatnonzero(np.isfinite(points).all(axis=1))
                if len(valid) >= 30 and coverage(pixels[valid], size) >= 4:
                    anchor = self._keyframe(
                        index, pose, pixels, desc, points, right_u, {}
                    )
                    status = "tracking"
                    info["tracking_ok"] = True
                    self.accepted_tracks = [
                        (
                            int(self.map.keyframes[anchor].landmark_ids[j]),
                            pixels[j].copy(),
                        )
                        for j in valid
                    ]
                    inlier_features = set(valid.tolist())
            elif self.initial is None:
                self.initial = (index, pixels.copy(), desc.copy())
            else:
                start, initial_pixels, initial_desc = self.initial
                pairs = match_descriptors(initial_desc, desc)
                info["num_matches"] = len(pairs)
                result = (
                    initialize_monocular(
                        initial_pixels[pairs[:, 0]], pixels[pairs[:, 1]], self.K, size
                    )
                    if len(pairs)
                    else None
                )
                if result is not None:
                    pose, positions, valid = result
                    first = MappingKeyframe(
                        0,
                        start,
                        np.eye(4),
                        initial_pixels,
                        initial_desc,
                        np.full(len(initial_pixels), -1, int),
                        depth_points=np.full((len(initial_pixels), 3), np.nan),
                    )
                    second = MappingKeyframe(
                        1,
                        index,
                        pose.copy(),
                        pixels.copy(),
                        desc.copy(),
                        np.full(len(pixels), -1, int),
                        depth_points=np.full((len(pixels), 3), np.nan),
                    )
                    self.map.keyframes.update({0: first, 1: second})
                    first.image_size = second.image_size = size
                    for position, j in zip(positions, valid):
                        a, b = pairs[j]
                        lid = self.map.add_landmark(
                            position,
                            desc[b],
                            0,
                            {
                                0: Observation(initial_pixels[a].copy()),
                                1: Observation(pixels[b].copy()),
                            },
                        )
                        first.landmark_ids[a] = lid
                        second.landmark_ids[b] = lid
                        first.depth_points[a] = position
                        second.depth_points[b] = pose[:3, :3].T @ (
                            position - pose[:3, 3]
                        )
                    self.last_keyframe = anchor = 1
                    self.map.revision += 1
                    self.map.pose_anchors[start] = 0
                    status = "tracking"
                    info.update(
                        tracking_ok=True,
                        num_inliers=len(valid),
                        inlier_ratio=len(valid) / len(pairs),
                    )
                    self.accepted_tracks = [
                        (
                            int(second.landmark_ids[pairs[j, 1]]),
                            pixels[pairs[j, 1]].copy(),
                        )
                        for j in valid
                    ]
                    inlier_features = set(pairs[valid, 1].tolist())
        else:
            result, stats = self._track(pixels, desc, size)
            info.update(stats)
            recovered = False
            stereo_reference = False
            if self.previous_stereo_geometry is not None:
                previous_frame, previous_index = self.previous_stereo_geometry
                if index - previous_index <= 3:
                    measured = StereoLoopFrame(pixels, points, desc, size)
                    prior = (
                        np.linalg.inv(self.map.poses[previous_index])
                        @ self._motion_prediction()
                    )
                    verified = estimate_stereo_reference(
                        previous_frame,
                        measured,
                        self.K,
                        min_inliers=self.config.min_inliers,
                        initial_pose=prior,
                    )
                    if verified is not None:
                        if verified["reverse_checked"] and index-previous_index == 1:
                            self.verified_stereo_motion = (verified["measurement"].copy(), index)
                        if verified["reverse_checked"]:
                            verified_motion = (previous_index, verified["measurement"].copy())
                        reference_pose = (
                            self.map.poses[previous_index] @ verified["measurement"]
                        )
                        info["stereo_reference_verified"] = True
                        info["stereo_bidirectional_refinement"] = verified.get("bidirectional_refinement")
                        info["stereo_reference_verification"] = (
                            "bidirectional_pnp"
                            if verified["reverse_checked"]
                            else "source_depth_pnp"
                        )
                        conflict = False
                        if result is not None:
                            difference = np.linalg.inv(reference_pose) @ result[0]
                            translation_error = float(np.linalg.norm(difference[:3, 3]))
                            rotation_error = float(
                                np.degrees(
                                    np.linalg.norm(cv.Rodrigues(difference[:3, :3])[0])
                                )
                            )
                            info.update(
                                map_reference_translation_error_m=translation_error,
                                map_reference_rotation_error_deg=rotation_error,
                            )
                            # Use the same disagreement limits as bidirectional
                            # stereo verification. A low left reprojection error
                            # cannot justify contradicting independent metric motion.
                            conflict = translation_error > 0.5 or rotation_error > 1.5
                        if result is None or conflict:
                            result = (reference_pose, {})
                            self.accepted_tracks = []
                            info.update(
                                num_matches=verified["matches"],
                                num_inliers=verified["inliers"],
                                inlier_ratio=verified["inliers"] / verified["matches"],
                                reprojection_error=verified["median_reprojection_px"],
                                pose_source="stereo_tracking_reference",
                                reference_inlier_features=verified["target_features"],
                                map_pose_rejected_for_stereo_conflict=conflict,
                            )
                            stereo_reference = True
            if self.stereo is not None and result is None:
                reference, reference_stats = self._keyframe_stereo_reference(
                    pixels, desc, points, size
                )
                if reference is not None:
                    recovered = True
                    result = reference
                    self.accepted_tracks = []
                    info.update(reference_stats)
                    stereo_reference = True
            if result is None:
                result, stats = self._relocalize(pixels, desc, size, points)
                if result is not None:
                    info.update(stats)
                    recovered = True
                    stereo_reference = (
                        stats.get("pose_source") == "keyframe_stereo_reference"
                    )
            if result is not None:
                pose, associations = result
                inlier_features = set(associations) or set(
                    info.get("reference_inlier_features", [])
                )
                recovered = recovered or (
                    bool(self.map.statuses) and self.map.statuses[-1] == "lost"
                )
                status = "relocalized" if recovered else "tracking"
                info["tracking_ok"] = True
                last = self.map.keyframes[self.last_keyframe]
                # Verified flow observations may have no current detector index.
                # Count each live landmark once across both correspondence sources.
                tracked_landmarks = {
                    lid for lid in associations.values() if lid in self.map.landmarks
                } | {
                    lid for lid, _ in self.accepted_tracks if lid in self.map.landmarks
                }
                info["tracked_landmarks"] = len(tracked_landmarks)
                if (
                    stereo_reference
                    or index - last.frame >= self.config.keyframe_interval
                    or (index - last.frame >= 2 and len(tracked_landmarks) < 80)
                ):
                    anchor = self._keyframe(
                        index, pose, pixels, desc, points, right_u, associations
                    )
        if info["tracking_ok"]:
            self.previous_gray = self.current_gray.copy()
            tracks = {}
            if anchor is not None and self.map.keyframes[anchor].frame == index:
                for j, lid in enumerate(self.map.keyframes[anchor].landmark_ids):
                    if lid >= 0:
                        tracks[int(lid)] = self.map.keyframes[anchor].pixels[j].copy()
            # Never replace a verified flow measurement with a nearby detector pixel.
            tracks.update({lid: p for lid, p in self.accepted_tracks})
            self.previous_tracks = list(tracks.items())
        self.map.record(pose, status, anchor)
        if info["tracking_ok"] and verified_motion is not None:
            previous_index, measurement = verified_motion
            self.map.add_stereo_motion(previous_index, index, measurement)
        if (
            info["tracking_ok"]
            and anchor is not None
            and self.map.keyframes[anchor].frame == index
            and len(self.map.keyframes) >= 3
        ):
            with self.profile.measure("local_bundle"):
                report = local_bundle_adjustment(
                    self.map,
                    self.K,
                    self.stereo.baseline if self.stereo is not None else 0.0,
                    window=self.config.bundle_window,
                    disparity_offset=self.stereo.disparity_offset if self.stereo is not None else 0.,
                ) if self.config.bundle_enabled else {"applied": False, "reason": "diagnostic_ablation"}
            self.bundle_reports.append({"frame": index, **report})
            pose = self.map.poses[-1].copy()
            self.loop_worker.schedule(self.map)
        if info["tracking_ok"] and self.stereo is not None:
            self.previous_stereo_geometry = (
                StereoLoopFrame(pixels.copy(), points.copy(), desc.copy(), size),
                index,
            )
        # Bound the tracking working set, not the persistent map. Old, valid
        # landmarks remain available to geometric relocalization and loop correction.
        self.map.landmarks = {
            i: l for i, l in self.map.landmarks.items() if l.misses < 5
        }
        removed = False
        for keyframe in self.map.keyframes.values():
            stale = np.array(
                [
                    lid >= 0 and lid not in self.map.landmarks
                    for lid in keyframe.landmark_ids
                ]
            )
            if stale.any():
                keyframe.landmark_ids[stale] = -1
                removed = True
        if removed:
            self.map.revision += 1
            self.map.geometry_revision += 1
        info.update(
            state=status,
            motion_prediction_source=self.motion_prediction_source,
            map_revision=self.map.revision,
            translation_scale="metric" if self.map.metric else "arbitrary",
            feature_points=pixels.tolist(),
            inlier_mask=[j in inlier_features for j in range(len(pixels))],
        )
        self.diagnostics.append(
            {
                k: v
                for k, v in info.items()
                if k not in ("feature_points", "inlier_mask")
            }
        )
        return pose, info

    def close(self, finish=True):
        self.loop_worker.close(self.map, finish=finish)

    def map_state(self):
        return {
            "keyframes": len(self.map.keyframes),
            "map_points": len(self.map.landmarks),
            "revision": self.map.revision,
        }

    def map_points_sample(self, max_points=1000):
        points = [l.position.tolist() for l in self.map.landmarks.values()]
        return points[:: max(1, len(points) // max(1, max_points))][:max_points]
