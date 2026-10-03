"""Persistent stereo/monocular image-only tracking and local mapping."""

from dataclasses import dataclass
import hashlib
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
from keyframe_index import KeyframeIndex
from descriptor_matching import DescriptorMatcher
from performance import PerformanceConfig
from stereo_depth import StereoSearchConfig, verify_stereo_depth_candidates
from stereo_pose_arbitration import SupportedStereoFrame, SupportedStereoHoldout, arbitrate_stereo_pose


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
    stereo_depth_policy: str = "supported"
    stereo_pose_arbitration: bool = False


class SharedSlam:
    def __init__(self, matrix, stereo=None, config=None, performance=None):
        self.performance = performance or PerformanceConfig()
        self.matcher = DescriptorMatcher(self.performance.matching_backend)
        self._landmark_cache = None
        self._last_map_track_inlier_misses = None
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
        if self.config.stereo_depth_policy not in ('supported', 'verified_fallback'):
            raise ValueError('Invalid stereo depth policy')
        if stereo is None and self.config.stereo_depth_policy != 'supported':
            raise ValueError('Verified stereo depth requires two calibrated cameras')
        if stereo is None and self.config.stereo_pose_arbitration:
            raise ValueError('Stereo pose arbitration requires two calibrated cameras')
        self.previous_supported_stereo = None
        self.current_supported_stereo = None
        self._supported_extraction = None
        self._arbitration_context = None
        identity = hashlib.sha256()
        for value in (self.K, self.stereo.Q if self.stereo is not None else np.empty(0)):
            identity.update(value.dtype.str.encode()); identity.update(value.tobytes())
        if self.stereo is not None:
            identity.update(np.float64(self.stereo.baseline).tobytes())
        self.stereo_calibration_identity = identity.hexdigest()
        self.stereo_search_config = StereoSearchConfig()
        self.current_left_gray = self.current_right_gray = None
        self.stereo_depth_verification = {}
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
        self.profile = StageProfile(detailed=self.performance.profile)
        self.loop_worker = LiveLoopWorker(self.K, self.map.metric, mode=self.config.loop_mode,
                                          performance=self.performance, profiler=self.profile, matcher=self.matcher)
        self.retrieval = KeyframeRetrieval()
        self.retrieval_index = KeyframeIndex() if self.performance.retrieval == 'indexed' else None
        for name in ("_extract", "_track", "_keyframe", "_relocalize", "_keyframe_stereo_reference"):
            setattr(self, name, self.profile.wrap(name, getattr(self, name)))
        self.previous_gray = None
        self.previous_tracks = []
        self.accepted_tracks = []
        self.previous_stereo_geometry = None
        self.verified_stereo_motion = None
        self.motion_prediction_source = "held_pose"
        self.current_disparity = None

    def _match(self, first, second):
        self.profile.count('descriptor_pairs', len(first) * len(second))
        return self.profile.call('matching', self.matcher, first, second)

    def _cached_landmarks(self):
        # A correction and its revision must be observed as one snapshot. A
        # mixed position array must never be cached under a newer revision.
        with self.map.lock:
            structure = (len(self.map.landmarks), self.map.next_landmark)
            cache = self._landmark_cache
            if cache is None or cache[0] != structure:
                landmarks = list(self.map.landmarks.values())
                cache = [structure, -1, landmarks, None]
                self._landmark_cache = cache
            if cache[1] != self.map.revision:
                cache[3] = np.asarray([l.position for l in cache[2]])
                cache[1] = self.map.revision
            return cache

    def _measure_supported_stereo_pixels(self, pixels):
        return self._measure_stereo_pixels(pixels, supported_only=True)

    def _measure_stereo_pixels(self, pixels, supported_only=False, capture_supported=False):
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
        if capture_supported:
            self._supported_extraction = (points.copy(), right_u.copy())
        if self.config.stereo_depth_policy == 'verified_fallback' and not supported_only:
            candidates = np.flatnonzero(inside & ~np.isfinite(points).all(axis=1))
            if len(candidates):
                nearest = np.rint(pixels[candidates]).astype(int)
                raw = self.current_disparity[nearest[:, 1], nearest[:, 0]]
                den = self.stereo.Q[3, 2]*raw + self.stereo.Q[3, 3]
                z = np.divide(self.stereo.Q[2, 3], den,
                              out=np.full(len(raw), np.nan), where=den > 0)
                eligible = np.isfinite(raw) & (raw > 0) & (raw < 96)
                eligible &= np.isfinite(z) & (z > .1) & (z < 100)
                candidates = candidates[eligible]
                # Nearest disparity decides eligibility only. Independent full
                # image searches supply the returned correspondence and depth.
                q = self.stereo.Q
                if len(candidates) and q[3, 2] > 0 and q[2, 3] > 0:
                    bf, offset = q[2, 3]/q[3, 2], -q[3, 3]/q[3, 2]
                    bounds = (max(0., offset+bf/100), min(96., offset+bf/.1))
                    if bounds[0] < bounds[1]:
                        measured, counters = self.profile.call(
                            'stereo_depth_verification', verify_stereo_depth_candidates,
                            self.current_left_gray, self.current_right_gray,
                            pixels[candidates], self.stereo_search_config, bounds)
                        for key, value in counters.items():
                            self.stereo_depth_verification[key] = self.stereo_depth_verification.get(key, 0)+value
                        d = pixels[candidates, 0]-measured
                        den = q[3, 2]*d+q[3, 3]
                        z = np.divide(q[2, 3], den, out=np.full(len(d), np.nan), where=den > 0)
                        good = np.isfinite(z) & (d > 0) & (d < 96) & (z > .1) & (z < 100)
                        accepted = candidates[good]
                        rays = np.c_[pixels[accepted], np.ones(len(accepted))] @ self.inverse_K.T
                        points[accepted] = rays*z[good, None]
                        right_u[accepted] = measured[good]
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
            points, right_u = self._measure_stereo_pixels(
                pixels, capture_supported=self.config.stereo_pose_arbitration)
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
        detected_features = len(desc)
        observed = set(associations.values())
        appended_pixels, appended_desc = [], []
        for lid, pixel in self.accepted_tracks:
            if lid in self.map.landmarks and lid not in observed:
                feature = len(pixels) + len(appended_pixels)
                appended_pixels.append(pixel)
                appended_desc.append(self.map.landmarks[lid].descriptor)
                associations[feature] = lid
                observed.add(lid)
        if appended_pixels:
            pixels = np.vstack([pixels, appended_pixels])
            desc = np.vstack([desc, appended_desc])
            points = np.vstack([points, np.full((len(appended_pixels), 3), np.nan)])
            right_u = np.r_[right_u, np.full(len(appended_pixels), np.nan)]
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
        frame.retrieval_descriptors = frame.descriptors[:detected_features]
        frame.pixels.setflags(write=False)
        frame.descriptors.setflags(write=False)
        frame.retrieval_descriptors.setflags(write=False)
        self.map.keyframes[ident] = frame
        tracked_pixels = {lid: pixel for lid, pixel in self.accepted_tracks}
        measured_rights = {}
        if self.performance.cpu_optimizations and self.stereo is not None and self.current_disparity is not None:
            features = [f for f, lid in associations.items() if lid in self.map.landmarks]
            if features:
                observations = np.asarray([tracked_pixels.get(associations[f], pixels[f]) for f in features])
                _, measured = self._measure_stereo_pixels(observations)
                measured_rights = dict(zip(features, measured))
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
                    if feature in measured_rights:
                        measured = [measured_rights[feature]]
                    else:
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
            measured_pairs = self._match(previous.descriptors, desc)
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
                pairs = self._match(previous.descriptors, desc)
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
        # Bind any pending miss snapshot to this exact successful direct track
        # result. Ranked recovery may call _track several times in one frame.
        self._last_map_track_inlier_misses = None
        cache = None
        if self.performance.cpu_optimizations:
            cache = self._cached_landmarks()
            landmarks = cache[2]
        else:
            landmarks = list(self.map.landmarks.values())
        if candidate_ids is not None:
            landmarks = [l for l in landmarks if l.id in candidate_ids]
            cache = None
        if not landmarks:
            return None, {}
        if not relocalize:
            predicted = self._motion_prediction()
            projected, z = project(
                cache[3] if cache is not None else np.array([l.position for l in landmarks]), predicted, self.K
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
        # Keep descriptor storage bounded by the already-filtered tracking set.
        # A persistent dense matrix here duplicates every descriptor in the map,
        # even though tracking truncates the active list to max_landmarks above.
        descriptors = np.array([l.descriptor for l in landmarks])
        pairs = self._match(descriptors, desc)
        arbitration = self._arbitration_context if self.config.stereo_pose_arbitration else None
        if arbitration is not None:
            # A held detector observation may map to a different older landmark
            # than the source snapshot. Exclude every linked descriptor/flow ID.
            arbitration['excluded_landmarks'].update(
                landmarks[a].id for a, b in pairs if b in arbitration['excluded_targets'])
            pairs = np.asarray([(a, b) for a, b in pairs
                                if b not in arbitration['excluded_targets']
                                and landmarks[a].id not in arbitration['excluded_landmarks']], int).reshape(-1, 2)
        candidates = {landmarks[a].id: (pixels[b], int(b)) for a, b in pairs}
        descriptor_candidates = candidates.copy()
        flow_conflicts = 0
        flow_visibility_rejections = 0
        previous_tracks = (self.previous_tracks if arbitration is None else
                           [(lid, p) for lid, p in self.previous_tracks
                            if lid not in arbitration['excluded_landmarks']])
        if not relocalize and self.previous_gray is not None and previous_tracks:
            old_pixels = np.array(
                [p for _, p in previous_tracks], np.float32
            ).reshape(-1, 1, 2)
            # Seed flow using map geometry and the same motion prediction as PnP.
            # Starting at the previous pixel can lock onto an adjacent repeat at
            # high image velocities even when forward/backward flow agrees.
            predicted_pixels = old_pixels.copy()
            flow_visible = np.zeros(len(old_pixels), bool)
            tracked_ids = [
                j
                for j, (lid, _) in enumerate(previous_tracks)
                if lid in self.map.landmarks
            ]
            if tracked_ids:
                world = np.array(
                    [
                        self.map.landmarks[previous_tracks[j][0]].position
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
                for j, (lid, _) in enumerate(previous_tracks):
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
                        if arbitration is not None and (
                                feature in arbitration['excluded_targets']
                                or self._physical_pixel_key(p) in arbitration['excluded_target_pixels']):
                            continue
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
            if arbitration is not None:
                arbitration['map_fit_landmarks'].update(identifiers)
                arbitration['map_fit_targets'].update(
                    measurements[lid][1] for lid in identifiers if measurements[lid][1] >= 0)
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
        if self.config.stereo_pose_arbitration and not relocalize:
            self._last_map_track_inlier_misses = {
                int(ids[j]): int(self.map.landmarks[ids[j]].misses)
                for j in inlier_set
            }
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
                ((len(self._match(k.descriptors, desc)), k.id) for k in local),
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
        eligible = list(self.map.keyframes)
        if self.performance.retrieval == 'current':
            self.retrieval.update(self.map.keyframes)
            candidate_ids = (self.retrieval.query(desc, self.config.retrieval_candidates)
                             if self.config.retrieval_candidates > 0 else eligible)
        elif self.retrieval_index is not None:
            for ident, frame in self.map.keyframes.items():
                appearance = getattr(frame, 'retrieval_descriptors', frame.descriptors)
                self.retrieval_index.upsert(ident, frame.frame, appearance)
            candidate_ids = self.retrieval_index.query(desc, eligible, limit=20)
            if candidate_ids is None:
                candidate_ids = eligible
        else:
            candidate_ids = eligible
        ranked = sorted(
            (
                (len(self._match(k.descriptors, desc)), k.id)
                for k in (self.map.keyframes[i] for i in candidate_ids)
            ),
            reverse=True,
        )
        result, stats = self._recover_ranked(pixels, desc, size, points, ranked)
        if result is None and self.retrieval_index is not None and len(candidate_ids) < len(eligible):
            cached_ids = set(candidate_ids)
            ranked = sorted(ranked + [(len(self._match(self.map.keyframes[i].descriptors, desc)), i)
                                      for i in eligible if i not in cached_ids], reverse=True)
            self.profile.count('recovery_exhaustive_fallbacks')
            return self._recover_ranked(pixels, desc, size, points, ranked)
        return result, stats

    def _recover_ranked(self, pixels, desc, size, points, ranked):
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

    def _prepare_frame_images(self, image, right):
        if image is None:
            raise ValueError('Missing image')
        self.current_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
        self.stereo_depth_verification = {}
        if self.config.stereo_depth_policy == 'verified_fallback' and self.stereo is not None:
            # Refresh before extraction, including diagnostic cache hits.
            self.current_left_gray = np.asarray(self.current_gray, np.float32)
            self.current_right_gray = (None if right is None else np.asarray(
                cv.cvtColor(right, cv.COLOR_BGR2GRAY) if right.ndim == 3 else right, np.float32))

    @staticmethod
    def _physical_pixel_key(pixel):
        return tuple(np.asarray(pixel, np.float32).tolist())

    def _capture_supported_stereo(self, index, pixels, desc, size):
        # Native extraction copies the supported surface before restoration.
        # A cache hit skips that producer, so derive it from raw disparity only.
        raw = self._supported_extraction
        if raw is None:
            raw = self._measure_supported_stereo_pixels(pixels)
        return SupportedStereoFrame(pixels, desc, raw[0], raw[1],
                                    np.full(len(pixels), -1, int), index, size,
                                    self.stereo_calibration_identity)

    def _prepare_stereo_arbitration(self, index, current):
        report = {'choice': 'map', 'reason': 'missing_previous_supported_frame'}
        previous = self.previous_supported_stereo
        if (previous is None or self.previous_stereo_geometry is None
                or previous.frame != self.previous_stereo_geometry[1]
                or not 0 < index-previous.frame <= 3
                or previous.frame >= len(self.map.poses)
                or self.map.statuses[previous.frame] not in ('tracking', 'relocalized')
                or previous.calibration_identity != current.calibration_identity):
            return None, report
        pairs = self._match(previous.descriptors, current.descriptors)
        if not len(pairs):
            report['reason'] = 'insufficient_supported_pool'
            return None, report
        # Multiple SIFT orientations can describe one physical pixel. Drop every
        # duplicate in either full extraction before splitting the matching pool.
        _, previous_inverse, previous_counts = np.unique(
            np.asarray(previous.pixels, np.float32), axis=0, return_inverse=True, return_counts=True)
        _, current_inverse, current_counts = np.unique(
            np.asarray(current.pixels, np.float32), axis=0, return_inverse=True, return_counts=True)
        a, b = pairs.T
        valid = (np.isfinite(previous.points[a]).all(axis=1)
                 & np.isfinite(current.points[b]).all(axis=1)
                 & np.isfinite(previous.right_u[a]) & np.isfinite(current.right_u[b])
                 & (previous.right_u[a] >= 0) & (previous.right_u[a] < previous.image_size[0])
                 & (current.right_u[b] >= 0) & (current.right_u[b] < current.image_size[0])
                 & (previous_counts[previous_inverse[a]] == 1)
                 & (current_counts[current_inverse[b]] == 1))
        pool = pairs[valid]
        fit = pool[pool[:, 0] % 2 == 0]
        held = pool[pool[:, 0] % 2 == 1]
        minimum = self.config.min_inliers
        report.update(supported_pool=len(pool), fit_count=len(fit), holdout_count=len(held),
                      dropped_duplicate_matches=int(np.sum(
                          (previous_counts[previous_inverse[a]] > 1)
                          | (current_counts[current_inverse[b]] > 1))),
                      source_frame=previous.frame, target_frame=index)
        if (min(len(fit), len(held)) < minimum
                or any(coverage(frame.pixels[subset[:, column]], frame.image_size) < 3
                       for subset in (fit, held)
                       for frame, column in ((previous, 0), (current, 1)))):
            report['reason'] = 'insufficient_reserved_support'
            return None, report
        source, target = held.T
        source_pixels = {self._physical_pixel_key(p) for p in previous.pixels[source]}
        target_pixels = {self._physical_pixel_key(p) for p in current.pixels[target]}
        excluded_targets = {j for j, p in enumerate(current.pixels)
                            if self._physical_pixel_key(p) in target_pixels}
        excluded_landmarks = set(previous.landmark_ids[source].tolist()) - {-1}
        excluded_landmarks.update(lid for lid, p in self.previous_tracks
                                  if self._physical_pixel_key(p) in source_pixels)
        evidence = SupportedStereoHoldout(previous.points[source], current.pixels[target],
                                         current.right_u[target], source, target,
                                         previous.landmark_ids[source],
                                         'immutable_supported_extraction', previous.frame,
                                         previous.calibration_identity)
        context = {'previous': previous, 'current': current, 'fit': fit, 'evidence': evidence,
                   'excluded_targets': excluded_targets, 'excluded_landmarks': excluded_landmarks,
                   'excluded_target_pixels': target_pixels,
                   'map_fit_targets': set(), 'map_fit_landmarks': set(), 'report': report}
        source_frame = StereoLoopFrame(previous.pixels, previous.points, previous.descriptors, previous.image_size)
        target_frame = StereoLoopFrame(current.pixels, current.points, current.descriptors, current.image_size)
        verified = self.profile.call(
            'stereo_pose_arbitration_fit', estimate_stereo_reference,
            source_frame, target_frame, self.K, min_inliers=minimum,
            initial_pose=None, matcher=lambda first, second: fit.copy())
        if verified is None or not verified['reverse_checked']:
            report['reason'] = 'independent_training_failed'
            return None, report
        context['verified'] = verified
        report['reason'] = 'reserved_supported_evidence'
        return context, report

    def _arbitrate_supported_pose(self, context, map_pose, measurement):
        previous = context['previous']
        with self.map.lock:
            relative = np.linalg.inv(self.map.poses[previous.frame]) @ map_pose
        report = arbitrate_stereo_pose(
            relative, measurement, context['evidence'], self.K, self.stereo.baseline,
            context['current'].image_size, disparity_offset=self.stereo.disparity_offset,
            minimum_inliers=self.config.min_inliers,
            calibration_identity=self.stereo_calibration_identity,
            independent_training_verified=True, map_holdout_excluded=True,
            independent_fit_source_ids=context['fit'][:, 0],
            independent_fit_target_ids=context['fit'][:, 1],
            map_fit_target_ids=list(context['map_fit_targets']),
            map_fit_landmark_ids=list(context['map_fit_landmarks']))
        return {**context['report'], **report,
                'independent_fit_sha256': hashlib.sha256(
                    np.ascontiguousarray(context['fit'], dtype='<i8').tobytes()).hexdigest(),
                'map_fit_landmarks': len(context['map_fit_landmarks']),
                'map_fit_target_features': len(context['map_fit_targets']),
                'physical_identity': 'exact_float32_pixels_duplicates_dropped'}

    def _validate_stereo_associations_at_pose(
        self, pose, associations, accepted_tracks, pixels, size, final_solve_positions,
        previous_inlier_misses=None,
    ):
        """Keep only existing map links that fit a selected fixed stereo pose.

        This is deliberately a validation pass: it never estimates or changes the
        selected pose. Current right-image measurements are sampled at each
        accepted observation, including subpixel flow coordinates.
        """
        track_pixels = {}
        for landmark_id, pixel in accepted_tracks:
            track_pixels.setdefault(int(landmark_id), np.asarray(pixel, float).copy())
        feature_for_landmark = {}
        for feature, landmark_id in associations.items():
            feature_for_landmark.setdefault(int(landmark_id), int(feature))
        candidate_ids = list(dict.fromkeys(
            [int(lid) for lid, _ in accepted_tracks]
            + [int(lid) for lid in associations.values()]
        ))
        rejected = {
            'missing_landmark': 0, 'invalid_observation': 0,
            'invalid_stereo_measurement': 0, 'behind_camera': 0,
            'outside_image': 0, 'left_residual': 0, 'right_residual': 0,
        }
        left_errors = []
        right_errors = []
        retained_pixels = []
        retained_ids = []
        valid_ids = []
        observed_pixels = []
        predicted_left = []
        predicted_right = []
        with self.map.lock:
            for landmark_id in candidate_ids:
                landmark = self.map.landmarks.get(landmark_id)
                if landmark is None:
                    rejected['missing_landmark'] += 1
                    continue
                if landmark_id in track_pixels:
                    observed = track_pixels[landmark_id]
                else:
                    feature = feature_for_landmark.get(landmark_id, -1)
                    if feature < 0 or feature >= len(pixels):
                        rejected['invalid_observation'] += 1
                        continue
                    observed = np.asarray(pixels[feature], float)
                world = np.asarray(landmark.position, float)
                if (observed.shape != (2,) or not np.isfinite(observed).all()
                        or world.shape != (3,) or not np.isfinite(world).all()
                        or observed[0] < 0 or observed[0] >= size[0]
                        or observed[1] < 0 or observed[1] >= size[1]):
                    rejected['invalid_observation'] += 1
                    continue
                camera = (world - pose[:3, 3]) @ pose[:3, :3]
                if not np.isfinite(camera).all() or camera[2] <= 0:
                    rejected['behind_camera'] += 1
                    continue
                homogeneous = self.K @ camera
                projected = homogeneous[:2] / homogeneous[2]
                right = (projected[0] - self.K[0, 0] * self.stereo.baseline / camera[2]
                         - self.stereo.disparity_offset)
                if (not np.isfinite(projected).all() or not np.isfinite(right)
                        or projected[0] < 0 or projected[0] >= size[0]
                        or projected[1] < 0 or projected[1] >= size[1]
                        or right < 0 or right >= size[0]):
                    rejected['outside_image'] += 1
                    continue
                valid_ids.append(landmark_id)
                observed_pixels.append(observed)
                predicted_left.append(projected)
                predicted_right.append(right)

        if observed_pixels:
            observed_array = np.asarray(observed_pixels, float)
            sampled_points, sampled_right = self._measure_supported_stereo_pixels(observed_array)
            for i, landmark_id in enumerate(valid_ids):
                if (not np.isfinite(sampled_points[i]).all()
                        or not np.isfinite(sampled_right[i])
                        or sampled_right[i] < 0 or sampled_right[i] >= size[0]):
                    rejected['invalid_stereo_measurement'] += 1
                    continue
                left_error = float(np.linalg.norm(predicted_left[i] - observed_array[i]))
                right_error = float(abs(predicted_right[i] - sampled_right[i]))
                if left_error > 2.0:
                    rejected['left_residual'] += 1
                    continue
                if right_error > 2.0:
                    rejected['right_residual'] += 1
                    continue
                retained_ids.append(landmark_id)
                retained_pixels.append(observed_array[i])
                left_errors.append(left_error)
                right_errors.append(right_error)

        retained_ids = set(retained_ids)
        retained_pixels = np.asarray(retained_pixels, float).reshape(-1, 2)
        left_errors = np.asarray(left_errors, float)
        right_errors = np.asarray(right_errors, float)
        count = len(retained_ids)
        denominator = int(final_solve_positions) if final_solve_positions is not None else 0
        ratio = count / denominator if denominator > 0 else 0.0
        spatial_coverage = coverage(retained_pixels, size)
        med_left = float(np.median(left_errors)) if len(left_errors) else None
        med_right = float(np.median(right_errors)) if len(right_errors) else None
        failed_gates = []
        if count < self.config.min_inliers:
            failed_gates.append('insufficient_inliers')
        if denominator <= 0:
            failed_gates.append('missing_final_solve_positions')
        if ratio < 0.25:
            failed_gates.append('insufficient_retention_ratio')
        if spatial_coverage < 3:
            failed_gates.append('insufficient_spatial_coverage')
        if med_left is None or med_left > 1.5:
            failed_gates.append('median_left_residual')
        if med_right is None or med_right > 1.5:
            failed_gates.append('median_right_residual')
        eligible = not failed_gates
        reason = 'retained' if eligible else failed_gates[0]
        with self.map.lock:
            if previous_inlier_misses is not None:
                # _track provisionally reset these inliers before arbitration.
                # Restore one outcome from their pre-frame count so repeated
                # independent-pose rejection can reach the normal cull limit.
                for landmark_id, previous_misses in previous_inlier_misses.items():
                    landmark = self.map.landmarks.get(landmark_id)
                    if landmark is not None:
                        landmark.misses = (
                            0 if eligible and landmark_id in retained_ids
                            else int(previous_misses) + 1
                        )
            else:
                # Direct helper calls without a successful map solve do not
                # have a pre-frame snapshot to reconcile.
                for landmark_id in candidate_ids:
                    landmark = self.map.landmarks.get(landmark_id)
                    if landmark is not None:
                        landmark.misses = 0 if eligible and landmark_id in retained_ids else max(landmark.misses, 1)
        if eligible:
            kept_associations = {
                feature: landmark_id for feature, landmark_id in associations.items()
                if int(landmark_id) in retained_ids
            }
            kept_tracks = [
                (int(landmark_id), np.asarray(pixel).copy())
                for landmark_id, pixel in accepted_tracks
                if int(landmark_id) in retained_ids
            ]
        else:
            kept_associations, kept_tracks = {}, []
        diagnostics = {
            'eligible': bool(eligible), 'reason': reason,
            'candidate_landmarks': len(candidate_ids), 'retained_landmarks': count,
            'retained_ratio': float(ratio), 'final_solve_positions': denominator,
            'spatial_coverage': int(spatial_coverage),
            'median_left_residual_px': med_left,
            'median_right_residual_px': med_right,
            'rejected': rejected,
        }
        return kept_associations, kept_tracks, diagnostics

    @staticmethod
    def _is_proper_se3(value):
        try:
            raw = np.asarray(value)
            if np.iscomplexobj(raw):
                return False
            pose = np.asarray(raw, float)
            return bool(
                pose.shape == (4, 4)
                and np.isfinite(pose).all()
                and np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-8, rtol=0.)
                and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0.)
                and np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6, rtol=0.)
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return False

    def _live_stereo_calibration_identity(self):
        """Hash the calibration currently used by the estimator and validator."""
        digest = hashlib.sha256()
        matrix = np.asarray(self.K)
        q = np.asarray(self.stereo.Q) if self.stereo is not None else np.empty(0)
        for value in (matrix, q):
            digest.update(value.dtype.str.encode())
            digest.update(value.tobytes())
        if self.stereo is not None:
            digest.update(np.float64(self.stereo.baseline).tobytes())
        return digest.hexdigest()

    def _full_supported_match_pairs(
        self, source, target, previous_index, index, previous_frame, measured_frame, size,
    ):
        """Build all mutual descriptor pairs with unique physical endpoints.

        Depth availability is intentionally not a match filter: the existing
        reference estimator independently gates forward and reverse PnP rows.
        """
        report = {
            'attempted': True,
            'fit_source': 'full_supported_reference',
            'fit_depth_policy': 'supported_raw',
            'held_out_arbitration_used': False,
        }

        def fail(reason):
            return None, {**report, 'eligible': False, 'reason': reason}

        if (not self._supported_record_is_well_formed(source, size)
                or not self._supported_record_is_well_formed(target, size)):
            return fail('malformed_supported_endpoint')
        if (not isinstance(previous_index, (int, np.integer))
                or isinstance(previous_index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or isinstance(index, (bool, np.bool_))
                or source.frame != int(previous_index) or target.frame != int(index)
                or int(index)-int(previous_index) < 1 or int(index)-int(previous_index) > 3):
            return fail('supported_endpoint_id_mismatch')
        try:
            expected_size = tuple(size)
            live_identity = self._live_stereo_calibration_identity()
            disparity = np.asarray(self.current_disparity)
            source_geometry, geometry_index = self.previous_stereo_geometry
            source_geometry_size = tuple(source_geometry.image_size)
            fit_source_size = tuple(previous_frame.image_size)
            fit_target_size = tuple(measured_frame.image_size)
            source_geometry_pixels = np.asarray(source_geometry.pixels)
            source_geometry_descriptors = np.asarray(source_geometry.descriptors)
            fit_source_pixels = np.asarray(previous_frame.pixels)
            fit_source_descriptors = np.asarray(previous_frame.descriptors)
            fit_target_pixels = np.asarray(measured_frame.pixels)
            fit_target_descriptors = np.asarray(measured_frame.descriptors)
        except (AttributeError, TypeError, ValueError, IndexError):
            return fail('missing_live_supported_endpoint')
        if (live_identity != self.stereo_calibration_identity
                or source.calibration_identity != live_identity
                or target.calibration_identity != live_identity):
            return fail('stale_supported_calibration_identity')
        if (len(expected_size) != 2 or not all(
                    isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and v > 0
                    for v in expected_size)
                or any(value != expected_size for value in
                       (source_geometry_size, fit_source_size, fit_target_size))
                or not isinstance(geometry_index, (int, np.integer))
                or isinstance(geometry_index, (bool, np.bool_))
                or int(geometry_index) != int(previous_index)
                or disparity.ndim != 2
                or disparity.shape != (int(expected_size[1]), int(expected_size[0]))):
            return fail('supported_endpoint_domain_mismatch')
        if (not np.array_equal(source_geometry_pixels, source.pixels)
                or not np.array_equal(source_geometry_descriptors, source.descriptors)
                or not np.array_equal(fit_source_pixels, source.pixels)
                or not np.array_equal(fit_source_descriptors, source.descriptors)
                or not np.array_equal(fit_target_pixels, target.pixels)
                or not np.array_equal(fit_target_descriptors, target.descriptors)):
            return fail('supported_endpoint_pixels_or_descriptors_mismatch')
        with self.map.lock:
            poses = self.map.poses
            statuses = self.map.statuses
            accepted = [i for i, status in enumerate(statuses)
                        if status in ('tracking', 'relocalized')]
            if (len(statuses) != len(poses) or int(index) != len(poses)
                    or int(previous_index) < 0 or int(previous_index) >= len(poses)
                    or statuses[int(previous_index)] not in ('tracking', 'relocalized')
                    or not accepted or accepted[-1] != int(previous_index)):
                return fail('supported_source_not_accepted')
        try:
            pairs = np.asarray(self._match(source.descriptors, target.descriptors))
        except (cv.error, TypeError, ValueError, IndexError):
            return fail('full_supported_descriptor_match_failed')
        if (pairs.ndim != 2 or pairs.shape[1:] != (2,)
                or not np.issubdtype(pairs.dtype, np.integer)
                or np.issubdtype(pairs.dtype, np.bool_)):
            return fail('malformed_full_supported_matches')
        if len(pairs) and (np.any(pairs < 0)
                           or np.any(pairs[:, 0] >= len(source.pixels))
                           or np.any(pairs[:, 1] >= len(target.pixels))):
            return fail('full_supported_match_index_out_of_range')
        raw_count = len(pairs)
        if raw_count:
            _, pair_inverse, pair_counts = np.unique(
                pairs, axis=0, return_inverse=True, return_counts=True)
            _, source_match_inverse, source_match_counts = np.unique(
                pairs[:, 0], return_inverse=True, return_counts=True)
            _, target_match_inverse, target_match_counts = np.unique(
                pairs[:, 1], return_inverse=True, return_counts=True)
            _, source_inverse, source_counts = np.unique(
                np.asarray(source.pixels, np.float32), axis=0,
                return_inverse=True, return_counts=True)
            _, target_inverse, target_counts = np.unique(
                np.asarray(target.pixels, np.float32), axis=0,
                return_inverse=True, return_counts=True)
            source_ids, target_ids = pairs.T
            unique = (
                (pair_counts[pair_inverse] == 1)
                & (source_match_counts[source_match_inverse] == 1)
                & (target_match_counts[target_match_inverse] == 1)
                & (source_counts[source_inverse[source_ids]] == 1)
                & (target_counts[target_inverse[target_ids]] == 1)
            )
            pairs = pairs[unique].copy()
        report.update(
            raw_mutual_matches=raw_count,
            unique_physical_matches=len(pairs),
            dropped_duplicate_matches=raw_count-len(pairs),
            source_frame=int(source.frame), target_frame=int(target.frame),
            calibration_identity=live_identity,
        )
        if not len(pairs):
            return fail('no_unique_full_supported_matches')
        return pairs, {**report, 'eligible': True, 'reason': 'full_supported_pairs_ready'}

    def _estimate_full_supported_reference(
        self, source, target, previous_index, index, previous_frame, measured_frame, size,
    ):
        pairs, report = self._full_supported_match_pairs(
            source, target, previous_index, index, previous_frame, measured_frame, size)
        if pairs is None:
            return None, report
        source_frame = StereoLoopFrame(source.pixels, source.points,
                                       source.descriptors, source.image_size)
        target_frame = StereoLoopFrame(target.pixels, target.points,
                                       target.descriptors, target.image_size)
        try:
            verified = self.profile.call(
                'stereo_pose_full_supported_fallback_fit', estimate_stereo_reference,
                source_frame, target_frame, self.K,
                min_inliers=self.config.min_inliers, initial_pose=None,
                matcher=lambda first, second: pairs.copy())
        except (cv.error, np.linalg.LinAlgError, TypeError, ValueError, IndexError):
            return None, {**report, 'eligible': False,
                          'reason': 'full_supported_reference_fit_error'}
        if not isinstance(verified, dict):
            return None, {**report, 'eligible': False,
                          'reason': ('full_supported_reference_failed' if verified is None
                                     else 'malformed_full_supported_reference_result'),
                          'reverse_checked': False}
        if verified.get('reverse_checked') is not True:
            return None, {**report, 'eligible': False,
                          'reason': 'full_supported_reference_not_reverse_checked',
                          'reverse_checked': bool(verified.get('reverse_checked'))}
        if not self._is_proper_se3(verified.get('measurement')):
            return None, {**report, 'eligible': False,
                          'reason': 'full_supported_reference_invalid_pose',
                          'reverse_checked': True}
        try:
            matches = verified.get('matches')
            inliers = verified.get('inliers')
            median_error = float(verified.get('median_reprojection_px'))
            target_features = np.asarray(verified.get('target_features'))
        except (TypeError, ValueError, OverflowError):
            matches = inliers = None
            median_error = np.nan
            target_features = np.empty(0)
        if (not isinstance(matches, (int, np.integer))
                or isinstance(matches, (bool, np.bool_))
                or not isinstance(inliers, (int, np.integer))
                or isinstance(inliers, (bool, np.bool_))
                or matches <= 0 or inliers < self.config.min_inliers or inliers > matches
                or not np.isfinite(median_error) or median_error < 0
                or target_features.ndim != 1 or len(target_features) != inliers
                or not np.issubdtype(target_features.dtype, np.integer)
                or np.issubdtype(target_features.dtype, np.bool_)
                or np.any(target_features < 0)
                or np.any(target_features >= len(target.pixels))
                or len(np.unique(target_features)) != len(target_features)):
            return None, {**report, 'eligible': False,
                          'reason': 'malformed_full_supported_reference_result',
                          'reverse_checked': True}
        return verified, {**report, 'eligible': True,
                          'reason': 'full_supported_reference_verified',
                          'reverse_checked': True,
                          'reference_matches': int(matches),
                          'reference_inliers': int(inliers),
                          'reference_median_reprojection_px': median_error,
                          'reference_inlier_features': target_features.astype(int).tolist()}

    @staticmethod
    def _supported_record_is_well_formed(record, size):
        if record is None:
            return False
        try:
            pixels = np.asarray(record.pixels)
            descriptors = np.asarray(record.descriptors)
            points = np.asarray(record.points)
            right_u = np.asarray(record.right_u)
            landmark_ids = np.asarray(record.landmark_ids)
            image_size = tuple(record.image_size)
            frame = record.frame
            calibration_identity = record.calibration_identity
        except (AttributeError, TypeError, ValueError):
            return False
        try:
            n = len(pixels) if pixels.ndim else -1
            return bool(
                n > 0
                and pixels.shape == (n, 2)
                and descriptors.ndim == 2 and descriptors.shape[0] == n
                and descriptors.shape[1] > 0
                and points.shape == (n, 3)
                and right_u.shape == (n,)
                and landmark_ids.shape == (n,)
                and all(np.issubdtype(a.dtype, np.integer)
                        or np.issubdtype(a.dtype, np.floating)
                        for a in (pixels, descriptors, points, right_u))
                and np.issubdtype(landmark_ids.dtype, np.integer)
                and np.isfinite(pixels).all()
                and np.isfinite(descriptors).all()
                and not np.isinf(points).any()
                and not np.isinf(right_u).any()
                and np.all(pixels >= 0)
                and np.all(pixels < np.asarray(size, float))
                and len(image_size) == 2
                and all(isinstance(v, (int, np.integer))
                        and not isinstance(v, (bool, np.bool_)) and v > 0
                        for v in image_size)
                and tuple(image_size) == tuple(size)
                and isinstance(frame, (int, np.integer)) and not isinstance(frame, (bool, np.bool_))
                and frame >= 0
                and isinstance(calibration_identity, str) and bool(calibration_identity)
            )
        except (TypeError, ValueError, OverflowError):
            return False

    def _hard_reference_retention_guard(
        self, index, previous_index, previous_frame, measured_frame, verified,
        source_pose, reference_pose, source_revision, size,
    ):
        """Fail closed unless a reverse-verified reference is tied to live endpoints.

        Caller holds ``map.lock`` from source-pose capture through this check and
        the subsequent fixed-pose association validation.
        """
        report = {'eligible': False, 'reason': 'invalid_reference_metadata'}
        source = self.previous_supported_stereo
        current = self.current_supported_stereo
        try:
            live_identity = self._live_stereo_calibration_identity()
            q = np.asarray(self.stereo.Q, float)
            matrix = np.asarray(self.K, float)
            baseline = float(self.stereo.baseline)
            offset = float(self.stereo.disparity_offset)
            disparity = np.asarray(self.current_disparity)
            source_geometry, geometry_index = self.previous_stereo_geometry
            statuses = self.map.statuses
            poses = self.map.poses
        except (AttributeError, TypeError, ValueError, IndexError):
            return {**report, 'reason': 'missing_live_endpoint_or_calibration'}

        def fail(reason):
            return {**report, 'reason': reason}

        if (not self.config.stereo_pose_arbitration or self.stereo is None
                or not isinstance(verified, dict) or verified.get('reverse_checked') is not True):
            return fail('reverse_verification_or_feature_flag_missing')
        if not self._supported_record_is_well_formed(source, size):
            return fail('malformed_source_record')
        if not self._supported_record_is_well_formed(current, size):
            return fail('malformed_target_record')
        if (not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_))
                or not isinstance(previous_index, (int, np.integer))
                or isinstance(previous_index, (bool, np.bool_))
                or not isinstance(source_revision, (int, np.integer))
                or isinstance(source_revision, (bool, np.bool_))
                or not isinstance(self.map.revision, (int, np.integer))
                or isinstance(self.map.revision, (bool, np.bool_))):
            return fail('malformed_endpoint_or_revision_type')
        index = int(index)
        previous_index = int(previous_index)
        if (not isinstance(geometry_index, (int, np.integer))
                or isinstance(geometry_index, (bool, np.bool_))
                or int(geometry_index) != int(previous_index)
                or int(source.frame) != int(previous_index)
                or int(current.frame) != int(index)
                or int(index) != len(poses)
                or int(previous_index) < 0 or int(previous_index) >= len(poses)
                or not 1 <= int(index)-int(previous_index) <= 3):
            return fail('endpoint_or_frame_gap_mismatch')
        try:
            source_geometry_size = tuple(source_geometry.image_size)
            fit_source_size = tuple(previous_frame.image_size)
            fit_target_size = tuple(measured_frame.image_size)
        except (AttributeError, TypeError, ValueError):
            return fail('malformed_endpoint_image_size')
        if any(len(value) != 2 or not all(
                   isinstance(v, (int, np.integer))
                   and not isinstance(v, (bool, np.bool_)) and v > 0
                   for v in value) or value != tuple(size) for value in
               (source_geometry_size, fit_source_size, fit_target_size)):
            return fail('endpoint_image_size_mismatch')
        if len(statuses) != len(poses) or statuses[previous_index] not in ('tracking', 'relocalized'):
            return fail('source_frame_not_accepted')
        latest_accepted = [i for i, status in enumerate(statuses)
                           if status in ('tracking', 'relocalized')]
        if not latest_accepted or latest_accepted[-1] != int(previous_index):
            return fail('source_frame_not_latest_accepted')
        if (not np.isfinite([baseline, offset]).all() or baseline <= 0
                or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0., 0., 1.], atol=1e-8, rtol=0.)
                or q.shape != (4, 4) or not np.isfinite(q).all()
                or q[3, 2] == 0 or q[2, 3] <= 0):
            return fail('invalid_live_calibration')
        if (live_identity != self.stereo_calibration_identity
                or source.calibration_identity != live_identity
                or current.calibration_identity != live_identity):
            return fail('stale_calibration_identity')
        if (disparity.ndim != 2
                or disparity.shape != (int(size[1]), int(size[0]))):
            return fail('invalid_current_disparity_shape')
        # Shape/content checks bind the immutable raw records to the exact frames
        # used for reference fitting, while allowing map-fit depth restoration.
        try:
            geometry_pixels = np.asarray(source_geometry.pixels)
            geometry_descriptors = np.asarray(source_geometry.descriptors)
            fit_source_pixels = np.asarray(previous_frame.pixels)
            fit_source_descriptors = np.asarray(previous_frame.descriptors)
            fit_target_pixels = np.asarray(measured_frame.pixels)
            fit_target_descriptors = np.asarray(measured_frame.descriptors)
        except (AttributeError, TypeError, ValueError):
            return fail('malformed_endpoint_pixels_or_descriptors')
        if (geometry_pixels.shape != source.pixels.shape
                or geometry_descriptors.shape != source.descriptors.shape
                or fit_source_pixels.shape != source.pixels.shape
                or fit_source_descriptors.shape != source.descriptors.shape
                or fit_target_pixels.shape != current.pixels.shape
                or fit_target_descriptors.shape != current.descriptors.shape
                or not np.array_equal(geometry_pixels, source.pixels)
                or not np.array_equal(geometry_descriptors, source.descriptors)
                or not np.array_equal(fit_source_pixels, source.pixels)
                or not np.array_equal(fit_source_descriptors, source.descriptors)
                or not np.array_equal(fit_target_pixels, current.pixels)
                or not np.array_equal(fit_target_descriptors, current.descriptors)):
            return fail('endpoint_pixels_or_descriptors_mismatch')
        if (not self._is_proper_se3(source_pose)
                or not self._is_proper_se3(verified.get('measurement'))
                or not self._is_proper_se3(reference_pose)):
            return fail('invalid_se3_pose')
        live_source_pose = np.asarray(poses[previous_index], float)
        if (not self._is_proper_se3(live_source_pose)
                or not np.array_equal(live_source_pose, np.asarray(source_pose, float))):
            return fail('source_pose_epoch_changed')
        expected_pose = np.asarray(source_pose, float) @ np.asarray(verified['measurement'], float)
        if not np.allclose(expected_pose, reference_pose, atol=1e-10, rtol=1e-10):
            return fail('reference_pose_composition_mismatch')
        if source_revision != self.map.revision:
            return fail('map_revision_changed')
        return {
            'eligible': True, 'reason': 'validated_reverse_reference_endpoints',
            'source_frame': int(source.frame), 'target_frame': int(current.frame),
            'gap': int(index)-int(previous_index),
            'calibration_identity': live_identity,
            'source_map_revision': int(source_revision),
            'reference_reverse_checked': True,
            'current_disparity_shape': list(disparity.shape),
        }

    def _age_provisional_map_inliers(self, previous_inlier_misses):
        """Age a successful map hypothesis once when a hard stereo conflict rejects it."""
        if previous_inlier_misses is None:
            return
        with self.map.lock:
            for landmark_id, previous_misses in previous_inlier_misses.items():
                landmark = self.map.landmarks.get(landmark_id)
                if landmark is not None:
                    landmark.misses = int(previous_misses) + 1

    def _supported_stereo_after_acceptance(self, record, associations):
        linked = np.full(len(record.pixels), -1, int)
        last = self.map.keyframes.get(self.last_keyframe)
        if last is not None and last.frame == record.frame:
            linked[:] = last.landmark_ids[:len(linked)]
        else:
            for feature, lid in associations.items():
                if 0 <= feature < len(linked):
                    linked[feature] = lid
        return SupportedStereoFrame(record.pixels, record.descriptors, record.points,
                                    record.right_u, linked, record.frame, record.image_size,
                                    record.calibration_identity)

    def process(self, index, image, right=None):
        if index != len(self.map.poses):
            raise ValueError(
                "Frames must arrive in consecutive order, starting at zero"
            )
        self.loop_worker.poll(self.map)
        self._prepare_frame_images(image, right)
        if self.config.stereo_pose_arbitration:
            self._supported_extraction = None
            self._arbitration_context = None
        pixels, desc, points, right_u = self._extract(image, right)
        size = (image.shape[1], image.shape[0])
        arbitration_report = None
        arbitration_measurement = None
        if self.config.stereo_pose_arbitration:
            self.current_supported_stereo = self._capture_supported_stereo(index, pixels, desc, size)
            self._arbitration_context, arbitration_report = self._prepare_stereo_arbitration(
                index, self.current_supported_stereo)
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
        associations = {}
        map_inlier_miss_snapshot = None
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
                pairs = self._match(initial_desc, desc)
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
            map_inlier_miss_snapshot = self._last_map_track_inlier_misses
            info.update(stats)
            recovered = False
            stereo_reference = False
            full_supported_fallback_attempted = False
            full_supported_fallback_report = None
            arbitration_selected = False
            pre_full_reserved_arbitration = None
            reserved_arbitration_attempted = False
            if self.previous_stereo_geometry is not None:
                previous_frame, previous_index = self.previous_stereo_geometry
                if index - previous_index <= 3:
                    measured = StereoLoopFrame(pixels, points, desc, size)
                    arbitration = self._arbitration_context
                    reference_matcher = self._match
                    prior = (
                        np.linalg.inv(self.map.poses[previous_index])
                        @ self._motion_prediction()
                    )
                    if arbitration is not None:
                        raw_previous = arbitration['previous']
                        raw_current = arbitration['current']
                        previous_frame = StereoLoopFrame(raw_previous.pixels, raw_previous.points,
                                                         raw_previous.descriptors, raw_previous.image_size)
                        measured = StereoLoopFrame(raw_current.pixels, raw_current.points,
                                                   raw_current.descriptors, raw_current.image_size)
                        # All forward/reverse/refinement rows derive from this
                        # pre-reserved fitting pool. No holdout row enters PnP.
                        reference_matcher = lambda first, second: arbitration['fit'].copy()
                        prior = None
                    verified = (arbitration['verified'] if arbitration is not None else estimate_stereo_reference(
                        previous_frame,
                        measured,
                        self.K,
                        min_inliers=self.config.min_inliers,
                        initial_pose=prior,
                        matcher=reference_matcher,
                    ))
                    if verified is not None:
                        with self.map.lock:
                            source_pose = self.map.poses[previous_index].copy()
                            source_map_revision = self.map.revision
                            reference_pose = source_pose @ verified["measurement"]
                        conflict = False
                        half_conflict = False
                        half_translation_error = half_rotation_error = None
                        if result is not None:
                            difference = np.linalg.inv(reference_pose) @ result[0]
                            half_translation_error = float(np.linalg.norm(difference[:3, 3]))
                            half_rotation_error = float(
                                np.degrees(
                                    np.linalg.norm(cv.Rodrigues(difference[:3, :3])[0])
                                )
                            )
                            # Use the same disagreement limits as bidirectional
                            # stereo verification. A low left reprojection error
                            # cannot justify contradicting independent metric motion.
                            half_conflict = (half_translation_error > 0.5
                                             or half_rotation_error > 1.5)
                            conflict = half_conflict

                        # A valid reserved half-pool fit can still be biased by its
                        # partition. When it hard-conflicts with a successful map
                        # solve, refit once on every unique physical supported row
                        # before installing motion, BA, or held-out diagnostics.
                        if (arbitration is not None and result is not None
                                and verified.get('reverse_checked') is True and half_conflict):
                            # Score the immutable reserved observations before
                            # consuming them in the full-pool retry. Only a
                            # provenance-valid strict held-out win can bypass
                            # that retry; every other outcome follows the
                            # established full-pool path below.
                            with self.map.lock:
                                source_guard = self._hard_reference_retention_guard(
                                    index, previous_index, previous_frame, measured,
                                    verified, source_pose, reference_pose,
                                    source_map_revision, size)
                                source_epoch_current = bool(source_guard['eligible'])
                                if source_epoch_current:
                                    pre_full_reserved_arbitration = (
                                        self._arbitrate_supported_pose(
                                            arbitration, result[0], verified['measurement']))
                                    reserved_arbitration_attempted = True
                                    source_guard = self._hard_reference_retention_guard(
                                        index, previous_index, previous_frame, measured,
                                        verified, source_pose, reference_pose,
                                        source_map_revision, size)
                                    source_epoch_current = bool(source_guard['eligible'])
                            if pre_full_reserved_arbitration is None:
                                pre_full_reserved_arbitration = {
                                    'choice': 'abstain',
                                    'reason': 'reserved_source_guard_failed_before_score',
                                    'source_guard': source_guard,
                                }
                            pre_full_reserved_arbitration = {
                                **pre_full_reserved_arbitration,
                                'fit_source': 'reserved_supported_training_rows',
                                'held_out_arbitration_used': reserved_arbitration_attempted,
                                'reservation_context_available': True,
                            }
                            if reserved_arbitration_attempted and not source_epoch_current:
                                pre_full_reserved_arbitration = {
                                    **pre_full_reserved_arbitration,
                                    'choice': 'abstain',
                                    'reason': 'reserved_source_guard_failed_after_score',
                                    'source_guard': source_guard,
                                }
                            arbitration_report = pre_full_reserved_arbitration
                            arbitration_selected = (
                                source_epoch_current
                                and arbitration_report.get('choice') == 'independent')
                        if (arbitration is not None and result is not None
                                and verified.get('reverse_checked') is True and half_conflict
                                and not arbitration_selected):
                            full_supported_fallback_attempted = True
                            # The reserved holdout was consumed by the full-pool
                            # retry. Prevent any later recovery probe in this
                            # process call from reusing its exclusions or score.
                            self._arbitration_context = None
                            full_verified, full_report = self._estimate_full_supported_reference(
                                arbitration['previous'], arbitration['current'],
                                previous_index, index, previous_frame, measured, size)
                            if full_verified is not None:
                                with self.map.lock:
                                    same_source_epoch = (
                                        self.map.revision == source_map_revision
                                        and np.array_equal(self.map.poses[previous_index], source_pose))
                                if not same_source_epoch:
                                    full_verified = None
                                    full_report = {**full_report, 'eligible': False,
                                                   'reason': 'source_map_epoch_changed'}
                            full_supported_fallback_report = {
                                **full_report,
                                'half_pool_conflicted_with_map': True,
                                'half_pool_translation_error_m': half_translation_error,
                                'half_pool_rotation_error_deg': half_rotation_error,
                                'reservation_context_available': True,
                                'held_out_arbitration_used': False,
                                'pre_full_reserved_arbitration': pre_full_reserved_arbitration,
                            }
                            arbitration_measurement = None
                            if full_verified is None:
                                # Never fall back to the rejected half fit as a pose
                                # or BA edge. Restore the successful map solve's
                                # original inlier miss counts exactly once, then let
                                # normal geometric recovery/loss proceed.
                                self._age_provisional_map_inliers(map_inlier_miss_snapshot)
                                result = None
                                associations = {}
                                self.accepted_tracks = []
                                verified = None
                                info['tracking_ok'] = False
                                arbitration_report = {
                                    'choice': 'abstain',
                                    'reason': 'full_supported_reference_failed',
                                    'reservation_context_available': True,
                                    'held_out_arbitration_used': False,
                                    'fit_source': 'full_supported_reference',
                                    'fit_depth_policy': 'supported_raw',
                                    'full_supported_fallback': full_supported_fallback_report,
                                }
                                info['full_supported_reference_fallback'] = full_supported_fallback_report
                            else:
                                verified = full_verified
                                reference_pose = source_pose @ verified['measurement']
                                difference = np.linalg.inv(reference_pose) @ result[0]
                                translation_error = float(np.linalg.norm(difference[:3, 3]))
                                rotation_error = float(np.degrees(
                                    np.linalg.norm(cv.Rodrigues(difference[:3, :3])[0])))
                                conflict = translation_error > 0.5 or rotation_error > 1.5
                                arbitration_report = {
                                    'choice': 'existing_reference' if conflict else 'map',
                                    'reason': ('full_supported_reference_still_conflicts'
                                               if conflict else
                                               'full_supported_reference_agrees_with_map'),
                                    'reservation_context_available': True,
                                    'held_out_arbitration_used': False,
                                    'fit_source': 'full_supported_reference',
                                    'fit_depth_policy': 'supported_raw',
                                    'full_supported_fallback': full_supported_fallback_report,
                                }
                                info['full_supported_reference_fallback'] = full_supported_fallback_report
                                info.update(
                                    map_reference_translation_error_m=translation_error,
                                    map_reference_rotation_error_deg=rotation_error,
                                )

                        if verified is not None:
                            if (arbitration is not None and result is not None
                                    and not full_supported_fallback_attempted):
                                arbitration_measurement = verified['measurement'].copy()
                            if verified['reverse_checked'] and index-previous_index == 1:
                                self.verified_stereo_motion = (verified['measurement'].copy(), index)
                            if verified['reverse_checked']:
                                verified_motion = (previous_index, verified['measurement'].copy())
                            info['stereo_reference_verified'] = True
                            info['stereo_bidirectional_refinement'] = verified.get('bidirectional_refinement')
                            info['stereo_reference_verification'] = (
                                'bidirectional_pnp' if verified['reverse_checked']
                                else 'source_depth_pnp')
                            if result is not None and not full_supported_fallback_attempted:
                                translation_error = half_translation_error
                                rotation_error = half_rotation_error
                                info.update(
                                    map_reference_translation_error_m=translation_error,
                                    map_reference_rotation_error_deg=rotation_error,
                                )
                            if (arbitration is not None and result is not None
                                    and not full_supported_fallback_attempted
                                    and verified['reverse_checked'] and not conflict):
                                arbitration_report = self._arbitrate_supported_pose(
                                    arbitration, result[0], verified['measurement'])
                                arbitration_report = {
                                    **arbitration_report,
                                    'fit_source': 'reserved_supported_training_rows',
                                    'held_out_arbitration_used': True,
                                    'reservation_context_available': True,
                                }
                                arbitration_selected = arbitration_report['choice'] == 'independent'
                        if verified is not None and (result is None or conflict or arbitration_selected):
                            if full_supported_fallback_attempted:
                                arbitration_report = {
                                    **arbitration_report,
                                    'choice': 'existing_reference' if conflict else 'map',
                                    'reason': ('full_supported_reference_still_conflicts'
                                               if conflict else
                                               'full_supported_reference_selected'),
                                }
                            elif arbitration is not None and not arbitration_selected:
                                arbitration_report = {**arbitration['report'], 'choice': 'existing_reference',
                                                      'reason': ('hard_disagreement_fallback' if conflict
                                                                 else 'missing_map_hypothesis')}
                            elif (conflict and arbitration_report is not None
                                  and not arbitration_selected):
                                arbitration_report = {**arbitration_report,
                                                      'choice': 'existing_reference',
                                                      'reason': 'hard_disagreement_fallback'}
                            if arbitration_selected:
                                # `associations` is assigned from `result` only
                                # after this arbitration block. Snapshot the map
                                # solve output before replacing its pose.
                                map_associations = dict(result[1])
                                map_tracks = list(self.accepted_tracks)
                                associations, retained_tracks, association_validation = (
                                    self._validate_stereo_associations_at_pose(
                                        reference_pose, map_associations, map_tracks,
                                        pixels, size, info.get('valid_3d', 0),
                                        map_inlier_miss_snapshot))
                                self.accepted_tracks = retained_tracks
                                arbitration_report = {
                                    **arbitration_report,
                                    'association_validation': association_validation,
                                }
                                if conflict:
                                    info['reference_association_validation'] = association_validation
                                if not association_validation['eligible']:
                                    # With no connected map support, preserve the
                                    # established reference-keyframe recovery path.
                                    stereo_reference = True
                            else:
                                reference_validation = None
                                retained = False
                                if conflict and self.config.stereo_pose_arbitration and result is not None:
                                    # A reverse-verified hard-conflict candidate can
                                    # reconnect to its old map only when the exact raw
                                    # endpoints, live calibration, and composed pose
                                    # still match the accepted reference transaction.
                                    with self.map.lock:
                                        reference_guard = self._hard_reference_retention_guard(
                                            index, previous_index, previous_frame, measured,
                                            verified, source_pose, reference_pose,
                                            source_map_revision, size)
                                        if reference_guard['eligible']:
                                            map_associations = dict(result[1])
                                            map_tracks = list(self.accepted_tracks)
                                            associations, retained_tracks, connection = (
                                                self._validate_stereo_associations_at_pose(
                                                    reference_pose, map_associations, map_tracks,
                                                    pixels, size, info.get('valid_3d', 0),
                                                    map_inlier_miss_snapshot))
                                            self.accepted_tracks = retained_tracks
                                            retained = bool(connection['eligible'])
                                        else:
                                            connection = None
                                    reservation_context = arbitration is not None
                                    try:
                                        live_calibration_identity = self._live_stereo_calibration_identity()
                                    except (AttributeError, TypeError, ValueError):
                                        live_calibration_identity = None
                                    reference_validation = {
                                        **reference_guard,
                                        'eligible': bool(reference_guard['eligible'] and retained),
                                        'reason': ('retained' if retained else
                                                   connection['reason'] if connection is not None
                                                   else reference_guard['reason']),
                                        'source_frame': int(previous_index),
                                        'target_frame': int(index),
                                        'gap': int(index)-int(previous_index),
                                        'live_calibration_identity': live_calibration_identity,
                                        'calibration_identity': reference_guard.get(
                                            'calibration_identity', live_calibration_identity),
                                        'cached_calibration_identity': self.stereo_calibration_identity,
                                        'source_calibration_identity': getattr(
                                            self.previous_supported_stereo, 'calibration_identity', None),
                                        'record_calibration_identity': getattr(
                                            self.current_supported_stereo, 'calibration_identity', None),
                                        'reference_reverse_checked': bool(verified.get('reverse_checked')),
                                        'reservation_context_available': reservation_context,
                                        'held_out_arbitration_used': False,
                                        'independent_fit_source': (
                                            'full_supported_reference' if full_supported_fallback_attempted
                                            else 'reserved_supported_training_rows' if reservation_context
                                            else 'map_coordinate_independent_stereo_reference'),
                                        'independent_fit_depth_policy': (
                                            'supported_raw' if (reservation_context
                                                                or full_supported_fallback_attempted)
                                            else self.config.stereo_depth_policy),
                                        'fit_depth_policy': (
                                            'supported_raw' if (reservation_context
                                                                or full_supported_fallback_attempted)
                                            else self.config.stereo_depth_policy),
                                        'map_fit_depth_policy': self.config.stereo_depth_policy,
                                        'prediction_seed_supplied': (
                                            False if full_supported_fallback_attempted
                                            else not reservation_context),
                                        'current_right_measurement': (
                                            'supported_only_raw_disparity_at_actual_observation'
                                            if connection is not None else 'not_remeasured_guard_rejected'),
                                        'original_final_solve_positions': int(info.get('valid_3d', 0)),
                                        'association_validation': connection,
                                    }
                                    if not reference_validation['eligible']:
                                        if connection is None:
                                            # The validator did not run, so reconcile
                                            # the provisional _track reset exactly once.
                                            self._age_provisional_map_inliers(
                                                map_inlier_miss_snapshot)
                                        associations = {}
                                        self.accepted_tracks = []
                                        stereo_reference = True
                                else:
                                    if conflict:
                                        # No immutable reverse-verified endpoint
                                        # proof: retain the established forced-KF path.
                                        self._age_provisional_map_inliers(
                                            map_inlier_miss_snapshot)
                                    associations = {}
                                    self.accepted_tracks = []
                                    stereo_reference = True
                                if reference_validation is not None:
                                    info['reference_association_validation'] = reference_validation
                                    if arbitration_report is not None:
                                        arbitration_report['reference_association_validation'] = reference_validation
                            result = (reference_pose, associations)
                            info.update(
                                num_matches=verified["matches"],
                                num_inliers=verified["inliers"],
                                inlier_ratio=verified["inliers"] / verified["matches"],
                                reprojection_error=verified["median_reprojection_px"],
                                pose_source="stereo_tracking_reference",
                                reference_inlier_features=verified["target_features"],
                                map_pose_rejected_for_stereo_conflict=conflict,
                            )
                            if arbitration_selected:
                                info['pose_source'] = 'reserved_stereo_arbitration'
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
        if arbitration_report is not None:
            info['stereo_pose_arbitration'] = arbitration_report
        if arbitration_measurement is not None and info['tracking_ok']:
            # This diagnostic does not withhold rows from later BA or protect
            # against subsequent corrections. Sample the two poses atomically.
            with self.map.lock:
                before_bundle = self._arbitrate_supported_pose(
                    self._arbitration_context, self.map.poses[-1], arbitration_measurement)
            info['stereo_pose_arbitration_before_bundle'] = before_bundle['map']
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
                    optimized=self.performance.cpu_optimizations,
                ) if self.config.bundle_enabled else {"applied": False, "reason": "diagnostic_ablation"}
            self.bundle_reports.append({"frame": index, **report})
            pose = self.map.poses[-1].copy()
            self.loop_worker.schedule(self.map)
        if arbitration_measurement is not None and info['tracking_ok']:
            with self.map.lock:
                after_bundle = self._arbitrate_supported_pose(
                    self._arbitration_context, self.map.poses[-1], arbitration_measurement)
            info['stereo_pose_arbitration_after_bundle'] = after_bundle['map']
        if info["tracking_ok"] and self.stereo is not None:
            self.previous_stereo_geometry = (
                StereoLoopFrame(pixels.copy(), points.copy(), desc.copy(), size),
                index,
            )
            if self.config.stereo_pose_arbitration:
                self.previous_supported_stereo = self._supported_stereo_after_acceptance(
                    self.current_supported_stereo, associations)
        # Bound the tracking working set, not the persistent map. Old, valid
        # landmarks remain available to geometric relocalization and loop correction.
        dropped = {i:l for i,l in self.map.landmarks.items() if l.misses >= 5}
        self.map.landmarks = {
            i: l for i, l in self.map.landmarks.items() if l.misses < 5
        }
        removed = False
        affected = ({k for l in dropped.values() for k in l.observations}
                    if self.performance.cpu_optimizations else set(self.map.keyframes))
        for keyframe in (self.map.keyframes[k] for k in sorted(affected) if k in self.map.keyframes):
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
        if self.config.stereo_depth_policy == 'verified_fallback':
            info['stereo_depth_verification'] = dict(self.stereo_depth_verification)
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
