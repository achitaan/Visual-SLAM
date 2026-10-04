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
from stereo_training_factors import EndpointPose, build_stereo_training_factors

MAX_SIFT_FEATURES = 10_000


def validate_feature_budget(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 1 <= int(value) <= MAX_SIFT_FEATURES):
        raise ValueError(f"features must be an integer between 1 and {MAX_SIFT_FEATURES}")
    return int(value)


def _owned_diagnostic_value(value):
    """Copy estimator values into finite JSON primitives with no live array views."""
    if isinstance(value, np.ndarray):
        return _owned_diagnostic_value(value.tolist())
    if isinstance(value, np.generic):
        return _owned_diagnostic_value(value.item())
    if isinstance(value, dict):
        return {str(key): _owned_diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_owned_diagnostic_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Non-finite value in bundle diagnostic snapshot")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _reserved_training_selected_as_edge(
    pose_source, arbitration_choice, final_source_frame,
    reserved_source_frame, final_measurement, reserved_measurement,
    consumed_full_pool,
):
    """Require source-role selection as well as equal measurement provenance."""
    if consumed_full_pool:
        return False
    role_selected = (
        pose_source == "reserved_stereo_arbitration"
        and arbitration_choice == "independent"
    ) or (
        pose_source == "stereo_tracking_reference"
        and arbitration_choice == "existing_reference"
    )
    if not role_selected or final_measurement is None or reserved_measurement is None:
        return False
    try:
        return bool(
            int(final_source_frame) == int(reserved_source_frame)
            and np.array_equal(np.asarray(final_measurement),
                               np.asarray(reserved_measurement))
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _filter_owned_training_edges(forward, reverse, target_claims, source_rows,
                                 reusable_landmark_ids):
    """Fail closed on ambiguous or unselected actual map Observation owners."""
    forward = np.asarray(forward, dtype=np.int64).reshape(-1, 2)
    reverse = np.asarray(reverse, dtype=np.int64).reshape(-1, 2)
    reusable = {int(value) for value in reusable_landmark_ids}
    allowed = {"unowned", "reusable_existing_selected_point", "existing_single_view_target"}
    safe, rejected = set(), 0
    for i, j in sorted({tuple(map(int, pair)) for pair in np.vstack((forward, reverse))}):
        target = target_claims.get(str(j), {"classification": "unowned"})
        source = source_rows.get(i, {"classification": "unowned"})
        if (target.get("classification", "ambiguous_or_measurement_mismatch") not in allowed
                or source.get("classification", "unowned") in (
                    "ambiguous_or_measurement_mismatch", "existing_owned_not_selected")):
            rejected += 1
            continue
        if source.get("classification") == "exact_source_observation":
            exact_ids = {
                int(claim["landmark_id"])
                for claim in source.get("authoritative_observations", [])
                if claim.get("existing_observation_certificate") is not None
            }
            if exact_ids - reusable:
                rejected += 1
                continue
        if target.get("classification") in (
                "reusable_existing_selected_point", "existing_single_view_target"):
            exact_ids = {
                int(claim["landmark_id"])
                for claim in target.get("claims", [])
                if claim.get("existing_observation_certificate") is not None
                and claim.get("landmark_id") is not None
            }
            if not exact_ids or not exact_ids.issubset(reusable):
                rejected += 1
                continue
        safe.add((i, j))
    fwd = np.asarray([pair for pair in forward if tuple(map(int, pair)) in safe], dtype=np.int64).reshape(-1, 2)
    rev = np.asarray([pair for pair in reverse if tuple(map(int, pair)) in safe], dtype=np.int64).reshape(-1, 2)
    return fwd, rev, rejected


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
    stereo_raw_reference_retry: bool = False
    stereo_owned_image_bundle: bool = False
    stereo_source_history_bundle: bool = False
    stereo_retained_source_observations: bool = False
    stereo_mapping_observation_retention: bool = False
    stereo_physical_match_pool: bool = False
    bundle_solver_accuracy: str = "default"
    stereo_bundle_gauge_mode: str = "veto"

    def __post_init__(self):
        object.__setattr__(self, "features", validate_feature_budget(self.features))
        if self.bundle_solver_accuracy not in ("default", "precise"):
            raise ValueError("bundle_solver_accuracy must be 'default' or 'precise'")
        if self.stereo_bundle_gauge_mode not in ("veto", "canonical_two_bridge"):
            raise ValueError("Invalid stereo bundle gauge mode")
        if self.stereo_bundle_gauge_mode != "veto" and (
                self.stereo_owned_image_bundle or self.stereo_source_history_bundle
                or self.stereo_retained_source_observations):
            raise ValueError("Gauge canonicalization requires the original stereo image model")
        if self.stereo_source_history_bundle and not self.stereo_owned_image_bundle:
            raise ValueError(
                "stereo_source_history_bundle requires stereo_owned_image_bundle"
            )
        if self.stereo_retained_source_observations and not (
                self.stereo_owned_image_bundle and self.stereo_source_history_bundle):
            raise ValueError(
                "stereo_retained_source_observations requires owned image and source-history bundles"
            )
        if self.stereo_mapping_observation_retention and not (
                self.stereo_pose_arbitration or self.stereo_raw_reference_retry):
            raise ValueError(
                "stereo_mapping_observation_retention requires an independent stereo reference path"
            )
        if self.stereo_physical_match_pool and not self.stereo_pose_arbitration:
            raise ValueError("stereo_physical_match_pool requires stereo_pose_arbitration")


class SharedSlam:
    def __init__(self, matrix, stereo=None, config=None, performance=None,
                 bundle_diagnostic_writer=None, tracking_diagnostics_writer=None):
        self.performance = performance or PerformanceConfig()
        self.bundle_diagnostic_writer = bundle_diagnostic_writer
        self.bundle_diagnostic_errors = []
        self.tracking_diagnostics_writer = tracking_diagnostics_writer
        self.tracking_diagnostic_errors = []
        self._tracking_trace_probe_label = None
        self._tracking_trace_cohort_frame = None
        self.matcher = DescriptorMatcher(self.performance.matching_backend)
        self._landmark_cache = None
        self._identity_conflict_cache = {}
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
        validate_feature_budget(self.config.features)
        if self.config.bundle_solver_accuracy not in ("default", "precise"):
            raise ValueError("bundle_solver_accuracy must be 'default' or 'precise'")
        if (getattr(self.config, "stereo_bundle_gauge_mode", "veto") != "veto"
                and stereo is None):
            raise ValueError("Gauge canonicalization requires calibrated stereo")
        if self.config.stereo_depth_policy not in ('supported', 'verified_fallback', 'verified_all'):
            raise ValueError('Invalid stereo depth policy')
        if stereo is None and self.config.stereo_depth_policy != 'supported':
            raise ValueError('Verified stereo depth requires two calibrated cameras')
        if stereo is None and self.config.stereo_pose_arbitration:
            raise ValueError('Stereo pose arbitration requires two calibrated cameras')
        if getattr(self.config, "stereo_physical_match_pool", False) and (
                stereo is None or not self.config.stereo_pose_arbitration):
            raise ValueError(
                'Physical stereo match pool requires calibrated stereo and pose arbitration'
            )
        if stereo is None and self.config.stereo_raw_reference_retry:
            raise ValueError('Raw stereo reference retry requires two calibrated cameras')
        if self.config.stereo_owned_image_bundle and (
                stereo is None or not self.config.stereo_pose_arbitration):
            raise ValueError('Owned stereo bundle requires calibrated stereo and pose arbitration')
        if getattr(self.config, "stereo_source_history_bundle", False) and (
                stereo is None or not self.config.stereo_owned_image_bundle):
            raise ValueError(
                'Source-history bundle requires calibrated stereo and owned image bundle'
            )
        if getattr(self.config, "stereo_retained_source_observations", False) and (
                stereo is None or not self.config.stereo_owned_image_bundle
                or not self.config.stereo_source_history_bundle):
            raise ValueError(
                'Retained source observations require calibrated stereo, owned image bundle, '
                'and source-history bundle'
            )
        if getattr(self.config, "stereo_mapping_observation_retention", False) and (
                stereo is None or not (self.config.stereo_pose_arbitration
                                       or self.config.stereo_raw_reference_retry)):
            raise ValueError(
                'Mapping observation retention requires calibrated stereo and an '
                'independent stereo reference path'
            )
        if (self.config.stereo_owned_image_bundle
                and self.config.stereo_depth_policy == 'verified_all'):
            raise ValueError('Owned stereo bundle does not accept verified_all measurement provenance')
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
        self._previous_tracks_frame = None
        self._source_history_capture = None
        self._source_history_diagnostic_cohort = None
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

    def _depth_fit_policy_label(self):
        policy = getattr(getattr(self, 'config', None), 'stereo_depth_policy', 'supported')
        return ('verified_all' if policy == 'verified_all'
                else 'supported_raw')

    def _depth_measurement_source_label(self):
        policy = getattr(getattr(self, 'config', None), 'stereo_depth_policy', 'supported')
        return ('verified_right_image_correspondence'
                if policy == 'verified_all'
                else 'raw_supported_stereo_disparity')

    def _measure_stereo_pixels(self, pixels, supported_only=False, capture_supported=False):
        """Sample metric depth at the actual left observations, including flow tracks."""
        points = np.full((len(pixels), 3), np.nan)
        right_u = np.full(len(pixels), np.nan)
        if self.stereo is None or not len(pixels):
            return points, right_u
        finite = np.isfinite(pixels).all(axis=1)
        if self.config.stereo_depth_policy == 'verified_all':
            if self.current_left_gray is None or self.current_right_gray is None:
                self.stereo_depth_verification['missing_images'] = (
                    self.stereo_depth_verification.get('missing_images', 0) + int(len(pixels)))
                if capture_supported:
                    self._supported_extraction = (points.copy(), right_u.copy())
                return points, right_u
            if (np.ndim(self.current_left_gray) != 2
                    or np.ndim(self.current_right_gray) != 2
                    or np.shape(self.current_left_gray) != np.shape(self.current_right_gray)):
                self.stereo_depth_verification['invalid_images'] = (
                    self.stereo_depth_verification.get('invalid_images', 0) + int(len(pixels)))
                if capture_supported:
                    self._supported_extraction = (points.copy(), right_u.copy())
                return points, right_u
            height, width = self.current_left_gray.shape
            inside = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] <= width-1)
            inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] <= height-1)
            ids = np.flatnonzero(inside)
            raw_q = np.asarray(self.stereo.Q)
            if (np.iscomplexobj(raw_q) or raw_q.shape != (4, 4)
                    or not np.issubdtype(raw_q.dtype, np.number)
                    or not np.isfinite(raw_q).all()):
                self.stereo_depth_verification['invalid_calibration'] = (
                    self.stereo_depth_verification.get('invalid_calibration', 0) + int(len(ids)))
                if capture_supported:
                    self._supported_extraction = (points.copy(), right_u.copy())
                return points, right_u
            q = raw_q.astype(float, copy=True)
            if (q[3, 2] <= 0 or q[2, 3] <= 0
                    or not np.isfinite(self.inverse_K).all()):
                self.stereo_depth_verification['invalid_calibration'] = (
                    self.stereo_depth_verification.get('invalid_calibration', 0) + int(len(ids)))
                if capture_supported:
                    self._supported_extraction = (points.copy(), right_u.copy())
                return points, right_u
            measured = np.full(len(ids), np.nan)
            counters = {'candidates': len(ids), 'verified': 0}
            if len(ids) and self.current_left_gray is not None and self.current_right_gray is not None:
                q32, q23 = float(q[3, 2]), float(q[2, 3])
                if np.isfinite([q32, q23, q[3, 3]]).all() and q32 > 0 and q23 > 0:
                    bf, offset = q23/q32, -float(q[3, 3])/q32
                    bounds = (max(0., offset+bf/100.), min(96., offset+bf/.1))
                    if bounds[0] < bounds[1]:
                        measured, counters = self.profile.call(
                            'stereo_depth_verification', verify_stereo_depth_candidates,
                            self.current_left_gray, self.current_right_gray,
                            np.asarray(pixels, float)[ids], self.stereo_search_config, bounds)
            for key, value in counters.items():
                self.stereo_depth_verification[key] = self.stereo_depth_verification.get(key, 0)+int(value)
            disparity = np.asarray(pixels, float)[ids, 0] - measured
            denominator = q[3, 2]*disparity + q[3, 3]
            depth = np.divide(q[2, 3], denominator,
                              out=np.full(len(ids), np.nan), where=denominator > 0)
            valid = np.isfinite(measured) & (disparity > 0.) & (disparity < 96.)
            valid &= np.isfinite(depth) & (depth > .1) & (depth < 100.)
            accepted = ids[valid]
            rays = np.c_[np.asarray(pixels, float)[accepted], np.ones(len(accepted))] @ self.inverse_K.T
            points[accepted] = rays * depth[valid, None]
            right_u[accepted] = measured[valid]
            if capture_supported:
                self._supported_extraction = (points.copy(), right_u.copy())
            return points, right_u
        if self.current_disparity is None:
            return points, right_u
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
        right_gray = None
        if (self.stereo is not None and right is not None
                and self.config.stereo_depth_policy == 'verified_all'):
            right_gray = cv.cvtColor(right, cv.COLOR_BGR2GRAY) if right.ndim == 3 else right
            self.current_left_gray = np.asarray(gray, np.float32)
            self.current_right_gray = np.asarray(right_gray, np.float32)
        keypoints, desc = self.detector.detectAndCompute(gray, None)
        pixels = np.array([k.pt for k in keypoints], np.float32).reshape(-1, 2)
        desc = desc if desc is not None else np.empty((0, 128), np.float32)
        points = np.full((len(pixels), 3), np.nan)
        right_u = np.full(len(pixels), np.nan)
        if self.stereo is not None:
            if right is None or right.shape[:2] != image.shape[:2]:
                raise ValueError("Stereo requires matching right image")
            if right_gray is None:
                right_gray = (
                    cv.cvtColor(right, cv.COLOR_BGR2GRAY) if right.ndim == 3 else right
                )
            if self.config.stereo_depth_policy == 'verified_all':
                self.current_left_gray = np.asarray(gray, np.float32)
                self.current_right_gray = np.asarray(right_gray, np.float32)
            disparity = (
                self.stereo.stereo.compute(gray, right_gray).astype(np.float32) / 16.0
            )
            self.current_disparity = disparity
            points, right_u = self._measure_stereo_pixels(
                pixels, capture_supported=(self.config.stereo_pose_arbitration
                                           or self.config.stereo_raw_reference_retry))
        return pixels, desc, points, right_u

    def _keyframe(self, index, pose, pixels, desc, points, right_u, associations):
        if (getattr(self, "_tracking_trace_frame", -1) == index
                and self._tracking_trace_enabled(index)
                and not getattr(self, "_tracking_trace_selected_emitted", False)):
            self._tracking_trace_selected(index, "selected_pre_keyframe", pose,
                "pending_acceptance", associations, pixels, right_u, None, {})
        pixels, desc, points, right_u = (
            pixels.copy(),
            desc.copy(),
            points.copy(),
            right_u.copy(),
        )
        # A valid tracked landmark need not coincide with a freshly detected SIFT keypoint.
        # Preserve its actual image observation rather than dropping it from bundle adjustment.
        detected_features = len(desc)
        association_output = associations
        associations = {int(feature): int(lid) for feature, lid in associations.items()
                        if int(lid) in self.map.landmarks
                        and 0 <= int(feature) < len(pixels)}
        tracks = self._unique_physical_tracks(self.accepted_tracks)

        # A landmark can only have one physical observation in a keyframe. If
        # malformed or legacy associations connect it to multiple detector
        # pixels, keep an exact accepted-flow pixel when it identifies one row;
        # otherwise preserve the flow observation as its own row below.
        tracked_pixels = {int(lid): np.asarray(pixel).reshape(2) for lid, pixel in tracks}
        groups, group_ids = self._physical_pixel_groups(pixels)
        physical_group_lookup = {
            self._physical_pixel_key(pixels[members[0]]): group_id
            for group_id, members in enumerate(groups)
            if np.isfinite(pixels[members[0]]).all()
        }
        group_for_feature = {int(feature): int(group_ids[feature])
                             for feature in range(len(pixels))}
        linked_groups = {}
        for feature, lid in associations.items():
            linked_groups.setdefault(lid, set()).add(group_for_feature[feature])
        blocked_groups = set()
        for lid, linked in list(linked_groups.items()):
            if len(linked) <= 1:
                continue
            flow = tracked_pixels.get(lid)
            flow_group = (self._physical_pixel_key(flow) if flow is not None
                          and np.isfinite(flow).all() else None)
            keep = physical_group_lookup.get(flow_group)
            if keep is None:
                blocked_groups.update(linked)
            associations = {feature: linked_lid for feature, linked_lid in associations.items()
                            if linked_lid != lid or group_for_feature[feature] == keep}

        # If an accepted flow point is exactly a detector pixel, reuse that
        # physical feature group. Otherwise append its real measured location.
        appended_pixels, appended_desc = [], []
        observed = set(associations.values())
        for lid, pixel in tracks:
            if lid not in self.map.landmarks or lid in observed:
                continue
            key = self._physical_pixel_key(pixel)
            exact_group = physical_group_lookup.get(key)
            exact_features = groups[exact_group] if exact_group is not None else []
            if exact_features:
                existing_ids = {associations[feature] for feature in exact_features
                                if feature in associations}
                if not existing_ids or existing_ids == {int(lid)}:
                    associations[exact_features[0]] = int(lid)
                    observed.add(int(lid))
                    continue
            feature = len(pixels) + len(appended_pixels)
            appended_pixels.append(np.asarray(pixel).reshape(2))
            appended_desc.append(self.map.landmarks[lid].descriptor)
            associations[feature] = int(lid)
            observed.add(int(lid))
        if appended_pixels:
            pixels = np.vstack([pixels, appended_pixels])
            desc = np.vstack([desc, appended_desc])
            points = np.vstack([points, np.full((len(appended_pixels), 3), np.nan)])
            right_u = np.r_[right_u, np.full(len(appended_pixels), np.nan)]

        # Propagate a single existing identity across every SIFT orientation
        # row at that exact pixel. Conflicting IDs are ambiguous: leave that
        # pixel group unlinked and do not create a replacement landmark.
        groups, group_ids = self._physical_pixel_groups(pixels)
        physical_group_lookup = {
            self._physical_pixel_key(pixels[members[0]]): group_id
            for group_id, members in enumerate(groups)
            if np.isfinite(pixels[members[0]]).all()
        }
        clean_associations = {}
        for group_id, members in enumerate(groups):
            ids = {associations[feature] for feature in members
                   if feature in associations and associations[feature] in self.map.landmarks}
            if len(ids) > 1:
                blocked_groups.add(group_id)
                continue
            if ids:
                lid = next(iter(ids))
                clean_associations.update({int(feature): int(lid) for feature in members})

        # Distinct pixels cannot silently overwrite one landmark's observation
        # in the per-keyframe observation dictionary. An accepted exact flow
        # measurement can select its own pixel; otherwise drop the ambiguous ID.
        associated_groups = {}
        for feature, lid in clean_associations.items():
            associated_groups.setdefault(lid, set()).add(int(group_ids[feature]))
        for lid, linked in list(associated_groups.items()):
            if len(linked) <= 1:
                continue
            flow = tracked_pixels.get(lid)
            flow_key = (self._physical_pixel_key(flow) if flow is not None
                        and np.isfinite(flow).all() else None)
            keep = physical_group_lookup.get(flow_key)
            if keep not in linked:
                keep = None
            if keep is None:
                blocked_groups.update(linked)
            clean_associations = {feature: value for feature, value in clean_associations.items()
                                  if value != lid or int(group_ids[feature]) == keep}

        # Track aliases are represented once for subsequent LK and prediction.
        self.accepted_tracks = self._unique_physical_tracks(tracks)
        association_output.clear()
        association_output.update(clean_associations)
        associations = association_output
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
        observed_by_landmark = {}
        for feature, lid in associations.items():
            observed_by_landmark.setdefault(lid, feature)
        for lid, feature in observed_by_landmark.items():
            observed_pixel = tracked_pixels.get(lid, pixels[feature])
            same_pixel = np.array_equal(np.asarray(observed_pixel, np.float32),
                                        np.asarray(pixels[feature], np.float32))
            measured_right = (float(right_u[feature])
                              if same_pixel and np.isfinite(right_u[feature]) else None)
            if self.stereo is not None and self.current_disparity is not None:
                # The disparity at a nearby detector feature is a different
                # observation. Re-measure at the accepted flow coordinate.
                if feature in measured_rights:
                    measured = [measured_rights[feature]]
                else:
                    _, measured = self._measure_stereo_pixels(observed_pixel[None])
                measured_right = float(measured[0]) if np.isfinite(measured[0]) else None
            self.map.landmarks[lid].observations[ident] = Observation(
                observed_pixel.copy(), measured_right
            )
        if self.stereo is not None:
            for group_id, members in enumerate(groups):
                if group_id in blocked_groups or any(
                        frame.landmark_ids[feature] >= 0 for feature in members):
                    continue
                values = points[members]
                rights = right_u[members]
                finite = np.isfinite(values).all(axis=1) & np.isfinite(rights)
                if not finite.all() or not np.allclose(values, values[0], rtol=1e-6, atol=1e-6) \
                        or not np.allclose(rights, rights[0], rtol=1e-6, atol=1e-6):
                    continue
                feature = members[0]
                world = pose[:3, :3] @ values[0] + pose[:3, 3]
                lid = self.map.add_landmark(
                    world,
                    desc[feature],
                    ident,
                    {
                        ident: Observation(
                            pixels[feature].copy(), float(rights[0])
                        )
                    },
                )
                frame.landmark_ids[members] = lid
        elif self.last_keyframe is not None:
            previous = self.map.keyframes[self.last_keyframe]
            measured_pairs = self._match(previous.descriptors, desc)
            measured_pairs = self._collapse_physical_matches(
                measured_pairs, previous.pixels, pixels,
                previous.landmark_ids, frame.landmark_ids,
            )
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
                pairs = self._collapse_physical_matches(
                    pairs, previous.pixels, pixels,
                    previous.landmark_ids, frame.landmark_ids,
                )
                previous_groups, previous_group_ids = self._physical_pixel_groups(previous.pixels)
                current_groups, current_group_ids = self._physical_pixel_groups(pixels)
                pairs = np.asarray([
                    (a, b) for a, b in pairs
                    if all(previous.landmark_ids[row] < 0
                           for row in previous_groups[previous_group_ids[a]])
                    and all(frame.landmark_ids[row] < 0
                            for row in current_groups[current_group_ids[b]])
                ], int).reshape(-1, 2)
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
                    previous.landmark_ids[previous_groups[previous_group_ids[ia]]] = lid
                    frame.landmark_ids[current_groups[current_group_ids[ib]]] = lid
                    if previous.depth_points is None:
                        previous.depth_points = np.full(
                            (len(previous.pixels), 3), np.nan
                        )
                    previous.depth_points[previous_groups[previous_group_ids[ia]]] = previous.pose[:3, :3].T @ (
                        position - previous.pose[:3, 3]
                    )
                    frame.depth_points[current_groups[current_group_ids[ib]]] = pose[:3, :3].T @ (position - pose[:3, 3])
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
        ambiguous_landmarks = self._landmark_pixel_identity_conflicts(landmarks)
        if ambiguous_landmarks:
            landmarks = [l for l in landmarks if l.id not in ambiguous_landmarks]
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
        # Keep geometry unique by LM while retaining linked SIFT orientation
        # descriptors from bounded anchor/latest keyframes for appearance.
        descriptors, descriptor_landmark_ids = self._landmark_descriptor_bank(landmarks)
        pairs = self._match(descriptors, desc)
        arbitration = self._arbitration_context if self.config.stereo_pose_arbitration else None
        if arbitration is not None:
            # A held detector observation may map to a different older landmark
            # than the source snapshot. Exclude every linked descriptor/flow ID.
            arbitration['excluded_landmarks'].update(
                int(descriptor_landmark_ids[a])
                for a, b in pairs if b in arbitration['excluded_targets'])
            pairs = np.asarray([(a, b) for a, b in pairs
                                if b not in arbitration['excluded_targets']
                                and int(descriptor_landmark_ids[a])
                                not in arbitration['excluded_landmarks']], int).reshape(-1, 2)
        pairs = self._collapse_landmark_matches(pairs, descriptor_landmark_ids, pixels)
        candidates = {int(descriptor_landmark_ids[a]): (pixels[b], int(b))
                      for a, b in pairs}
        descriptor_candidates = candidates.copy()
        flow_conflicts = 0
        flow_visibility_rejections = 0
        previous_tracks = (self.previous_tracks if arbitration is None else
                           [(lid, p) for lid, p in self.previous_tracks
                            if lid not in arbitration['excluded_landmarks']])
        previous_tracks = [
            (lid, p) for lid, p in previous_tracks if lid not in ambiguous_landmarks
        ]
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
        candidates = self._collapse_track_candidates(candidates, pixels)
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
            measured_right = None
            if solution is not None and self.stereo is not None:
                _, measured_right = self._measure_stereo_pixels(observations)
                solution, stereo_diagnostics = refine_stereo_map_pose(
                    solution, positions, observations, measured_right, self.K,
                    self.stereo.baseline, size,
                    self.config.min_inliers if not relocalize else 20,
                    self.stereo.disparity_offset,
                )
                diagnostics.update(stereo_diagnostics)
            if self._tracking_trace_enabled(getattr(self, "_tracking_trace_frame", -1)):
                valid_indices = (
                    np.asarray(solution[1], dtype=int).reshape(-1)
                    if solution is not None else np.empty(0, dtype=int)
                )
                inlier_ids = [int(identifiers[j]) for j in valid_indices
                              if 0 <= j < len(identifiers)]
                rows = []
                for row_index, (landmark_id, point, pixel) in enumerate(
                        zip(identifiers, positions, observations)):
                    feature = int(measurements[landmark_id][1])
                    right_value = None
                    right_valid = False
                    if measured_right is not None and row_index < len(measured_right):
                        candidate_right = float(measured_right[row_index])
                        if np.isfinite(candidate_right):
                            right_value = candidate_right
                            right_valid = True
                    rows.append({
                        "landmark_id": int(landmark_id),
                        "world_position": np.asarray(point, dtype=float).copy(),
                        "pixel": np.asarray(pixel, dtype=np.float32).copy(),
                        "detector_feature_index": feature,
                        "right_u": right_value,
                        "right_u_valid": right_valid,
                        "right_role": "map_refinement_measurement" if right_valid else "not_measured",
                    })
                self._tracking_trace_event(self._tracking_trace_frame, "solve_probe", {
                    "attempt_index": int(getattr(self, "_tracking_trace_probe_count", 0)),
                    "kind": str(getattr(self, "_tracking_trace_probe_kind", "map_pose_solve")),
                    "relocalize": bool(relocalize),
                    "pool_complete": True,
                    "input_row_count": len(rows),
                    "rows": rows,
                    "returned_status": "failed" if solution is None else "accepted",
                    "inlier_landmark_ids": inlier_ids,
                    "pose_diagnostics": diagnostics,
                    "candidate_pose": None if solution is None else np.asarray(solution[0]).copy(),
                    "initial_pose": None if seed is None else np.asarray(seed).copy(),
                    "geometry_seed_dependency": not relocalize,
                })
                self._tracking_trace_probe_count = int(
                    getattr(self, "_tracking_trace_probe_count", 0)) + 1
            return solution, diagnostics, identifiers, positions, observations

        self._tracking_trace_probe_kind = (
            "flow_assisted_primary" if not relocalize else "relocalization_primary"
        )
        result, pose_diagnostics, ids, world, observed = solve(candidates)
        source = "descriptor_map" if relocalize else "flow_assisted_map"
        augmented_correspondences = len(candidates)
        flow_rejection = None
        if result is None and not relocalize and len(descriptor_candidates) >= self.config.min_inliers:
            # A large, coherent cluster of repeated-texture flow can dominate
            # RANSAC while failing spatial or stereo checks. Independently matched
            # map descriptors must still get a solve with the same acceptance gates.
            self._tracking_trace_probe_kind = "descriptor_fallback"
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
        self.accepted_tracks = self._unique_physical_tracks(
            [(ids[j], observed[j].copy()) for j in valid]
        )
        return (pose, associations), info

    def _keyframe_stereo_reference(self, pixels, desc, points, size, ranked=None):
        """Verify camera motion against raw keyframe stereo, without map points."""
        if self.stereo is None:
            return None, {}
        if ranked is None:
            local = list(self.map.keyframes.values())[-self.config.bundle_window :]
            ranked = sorted(
                ((self._physical_match_support(k, pixels, desc), k.id) for k in local),
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
            raw_source = StereoLoopFrame(
                keyframe.pixels, keyframe.depth_points, keyframe.descriptors,
                keyframe.image_size or size,
            )
            source, physical_query, matcher = self._physical_stereo_reference_inputs(
                raw_source, query, source_landmark_ids=keyframe.landmark_ids
            )
            pairs = matcher(source.descriptors, physical_query.descriptors)
            verified = verify_loop(
                source, physical_query, self.K, min_inliers=20, pairs=pairs
            )
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
                (self._physical_match_support(k, pixels, desc), k.id)
                for k in (self.map.keyframes[i] for i in candidate_ids)
            ),
            reverse=True,
        )
        result, stats = self._recover_ranked(pixels, desc, size, points, ranked)
        if result is None and self.retrieval_index is not None and len(candidate_ids) < len(eligible):
            cached_ids = set(candidate_ids)
            ranked = sorted(ranked + [(self._physical_match_support(self.map.keyframes[i], pixels, desc), i)
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
        if (self.config.stereo_depth_policy in ('verified_fallback', 'verified_all')
                and self.stereo is not None):
            # Refresh before extraction, including diagnostic cache hits.
            self.current_left_gray = np.asarray(self.current_gray, np.float32)
            self.current_right_gray = (None if right is None else np.asarray(
                cv.cvtColor(right, cv.COLOR_BGR2GRAY) if right.ndim == 3 else right, np.float32))

    @staticmethod
    def _physical_pixel_key(pixel):
        return tuple(np.asarray(pixel, np.float32).tolist())

    @classmethod
    def _physical_pixel_groups(cls, pixels):
        """Return stable exact-float32 pixel groups and a row-to-group map."""
        pixels = np.asarray(pixels, np.float32).reshape(-1, 2)
        keys, groups = {}, []
        row_groups = np.full(len(pixels), -1, int)
        for row, pixel in enumerate(pixels):
            if not np.isfinite(pixel).all():
                # Invalid coordinates are never evidence of shared identity.
                group = len(groups)
                groups.append([row])
                row_groups[row] = group
                continue
            key = cls._physical_pixel_key(pixel)
            group = keys.get(key)
            if group is None:
                group = len(groups)
                keys[key] = group
                groups.append([])
            groups[group].append(row)
            row_groups[row] = group
        return groups, row_groups

    @classmethod
    def _unique_supported_feature_rows(cls, pixels, points, right_u):
        """Return one row per pixel only when aliased metric readings agree."""
        pixels = np.asarray(pixels, np.float32).reshape(-1, 2)
        points = np.asarray(points, float).reshape(-1, 3)
        right_u = np.asarray(right_u, float).reshape(-1)
        if len(points) != len(pixels) or len(right_u) != len(pixels):
            return np.empty(0, int)
        groups, _ = cls._physical_pixel_groups(pixels)
        selected = []
        for members in groups:
            values = points[members]
            rights = right_u[members]
            finite = np.isfinite(values).all(axis=1) & np.isfinite(rights)
            if (finite.all()
                    and np.allclose(values, values[0], rtol=1e-6, atol=1e-6)
                    and np.allclose(rights, rights[0], rtol=1e-6, atol=1e-6)):
                selected.append(members[0])
        return np.asarray(selected, int)

    @classmethod
    def _unique_physical_geometry_rows(cls, pixels, points):
        """Select one appearance row per physical pixel for geometric fitting."""
        pixels = np.asarray(pixels, np.float32).reshape(-1, 2)
        points = np.asarray(points, float).reshape(-1, 3)
        if len(points) != len(pixels):
            return np.empty(0, int)
        groups, _ = cls._physical_pixel_groups(pixels)
        selected = []
        for members in groups:
            finite_rows = [row for row in members if np.isfinite(points[row]).all()]
            if finite_rows:
                finite_points = points[finite_rows]
                if not np.allclose(finite_points, finite_points[0], rtol=1e-6, atol=1e-6):
                    continue
                selected.append(finite_rows[0])
            else:
                selected.append(members[0])
        return np.asarray(selected, int)

    @classmethod
    def _canonical_physical_geometry_points(cls, pixels, points):
        """Share one consistent 3D measurement across exact-pixel aliases."""
        pixels = np.asarray(pixels, np.float32).reshape(-1, 2)
        points = np.asarray(points, float)
        if points.shape != (len(pixels), 3):
            return np.full((len(pixels), 3), np.nan), np.zeros(len(pixels), bool)
        result = points.copy()
        usable = np.isfinite(pixels).all(axis=1)
        groups, _ = cls._physical_pixel_groups(pixels)
        for members in groups:
            finite_rows = [row for row in members if np.isfinite(points[row]).all()]
            if not finite_rows:
                result[members] = np.nan
                continue
            finite_points = points[finite_rows]
            if not np.allclose(finite_points, finite_points[0], rtol=1e-6, atol=1e-6):
                result[members] = np.nan
                usable[members] = False
                continue
            # Orientation rows at one exact float32 pixel describe one camera
            # measurement. Preserve their descriptors and reuse that one value.
            result[members] = finite_points[0]
        return result, usable

    def _physical_stereo_reference_inputs(
        self, source, target, source_landmark_ids=None, target_landmark_ids=None
    ):
        """Keep every appearance row while returning unique physical match edges."""
        source_points, source_usable = self._canonical_physical_geometry_points(
            source.pixels, source.points
        )
        target_points, target_usable = self._canonical_physical_geometry_points(
            target.pixels, target.points
        )
        source_frame = StereoLoopFrame(
            np.asarray(source.pixels, np.float32), source_points,
            np.asarray(source.descriptors), source.image_size,
        )
        target_frame = StereoLoopFrame(
            np.asarray(target.pixels, np.float32), target_points,
            np.asarray(target.descriptors), target.image_size,
        )

        def matcher(first, second):
            pairs = self._match(first, second)
            pairs = self._collapse_physical_matches(
                pairs, source_frame.pixels, target_frame.pixels,
                source_landmark_ids, target_landmark_ids,
            )
            if len(pairs):
                pairs = pairs[source_usable[pairs[:, 0]] & target_usable[pairs[:, 1]]]
            return pairs

        return source_frame, target_frame, matcher

    @classmethod
    def _collapse_physical_matches(
        cls, pairs, first_pixels, second_pixels,
        first_landmark_ids=None, second_landmark_ids=None,
    ):
        """Keep one deterministic descriptor pair per unambiguous pixel edge.

        Mutual descriptor matching is unique by descriptor row, not physical
        image point. Conflicting endpoint relations are rejected rather than
        arbitrarily assigning one physical observation to another.
        """
        first_pixels = np.asarray(first_pixels, np.float32).reshape(-1, 2)
        second_pixels = np.asarray(second_pixels, np.float32).reshape(-1, 2)
        pairs = np.asarray(pairs)
        if (pairs.ndim != 2 or pairs.shape[1:] != (2,)
                or not np.issubdtype(pairs.dtype, np.integer)
                or np.issubdtype(pairs.dtype, np.bool_)):
            return np.empty((0, 2), int)
        pairs = pairs.astype(int, copy=False)
        if (len(pairs) and (np.any(pairs < 0)
                            or np.any(pairs[:, 0] >= len(first_pixels))
                            or np.any(pairs[:, 1] >= len(second_pixels)))):
            return np.empty((0, 2), int)
        first_groups, first_row_group = cls._physical_pixel_groups(first_pixels)
        second_groups, second_row_group = cls._physical_pixel_groups(second_pixels)

        def conflicting_groups(groups, row_groups, landmark_ids):
            conflicts = set()
            if landmark_ids is None:
                return conflicts
            landmark_ids = np.asarray(landmark_ids)
            if landmark_ids.shape != (len(row_groups),):
                return set(range(len(groups)))
            for group_id, rows in enumerate(groups):
                identifiers = {int(landmark_ids[row]) for row in rows
                               if int(landmark_ids[row]) >= 0}
                if len(identifiers) > 1:
                    conflicts.add(group_id)
            return conflicts

        bad_first = conflicting_groups(first_groups, first_row_group, first_landmark_ids)
        bad_second = conflicting_groups(second_groups, second_row_group, second_landmark_ids)
        edges = {}
        first_to_second, second_to_first = {}, {}
        for first, second in pairs:
            first_group = int(first_row_group[first])
            second_group = int(second_row_group[second])
            if first_group in bad_first or second_group in bad_second:
                continue
            edge = (first_group, second_group)
            edges.setdefault(edge, []).append((int(first), int(second)))
            first_to_second.setdefault(first_group, set()).add(second_group)
            second_to_first.setdefault(second_group, set()).add(first_group)
        ambiguous_first = {group for group, targets in first_to_second.items()
                           if len(targets) > 1}
        ambiguous_second = {group for group, sources in second_to_first.items()
                            if len(sources) > 1}
        selected = [min(rows) for (first_group, second_group), rows in edges.items()
                    if first_group not in ambiguous_first
                    and second_group not in ambiguous_second]
        return np.asarray(sorted(selected), int).reshape(-1, 2)

    def _physical_match_support(self, keyframe, pixels, descriptors):
        pairs = self._match(keyframe.descriptors, descriptors)
        source_pixels = getattr(keyframe, "pixels", None)
        if source_pixels is None:
            return len(pairs)
        return len(self._collapse_physical_matches(
            pairs, source_pixels, pixels, getattr(keyframe, "landmark_ids", None)
        ))

    @staticmethod
    def _collapse_landmark_matches(pairs, landmark_ids, target_pixels):
        """Collapse descriptor-bank aliases to one row per LM/target pixel."""
        landmark_ids = np.asarray(landmark_ids, int)
        target_pixels = np.asarray(target_pixels, np.float32).reshape(-1, 2)
        pairs = np.asarray(pairs)
        if (pairs.ndim != 2 or pairs.shape[1:] != (2,)
                or not np.issubdtype(pairs.dtype, np.integer)
                or np.issubdtype(pairs.dtype, np.bool_)):
            return np.empty((0, 2), int)
        pairs = pairs.astype(int, copy=False)
        if (landmark_ids.ndim != 1 or (len(pairs) and (np.any(pairs < 0)
                or np.any(pairs[:, 0] >= len(landmark_ids))
                or np.any(pairs[:, 1] >= len(target_pixels))))):
            return np.empty((0, 2), int)
        _, target_row_group = SharedSlam._physical_pixel_groups(target_pixels)
        edges = {}
        lm_to_target, target_to_lm = {}, {}
        for row, target in pairs:
            landmark_id = int(landmark_ids[row])
            target_group = int(target_row_group[target])
            if landmark_id < 0:
                continue
            edges.setdefault((landmark_id, target_group), []).append((int(row), int(target)))
            lm_to_target.setdefault(landmark_id, set()).add(target_group)
            target_to_lm.setdefault(target_group, set()).add(landmark_id)
        ambiguous_lms = {landmark_id for landmark_id, targets in lm_to_target.items()
                         if len(targets) > 1}
        ambiguous_targets = {target for target, landmarks in target_to_lm.items()
                             if len(landmarks) > 1}
        selected = [min(rows) for (landmark_id, target_group), rows in edges.items()
                    if landmark_id not in ambiguous_lms
                    and target_group not in ambiguous_targets]
        return np.asarray(sorted(selected), int).reshape(-1, 2)

    @classmethod
    def _collapse_track_candidates(cls, candidates, target_pixels):
        """Keep unique flow rays and clear only ambiguous detector labels."""
        target_pixels = np.asarray(target_pixels, np.float32).reshape(-1, 2)
        group_claims, candidate_groups, feature_claims = {}, {}, {}
        for landmark_id, (pixel, feature) in candidates.items():
            # The LK measurement is the geometry observation. Its nearest SIFT
            # row is only an appearance association and may be shared by nearby
            # but physically distinct flow tracks.
            key = cls._physical_pixel_key(np.asarray(pixel, np.float32).reshape(2))
            candidate_groups[int(landmark_id)] = key
            group_claims.setdefault(key, set()).add(int(landmark_id))
            if 0 <= int(feature) < len(target_pixels):
                detector_key = cls._physical_pixel_key(target_pixels[int(feature)])
                feature_claims.setdefault(detector_key, []).append(int(landmark_id))
        conflicts = {key for key, identifiers in group_claims.items()
                     if len(identifiers) > 1}
        kept = {int(landmark_id): value for landmark_id, value in candidates.items()
                if candidate_groups[int(landmark_id)] not in conflicts}
        # Two different physical observations cannot both claim the same
        # detector identity. Preserve their actual coordinates as flow-only
        # rows; keep an exact-pixel claimant linked when it is unambiguous.
        for detector_key, identifiers in feature_claims.items():
            identifiers = [landmark_id for landmark_id in identifiers if landmark_id in kept]
            keys = {candidate_groups[landmark_id] for landmark_id in identifiers}
            if len(keys) <= 1:
                continue
            exact = [landmark_id for landmark_id in identifiers
                     if candidate_groups[landmark_id] == detector_key]
            keep_exact = exact[0] if len(exact) == 1 else None
            for landmark_id in identifiers:
                if landmark_id == keep_exact:
                    continue
                pixel, _ = kept[landmark_id]
                kept[landmark_id] = (pixel, -1)
        return kept

    def _landmark_descriptor_bank(self, landmarks):
        """Use anchor/latest linked SIFT rows without retaining a dense map copy."""
        frames_for_landmark, requested_by_frame = {}, {}
        for landmark in landmarks:
            frame_ids = []
            for frame_id in (landmark.anchor, max(landmark.observations, default=landmark.anchor)):
                if frame_id not in frame_ids:
                    frame_ids.append(frame_id)
            frames_for_landmark[int(landmark.id)] = frame_ids
            for frame_id in frame_ids:
                if frame_id in self.map.keyframes:
                    requested_by_frame.setdefault(frame_id, set()).add(int(landmark.id))

        rows_by_landmark = {}
        for frame_id, requested in requested_by_frame.items():
            keyframe = self.map.keyframes[frame_id]
            rows = np.flatnonzero(np.isin(keyframe.landmark_ids, list(requested)))
            for row in rows:
                landmark_id = int(keyframe.landmark_ids[row])
                rows_by_landmark.setdefault(landmark_id, {}).setdefault(frame_id, []).append(
                    keyframe.descriptors[row]
                )

        descriptors, identifiers = [], []
        for landmark in landmarks:
            found = False
            seen_descriptors = set()
            rows = rows_by_landmark.get(int(landmark.id), {})
            for frame_id in frames_for_landmark[int(landmark.id)]:
                for descriptor in rows.get(frame_id, ()):
                    signature = (descriptor.dtype.str, descriptor.shape, descriptor.tobytes())
                    if signature in seen_descriptors:
                        continue
                    seen_descriptors.add(signature)
                    descriptors.append(descriptor)
                    identifiers.append(int(landmark.id))
                    found = True
            if not found:
                descriptors.append(landmark.descriptor)
                identifiers.append(int(landmark.id))
        if not descriptors:
            return np.empty((0, 128), np.float32), np.empty(0, int)
        return np.asarray(descriptors), np.asarray(identifiers, int)

    def _landmark_pixel_identity_conflicts(self, landmarks):
        """Find distinct LMs claiming one exact pixel in one keyframe."""
        claims = {}
        conflicts = set()
        cached_keyframes = set()
        # Inspect stored detector rows, not just the candidate subset. A
        # relocalization probe may contain only one of two legacy IDs that
        # claim the same source pixel. Cache the conflict set per keyframe;
        # exact content hashes detect old-keyframe ID edits without regrouping
        # every historical row on every frame.
        for keyframe_id, keyframe in self.map.keyframes.items():
            cached_keyframes.add(keyframe_id)
            pixels = np.asarray(getattr(keyframe, "pixels", np.empty((0, 2))), np.float32)
            identifiers = np.asarray(
                getattr(keyframe, "landmark_ids", np.empty(0, int))
            )
            if (pixels.ndim != 2 or pixels.shape[1:] != (2,)
                    or identifiers.shape != (len(pixels),)
                    or not np.issubdtype(identifiers.dtype, np.integer)
                    or np.issubdtype(identifiers.dtype, np.bool_)):
                self._identity_conflict_cache.pop(keyframe_id, None)
                continue
            pixel_bytes = np.ascontiguousarray(pixels)
            id_bytes = np.ascontiguousarray(identifiers)
            digest = hashlib.sha256()
            digest.update(str((pixel_bytes.dtype.str, pixel_bytes.shape)).encode())
            if pixel_bytes.size:
                digest.update(memoryview(pixel_bytes).cast("B"))
            digest.update(str((id_bytes.dtype.str, id_bytes.shape)).encode())
            if id_bytes.size:
                digest.update(memoryview(id_bytes).cast("B"))
            fingerprint = digest.digest()
            cached = self._identity_conflict_cache.get(keyframe_id)
            if cached is not None and cached[0] == fingerprint:
                conflicts.update(cached[1])
                continue
            frame_conflicts = set()
            groups, _ = self._physical_pixel_groups(pixels)
            for members in groups:
                ids = {int(identifiers[row]) for row in members
                       if int(identifiers[row]) >= 0}
                if len(ids) > 1:
                    frame_conflicts.update(ids)
            self._identity_conflict_cache[keyframe_id] = (
                fingerprint, frozenset(frame_conflicts)
            )
            conflicts.update(frame_conflicts)
        for stale_keyframe in set(self._identity_conflict_cache) - cached_keyframes:
            del self._identity_conflict_cache[stale_keyframe]
        for landmark in landmarks:
            for keyframe_id, observation in landmark.observations.items():
                # Synthetic map-only clients can provide observations without
                # keyframes; there is no detector-row identity to disambiguate.
                if keyframe_id not in self.map.keyframes:
                    continue
                pixel = np.asarray(observation.pixel, np.float32).reshape(-1)
                if pixel.shape == (2,) and np.isfinite(pixel).all():
                    claims.setdefault((int(keyframe_id), self._physical_pixel_key(pixel)), set()).add(
                        int(landmark.id)
                    )
        conflicts.update(landmark_id for identifiers in claims.values()
                         if len(identifiers) > 1 for landmark_id in identifiers)
        return conflicts

    @classmethod
    def _unique_physical_tracks(cls, tracks):
        """Keep one LM per exact accepted-flow pixel and one pixel per LM."""
        by_landmark, by_pixel = {}, {}
        normalized = []
        for landmark_id, pixel in tracks:
            landmark_id = int(landmark_id)
            pixel = np.asarray(pixel).reshape(2)
            if not np.isfinite(pixel).all():
                continue
            key = cls._physical_pixel_key(pixel)
            by_landmark.setdefault(landmark_id, set()).add(key)
            by_pixel.setdefault(key, set()).add(landmark_id)
            normalized.append((landmark_id, pixel.copy(), key))
        ambiguous_landmarks = {landmark_id for landmark_id, keys in by_landmark.items()
                               if len(keys) > 1}
        ambiguous_pixels = {key for key, identifiers in by_pixel.items()
                            if len(identifiers) > 1}
        result, seen = [], set()
        for landmark_id, pixel, key in normalized:
            if (landmark_id in ambiguous_landmarks or key in ambiguous_pixels
                    or landmark_id in seen):
                continue
            result.append((landmark_id, pixel))
            seen.add(landmark_id)
        return result

    def _capture_supported_stereo(self, index, pixels, desc, size):
        # Native extraction snapshots policy-selected measurements before any
        # later restoration. A cache hit remeasures from the current stereo pair.
        raw = self._supported_extraction
        if raw is None:
            raw = self._measure_supported_stereo_pixels(pixels)
        return SupportedStereoFrame(pixels, desc, raw[0], raw[1],
                                    np.full(len(pixels), -1, int), index, size,
                                    self.stereo_calibration_identity)

    @classmethod
    def _partition_physical_stereo_matches(cls, source, target, raw_pairs):
        """Partition raw mutual descriptor matches by validated physical edges.

        Endpoint topology is built before consulting metric readings or landmark
        labels. A physical pixel alias group is admitted only when every row in
        that full extraction group agrees and is independently valid.
        """
        report = {
            "physical_match_pool_mode": "strict_full_alias_edges_v1",
            "raw_descriptor_pair_count": 0,
            "physical_edges_before_validation": 0,
            "physical_edges_accepted": 0,
            "dropped_competing_relations": 0,
            "dropped_invalid_source_groups": 0,
            "dropped_invalid_target_groups": 0,
            "dropped_conflicting_landmark_edges": 0,
            "fit_group_count": 0,
            "holdout_group_count": 0,
            "partition_rule": "source_physical_group_ordinal_modulo_2",
        }

        def frame_arrays(frame):
            pixels_raw = np.asarray(frame.pixels)
            points_raw = np.asarray(frame.points)
            right_raw = np.asarray(frame.right_u)
            ids_raw = np.asarray(frame.landmark_ids)
            if (pixels_raw.ndim != 2 or pixels_raw.shape[1:] != (2,)
                    or pixels_raw.dtype.kind not in "fiu"
                    or points_raw.shape != (len(pixels_raw), 3)
                    or points_raw.dtype.kind not in "fiu"
                    or right_raw.shape != (len(pixels_raw),)
                    or right_raw.dtype.kind not in "fiu"
                    or ids_raw.shape != (len(pixels_raw),)
                    or ids_raw.dtype.kind not in "iu"):
                raise ValueError("invalid_real_endpoint_arrays")
            if (ids_raw.dtype.kind == "u"
                    and np.any(ids_raw > np.iinfo(np.int64).max)):
                raise ValueError("landmark_id_out_of_range")
            if ids_raw.dtype.kind == "i" and np.any(ids_raw < -1):
                raise ValueError("invalid_negative_landmark_id")
            pixels = np.asarray(pixels_raw, dtype=np.float32)
            points = np.asarray(points_raw, dtype=np.float64)
            right = np.asarray(right_raw, dtype=np.float64)
            ids = np.asarray(ids_raw, dtype=np.int64)
            try:
                size_raw = np.asarray(frame.image_size)
                if (size_raw.shape != (2,) or size_raw.dtype.kind not in "fiu"
                        or not np.isfinite(size_raw).all()):
                    raise ValueError
                size = np.asarray(size_raw, dtype=np.float64)
            except Exception as error:
                raise ValueError("invalid_endpoint_image_size") from error
            if not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError("invalid_endpoint_image_size")
            return pixels, points, right, ids, size

        try:
            source_pixels, source_points, source_right, source_ids, source_size = frame_arrays(source)
            target_pixels, target_points, target_right, target_ids, target_size = frame_arrays(target)
        except Exception as error:
            report.update(reason=str(error), invalid_inputs=True)
            return {
                "fit_pairs": np.empty((0, 2), np.int64),
                "held_pairs": np.empty((0, 2), np.int64),
                "fit_edges": [], "held_edges": [], "report": report,
            }

        pairs = np.asarray(raw_pairs)
        if (pairs.ndim != 2 or pairs.shape[1:] != (2,)
                or not np.issubdtype(pairs.dtype, np.integer)
                or np.issubdtype(pairs.dtype, np.bool_)):
            report.update(reason="invalid_raw_match_pairs", invalid_inputs=True)
            return {
                "fit_pairs": np.empty((0, 2), np.int64),
                "held_pairs": np.empty((0, 2), np.int64),
                "fit_edges": [], "held_edges": [], "report": report,
            }
        if (pairs.dtype.kind == "u"
                and np.any(pairs > np.iinfo(np.int64).max)):
            report.update(reason="raw_match_index_out_of_range", invalid_inputs=True)
            return {
                "fit_pairs": np.empty((0, 2), np.int64),
                "held_pairs": np.empty((0, 2), np.int64),
                "fit_edges": [], "held_edges": [], "report": report,
            }
        pairs = np.asarray(pairs, dtype=np.int64)
        report["raw_descriptor_pair_count"] = int(len(pairs))
        if (len(pairs) and (np.any(pairs < 0)
                            or np.any(pairs[:, 0] >= len(source_pixels))
                            or np.any(pairs[:, 1] >= len(target_pixels)))):
            report.update(reason="raw_match_index_out_of_bounds", invalid_inputs=True)
            return {
                "fit_pairs": np.empty((0, 2), np.int64),
                "held_pairs": np.empty((0, 2), np.int64),
                "fit_edges": [], "held_edges": [], "report": report,
            }
        if not len(pairs):
            report["reason"] = "empty_raw_match_pool"
            return {
                "fit_pairs": np.empty((0, 2), np.int64),
                "held_pairs": np.empty((0, 2), np.int64),
                "fit_edges": [], "held_edges": [], "report": report,
            }

        source_groups, source_row_group = cls._physical_pixel_groups(source_pixels)
        target_groups, target_row_group = cls._physical_pixel_groups(target_pixels)

        def claims_by_group(groups, ids):
            result = []
            id_groups = {}
            for group_id, members in enumerate(groups):
                claims = {int(ids[row]) for row in members if int(ids[row]) >= 0}
                result.append(claims)
                for landmark_id in claims:
                    id_groups.setdefault(landmark_id, set()).add(group_id)
            conflicts = {group_id for group_ids in id_groups.values() if len(group_ids) > 1
                         for group_id in group_ids}
            conflicts.update(group_id for group_id, claims in enumerate(result)
                             if len(claims) > 1)
            return result, conflicts

        source_claims, source_identity_conflicts = claims_by_group(source_groups, source_ids)
        target_claims, target_identity_conflicts = claims_by_group(target_groups, target_ids)

        # Construct the complete relation graph from raw descriptor matches.
        # No endpoint measurement/identity filtering is allowed at this stage.
        edge_matches = {}
        source_relations, target_relations = {}, {}
        for source_row, target_row in sorted({tuple(map(int, pair)) for pair in pairs}):
            source_group = int(source_row_group[source_row])
            target_group = int(target_row_group[target_row])
            edge_matches.setdefault((source_group, target_group), []).append(
                (source_row, target_row))
            source_relations.setdefault(source_group, set()).add(target_group)
            target_relations.setdefault(target_group, set()).add(source_group)
        report["physical_edges_before_validation"] = int(len(edge_matches))
        competing_source = {group for group, neighbors in source_relations.items()
                            if len(neighbors) > 1}
        competing_target = {group for group, neighbors in target_relations.items()
                            if len(neighbors) > 1}
        report["dropped_competing_relations"] = int(sum(
            source_group in competing_source or target_group in competing_target
            for source_group, target_group in edge_matches))

        def valid_group(frame_pixels, frame_points, frame_right, image_size, groups, group_id):
            members = groups[group_id]
            pixels = frame_pixels[members]
            points = frame_points[members]
            right = frame_right[members]
            inside = (
                np.isfinite(pixels).all(axis=1)
                & (pixels[:, 0] >= 0) & (pixels[:, 0] < image_size[0])
                & (pixels[:, 1] >= 0) & (pixels[:, 1] < image_size[1])
            )
            return bool(
                np.all(inside)
                and np.isfinite(points).all()
                and np.all(points[:, 2] > 0)
                and np.isfinite(right).all()
                and np.all((right >= 0) & (right < image_size[0]))
                and np.allclose(points, points[0], rtol=1e-6, atol=1e-6)
                and np.allclose(right, right[0], rtol=1e-6, atol=1e-6)
            )

        valid_source = {}
        valid_target = {}
        fit_edges, held_edges = [], []
        for (source_group, target_group), matches in sorted(edge_matches.items()):
            if source_group in competing_source or target_group in competing_target:
                continue
            if source_group not in valid_source:
                valid_source[source_group] = valid_group(
                    source_pixels, source_points, source_right, source_size,
                    source_groups, source_group)
            if target_group not in valid_target:
                valid_target[target_group] = valid_group(
                    target_pixels, target_points, target_right, target_size,
                    target_groups, target_group)
            if not valid_source[source_group]:
                report["dropped_invalid_source_groups"] += 1
                continue
            if not valid_target[target_group]:
                report["dropped_invalid_target_groups"] += 1
                continue
            if (source_group in source_identity_conflicts
                    or target_group in target_identity_conflicts):
                report["dropped_conflicting_landmark_edges"] += 1
                continue
            source_aliases = tuple(map(int, source_groups[source_group]))
            target_aliases = tuple(map(int, target_groups[target_group]))
            pair = min(matches)
            source_claim_ids = tuple(sorted(source_claims[source_group]))
            target_claim_ids = tuple(sorted(target_claims[target_group]))
            edge = {
                "pair": pair,
                "source_group_id": int(source_group),
                "target_group_id": int(target_group),
                "source_alias_rows": source_aliases,
                "target_alias_rows": target_aliases,
                "source_landmark_ids": source_claim_ids,
                "target_landmark_ids": target_claim_ids,
                "partition_source_group_ordinal": int(source_group),
            }
            (fit_edges if source_group % 2 == 0 else held_edges).append(edge)

        fit_edges.sort(
            key=lambda edge: (edge["source_group_id"], edge["target_group_id"])
        )
        held_edges.sort(
            key=lambda edge: (edge["source_group_id"], edge["target_group_id"])
        )
        fit_source_groups = {edge["source_group_id"] for edge in fit_edges}
        held_source_groups = {edge["source_group_id"] for edge in held_edges}
        fit_target_groups = {edge["target_group_id"] for edge in fit_edges}
        held_target_groups = {edge["target_group_id"] for edge in held_edges}
        if (fit_source_groups & held_source_groups
                or fit_target_groups & held_target_groups):
            report.update(reason="physical_partition_overlap", invalid_inputs=True)
            fit_edges, held_edges = [], []

        fit_pairs = np.asarray([edge["pair"] for edge in fit_edges], dtype=np.int64).reshape(-1, 2)
        held_pairs = np.asarray([edge["pair"] for edge in held_edges], dtype=np.int64).reshape(-1, 2)
        report.update(
            physical_edges_accepted=int(len(fit_edges) + len(held_edges)),
            fit_group_count=int(len(fit_edges)),
            holdout_group_count=int(len(held_edges)),
            dropped_duplicate_descriptor_pairs=max(0, int(len(pairs) - len(edge_matches))),
            reason=("physical_edges_partitioned" if len(fit_edges) and len(held_edges)
                    else "insufficient_physical_fit_or_holdout"),
        )
        return {
            "fit_pairs": fit_pairs,
            "held_pairs": held_pairs,
            "fit_edges": fit_edges,
            "held_edges": held_edges,
            "report": report,
        }

    def _prepare_stereo_arbitration(self, index, current, capture_rows=False):
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
        physical_mode = bool(getattr(self.config, "stereo_physical_match_pool", False))
        fit_edges = held_edges = None
        if physical_mode:
            partition = self._partition_physical_stereo_matches(previous, current, pairs)
            report.update(partition["report"])
            report["physical_identity"] = "strict_full_alias_edges_v1"
            if partition["report"].get("invalid_inputs"):
                return None, report
            fit = partition["fit_pairs"]
            held = partition["held_pairs"]
            fit_edges = partition["fit_edges"]
            held_edges = partition["held_edges"]
            if not len(pairs):
                report["reason"] = "insufficient_supported_pool"
                return None, report
            report.update(supported_pool=len(fit) + len(held), fit_count=len(fit),
                          holdout_count=len(held), source_frame=previous.frame,
                          target_frame=index)
        else:
            if not len(pairs):
                report['reason'] = 'insufficient_supported_pool'
                return None, report
            # Preserve the default numerical path exactly. It drops duplicate
            # endpoint pixels before splitting descriptor rows by source index.
            _, previous_inverse, previous_counts = np.unique(
                np.asarray(previous.pixels, np.float32), axis=0,
                return_inverse=True, return_counts=True)
            _, current_inverse, current_counts = np.unique(
                np.asarray(current.pixels, np.float32), axis=0,
                return_inverse=True, return_counts=True)
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
            report.update(supported_pool=len(pool), fit_count=len(fit), holdout_count=len(held),
                          dropped_duplicate_matches=int(np.sum(
                              (previous_counts[previous_inverse[a]] > 1)
                              | (current_counts[current_inverse[b]] > 1))),
                          source_frame=previous.frame, target_frame=index)
        minimum = self.config.min_inliers
        if (min(len(fit), len(held)) < minimum
                or any(coverage(frame.pixels[subset[:, column]], frame.image_size) < 3
                       for subset in (fit, held)
                       for frame, column in ((previous, 0), (current, 1)))):
            report['reason'] = 'insufficient_reserved_support'
            return None, report
        source, target = held.T
        if physical_mode:
            held_source_aliases = sorted({row for edge in held_edges
                                          for row in edge["source_alias_rows"]})
            held_target_aliases = sorted({row for edge in held_edges
                                          for row in edge["target_alias_rows"]})
            source_pixels = {self._physical_pixel_key(previous.pixels[row])
                             for row in held_source_aliases}
            target_pixels = {self._physical_pixel_key(current.pixels[row])
                             for row in held_target_aliases}
            excluded_targets = set(held_target_aliases)
            excluded_landmarks = {
                landmark_id for edge in held_edges
                for landmark_id in (edge["source_landmark_ids"]
                                    + edge["target_landmark_ids"])
            }
            excluded_landmarks.update(
                int(landmark_id) for landmark_id, pixel in self.previous_tracks
                if self._physical_pixel_key(pixel) in source_pixels
            )
            held_landmark_ids = np.asarray([
                edge["source_landmark_ids"][0]
                if len(edge["source_landmark_ids"]) == 1 else -1
                for edge in held_edges
            ], dtype=np.int64)
        else:
            source_pixels = {self._physical_pixel_key(p) for p in previous.pixels[source]}
            target_pixels = {self._physical_pixel_key(p) for p in current.pixels[target]}
            excluded_targets = {j for j, p in enumerate(current.pixels)
                                if self._physical_pixel_key(p) in target_pixels}
            excluded_landmarks = set(previous.landmark_ids[source].tolist()) - {-1}
            excluded_landmarks.update(lid for lid, p in self.previous_tracks
                                      if self._physical_pixel_key(p) in source_pixels)
            held_landmark_ids = previous.landmark_ids[source]
        evidence = SupportedStereoHoldout(previous.points[source], current.pixels[target],
                                         current.right_u[target], source, target,
                                         held_landmark_ids,
                                         'immutable_supported_extraction', previous.frame,
                                         previous.calibration_identity)
        fit_epoch = None
        fit_source_state = None
        if capture_rows:
            with self.map.lock:
                fit_epoch = (int(self.map.revision), int(self.map.geometry_revision))
                source_anchor = self.map.pose_anchors[previous.frame]
                fit_source_state = {
                    'source_pose': self.map.poses[previous.frame].copy(),
                    'source_status': self.map.statuses[previous.frame],
                    'source_anchor_keyframe_id': source_anchor,
                    'source_anchor_pose': (
                        self.map.keyframes[source_anchor].pose.copy()
                        if source_anchor in self.map.keyframes else None
                    ),
                }
        context = {'previous': previous, 'current': current, 'fit': fit, 'evidence': evidence,
                   'excluded_targets': excluded_targets, 'excluded_landmarks': excluded_landmarks,
                   'excluded_target_pixels': target_pixels,
                   'map_fit_targets': set(), 'map_fit_landmarks': set(), 'report': report}
        if capture_rows:
            context['held_pairs'] = np.array(held, dtype=np.int64, copy=True)
        source_frame = StereoLoopFrame(previous.pixels, previous.points, previous.descriptors, previous.image_size)
        target_frame = StereoLoopFrame(current.pixels, current.points, current.descriptors, current.image_size)
        if capture_rows:
            verified = self.profile.call(
                'stereo_pose_arbitration_fit', estimate_stereo_reference,
                source_frame, target_frame, self.K, min_inliers=minimum,
                initial_pose=None, matcher=lambda first, second: fit.copy(),
                capture_rows=True,
            )
        else:
            verified = self.profile.call(
                'stereo_pose_arbitration_fit', estimate_stereo_reference,
                source_frame, target_frame, self.K, min_inliers=minimum,
                initial_pose=None, matcher=lambda first, second: fit.copy())
        if verified is None or not verified['reverse_checked']:
            report['reason'] = 'independent_training_failed'
            return None, report
        context['verified'] = verified
        if capture_rows:
            context['training_rows'] = verified.get('training_rows')
            context['fit_source_epoch'] = fit_epoch
            context['fit_source_state'] = fit_source_state
            context['fit_pairs_snapshot'] = np.array(fit, dtype=np.int64, copy=True)
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
        physical_identity = (
            "strict_full_alias_edges_v1"
            if getattr(self.config, "stereo_physical_match_pool", False)
            else "exact_float32_pixels_duplicates_dropped"
        )
        return {**context['report'], **report,
                'independent_fit_sha256': hashlib.sha256(
                    np.ascontiguousarray(context['fit'], dtype='<i8').tobytes()).hexdigest(),
                'map_fit_landmarks': len(context['map_fit_landmarks']),
                'map_fit_target_features': len(context['map_fit_targets']),
                'physical_identity': physical_identity}

    def _stereo_mapping_retention_context_is_current(self, pose, context):
        """Revalidate the exact independent-pose transaction before map retention."""
        if not isinstance(context, dict):
            return False, 'missing_independent_pose_context'
        guard = context.get('guard')
        verified = context.get('verified')
        try:
            raw_integers = [context['source_frame'], context['target_frame'],
                            context['source_map_revision'], context['source_geometry_revision']]
            if any(not isinstance(value, (int, np.integer))
                   or isinstance(value, (bool, np.bool_)) for value in raw_integers):
                return False, 'malformed_independent_pose_integer_fields'
            source_frame, target_frame, source_revision, geometry_revision = map(int, raw_integers)
            raw_poses = [context['source_pose'], context['reference_pose'], pose,
                         verified.get('measurement') if isinstance(verified, dict) else None]
            if any(value is None or np.iscomplexobj(np.asarray(value)) for value in raw_poses):
                return False, 'malformed_independent_pose_values'
            source_pose = np.asarray(raw_poses[0], float)
            reference_pose = np.asarray(raw_poses[1], float)
            selected_pose = np.asarray(raw_poses[2], float)
            previous_frame = context['previous_frame']
            measured_frame = context['measured_frame']
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, 'malformed_independent_pose_context'
        if (not isinstance(guard, dict) or guard.get('eligible') is not True
                or guard.get('reference_reverse_checked') is not True
                or not isinstance(verified, dict) or verified.get('reverse_checked') is not True):
            return False, 'independent_pose_certificate_missing'
        if (source_frame < 0 or target_frame != len(self.map.poses)
                or target_frame - source_frame < 1
                or source_frame >= len(self.map.poses)
                or source_revision != self.map.revision
                or geometry_revision != self.map.geometry_revision
                or guard.get('source_map_revision') != source_revision
                or guard.get('source_frame') != source_frame
                or guard.get('target_frame') != target_frame):
            return False, 'map_or_geometry_epoch_changed'
        try:
            raw_excluded_ids = context.get('excluded_landmark_ids', ())
            if not isinstance(raw_excluded_ids, (set, frozenset, list, tuple)):
                return False, 'malformed_held_out_exclusion'
            if any(not isinstance(value, (int, np.integer))
                   or isinstance(value, (bool, np.bool_)) or int(value) < 0
                   for value in raw_excluded_ids):
                return False, 'malformed_held_out_exclusion'
            raw_excluded_pixels = context.get('excluded_target_pixels', ())
            if isinstance(raw_excluded_pixels, (set, frozenset)):
                raw_excluded_pixels = list(raw_excluded_pixels)
            excluded_array = np.asarray(raw_excluded_pixels)
            if excluded_array.size == 0:
                excluded_array = np.empty((0, 2), float)
            if (np.iscomplexobj(excluded_array)
                    or excluded_array.ndim != 2 or excluded_array.shape[1] != 2
                    or not np.isfinite(np.asarray(excluded_array, float)).all()):
                return False, 'malformed_held_out_exclusion'
        except (TypeError, ValueError, OverflowError):
            return False, 'malformed_held_out_exclusion'
        if (not self._is_proper_se3(source_pose)
                or not self._is_proper_se3(reference_pose)
                or not self._is_proper_se3(selected_pose)
                or not self._is_proper_se3(verified.get('measurement'))):
            return False, 'invalid_reference_pose'
        if (not np.array_equal(np.asarray(self.map.poses[source_frame], float), source_pose)
                or not np.allclose(selected_pose, reference_pose, atol=1e-12, rtol=0.)
                or not np.allclose(source_pose @ np.asarray(verified['measurement'], float),
                                   reference_pose, atol=1e-10, rtol=1e-10)):
            return False, 'source_or_reference_pose_changed'
        try:
            live_identity = self._live_stereo_calibration_identity()
            source_record = self.previous_supported_stereo
            target_record = self.current_supported_stereo
        except (AttributeError, TypeError, ValueError):
            return False, 'live_calibration_or_record_missing'
        identity = guard.get('calibration_identity')
        if (not isinstance(identity, str) or not identity
                or live_identity != identity
                or self.stereo_calibration_identity != identity
                or getattr(source_record, 'calibration_identity', None) != identity
                or getattr(target_record, 'calibration_identity', None) != identity
                or getattr(source_record, 'frame', None) != source_frame
                or getattr(target_record, 'frame', None) != target_frame):
            return False, 'calibration_or_endpoint_epoch_changed'
        # Re-run the production guard after stereo sampling. This catches a
        # sampler or concurrent update that mutates source geometry, endpoint
        # ownership, status, pose, or calibration during row validation.
        fresh = self._hard_reference_retention_guard(
            target_frame, source_frame, previous_frame, measured_frame, verified,
            source_pose, reference_pose, source_revision,
            tuple(self.current_supported_stereo.image_size))
        if fresh.get('eligible') is not True:
            return False, 'fresh_reference_guard_rejected:' + str(fresh.get('reason'))
        if (int(self.map.geometry_revision) != geometry_revision
                or int(self.map.revision) != source_revision):
            return False, 'map_or_geometry_epoch_changed'
        return True, 'independent_pose_transaction_current'

    def _validate_stereo_associations_at_pose(
        self, pose, associations, accepted_tracks, pixels, size, final_solve_positions,
        previous_inlier_misses=None, retention_context=None,
    ):
        """Keep only existing map links that fit a selected fixed stereo pose.

        This is deliberately a validation pass: it never estimates or changes the
        selected pose. Current right-image measurements are sampled at each
        accepted observation, including subpixel flow coordinates.
        """
        track_pixels = {}
        all_track_pixels = {}
        for landmark_id, pixel in accepted_tracks:
            track_pixels.setdefault(int(landmark_id), np.asarray(pixel, float).copy())
            all_track_pixels.setdefault(int(landmark_id), []).append(
                np.asarray(pixel, float).copy())
        feature_for_landmark = {}
        features_for_landmark = {}
        for feature, landmark_id in associations.items():
            feature_for_landmark.setdefault(int(landmark_id), int(feature))
            features_for_landmark.setdefault(int(landmark_id), []).append(int(feature))
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
        errors_by_id = {}
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
                errors_by_id[int(landmark_id)] = (
                    observed_array[i].copy(), left_error, right_error)

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

        # This path can preserve mapping observations only. It never promotes
        # the independent pose into pose-support eligibility, and is reachable
        # only when the original aggregate failure was spatial coverage alone.
        mapping_retained_ids = set()
        mapping_retention_allowed = False
        mapping_reason = 'disabled_or_not_coverage_only'
        if (not eligible and failed_gates == ['insufficient_spatial_coverage']
                and getattr(self.config, 'stereo_mapping_observation_retention', False)):
            context_ok, mapping_reason = self._stereo_mapping_retention_context_is_current(
                pose, retention_context)
            if context_ok:
                # Physical identities are exact float32 coordinates. Repeated
                # identical LK rows collapse to one observation; ambiguous
                # landmark or pixel ownership is rejected for every claimant.
                detector_keys = {}
                flow_keys = {}
                ids_by_key = {}
                ambiguous = set()
                for landmark_id in candidate_ids:
                    keys = set()
                    for feature in features_for_landmark.get(landmark_id, []):
                        if 0 <= feature < len(pixels):
                            point = np.asarray(pixels[feature], np.float32)
                            if point.shape == (2,) and np.isfinite(point).all():
                                keys.add(self._physical_pixel_key(point))
                    detector_keys[landmark_id] = keys
                    flow_rows = all_track_pixels.get(landmark_id, [])
                    flow_key_set = set()
                    for point in flow_rows:
                        if point.shape == (2,) and np.isfinite(point).all():
                            flow_key_set.add(self._physical_pixel_key(point))
                    flow_keys[landmark_id] = flow_key_set
                    if len(flow_key_set) > 1 or len(keys) > 1:
                        ambiguous.add(landmark_id)
                    for key in keys | flow_key_set:
                        ids_by_key.setdefault(key, set()).add(landmark_id)
                for owners in ids_by_key.values():
                    if len(owners) > 1:
                        ambiguous.update(owners)

                try:
                    excluded_ids = {int(value) for value in
                                    retention_context.get('excluded_landmark_ids', ())}
                    excluded_pixels = {
                        self._physical_pixel_key(np.asarray(value, np.float32))
                        for value in retention_context.get('excluded_target_pixels', ())
                        if np.asarray(value).shape == (2,)
                        and np.isfinite(np.asarray(value, float)).all()
                    }
                except (TypeError, ValueError, OverflowError):
                    excluded_ids, excluded_pixels = set(candidate_ids), set()
                    mapping_reason = 'malformed_held_out_exclusion'

                for landmark_id in candidate_ids:
                    if (landmark_id in ambiguous or landmark_id in excluded_ids
                            or landmark_id not in errors_by_id):
                        continue
                    flow_key_set = flow_keys.get(landmark_id, set())
                    detector_key_set = detector_keys.get(landmark_id, set())
                    if flow_key_set:
                        canonical_key = next(iter(flow_key_set))
                    elif len(detector_key_set) == 1:
                        canonical_key = next(iter(detector_key_set))
                    else:
                        continue
                    tested_key = self._physical_pixel_key(errors_by_id[landmark_id][0])
                    if (canonical_key != tested_key or canonical_key in excluded_pixels
                            or bool(detector_key_set & excluded_pixels)):
                        continue
                    mapping_retained_ids.add(landmark_id)

                # Re-check every non-spatial gate on the exact rows that will
                # actually be written to the map.
                selected_rows = [errors_by_id[lid] for lid in candidate_ids
                                 if lid in mapping_retained_ids]
                map_count = len(mapping_retained_ids)
                map_ratio = map_count / denominator if denominator > 0 else 0.0
                map_pixels = np.asarray([row[0] for row in selected_rows], float).reshape(-1, 2)
                map_coverage = coverage(map_pixels, size)
                map_left = np.asarray([row[1] for row in selected_rows], float)
                map_right = np.asarray([row[2] for row in selected_rows], float)
                map_med_left = float(np.median(map_left)) if len(map_left) else None
                map_med_right = float(np.median(map_right)) if len(map_right) else None
                if (map_count >= self.config.min_inliers and denominator > 0
                        and map_ratio >= 0.25 and map_med_left is not None
                        and map_med_left <= 1.5 and map_med_right is not None
                        and map_med_right <= 1.5 and map_coverage < 3):
                    # Recheck the certificate after constructing the exact row
                    # set, immediately before applying the retention outcome.
                    context_ok, mapping_reason = self._stereo_mapping_retention_context_is_current(
                        pose, retention_context)
                    mapping_retention_allowed = bool(context_ok)
                else:
                    mapping_reason = 'canonical_rows_fail_nonspatial_aggregate_gates'
                    mapping_retained_ids.clear()
            elif self.config.stereo_mapping_observation_retention:
                mapping_reason = 'independent_pose_context_rejected:' + mapping_reason
        with self.map.lock:
            if mapping_retention_allowed:
                # Keep the final certificate check inside the same critical
                # section as the once-per-frame miss outcome.
                context_ok, mapping_reason = self._stereo_mapping_retention_context_is_current(
                    pose, retention_context)
                mapping_retention_allowed = bool(context_ok)
                if not mapping_retention_allowed:
                    mapping_retained_ids.clear()
            if previous_inlier_misses is not None:
                # _track provisionally reset these inliers before arbitration.
                # Restore one outcome from their pre-frame count so repeated
                # independent-pose rejection can reach the normal cull limit.
                for landmark_id, previous_misses in previous_inlier_misses.items():
                    landmark = self.map.landmarks.get(landmark_id)
                    if landmark is not None:
                        landmark.misses = (
                            0 if (eligible and landmark_id in retained_ids)
                            or (mapping_retention_allowed and landmark_id in mapping_retained_ids)
                            else int(previous_misses) + 1
                        )
            else:
                # Direct helper calls without a successful map solve do not
                # have a pre-frame snapshot to reconcile.
                for landmark_id in candidate_ids:
                    landmark = self.map.landmarks.get(landmark_id)
                    if landmark is not None:
                        landmark.misses = (
                            0 if (eligible and landmark_id in retained_ids)
                            or (mapping_retention_allowed and landmark_id in mapping_retained_ids)
                            else max(landmark.misses, 1))
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
        elif mapping_retention_allowed:
            kept_tracks = []
            for landmark_id in candidate_ids:
                if landmark_id not in mapping_retained_ids:
                    continue
                flow_rows = all_track_pixels.get(landmark_id, [])
                if flow_rows:
                    # Exact duplicate LK rows represent one physical sample.
                    kept_tracks.append((int(landmark_id), np.asarray(flow_rows[0]).copy()))
            kept_associations = {
                int(feature): int(landmark_id)
                for feature, landmark_id in associations.items()
                if int(landmark_id) in mapping_retained_ids
                and 0 <= int(feature) < len(pixels)
                and self._physical_pixel_key(pixels[int(feature)])
                    in detector_keys.get(int(landmark_id), set())
            }
        else:
            kept_associations, kept_tracks = {}, []
        diagnostics = {
            'eligible': bool(eligible), 'reason': reason,
            'pose_support_eligible': bool(eligible),
            'aggregate_failure': failed_gates[0] if failed_gates else None,
            'mapping_observation_retention_allowed': bool(mapping_retention_allowed),
            'mapping_observation_retention_reason': mapping_reason,
            'mapping_retained_landmarks': (len(mapping_retained_ids)
                                           if mapping_retention_allowed else 0),
            'mapping_observation_role': 'tracking_fit_consumed',
            'held_out_rows_reused': False,
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
        physical_canonical=False,
    ):
        """Build mutual descriptor pairs with the selected physical identity policy.

        Depth availability is intentionally not a match filter: the existing
        reference estimator independently gates forward and reverse PnP rows.
        The reserved-conflict fallback keeps its conservative drop-alias policy;
        ordinary raw retry keeps all appearances through matching, then
        canonicalizes unambiguous exact-pixel edges and depth geometry.
        """
        report = {
            'attempted': True,
            'fit_source': ('raw_supported_reference_retry' if physical_canonical
                           else 'full_supported_reference'),
            'fit_depth_policy': self._depth_fit_policy_label(),
            'held_out_arbitration_used': False,
        }
        if physical_canonical:
            report['physical_identity'] = 'exact_float32_pixels_canonicalized'

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
        if physical_canonical:
            pairs = self._collapse_physical_matches(
                pairs, source.pixels, target.pixels)
            source_points, source_usable = self._canonical_physical_geometry_points(
                source.pixels, source.points)
            target_points, target_usable = self._canonical_physical_geometry_points(
                target.pixels, target.points)
            if len(pairs):
                pairs = pairs[source_usable[pairs[:, 0]] & target_usable[pairs[:, 1]]]
            report['finite_supported_geometry_rows'] = {
                'source': int(np.isfinite(source_points).all(axis=1).sum()),
                'target': int(np.isfinite(target_points).all(axis=1).sum()),
            }
        elif raw_count:
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
        physical_canonical=False,
    ):
        pairs, report = self._full_supported_match_pairs(
            source, target, previous_index, index, previous_frame, measured_frame, size,
            physical_canonical=physical_canonical)
        if pairs is None:
            return None, report
        source_points = source.points
        target_points = target.points
        if physical_canonical:
            source_points, _ = self._canonical_physical_geometry_points(
                source.pixels, source.points)
            target_points, _ = self._canonical_physical_geometry_points(
                target.pixels, target.points)
        source_frame = StereoLoopFrame(source.pixels, source_points,
                                       source.descriptors, source.image_size)
        target_frame = StereoLoopFrame(target.pixels, target_points,
                                       target.descriptors, target.image_size)
        try:
            verified = self.profile.call(
                ('stereo_pose_raw_supported_retry_fit' if physical_canonical
                 else 'stereo_pose_full_supported_fallback_fit'), estimate_stereo_reference,
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

    def _estimate_raw_supported_reference_retry(
        self, source, target, previous_index, index, previous_frame, measured_frame, size,
    ):
        """Retry a failed configured reference fit on immutable raw stereo records."""
        report = {
            'attempted': True,
            'gate': 'configured_reference_failed_and_reserved_context_unavailable',
            'configured_reference_rejection': 'configured_reference_returned_none',
            'reservation_context_available': False,
            'held_out_arbitration_used': False,
            'fit_source': 'raw_supported_reference_retry',
            'fit_depth_policy': self._depth_fit_policy_label(),
            'prediction_seed_supplied': False,
        }

        def fail(reason, **fields):
            return None, {**report, 'eligible': False, 'reason': reason, **fields}, None, None, None

        if not self.config.stereo_raw_reference_retry or self.stereo is None:
            return fail('raw_reference_retry_disabled')
        try:
            with self.map.lock:
                if (not isinstance(previous_index, (int, np.integer))
                        or isinstance(previous_index, (bool, np.bool_))
                        or not 0 <= int(previous_index) < len(self.map.poses)):
                    return fail('invalid_source_pose_index')
                source_pose_raw = np.asarray(self.map.poses[int(previous_index)])
                source_revision = self.map.revision
                if (not isinstance(source_revision, (int, np.integer))
                        or isinstance(source_revision, (bool, np.bool_))
                        or not self._is_proper_se3(source_pose_raw)):
                    return fail('invalid_source_map_epoch')
                source_pose = np.asarray(source_pose_raw, float).copy()
        except (AttributeError, TypeError, ValueError, IndexError):
            return fail('missing_source_map_epoch')

        try:
            verified, fit_report = self._estimate_full_supported_reference(
                source, target, previous_index, index, previous_frame,
                measured_frame, size, physical_canonical=True)
        except (cv.error, np.linalg.LinAlgError, TypeError, ValueError, IndexError):
            verified, fit_report = None, {
                'eligible': False, 'reason': 'raw_supported_reference_fit_error',
                'reverse_checked': False,
            }
        report = {**report, **fit_report}
        if verified is None:
            return None, {**report, 'eligible': False,
                          'reason': fit_report.get('reason', 'raw_supported_reference_failed')}, None, None, None

        try:
            with self.map.lock:
                reference_pose = source_pose @ np.asarray(verified['measurement'], float)
                source_guard = self._hard_reference_retention_guard(
                    index, previous_index, previous_frame, measured_frame,
                    verified, source_pose, reference_pose, source_revision, size)
        except (AttributeError, TypeError, ValueError, IndexError, np.linalg.LinAlgError):
            source_guard = {'eligible': False, 'reason': 'raw_reference_source_guard_error'}
            reference_pose = None
        if not source_guard.get('eligible'):
            return None, {**report, 'eligible': False,
                          'reason': source_guard.get('reason', 'raw_reference_source_guard_failed'),
                          'source_guard': source_guard,
                          'reverse_checked': True}, None, None, None
        return (verified,
                {**report, 'eligible': True,
                 'reason': 'raw_supported_reference_verified',
                 'source_guard': source_guard,
                 'source_frame': int(previous_index), 'target_frame': int(index),
                 'source_map_revision': int(source_revision),
                 'reverse_checked': True},
                source_pose, int(source_revision), reference_pose)

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

        if (not (self.config.stereo_pose_arbitration
                 or self.config.stereo_raw_reference_retry) or self.stereo is None
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
        live_source_pose_raw = np.asarray(poses[previous_index])
        if (not self._is_proper_se3(live_source_pose_raw)
                or not np.array_equal(np.asarray(live_source_pose_raw, float),
                                      np.asarray(source_pose, float))):
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

    def _emit_bundle_diagnostic(self, frame, phase, payload, image_size):
        """Isolate diagnostic I/O from estimator control flow and solver state."""
        writer = self.bundle_diagnostic_writer
        if writer is None:
            return
        try:
            writer.emit(frame=frame, phase=phase,
                        payload=_owned_diagnostic_value(payload),
                        image_size=tuple(image_size))
        except Exception as error:
            self.bundle_diagnostic_errors.append({
                "frame": int(frame), "phase": str(phase),
                "error_type": type(error).__name__,
            })

    def _skip_bundle_diagnostic(self, frame, reason):
        writer = self.bundle_diagnostic_writer
        if writer is None:
            return
        try:
            writer.mark_skipped(int(frame), str(reason))
        except Exception as error:
            self.bundle_diagnostic_errors.append({
                "frame": int(frame), "phase": "skip",
                "error_type": type(error).__name__,
            })

    def _source_history_cohort_diagnostic_state(self, current_ba_frame):
        """Snapshot the BA69 source-history cohort without changing map state."""
        packet = self._source_history_diagnostic_cohort
        if not isinstance(packet, dict):
            return None
        cohort = packet.get("source_history_observations")
        if not isinstance(cohort, dict):
            return None
        rows = cohort.get("rows", [])
        source_frame = cohort.get("source_frame")
        if (not isinstance(source_frame, int) or isinstance(source_frame, bool)
                or source_frame < 0 or not isinstance(rows, list)):
            raise ValueError("Malformed diagnostic source-history cohort")
        cohort_ids = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Malformed source-history cohort row")
            landmark_id = row.get("landmark_id")
            if (not isinstance(landmark_id, int) or isinstance(landmark_id, bool)
                    or landmark_id < 0):
                raise ValueError("Malformed source-history landmark ID")
            cohort_ids.append(landmark_id)
        if len(set(cohort_ids)) != len(cohort_ids):
            raise ValueError("Duplicate source-history cohort landmark ID")

        with self.map.lock:
            endpoint = self._source_history_endpoint(source_frame)
            if endpoint is None:
                source_pose = None
                source_status = None
                source_anchor_id = None
                source_anchor_pose = None
                source_reason = "source_endpoint_unavailable"
            else:
                source_pose = endpoint["pose"].tolist()
                source_status = endpoint["status"]
                source_anchor_id = endpoint["anchor_keyframe_id"]
                source_anchor_pose = endpoint["anchor_pose"].tolist()
                source_reason = None
            # The prepared table binds the full owned-bundle calibration,
            # including disparity offset; compare the same identity here.
            current_calibration = self._live_owned_bundle_calibration()
            same_calibration = (
                current_calibration is not None
                and current_calibration == cohort.get("source_calibration_identity")
            )
            cohort_rows = []
            for landmark_id in cohort_ids:
                landmark = self.map.landmarks.get(landmark_id)
                if landmark is None:
                    cohort_rows.append({
                        "landmark_id": landmark_id,
                        "available": False,
                        "reason": "culled_or_missing",
                        "current_world_position": None,
                        "current_anchor_id": None,
                        "actual_observations": [],
                    })
                    continue
                position = np.asarray(landmark.position)
                if (np.iscomplexobj(position) or position.shape != (3,)
                        or not np.issubdtype(position.dtype, np.number)
                        or not np.isfinite(position).all()):
                    # Let the diagnostic sink record an error; never turn an
                    # invalid geometric state into a plausible nullable row.
                    raise ValueError("Non-finite source-history cohort world position")
                observation_rows = []
                for keyframe_id, observation in sorted(landmark.observations.items()):
                    keyframe_id = int(keyframe_id)
                    keyframe = self.map.keyframes.get(keyframe_id)
                    pixel = np.asarray(observation.pixel)
                    if (np.iscomplexobj(pixel) or pixel.shape != (2,)
                            or not np.issubdtype(pixel.dtype, np.number)
                            or not np.isfinite(pixel).all()):
                        raise ValueError("Invalid stored cohort observation pixel")
                    raw_right = observation.right_u
                    right_valid = False
                    right_value = None
                    if raw_right is not None:
                        right_array = np.asarray(raw_right)
                        if (not np.iscomplexobj(right_array)
                                and np.issubdtype(right_array.dtype, np.number)
                                and right_array.shape == ()
                                and np.isfinite(right_array).item()):
                            right_value = float(right_array)
                            right_valid = True
                    observation_rows.append({
                        "keyframe_id": keyframe_id,
                        "image_frame": None if keyframe is None else int(keyframe.frame),
                        "pixel": np.asarray(pixel, dtype=np.float32).tolist(),
                        "right_u": right_value,
                        "right_u_valid": right_valid,
                        "dimensions": 3 if right_valid else 2,
                    })
                cohort_rows.append({
                    "landmark_id": landmark_id,
                    "available": True,
                    "reason": None,
                    "current_world_position": np.asarray(position, dtype=float).tolist(),
                    "current_anchor_id": int(landmark.anchor),
                    "actual_observations": observation_rows,
                })
            eligible_source = (
                endpoint is not None
                and source_status in ("tracking", "relocalized")
                and cohort.get("source_snapshot_valid") is True
                and cohort.get("status") == "active"
                and bool(cohort_ids)
            )
            return _owned_diagnostic_value({
                "schema": "source_history_cohort_state_v1",
                "cohort_BA_frame": packet["cohort_BA_frame"],
                "current_BA_frame": int(current_ba_frame),
                "cohort_revision": cohort.get("revision"),
                "cohort_geometry_revision": cohort.get("geometry_revision"),
                "revision": int(self.map.revision),
                "geometry_revision": int(self.map.geometry_revision),
                "source_frame": source_frame,
                "source_status": source_status,
                "source_anchor_id": source_anchor_id,
                "source_camera_to_world": source_pose,
                "source_anchor_camera_to_world": source_anchor_pose,
                "calibration_identity": current_calibration,
                "cohort_calibration_identity": cohort.get("source_calibration_identity"),
                "cross_phase_comparison_eligible": bool(eligible_source and same_calibration),
                "cross_phase_ineligible_reason": (
                    source_reason if endpoint is None else
                    ("source_status_not_accepted"
                     if source_status not in ("tracking", "relocalized") else
                     ("source_history_table_ineligible"
                      if cohort.get("source_snapshot_valid") is not True
                      or cohort.get("status") != "active" or not cohort_ids else
                      (None if same_calibration else "calibration_identity_changed")))
                ),
                "rows": cohort_rows,
            })

    def bundle_diagnostics_manifest(self):
        writer = self.bundle_diagnostic_writer
        if writer is None:
            return {"enabled": False, "selected_frames": []}
        try:
            return _owned_diagnostic_value(writer.manifest())
        except Exception as error:
            self.bundle_diagnostic_errors.append({
                "frame": None, "phase": "manifest",
                "error_type": type(error).__name__,
            })
            return {
                "schema_version": 1,
                "enabled": True,
                "selected_frames": list(writer.frames),
                "frames": [
                    {"frame": int(frame), "status": "error",
                     "reason": "manifest_write_failed", "phases": {},
                     "errors": [{"phase": "manifest",
                                 "error_type": type(error).__name__}]}
                    for frame in writer.frames
                ],
            }

    def tracking_diagnostics_manifest(self):
        writer = self.tracking_diagnostics_writer
        if writer is None:
            return {"enabled": False, "selected_frames": []}
        try:
            manifest = _owned_diagnostic_value(writer.manifest())
            if self.tracking_diagnostic_errors:
                manifest.setdefault("errors", []).extend(
                    _owned_diagnostic_value(self.tracking_diagnostic_errors)
                )
            return manifest
        except Exception as error:
            self.tracking_diagnostic_errors.append({
                "frame": None, "phase": "manifest",
                "error_type": type(error).__name__,
            })
            return {
                "enabled": True, "selected_frames": list(writer.frames),
                "frames": [], "errors": list(self.tracking_diagnostic_errors),
            }

    def _tracking_trace_enabled(self, frame):
        writer = getattr(self, "tracking_diagnostics_writer", None)
        if writer is None:
            return False
        try:
            return bool(writer.should_capture(int(frame)))
        except Exception as error:
            self.tracking_diagnostic_errors.append({
                "frame": int(frame), "phase": "select",
                "error_type": type(error).__name__,
            })
            return False

    def _tracking_trace_event(self, frame, phase, payload):
        writer = getattr(self, "tracking_diagnostics_writer", None)
        if writer is None:
            return False
        try:
            ok = writer.record_event(int(frame), str(phase), payload)
            if not ok:
                self.tracking_diagnostic_errors.append({
                    "frame": int(frame), "phase": str(phase),
                    "error_type": "EventRejected",
                })
            return bool(ok)
        except Exception as error:
            self.tracking_diagnostic_errors.append({
                "frame": int(frame), "phase": str(phase),
                "error_type": type(error).__name__,
            })
            try:
                writer.record_error(int(frame), str(phase), type(error).__name__)
            except Exception:
                pass
            return False

    def _tracking_trace_frame_header(self, frame):
        try:
            with self.map.lock:
                start = max(0, len(self.map.poses) - 4)
                limit = min(len(self.map.poses), len(self.map.statuses),
                            len(self.map.pose_anchors), len(self.map.relative_poses))
                recent = [
                    {"frame": int(fid), "status": str(self.map.statuses[fid]),
                     "anchor_keyframe_id": self.map.pose_anchors[fid],
                     "camera_to_world": np.asarray(self.map.poses[fid]).copy(),
                     "relative_pose": np.asarray(self.map.relative_poses[fid]).copy()}
                    for fid in range(start, limit)
                ]
                return {
                    "schema": "tracking_frame_start_v1", "frame": int(frame),
                    "revision": int(self.map.revision),
                    "geometry_revision": int(self.map.geometry_revision),
                    "metric": bool(self.map.metric),
                    "current_pose_count": len(self.map.poses),
                    "current_keyframe_count": len(self.map.keyframes),
                    "current_landmark_count": len(self.map.landmarks),
                    "recent_frame_states": recent,
                    "rows": [], "input_row_count": 0,
                    "camera_matrix": self.K.copy(),
                    "baseline": None if self.stereo is None else float(self.stereo.baseline),
                    "disparity_offset": None if self.stereo is None else float(self.stereo.disparity_offset),
                }
        except Exception as error:
            return {"schema": "tracking_frame_start_v1", "frame": int(frame),
                    "snapshot_valid": False, "snapshot_error_type": type(error).__name__,
                    "rows": [], "input_row_count": 0}

    def _tracking_trace_frame_state(self, frame, *, include_landmarks=False,
                                    landmark_ids=None, stage="frame_state"):
        """Copy a bounded map epoch/ownership snapshot under the state lock."""
        try:
            with self.map.lock:
                poses = [
                    {"frame": int(fid), "status": str(self.map.statuses[fid]),
                     "anchor_keyframe_id": self.map.pose_anchors[fid],
                     "camera_to_world": np.asarray(self.map.poses[fid]).copy(),
                     "relative_pose": np.asarray(self.map.relative_poses[fid]).copy()}
                    for fid in range(min(len(self.map.poses), len(self.map.statuses),
                                         len(self.map.pose_anchors), len(self.map.relative_poses)))
                ]
                keyframes = [
                    {"keyframe_id": int(kid), "frame": int(kf.frame),
                     "camera_to_world": np.asarray(kf.pose).copy()}
                    for kid, kf in sorted(self.map.keyframes.items())
                ]
                if landmark_ids is None:
                    ids = sorted(self.map.landmarks) if include_landmarks else []
                else:
                    ids = sorted({int(value) for value in landmark_ids
                                  if int(value) in self.map.landmarks})
                rows = []
                for lid in ids:
                    landmark = self.map.landmarks[lid]
                    rows.append({
                        "record_type": "landmark", "landmark_id": int(lid),
                        "position": np.asarray(landmark.position).copy(),
                        "persistent_anchor_keyframe_id": int(landmark.anchor),
                        "misses": int(landmark.misses),
                    })
                    for keyframe_id, observation in sorted(landmark.observations.items()):
                        keyframe = self.map.keyframes.get(int(keyframe_id))
                        feature_ids = []
                        feature_pixels = []
                        if keyframe is not None:
                            feature_ids = [int(value) for value in np.flatnonzero(
                                np.asarray(keyframe.landmark_ids) == int(lid)
                            )]
                            feature_pixels = [
                                np.asarray(keyframe.pixels[index], np.float32).copy()
                                for index in feature_ids
                            ]
                        right_u = observation.right_u
                        rows.append({
                            "record_type": "observation", "landmark_id": int(lid),
                            "keyframe_id": int(keyframe_id),
                            "frame_id": (None if keyframe is None else int(keyframe.frame)),
                            "pixel": np.asarray(observation.pixel, np.float32).copy(),
                            "right_u": (None if right_u is None else float(right_u)),
                            "right_u_valid": bool(right_u is not None and np.isfinite(right_u)),
                            "keyframe_feature_ids": feature_ids,
                            "keyframe_feature_pixels": feature_pixels,
                        })
                return {
                    "schema": "tracking_map_state_v1", "frame": int(frame),
                    "stage": str(stage), "revision": int(self.map.revision),
                    "geometry_revision": int(self.map.geometry_revision),
                    "metric": bool(self.map.metric),
                    "next_landmark_id": int(self.map.next_landmark),
                    "pose_rows": poses, "keyframe_rows": keyframes,
                    "rows": rows, "input_row_count": len(rows),
                    "landmark_count": len(self.map.landmarks),
                    "captured_landmark_count": len(ids),
                    "observation_count": int(sum(len(self.map.landmarks[lid].observations)
                                                  for lid in ids)),
                    "stereo_motion": [
                        {"source_frame": int(pair[0]), "target_frame": int(pair[1]),
                         "measurement": np.asarray(value).copy()}
                        for pair, value in sorted(self.map.stereo_motion.items())
                    ],
                }
        except Exception as error:
            self.tracking_diagnostic_errors.append({
                "frame": int(frame), "phase": str(stage),
                "error_type": type(error).__name__,
            })
            return {
                "schema": "tracking_map_state_v1", "frame": int(frame),
                "stage": str(stage), "snapshot_valid": False,
                "snapshot_error_type": type(error).__name__,
                "snapshot_complete": False, "rows": [], "input_row_count": 0,
            }

    def _tracking_trace_reference_context(self, frame, report, info):
        arbitration = self._arbitration_context
        if not isinstance(arbitration, dict):
            return {
                "reference_pool_status": "unknown_uninstrumented",
                "fit_consumption_complete": False,
                "unused_claim_allowed": False,
                "reason": "no_arbitration_pool_ledger",
            }
        previous = arbitration.get("previous")
        current = arbitration.get("current")
        training = arbitration.get("training_rows")
        fit = arbitration.get("fit_pairs_snapshot")
        held = arbitration.get("held_pairs")
        captured = isinstance(previous, SupportedStereoFrame) and isinstance(
            current, SupportedStereoFrame) and isinstance(training, dict)
        if not captured:
            return {
                "reference_pool_status": "unknown_uninstrumented",
                "fit_consumption_complete": False,
                "unused_claim_allowed": False,
                "reason": "reference_row_pool_incomplete",
            }

        def owned_frame(record):
            right = np.asarray(record.right_u, np.float32)
            points = np.asarray(record.points, np.float32)
            return {
                "frame_id": int(record.frame),
                "image_size": tuple(record.image_size),
                "pixels": np.asarray(record.pixels, np.float32).copy(),
                "right_u": [None if not np.isfinite(value) else float(value) for value in right],
                "right_u_valid": np.isfinite(right).tolist(),
                "points": [[None if not np.isfinite(value) else float(value) for value in row]
                           for row in points],
                "points_valid": np.isfinite(points).all(axis=1).tolist(),
                "landmark_id_claims": np.asarray(record.landmark_ids, np.int64).copy(),
                "calibration_identity": str(record.calibration_identity),
            }
        choice = report if isinstance(report, dict) else {}
        fit_disjoint = np.asarray(held if held is not None else [], dtype=np.int64).reshape(-1, 2)
        fit_pairs = np.asarray(fit if fit is not None else [], dtype=np.int64).reshape(-1, 2)
        train_pairs = np.asarray(training.get("fit_pairs", []), dtype=np.int64).reshape(-1, 2)
        pairs_match = np.array_equal(fit_pairs, train_pairs)
        arbitration_choice = choice.get("choice")
        full_consumed = bool(
            isinstance(info.get("full_supported_reference_fallback"), dict)
            or info.get("full_supported_reference_attempted", False)
        )
        selection_consumed = bool(
            "choice" in choice and arbitration_choice is not None
            and ("score" in choice or "independent" in choice or "map" in choice)
        )
        fit_rows = np.asarray(training.get("forward_inlier_pairs", []), dtype=np.int64).reshape(-1, 2)
        reverse_rows = np.asarray(training.get("reverse_inlier_pairs", []), dtype=np.int64).reshape(-1, 2)
        return {
            "reference_pool_status": (
                "reserved_ledger_captured_other_reference_unknown"
                if pairs_match else "unknown_uninstrumented"
            ),
            "reserved_pool_complete": bool(pairs_match),
            "fit_consumption_complete": False,
            "unused_claim_allowed": False,
            "reason": None if pairs_match else "fit_pair_ledger_mismatch",
            "source": owned_frame(previous), "target": owned_frame(current),
            "fit_pairs": fit_pairs, "forward_inlier_pairs": fit_rows,
            "reverse_inlier_pairs": reverse_rows,
            "fit_disjoint_rows": fit_disjoint,
            "fit_disjoint_role": "fit_disjoint",
            "selection_consumed": bool(selection_consumed),
            "estimator_unused": False,
            "holdout_status": ("fit_disjoint_selection_consumed" if selection_consumed
                               else "selection_use_unknown"),
            "arbitration_report": dict(choice),
            "pose_source": info.get("pose_source"),
            "full_supported_reference_consumed": bool(full_consumed),
        }

    def _tracking_trace_selected(self, frame, phase, pose, status, associations,
                                 pixels, right_u, arbitration_report, info):
        """Copy selected evidence without adding any measurement or fitting call."""
        if not self._tracking_trace_enabled(frame):
            return
        try:
            payload = self._tracking_trace_frame_state(
                frame, include_landmarks=True, stage=phase)
            payload.update(
                accepted_pose=np.asarray(pose).copy(), tracking_status=str(status),
                pose_source=info.get("pose_source"),
                accepted_associations=[{"detector_feature_index": int(feature),
                                        "landmark_id": int(lid)}
                                       for feature, lid in sorted(associations.items())],
                accepted_tracks=[{"landmark_id": int(lid),
                                  "pixel": np.asarray(pixel, np.float32).copy()}
                                 for lid, pixel in self.accepted_tracks],
                detector_pixels=np.asarray(pixels, np.float32).copy(),
                detector_right_u=[None if not np.isfinite(value) else float(value)
                                  for value in right_u],
                reference_context=self._tracking_trace_reference_context(
                    frame, arbitration_report, info),
                fit_consumption_complete=False,
                unused_evidence_certified=False,
            )
            self._tracking_trace_event(frame, phase, payload)
            if phase == "selected_pre_keyframe":
                self._tracking_trace_selected_emitted = True
        except Exception as error:
            self.tracking_diagnostic_errors.append({
                "frame": int(frame), "phase": phase, "error_type": type(error).__name__})

    def _bundle_diagnostic_tracking_context(self, frame, image_size):
        """Snapshot existing predictor inputs without invoking prediction logic."""
        with self.map.lock:
            accepted = [
                index for index, status in enumerate(self.map.statuses)
                if status in ("tracking", "relocalized")
            ]
            recent = list(range(max(0, len(self.map.poses) - 4), len(self.map.poses)))
            verified = self.verified_stereo_motion
            return _owned_diagnostic_value({
                "frame": int(frame),
                "image_size": tuple(image_size),
                "revision": int(self.map.revision),
                "geometry_revision": int(self.map.geometry_revision),
                "stereo_calibration_identity": self.stereo_calibration_identity,
                "motion_prediction_source": self.motion_prediction_source,
                "latest_accepted_frame": accepted[-1] if accepted else None,
                "latest_accepted_pose": (
                    self.map.poses[accepted[-1]].copy() if accepted else None
                ),
                "recent_frame_poses": [
                    {"frame": index, "state": self.map.statuses[index],
                     "pose_anchor_keyframe_id": self.map.pose_anchors[index],
                     "camera_to_world": self.map.poses[index].copy()}
                    for index in recent
                ],
                "verified_stereo_motion": (
                    None if verified is None else {
                        "measurement": verified[0].copy(),
                        "target_frame": int(verified[1]),
                    }
                ),
            })

    def _bundle_diagnostic_finish_context(
        self, frame, image_size, report, window_keyframe_ids, motion_checks
    ):
        with self.map.lock:
            anchor_ids = set(int(value) for value in window_keyframe_ids)
            for check in motion_checks:
                if not isinstance(check, dict):
                    continue
                anchor_ids.update(
                    int(value) for value in check.get("anchor_keyframe_ids", [])
                    if value is not None
                )
            if 0 <= int(frame) < len(self.map.pose_anchors):
                current_anchor = self.map.pose_anchors[int(frame)]
                if current_anchor is not None:
                    anchor_ids.add(int(current_anchor))
            frame_ids = {int(frame)}
            for check in motion_checks:
                if isinstance(check, dict):
                    frame_ids.update(int(value) for value in check.get("frame_ids", []))
            affected_frames = [
                {"frame": index, "state": self.map.statuses[index],
                 "pose_anchor_keyframe_id": self.map.pose_anchors[index],
                 "camera_to_world": self.map.poses[index].copy()}
                for index in sorted(frame_ids)
                if 0 <= index < len(self.map.poses)
            ]
            affected_anchors = [
                {"keyframe_id": key, "frame": self.map.keyframes[key].frame,
                 "camera_to_world": self.map.keyframes[key].pose.copy()}
                for key in sorted(anchor_ids) if key in self.map.keyframes
            ]
            return _owned_diagnostic_value({
                "frame": int(frame),
                "image_size": tuple(image_size),
                "revision": int(self.map.revision),
                "geometry_revision": int(self.map.geometry_revision),
                "stereo_calibration_identity": self.stereo_calibration_identity,
                "applied": bool(report.get("applied", False)),
                "affected_frame_poses": affected_frames,
                "affected_anchor_poses": affected_anchors,
            })

    def _bundle_training_observations_context(
        self, frame, image_size, arbitration, arbitration_report, prepared_payload
    ):
        """Snapshot exact reserved stereo training rows for one requested BA frame."""
        base = {
            "schema": "stereo_training_rows_v1",
            "frame": int(frame),
            "status": "skipped",
            "eligible": False,
            "reason": "reserved_training_context_unavailable",
        }
        if not isinstance(arbitration, dict):
            report = arbitration_report if isinstance(arbitration_report, dict) else {}
            base["reason"] = report.get("reason", "reserved_training_context_unavailable")
            return base
        previous = arbitration.get("previous")
        current = arbitration.get("current")
        training = arbitration.get("training_rows")
        fit = arbitration.get("fit_pairs_snapshot")
        held = arbitration.get("held_pairs")
        if (previous is None or current is None or not isinstance(training, dict)
                or not isinstance(fit, np.ndarray) or not isinstance(held, np.ndarray)):
            base["reason"] = "training_row_ledger_unavailable"
            return base
        def integer_pairs(value):
            array = np.asarray(value)
            if (not np.issubdtype(array.dtype, np.integer)
                    or np.issubdtype(array.dtype, np.bool_)):
                raise ValueError("pair IDs must be integers")
            return np.asarray(array, dtype=np.int64).reshape(-1, 2)

        def integer_vector(value):
            array = np.asarray(value)
            if (not np.issubdtype(array.dtype, np.integer)
                    or np.issubdtype(array.dtype, np.bool_)):
                raise ValueError("row IDs must be integers")
            return np.asarray(array, dtype=np.int64).reshape(-1)
        try:
            fit = integer_pairs(fit)
            held = integer_pairs(held)
            training_fit = integer_pairs(training.get("fit_pairs", []))
            forward = integer_pairs(training.get("forward_inlier_pairs", []))
            reverse = integer_pairs(training.get("reverse_inlier_pairs", []))
            forward_rows = integer_vector(training.get("forward_fit_row_indices", []))
            reverse_rows = integer_vector(training.get("reverse_fit_row_indices", []))
        except (TypeError, ValueError, OverflowError):
            base["reason"] = "malformed_training_row_ledger"
            return base
        if not np.array_equal(fit, training_fit):
            base["reason"] = "captured_fit_pairs_do_not_match_estimator_input"
            return base
        if (len(forward) != len(forward_rows) or len(reverse) != len(reverse_rows)
                or np.any(forward_rows < 0) or np.any(forward_rows >= len(fit))
                or np.any(reverse_rows < 0) or np.any(reverse_rows >= len(fit))
                or (len(forward_rows) and not np.array_equal(fit[forward_rows], forward))
                or (len(reverse_rows) and not np.array_equal(fit[reverse_rows], reverse))):
            base["reason"] = "inlier_membership_ledger_mismatch"
            return base
        if (previous.pixels.ndim != 2 or previous.pixels.shape[1:] != (2,)
                or current.pixels.ndim != 2 or current.pixels.shape[1:] != (2,)
                or previous.points.shape != (len(previous.pixels), 3)
                or current.points.shape != (len(current.pixels), 3)
                or previous.right_u.shape != (len(previous.pixels),)
                or current.right_u.shape != (len(current.pixels),)
                or any(np.any(rows[:, 0] < 0) or np.any(rows[:, 0] >= len(previous.pixels))
                       or np.any(rows[:, 1] < 0) or np.any(rows[:, 1] >= len(current.pixels))
                       for rows in (fit, held, forward, reverse))):
            base["reason"] = "malformed_raw_supported_endpoint_records"
            return base

        def maybe_vector(value):
            try:
                array = np.asarray(value, dtype=float).reshape(-1)
            except (TypeError, ValueError, OverflowError):
                return {"values": [], "valid_mask": [], "valid": False}
            mask = np.isfinite(array)
            return {
                "values": [float(item) if valid else None
                           for item, valid in zip(array, mask)],
                "valid_mask": [bool(valid) for valid in mask],
                "valid": bool(mask.all()),
            }

        def feature_row(source_id, target_id, roles):
            source_id, target_id = int(source_id), int(target_id)
            return {
                "source_feature_id": source_id,
                "target_feature_id": target_id,
                "roles": roles,
                "source_pixel": maybe_vector(previous.pixels[source_id]),
                "target_pixel": maybe_vector(current.pixels[target_id]),
                "source_right_u": maybe_vector([previous.right_u[source_id]]),
                "target_right_u": maybe_vector([current.right_u[target_id]]),
                "source_camera_point": maybe_vector(previous.points[source_id]),
                "target_camera_point": maybe_vector(current.points[target_id]),
                "source_depth_source": self._depth_measurement_source_label(),
                "target_depth_source": self._depth_measurement_source_label(),
                # A SupportedStereoFrame link is a claim carried forward by
                # tracking; the authoritative observation is recorded below.
                "source_claimed_landmark_id": int(previous.landmark_ids[source_id]),
            }

        fit_edges = [tuple(map(int, row)) for row in fit]
        forward_edges = {tuple(map(int, row)) for row in forward}
        reverse_edges = {tuple(map(int, row)) for row in reverse}
        refinement_forward = forward_edges if training.get("refinement_attempted") else set()
        refinement_reverse = reverse_edges if training.get("refinement_attempted") else set()
        candidate_edges = sorted(forward_edges | reverse_edges)
        if len(candidate_edges) > 256:
            base.update({
                "status": "skipped",
                "reason": "candidate_inlier_row_cap_exceeded",
                "candidate_row_cap": 256,
                "candidate_row_count": int(len(candidate_edges)),
                "fit_pairs": [list(map(int, pair)) for pair in fit],
                "forward_inlier_pairs": [list(map(int, pair)) for pair in forward],
                "reverse_inlier_pairs": [list(map(int, pair)) for pair in reverse],
                "heldout_pairs": [list(map(int, pair)) for pair in held],
                "excluded_target_feature_ids": sorted(
                    map(int, arbitration.get("excluded_targets", set()))
                ),
                "excluded_landmark_ids": sorted(
                    map(int, arbitration.get("excluded_landmarks", set()))
                ),
            })
            return _owned_diagnostic_value(base)
        candidate_rows = [feature_row(source_id, target_id, {
            "fit_match": True,
            "forward_inlier": (source_id, target_id) in forward_edges,
            "reverse_inlier": (source_id, target_id) in reverse_edges,
            "refinement_forward_input": (source_id, target_id) in refinement_forward,
            "refinement_reverse_input": (source_id, target_id) in refinement_reverse,
            "image_factor_candidate": True,
        }) for source_id, target_id in candidate_edges]
        fit_match_rows = [feature_row(source_id, target_id, {
            "fit_match": True,
            "forward_inlier": (source_id, target_id) in forward_edges,
            "reverse_inlier": (source_id, target_id) in reverse_edges,
            "refinement_forward_input": (source_id, target_id) in refinement_forward,
            "refinement_reverse_input": (source_id, target_id) in refinement_reverse,
            "image_factor_candidate": (source_id, target_id) in forward_edges
            or (source_id, target_id) in reverse_edges,
        }) for source_id, target_id in fit_edges]
        held_rows = [feature_row(source_id, target_id, {
            "fit_match": False, "forward_inlier": False, "reverse_inlier": False,
            "held_out": True, "image_factor_candidate": False,
        }) for source_id, target_id in held]

        def pixel_keys(rows, record, column):
            return {self._physical_pixel_key(record.pixels[int(row[column])])
                    for row in rows}

        fit_source_keys = pixel_keys(fit, previous, 0)
        fit_target_keys = pixel_keys(fit, current, 1)
        held_source_keys = pixel_keys(held, previous, 0)
        held_target_keys = pixel_keys(held, current, 1)
        overlap = sorted(("source" if side == 0 else "target", tuple(key))
                         for side, keys in ((0, fit_source_keys & held_source_keys),
                                            (1, fit_target_keys & held_target_keys))
                         for key in keys)

        selected_ids = {
            int(item["landmark_id"])
            for item in prepared_payload.get("selection", {}).get("selected_landmarks", [])
            if isinstance(item, dict) and item.get("landmark_id") is not None
        }
        single_view_ids = {
            int(item["landmark_id"])
            for item in prepared_payload.get("single_view_propagations", [])
            if isinstance(item, dict) and item.get("landmark_id") is not None
        }

        with self.map.lock:
            revision = int(self.map.revision)
            geometry_revision = int(self.map.geometry_revision)
            target_keyframes = [
                (int(key), keyframe) for key, keyframe in self.map.keyframes.items()
                if int(keyframe.frame) == int(frame)
            ]
            prepared_source_pose = (
                self.map.poses[previous.frame].copy()
                if 0 <= int(previous.frame) < len(self.map.poses) else None
            )
            prepared_source_status = (
                self.map.statuses[previous.frame]
                if 0 <= int(previous.frame) < len(self.map.statuses) else None
            )
            prepared_source_anchor = (
                self.map.pose_anchors[previous.frame]
                if 0 <= int(previous.frame) < len(self.map.pose_anchors) else None
            )
            prepared_source_anchor_pose = (
                self.map.keyframes[prepared_source_anchor].pose.copy()
                if prepared_source_anchor in self.map.keyframes else None
            )
            target_pose = self.map.poses[int(frame)].copy() if int(frame) < len(self.map.poses) else None
            target_status = (self.map.statuses[int(frame)]
                             if int(frame) < len(self.map.statuses) else None)
            target_anchor = (self.map.pose_anchors[int(frame)]
                             if int(frame) < len(self.map.pose_anchors) else None)
            target_anchor_pose = (
                self.map.keyframes[target_anchor].pose.copy()
                if target_anchor in self.map.keyframes else None
            )
            calibration_matrix = np.array(self.K, copy=True)
            calibration_q = (np.array(self.stereo.Q, copy=True)
                             if self.stereo is not None else np.empty(0))
            calibration_baseline = (None if self.stereo is None
                                    else float(self.stereo.baseline))
            calibration_offset = (None if self.stereo is None
                                  else float(self.stereo.disparity_offset))
            digest = hashlib.sha256()
            for value in (calibration_matrix, calibration_q):
                digest.update(value.dtype.str.encode())
                digest.update(value.tobytes())
            if calibration_baseline is not None:
                digest.update(np.float64(calibration_baseline).tobytes())
            live_calibration = digest.hexdigest()
            calibration_matches = bool(
                live_calibration == previous.calibration_identity
                == current.calibration_identity == self.stereo_calibration_identity
            )

            def target_claims(target_feature_id):
                pixel = np.asarray(current.pixels[int(target_feature_id)], np.float32)
                pixel_key = self._physical_pixel_key(pixel)
                claims = []
                for keyframe_id, keyframe in target_keyframes:
                    keyframe_pixels = np.asarray(keyframe.pixels, np.float32)
                    indices = [i for i, value in enumerate(keyframe_pixels)
                               if self._physical_pixel_key(value) == pixel_key]
                    for feature_id in indices:
                        landmark_id = int(keyframe.landmark_ids[feature_id])
                        landmark = self.map.landmarks.get(landmark_id)
                        observation = (None if landmark is None else
                                       landmark.observations.get(keyframe_id))
                        obs_pixel = None if observation is None else maybe_vector(observation.pixel)
                        obs_right = (None if observation is None or observation.right_u is None
                                     else maybe_vector([observation.right_u]))
                        exact_pixel = bool(
                            observation is not None
                            and np.array_equal(np.asarray(observation.pixel, np.float32), pixel)
                        )
                        exact_right = bool(
                            observation is not None and observation.right_u is not None
                            and np.isfinite(current.right_u[target_feature_id])
                            and float(observation.right_u) == float(current.right_u[target_feature_id])
                        )
                        world_position = (None if landmark is None
                                          else maybe_vector(landmark.position))
                        certificate = None
                        if (landmark_id >= 0 and exact_pixel and exact_right
                                and observation is not None):
                            world = np.asarray(landmark.position, float)
                            certificate = {
                                "frame_id": int(keyframe.frame),
                                "keyframe_id": int(keyframe_id),
                                "keyframe_feature_id": int(feature_id),
                                "landmark_id": int(landmark_id),
                                "anchor_keyframe_id": int(landmark.anchor),
                                "pixel": np.asarray(observation.pixel, np.float32).tolist(),
                                "right_u": float(observation.right_u),
                                "world_position": world.tolist()
                                if world.shape == (3,) and np.isfinite(world).all() else None,
                            }
                        if landmark_id < 0:
                            classification = "unowned"
                        elif not exact_pixel or not exact_right:
                            classification = "ambiguous_or_measurement_mismatch"
                        elif landmark_id in selected_ids:
                            classification = "reusable_existing_selected_point"
                        elif landmark_id in single_view_ids:
                            classification = "existing_single_view_target"
                        else:
                            classification = "existing_owned_not_selected"
                        claims.append({
                            "keyframe_id": keyframe_id,
                            "keyframe_feature_id": int(feature_id),
                            "landmark_id": None if landmark_id < 0 else landmark_id,
                            "anchor_keyframe_id": (None if landmark is None else int(landmark.anchor)),
                            "world_position": world_position,
                            "observation_pixel": obs_pixel,
                            "observation_right_u": obs_right,
                            "exact_pixel_match": exact_pixel,
                            "exact_right_u_match": exact_right,
                            "classification": classification,
                            "existing_observation_certificate": certificate,
                        })
                if not claims:
                    return {"classification": "unowned", "claims": []}
                classes = {claim["classification"] for claim in claims}
                classification = (next(iter(classes)) if len(classes) == 1
                                  else "ambiguous_or_measurement_mismatch")
                return {"classification": classification, "claims": claims}

            claims_by_target = {
                int(target_id): target_claims(int(target_id))
                for target_id in sorted({int(pair[1]) for pair in fit_edges}
                                        | {int(pair[1]) for pair in held})
            }

            source_ids = sorted({int(pair[0]) for pair in fit_edges}
                                | {int(pair[0]) for pair in held})
            source_claims_by_feature = {}
            source_keyframes = [
                (int(key), keyframe) for key, keyframe in self.map.keyframes.items()
                if int(keyframe.frame) == int(previous.frame)
            ]
            for source_id in source_ids:
                source_pixel = np.asarray(previous.pixels[source_id], np.float32)
                source_key = self._physical_pixel_key(source_pixel)
                claimed_id = int(previous.landmark_ids[source_id])
                authoritative = []
                for keyframe_id, keyframe in source_keyframes:
                    for feature_id, pixel_value in enumerate(
                        np.asarray(keyframe.pixels, np.float32)
                    ):
                        if self._physical_pixel_key(pixel_value) != source_key:
                            continue
                        landmark_id = int(keyframe.landmark_ids[feature_id])
                        landmark = self.map.landmarks.get(landmark_id)
                        observation = (None if landmark is None else
                                       landmark.observations.get(keyframe_id))
                        exact_pixel = bool(
                            observation is not None
                            and np.array_equal(
                                np.asarray(observation.pixel, np.float32), source_pixel
                            )
                        )
                        exact_right = bool(
                            observation is not None and observation.right_u is not None
                            and np.isfinite(previous.right_u[source_id])
                            and float(observation.right_u) == float(previous.right_u[source_id])
                        )
                        world = (np.asarray(landmark.position, float)
                                 if landmark is not None else np.empty(0))
                        certificate = None
                        if (landmark_id >= 0 and exact_pixel and exact_right
                                and observation is not None):
                            certificate = {
                                "frame_id": int(keyframe.frame),
                                "keyframe_id": int(keyframe_id),
                                "keyframe_feature_id": int(feature_id),
                                "landmark_id": int(landmark_id),
                                "anchor_keyframe_id": int(landmark.anchor),
                                "pixel": np.asarray(observation.pixel, np.float32).tolist(),
                                "right_u": float(observation.right_u),
                                "world_position": world.tolist()
                                if world.shape == (3,) and np.isfinite(world).all() else None,
                            }
                        authoritative.append({
                            "keyframe_id": keyframe_id,
                            "keyframe_feature_id": int(feature_id),
                            "landmark_id": None if landmark_id < 0 else landmark_id,
                            "anchor_keyframe_id": (None if landmark is None
                                                   else int(landmark.anchor)),
                            "world_position": (None if landmark is None
                                               else maybe_vector(landmark.position)),
                            "observation_pixel": (None if observation is None
                                                  else maybe_vector(observation.pixel)),
                            "observation_right_u": (
                                None if observation is None or observation.right_u is None
                                else maybe_vector([observation.right_u])
                            ),
                            "exact_pixel_match": exact_pixel,
                            "exact_right_u_match": exact_right,
                            "existing_observation_certificate": certificate,
                        })
                track_claims = [
                    {"landmark_id": int(landmark_id),
                     "source_frame": int(previous.frame),
                     "pixel": maybe_vector(pixel),
                     "exact_pixel_match": bool(np.array_equal(
                         np.asarray(pixel, np.float32), source_pixel)),
                     "role": "previous_tracks_claim_not_observation_certificate"}
                    for landmark_id, pixel in self.previous_tracks
                    if int(landmark_id) == claimed_id
                ]
                source_claims_by_feature[source_id] = {
                    "claimed_landmark_id": None if claimed_id < 0 else claimed_id,
                    "authoritative_observations": authoritative,
                    "previous_tracks_claims": track_claims,
                    "landmark_world_position": (
                        maybe_vector(self.map.landmarks[claimed_id].position)
                        if claimed_id in self.map.landmarks else None
                    ),
                    "classification": (
                        "exact_source_observation"
                        if any(item["exact_pixel_match"] and item["exact_right_u_match"]
                               for item in authoritative)
                        else "tracker_claim_not_observation_certificate"
                        if track_claims else
                        "unowned" if claimed_id < 0 else
                        "ambiguous_or_measurement_mismatch"
                    ),
                }

        for row in fit_match_rows + candidate_rows + held_rows:
            row["source_map_ownership"] = source_claims_by_feature.get(
                row["source_feature_id"], {"classification": "unowned"})
            target_id = row["target_feature_id"]
            row["target_map_ownership"] = claims_by_target.get(
                target_id, {"classification": "unowned", "claims": []})
            row["target_claimed_landmark_id"] = int(current.landmark_ids[target_id])

        observation_certificates = {}
        for source_claim in source_claims_by_feature.values():
            for claim in source_claim.get("authoritative_observations", []):
                certificate = claim.get("existing_observation_certificate")
                if certificate is not None:
                    key = (certificate["frame_id"], certificate["landmark_id"],
                           tuple(certificate["pixel"]), certificate["right_u"])
                    observation_certificates[key] = certificate
        for target_claim in claims_by_target.values():
            for claim in target_claim.get("claims", []):
                certificate = claim.get("existing_observation_certificate")
                if certificate is not None:
                    key = (certificate["frame_id"], certificate["landmark_id"],
                           tuple(certificate["pixel"]), certificate["right_u"])
                    observation_certificates[key] = certificate

        fit_state = arbitration.get("fit_source_state") or {}
        fit_epoch = arbitration.get("fit_source_epoch")
        fit_pose = fit_state.get("source_pose")
        source_pose_unchanged = bool(
            fit_pose is not None and prepared_source_pose is not None
            and np.array_equal(np.asarray(fit_pose), np.asarray(prepared_source_pose))
        )
        source_anchor_unchanged = (
            fit_state.get("source_anchor_keyframe_id") == prepared_source_anchor
        )
        fit_anchor_pose = fit_state.get("source_anchor_pose")
        anchor_pose_unchanged = bool(
            fit_anchor_pose is not None and prepared_source_anchor_pose is not None
            and np.array_equal(np.asarray(fit_anchor_pose),
                               np.asarray(prepared_source_anchor_pose))
        )
        fit_geometry_revision = (None if fit_epoch is None else int(fit_epoch[1]))
        geometry_revision_unchanged = fit_geometry_revision == geometry_revision
        source_status_unchanged = (
            fit_state.get("source_status") in ("tracking", "relocalized")
            and fit_state.get("source_status") == prepared_source_status
        )
        target_status_accepted = target_status in ("tracking", "relocalized")
        fit_rows_sha256 = hashlib.sha256(
            np.ascontiguousarray(fit, dtype="<i8").tobytes()
        ).hexdigest()
        base.update({
            "status": "captured",
            "eligible": bool(not overlap and calibration_matches
                             and source_pose_unchanged and source_anchor_unchanged
                             and anchor_pose_unchanged and geometry_revision_unchanged
                             and source_status_unchanged and target_status_accepted
                             and training.get("reverse_status") == "verified"),
            "reason": None if (not overlap and calibration_matches and source_pose_unchanged
                               and source_anchor_unchanged and anchor_pose_unchanged
                               and geometry_revision_unchanged and source_status_unchanged
                               and target_status_accepted
                               and training.get("reverse_status") == "verified")
            else ("fit_holdout_physical_pixel_overlap" if overlap else
                  "live_calibration_mismatch" if not calibration_matches else
                  "source_pose_or_anchor_changed_since_fit" if not source_pose_unchanged
                  or not source_anchor_unchanged or not anchor_pose_unchanged else
                  "source_geometry_changed_since_fit" if not geometry_revision_unchanged else
                  "source_or_target_not_accepted" if not source_status_unchanged
                  or not target_status_accepted else "reverse_verification_unavailable"),
            "fit_source": "reserved_supported_training_rows",
            "training_role": "reserved_fit_train",
            "partition": "source_feature_index_modulo_2",
            "fit_pairs_sha256": fit_rows_sha256,
            "fit_pairs": [list(map(int, pair)) for pair in fit],
            "fit_training_matches": fit_match_rows,
            "forward_inlier_pairs": [list(map(int, pair)) for pair in forward],
            "reverse_inlier_pairs": [list(map(int, pair)) for pair in reverse],
            "forward_fit_row_indices": forward_rows.tolist(),
            "reverse_fit_row_indices": reverse_rows.tolist(),
            "refinement_forward_pairs": [list(map(int, pair)) for pair in forward]
            if training.get("refinement_attempted") else [],
            "refinement_reverse_pairs": [list(map(int, pair)) for pair in reverse]
            if training.get("refinement_attempted") else [],
            "refinement_applied": bool(training.get("refinement_applied")),
            "reverse_checked": training.get("reverse_status") == "verified",
            "fit_source_epoch": None if fit_epoch is None else list(fit_epoch),
            "prepared_epoch": [revision, geometry_revision],
            "source_pose_at_fit": maybe_vector(fit_pose) if fit_pose is not None else None,
            "source_pose_at_prepared": maybe_vector(prepared_source_pose)
            if prepared_source_pose is not None else None,
            "source_pose_unchanged_since_fit": source_pose_unchanged,
            "source_anchor_pose_unchanged_since_fit": anchor_pose_unchanged,
            "source_geometry_revision_unchanged_since_fit": geometry_revision_unchanged,
            "source_status_at_fit": fit_state.get("source_status"),
            "source_frame": {
                "frame": int(previous.frame), "image_size": list(previous.image_size),
                "calibration_identity": previous.calibration_identity,
                "role": "previous_raw_supported_stereo",
                "anchor_keyframe_id_at_fit": fit_state.get("source_anchor_keyframe_id"),
                "anchor_pose_at_fit": (maybe_vector(fit_state["source_anchor_pose"])
                                        if fit_state.get("source_anchor_pose") is not None else None),
                "anchor_keyframe_id_at_prepared": prepared_source_anchor,
                "anchor_pose_at_prepared": (maybe_vector(prepared_source_anchor_pose)
                                             if prepared_source_anchor_pose is not None else None),
                "pose_anchor_unchanged_since_fit": source_anchor_unchanged,
            },
            "target_frame": {
                "frame": int(current.frame), "image_size": list(current.image_size),
                "calibration_identity": current.calibration_identity,
                "role": "current_raw_supported_stereo",
                "state_at_prepared": target_status,
                "camera_to_world_at_prepared": maybe_vector(target_pose)
                if target_pose is not None else None,
                "anchor_keyframe_id_at_prepared": target_anchor,
                "anchor_pose_at_prepared": (maybe_vector(target_anchor_pose)
                                             if target_anchor_pose is not None else None),
            },
            "calibration": {
                "identity": live_calibration,
                "matches_endpoints": calibration_matches,
                "matrix": maybe_vector(calibration_matrix),
                "matrix_shape": list(calibration_matrix.shape),
                "rectification_q": maybe_vector(calibration_q),
                "baseline": calibration_baseline,
                "disparity_offset": calibration_offset,
                "image_size": list(image_size),
            },
            "excluded_heldout_pairs": held_rows,
            "existing_observations": [observation_certificates[key]
                                      for key in sorted(observation_certificates)],
            "candidate_row_cap": 256,
            "candidate_row_count": int(len(candidate_rows)),
            "fit_match_count": int(len(fit)),
            "heldout_match_count": int(len(held)),
            "excluded_target_feature_ids": sorted(map(int, arbitration["excluded_targets"])),
            "excluded_landmark_ids": sorted(map(int, arbitration["excluded_landmarks"])),
            "fit_holdout_physical_pixel_overlap": [
                {"endpoint": endpoint, "pixel_key": list(key)}
                for endpoint, key in overlap
            ],
            "rows": candidate_rows,
            "target_map_claims_by_feature": {
                str(target_id): claims_by_target[target_id]
                for target_id in sorted(claims_by_target)
            },
            "previous_track_claims": [
                {"landmark_id": int(landmark_id), "source_pixel": maybe_vector(pixel),
                 "source_frame": int(previous.frame), "role": "tracker_claim_not_observation_certificate"}
                for landmark_id, pixel in self.previous_tracks
            ],
        })
        return _owned_diagnostic_value(base)

    def _owned_stereo_training_provider(
        self, frame, image_size, arbitration, arbitration_report, info,
        verified_motion, consumed_full_pool,
    ):
        """Build immutable optional factors from the already-selected raw ledger."""
        captured = {}

        def reject(reason, details=None):
            result = {"status": "rejected", "reason": reason,
                      "model": "correlated_physical_stereo_image_rows_v1",
                      "covariance_claim": False, "intentionally_reuses_sensor_evidence": True,
                      "selected_factors": 0}
            if details:
                result.update(details)
            return (), result

        def provider(prepared_payload):
            if not isinstance(prepared_payload, dict):
                return reject("malformed_bundle_prepared_payload")
            phase = prepared_payload.get("training_factor_phase", "prepare")
            if phase == "validate":
                expected = captured.get("expected")
                if expected is None:
                    return reject("factor_validation_without_prepare")
                with self.map.lock:
                    if (prepared_payload.get("revision") != expected["revision"]
                            or prepared_payload.get("geometry_revision") != expected["geometry_revision"]):
                        return reject("prepared_packet_changed_before_factor_apply")
                    if (int(self.map.revision) != expected["revision"]
                            or int(self.map.geometry_revision) != expected["geometry_revision"]):
                        return reject("map_epoch_changed_before_factor_apply")
                    if self._live_owned_bundle_calibration() != expected["calibration_identity"]:
                        return reject("calibration_changed_before_factor_apply")
                    for frame_id, state in expected["endpoints"].items():
                        if (frame_id >= len(self.map.poses)
                                or not np.array_equal(self.map.poses[frame_id], state["pose"])
                                or self.map.statuses[frame_id] != state["status"]
                                or self.map.pose_anchors[frame_id] != state["anchor"]):
                            return reject("endpoint_changed_before_factor_apply")
                    for anchor_id, pose in expected["anchor_poses"].items():
                        keyframe = self.map.keyframes.get(anchor_id)
                        if keyframe is None or not np.array_equal(keyframe.pose, pose):
                            return reject("anchor_changed_before_factor_apply")
                    for landmark_id, point in expected["world_positions"].items():
                        landmark = self.map.landmarks.get(landmark_id)
                        if landmark is None or not np.array_equal(landmark.position, point):
                            return reject("reused_landmark_changed_before_factor_apply")
                    for cert in expected["observations"]:
                        landmark = self.map.landmarks.get(cert["landmark_id"])
                        observation = (None if landmark is None else
                                       landmark.observations.get(cert["keyframe_id"]))
                        keyframe = self.map.keyframes.get(cert["keyframe_id"])
                        if (observation is None or keyframe is None
                                or int(keyframe.frame) != cert["frame_id"]
                                or cert["keyframe_feature_id"] < 0
                                or cert["keyframe_feature_id"] >= len(keyframe.landmark_ids)
                                or int(keyframe.landmark_ids[cert["keyframe_feature_id"]]) != cert["landmark_id"]
                                or not np.array_equal(
                                    np.asarray(keyframe.pixels[cert["keyframe_feature_id"]], np.float32),
                                    cert["pixel"],
                                )
                                or not np.array_equal(np.asarray(observation.pixel, np.float32), cert["pixel"])
                                or observation.right_u is None
                                or np.float32(observation.right_u) != cert["right_u"]):
                            return reject("observation_ownership_changed_before_factor_apply")
                    return (), {"status": "validated", "reason": None,
                                "model": "correlated_physical_stereo_image_rows_v1",
                                "covariance_claim": False,
                                "intentionally_reuses_sensor_evidence": True}
            if phase != "prepare":
                return reject("unknown_training_factor_phase")
            if (not isinstance(arbitration, dict) or not isinstance(verified_motion, tuple)
                    or len(verified_motion) != 2):
                return reject("reserved_training_context_unavailable")
            saved_verified = arbitration.get("verified")
            if not isinstance(saved_verified, dict):
                return reject("reserved_reference_not_verified")
            selected = _reserved_training_selected_as_edge(
                info.get("pose_source"),
                arbitration_report.get("choice")
                if isinstance(arbitration_report, dict) else None,
                verified_motion[0], arbitration.get("previous").frame,
                verified_motion[1], saved_verified.get("measurement"),
                bool(consumed_full_pool),
            )
            if not selected:
                return reject("reserved_reference_not_selected_or_full_pool_consumed")

            # The detailed capture routine copies only the raw candidate rows and
            # actual exact Observation certificates. It is used only behind this
            # explicit opt-in, and is executed under the map lock for one epoch.
            with self.map.lock:
                snapshot = self._bundle_training_observations_context(
                    frame, image_size, arbitration, arbitration_report, prepared_payload
                )
                if not snapshot.get("eligible") or not snapshot.get("reverse_checked"):
                    return reject(snapshot.get("reason") or "reserved_rows_ineligible",
                                  {"capture_status": snapshot.get("status")})
                if snapshot.get("consumed_full_pool") or not snapshot.get("reserved_training_selected_as_edge", True):
                    return reject("reserved_reference_not_selected_or_full_pool_consumed")
                if (prepared_payload.get("revision") != snapshot.get("prepared_epoch", [None])[0]
                        or prepared_payload.get("geometry_revision") != snapshot.get("prepared_epoch", [None, None])[1]):
                    return reject("prepared_packet_epoch_mismatch")
                selection = prepared_payload.get("selection", {})
                selected_ids = {int(item["landmark_id"])
                                for item in selection.get("selected_landmarks", [])
                                if isinstance(item, dict) and item.get("landmark_id") is not None}
                single_rows = prepared_payload.get("single_view_propagations", [])
                single_ids = {int(item["landmark_id"]) for item in single_rows
                              if isinstance(item, dict) and item.get("landmark_id") is not None}
                reusable_ids = selected_ids | single_ids
                if not reusable_ids and snapshot.get("existing_observations"):
                    return reject("no_prepared_reusable_landmark_ids")

                previous = arbitration["previous"]
                current = arbitration["current"]
                training = arbitration.get("training_rows")
                if not isinstance(training, dict):
                    return reject("training_row_ledger_unavailable")
                target_claims = snapshot.get("target_map_claims_by_feature", {})
                source_rows = {int(row["source_feature_id"]): row.get("source_map_ownership", {})
                               for row in snapshot.get("fit_training_matches", [])}
                forward = np.asarray(training.get("forward_inlier_pairs", []), dtype=np.int64).reshape(-1, 2)
                reverse = np.asarray(training.get("reverse_inlier_pairs", []), dtype=np.int64).reshape(-1, 2)
                safe_forward, safe_reverse, excluded = _filter_owned_training_edges(
                    forward, reverse, target_claims, source_rows, reusable_ids
                )
                if not len(safe_forward) or not len(safe_reverse):
                    return reject("no_bidirectional_rows_after_map_ownership_checks",
                                  {"ownership_rejected_edges": excluded})

                fit_state = arbitration.get("fit_source_state") or {}
                fit_epoch = arbitration.get("fit_source_epoch")
                geometry_revision = int(self.map.geometry_revision)
                if (not isinstance(fit_epoch, (tuple, list)) or len(fit_epoch) != 2
                        or int(fit_epoch[1]) != geometry_revision):
                    return reject("source_geometry_changed_since_reserved_fit")
                endpoints = {}
                anchors = {}
                for endpoint in (previous, current):
                    fid = int(endpoint.frame)
                    if (fid >= len(self.map.poses) or fid >= len(self.map.statuses)
                            or fid >= len(self.map.pose_anchors)):
                        return reject("endpoint_missing_from_live_map")
                    anchor_id = self.map.pose_anchors[fid]
                    if anchor_id not in self.map.keyframes:
                        return reject("endpoint_anchor_missing")
                    status = self.map.statuses[fid]
                    raw_pose = np.asarray(self.map.poses[fid])
                    raw_anchor_pose = np.asarray(self.map.keyframes[anchor_id].pose)
                    if np.iscomplexobj(raw_pose) or np.iscomplexobj(raw_anchor_pose):
                        return reject("complex_endpoint_geometry")
                    pose = np.array(raw_pose, dtype=float, copy=True)
                    anchor_pose = np.array(raw_anchor_pose, dtype=float, copy=True)
                    endpoints[fid] = {"pose": pose, "status": status, "anchor": int(anchor_id)}
                    anchors[int(anchor_id)] = anchor_pose
                fit_pose = fit_state.get("source_pose")
                if (fit_pose is None or not np.array_equal(np.asarray(fit_pose), endpoints[int(previous.frame)]["pose"])
                        or fit_state.get("source_anchor_keyframe_id") != endpoints[int(previous.frame)]["anchor"]
                        or not np.array_equal(np.asarray(fit_state.get("source_anchor_pose")),
                                              anchors[endpoints[int(previous.frame)]["anchor"]])):
                    return reject("source_pose_or_anchor_changed_since_reserved_fit")

                camera_rows = {int(item["keyframe_id"]): np.asarray(item["camera_to_world"], float)
                               for item in prepared_payload.get("camera_poses", [])}
                for anchor_id, pose in anchors.items():
                    if anchor_id not in camera_rows or not np.array_equal(camera_rows[anchor_id], pose):
                        return reject("prepared_anchor_pose_mismatch")
                source_endpoint = EndpointPose(
                    frame_id=int(previous.frame), pose=endpoints[int(previous.frame)]["pose"],
                    anchor_keyframe_id=endpoints[int(previous.frame)]["anchor"],
                    anchor_pose=anchors[endpoints[int(previous.frame)]["anchor"]],
                    status=endpoints[int(previous.frame)]["status"],
                    source_epoch=(int(self.map.revision), geometry_revision),
                )
                target_endpoint = EndpointPose(
                    frame_id=int(current.frame), pose=endpoints[int(current.frame)]["pose"],
                    anchor_keyframe_id=endpoints[int(current.frame)]["anchor"],
                    anchor_pose=anchors[endpoints[int(current.frame)]["anchor"]],
                    status=endpoints[int(current.frame)]["status"],
                    source_epoch=(int(self.map.revision), geometry_revision),
                )
                certs = []
                points = {}
                cert_keys = set()
                for cert in snapshot.get("existing_observations", []):
                    lid = int(cert["landmark_id"])
                    if lid not in reusable_ids:
                        continue
                    key = (int(cert["frame_id"]), lid,
                           tuple(np.asarray(cert["pixel"], np.float32).tolist()),
                           float(np.float32(cert["right_u"])))
                    if key in cert_keys:
                        continue
                    cert_keys.add(key)
                    certs.append({"frame_id": key[0], "landmark_id": lid,
                                  "pixel": np.asarray(cert["pixel"], np.float32).copy(),
                                  "right_u": np.float32(cert["right_u"])})
                    landmark = self.map.landmarks.get(lid)
                    if landmark is None:
                        return reject("certified_landmark_missing")
                    raw_point = np.asarray(landmark.position)
                    if np.iscomplexobj(raw_point):
                        return reject("complex_reused_landmark_position")
                    point = np.array(raw_point, dtype=float, copy=True)
                    if point.shape != (3,) or not np.isfinite(point).all():
                        return reject("certified_landmark_position_invalid")
                    if lid in points and not np.array_equal(points[lid], point):
                        return reject("conflicting_landmark_snapshot")
                    points[lid] = point
                # Prepared initial values bind every reused map point to the BA snapshot.
                prepared_points = {}
                for item in selection.get("selected_landmarks", []):
                    if isinstance(item, dict):
                        prepared_points[int(item["landmark_id"])] = np.asarray(item["initial_world_position"], float)
                for item in single_rows:
                    if isinstance(item, dict):
                        prepared_points[int(item["landmark_id"])] = np.asarray(item["initial_world_position"], float)
                if any(lid not in prepared_points or not np.array_equal(point, prepared_points[lid])
                       for lid, point in points.items()):
                    return reject("reused_landmark_differs_from_prepared_point")

                packet_calibration = prepared_payload.get("calibration", {})
                raw_matrix = np.asarray(self.K)
                raw_q = np.asarray(self.stereo.Q)
                if np.iscomplexobj(raw_matrix) or np.iscomplexobj(raw_q):
                    return reject("complex_live_calibration")
                try:
                    packet_matrix = np.asarray(packet_calibration.get("matrix"))
                    packet_baseline = float(packet_calibration.get("baseline"))
                    packet_offset = float(packet_calibration.get("disparity_offset"))
                    current_offset = float(self.stereo.disparity_offset)
                    if (np.iscomplexobj(packet_matrix)
                            or not np.array_equal(packet_matrix, raw_matrix)
                            or packet_baseline != float(self.stereo.baseline)
                            or packet_offset != current_offset):
                        return reject("prepared_calibration_mismatch")
                except (TypeError, ValueError, OverflowError):
                    return reject("malformed_prepared_calibration")

                factors, report = build_stereo_training_factors(
                    previous, current, source_endpoint, target_endpoint,
                    matrix=np.array(self.K, dtype=float, copy=True),
                    baseline=float(self.stereo.baseline),
                    disparity_offset=float(self.stereo.disparity_offset),
                    calibration_identity=self.stereo_calibration_identity,
                    fit_pairs=arbitration.get("fit_pairs_snapshot"),
                    forward_inlier_pairs=safe_forward,
                    reverse_inlier_pairs=safe_reverse,
                    heldout_pairs=arbitration.get("held_pairs"),
                    existing_observations=certs,
                    existing_landmark_points=points,
                )
                valid_factors = []
                for factor in factors:
                    if (factor.reused_landmark_id >= 0
                            and factor.reused_landmark_id not in reusable_ids):
                        continue
                    if (factor.target_landmark_id >= 0
                            and factor.target_landmark_id not in reusable_ids):
                        continue
                    valid_factors.append(factor)
                report.update({"ownership_rejected_edges": int(excluded),
                               "selected_factors": len(valid_factors),
                               "reserved_training_selected_as_edge": True,
                               "consumed_full_pool": False})
                if not valid_factors:
                    return (), report
                report["status"] = "accepted"
                live_calibration = self._live_owned_bundle_calibration()
                if live_calibration is None:
                    return reject("malformed_live_calibration")
                report["calibration_identity"] = live_calibration
                used_landmarks = {int(factor.reused_landmark_id) for factor in valid_factors
                                  if int(factor.reused_landmark_id) >= 0}
                used_observation_keys = set()
                for factor in valid_factors:
                    lid = int(factor.reused_landmark_id)
                    if lid < 0:
                        continue
                    if factor.source_has_existing_observation:
                        used_observation_keys.add((int(factor.source_frame), lid,
                                                   tuple(np.asarray(factor.source_pixel, np.float32).tolist()),
                                                   float(np.float32(factor.source_right_u))))
                    if factor.target_has_existing_observation:
                        used_observation_keys.add((int(factor.target_frame), lid,
                                                   tuple(np.asarray(factor.target_pixel, np.float32).tolist()),
                                                   float(np.float32(factor.target_right_u))))
                certificate_rows = {}
                for item in snapshot.get("existing_observations", []):
                    key = (int(item["frame_id"]), int(item["landmark_id"]),
                           tuple(np.asarray(item["pixel"], np.float32).tolist()),
                           float(np.float32(item["right_u"])))
                    if key in used_observation_keys:
                        certificate_rows[key] = {
                            "keyframe_id": int(item["keyframe_id"]),
                            "keyframe_feature_id": int(item["keyframe_feature_id"]),
                            "frame_id": key[0], "landmark_id": key[1],
                            "pixel": np.asarray(item["pixel"], np.float32).copy(),
                            "right_u": np.float32(item["right_u"]),
                        }
                if set(certificate_rows) != used_observation_keys:
                    return reject("included_factor_observation_certificate_missing")
                used_points = {lid: points[lid] for lid in used_landmarks if lid in points}
                if set(used_points) != used_landmarks:
                    return reject("included_factor_landmark_snapshot_missing")
                captured["expected"] = {
                    "revision": int(self.map.revision),
                    "geometry_revision": geometry_revision,
                    "calibration_identity": live_calibration,
                    "endpoints": endpoints, "anchor_poses": anchors,
                    "world_positions": used_points,
                    "observations": list(certificate_rows.values()),
                }
                return tuple(valid_factors), report

        return provider

    def _retained_source_exclusion_snapshot(self, index, image_size):
        """Own reserved pixels without turning invalid ownership into an empty set."""
        def invalid(reason):
            return {"valid": False, "reason": reason}

        context = self._arbitration_context
        if context is None:
            return invalid("reserved_context_unavailable")
        if not isinstance(context, dict):
            return invalid("reserved_context_invalid")
        try:
            previous = context["previous"]
            current = context["current"]
            evidence = context["evidence"]
            frame = previous.frame
            if (not isinstance(frame, (int, np.integer))
                    or isinstance(frame, (bool, np.bool_)) or frame < 0
                    or not isinstance(current.frame, (int, np.integer))
                    or isinstance(current.frame, (bool, np.bool_))
                    or current.frame != index or frame >= index
                    or not isinstance(evidence.source_frame, (int, np.integer))
                    or isinstance(evidence.source_frame, (bool, np.bool_))
                    or evidence.source_frame != frame):
                return invalid("reserved_frame_binding_invalid")
            for name in ("baseline", "disparity_offset"):
                value = np.asarray(getattr(self.stereo, name))
                if (value.shape != () or np.iscomplexobj(value)
                        or not np.issubdtype(value.dtype, np.number)
                        or not np.isfinite(value).all()
                        or (name == "baseline" and value <= 0)):
                    return invalid("reserved_calibration_invalid")
            calibration = self._live_source_calibration_identity()
            if (not isinstance(calibration, str) or not calibration
                    or calibration != self.stereo_calibration_identity
                    or previous.calibration_identity != calibration
                    or current.calibration_identity != calibration
                    or evidence.calibration_identity != calibration):
                return invalid("reserved_calibration_invalid")
            size = np.asarray(image_size)
            if (size.shape != (2,) or np.iscomplexobj(size)
                    or not np.issubdtype(size.dtype, np.number)
                    or not np.isfinite(size).all() or np.any(size <= 0)
                    or tuple(previous.image_size) != tuple(image_size)
                    or tuple(current.image_size) != tuple(image_size)):
                return invalid("reserved_image_domain_invalid")
            pixels = np.asarray(previous.pixels)
            source_ids = np.asarray(evidence.source_ids)
            if (np.iscomplexobj(pixels)
                    or not np.issubdtype(pixels.dtype, np.number)
                    or pixels.ndim != 2 or pixels.shape[1] != 2
                    or source_ids.ndim != 1
                    or not np.issubdtype(source_ids.dtype, np.integer)
                    or np.issubdtype(source_ids.dtype, np.bool_)
                    or np.any(source_ids < 0) or np.any(source_ids >= len(pixels))
                    or len(np.unique(source_ids)) != len(source_ids)):
                return invalid("reserved_source_ids_invalid")
            selected_pixels = pixels[source_ids]
            if (not np.isfinite(selected_pixels).all()
                    or np.any(selected_pixels < 0) or np.any(selected_pixels >= size)):
                return invalid("reserved_source_pixels_invalid")
            raw_ids = context["excluded_landmarks"]
            if (not isinstance(raw_ids, (list, tuple, set, frozenset, np.ndarray))
                    or (isinstance(raw_ids, np.ndarray) and raw_ids.ndim != 1)
                    or any(not isinstance(value, (int, np.integer))
                           or isinstance(value, (bool, np.bool_)) or value < 0
                           for value in raw_ids)):
                return invalid("reserved_landmark_ids_invalid")
            excluded_ids = {int(value) for value in raw_ids}
            evidence_ids = np.asarray(evidence.landmark_ids)
            if (evidence_ids.shape != source_ids.shape
                    or not np.issubdtype(evidence_ids.dtype, np.integer)
                    or np.issubdtype(evidence_ids.dtype, np.bool_)
                    or np.any(evidence_ids < -1)
                    or not {int(value) for value in evidence_ids if value >= 0}
                           .issubset(excluded_ids)):
                return invalid("reserved_landmark_binding_invalid")
            owned_pixels = np.array(selected_pixels, dtype=np.float32, copy=True)
            if not np.isfinite(owned_pixels).all():
                return invalid("reserved_source_pixels_invalid")
            return {
                "valid": True, "reason": None,
                "landmark_ids": sorted(excluded_ids),
                "source_frame": int(frame), "source_pixels": owned_pixels,
            }
        except (AttributeError, KeyError, TypeError, ValueError, IndexError, OverflowError):
            return invalid("reserved_context_invalid")

    def _live_owned_bundle_calibration(self):
        try:
            digest = hashlib.sha256()
            for value in (np.asarray(self.K),
                          self.stereo.Q if self.stereo is not None else np.empty(0)):
                array = np.asarray(value)
                if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
                    return None
                if not np.isfinite(array).all():
                    return None
                digest.update(array.dtype.str.encode())
                digest.update(array.tobytes())
            if self.stereo is not None:
                baseline = float(self.stereo.baseline)
                offset = float(self.stereo.disparity_offset)
                if not np.isfinite([baseline, offset]).all():
                    return None
                digest.update(np.float64(baseline).tobytes())
                digest.update(np.float64(offset).tobytes())
            return digest.hexdigest()
        except (TypeError, ValueError, OverflowError):
            return None

    def _live_source_calibration_identity(self):
        """Match the raw-frame identity contract captured by SupportedStereoFrame."""
        try:
            digest = hashlib.sha256()
            for value in (np.asarray(self.K), np.asarray(self.stereo.Q)):
                array = np.asarray(value)
                if (np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number)
                        or not np.isfinite(array).all()):
                    return None
                digest.update(array.dtype.str.encode())
                digest.update(array.tobytes())
            baseline = float(self.stereo.baseline)
            if not np.isfinite(baseline) or baseline <= 0:
                return None
            digest.update(np.float64(baseline).tobytes())
            return digest.hexdigest()
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None

    def _source_history_endpoint(self, frame_id):
        """Return a detached live endpoint snapshot for source-history guards."""
        frame_id = int(frame_id)
        if (frame_id < 0 or frame_id >= len(self.map.poses)
                or frame_id >= len(self.map.statuses)
                or frame_id >= len(self.map.pose_anchors)):
            return None
        pose_raw = np.asarray(self.map.poses[frame_id])
        anchor_id = self.map.pose_anchors[frame_id]
        keyframe = self.map.keyframes.get(anchor_id)
        if (np.iscomplexobj(pose_raw) or not self._is_proper_se3(pose_raw)
                or keyframe is None):
            return None
        anchor_raw = np.asarray(keyframe.pose)
        if np.iscomplexobj(anchor_raw) or not self._is_proper_se3(anchor_raw):
            return None
        return {
            "frame": frame_id,
            "pose": np.array(pose_raw, dtype=float, copy=True),
            "status": self.map.statuses[frame_id],
            "anchor_keyframe_id": int(anchor_id),
            "anchor_pose": np.array(anchor_raw, dtype=float, copy=True),
        }

    def _capture_source_history_snapshot(self, target_frame=None):
        """Own the last accepted tracking pixels before the target frame replaces them."""
        if not getattr(self.config, "stereo_source_history_bundle", False):
            return None
        snapshot = {"status": "skipped", "reason": "source_history_unavailable",
                    "tracks": []}
        try:
            frame_id = self._previous_tracks_frame
            if (frame_id is None or not isinstance(frame_id, (int, np.integer))
                    or isinstance(frame_id, (bool, np.bool_))):
                snapshot["reason"] = "accepted_track_frame_unavailable"
                return snapshot
            frame_id = int(frame_id)
            if target_frame is not None and frame_id != int(target_frame) - 1:
                snapshot["reason"] = "accepted_track_frame_not_immediate_source"
                return snapshot
            supported = self.previous_supported_stereo
            live_source_identity = self._live_source_calibration_identity()
            if (supported is None or int(supported.frame) != frame_id
                    or supported.calibration_identity != self.stereo_calibration_identity
                    or live_source_identity != self.stereo_calibration_identity):
                snapshot["reason"] = "supported_source_endpoint_mismatch"
                return snapshot
            width, height = map(int, supported.image_size)
            if (width <= 0 or height <= 0 or self.previous_gray is None
                    or np.asarray(self.previous_gray).shape[:2] != (height, width)):
                snapshot["reason"] = "source_image_domain_unavailable"
                return snapshot
            live_calibration = self._live_owned_bundle_calibration()
            if live_calibration is None:
                snapshot["reason"] = "invalid_source_calibration"
                return snapshot

            with self.map.lock:
                endpoint = self._source_history_endpoint(frame_id)
                if endpoint is None or endpoint["status"] not in ("tracking", "relocalized"):
                    snapshot["reason"] = "source_map_endpoint_unavailable"
                    return snapshot
                raw_tracks = list(self.previous_tracks)
                by_id, pixel_claims, conflicted_ids = {}, {}, set()
                for item in raw_tracks:
                    try:
                        if not isinstance(item, (tuple, list)) or len(item) != 2:
                            continue
                        raw_id, raw_pixel = item
                        if (not isinstance(raw_id, (int, np.integer))
                                or isinstance(raw_id, (bool, np.bool_))):
                            continue
                        landmark_id = int(raw_id)
                        raw_pixel = np.asarray(raw_pixel)
                        if (np.iscomplexobj(raw_pixel)
                                or not np.issubdtype(raw_pixel.dtype, np.number)):
                            continue
                        pixel = np.asarray(raw_pixel, dtype=np.float32)
                        if (landmark_id < 0 or landmark_id not in self.map.landmarks
                                or pixel.shape != (2,) or not np.isfinite(pixel).all()
                                or pixel[0] < 0 or pixel[0] >= width
                                or pixel[1] < 0 or pixel[1] >= height):
                            continue
                        landmark = self.map.landmarks[landmark_id]
                        key = self._physical_pixel_key(pixel)
                        if landmark_id in conflicted_ids:
                            continue
                        prior = by_id.get(landmark_id)
                        if prior is not None and self._physical_pixel_key(prior) != key:
                            by_id.pop(landmark_id, None)
                            conflicted_ids.add(landmark_id)
                            continue
                        by_id[landmark_id] = pixel.copy()
                        pixel_claims.setdefault(key, set()).add(landmark_id)
                    except (TypeError, ValueError, OverflowError):
                        continue
                conflicting_pixels = {
                    key for key, identifiers in pixel_claims.items() if len(identifiers) > 1
                }
                tracks = [
                    {"landmark_id": int(landmark_id), "pixel": pixel.copy()}
                    for landmark_id, pixel in sorted(by_id.items())
                    if self._physical_pixel_key(pixel) not in conflicting_pixels
                ]
                if not tracks:
                    snapshot["reason"] = "no_accepted_multiview_source_tracks"
                    return snapshot
                snapshot = {
                    "status": "eligible",
                    "reason": None,
                    "frame": frame_id,
                    "tracks": tracks,
                    "source_pose": endpoint["pose"].copy(),
                    "source_status": endpoint["status"],
                    "source_anchor_keyframe_id": endpoint["anchor_keyframe_id"],
                    "source_anchor_pose": endpoint["anchor_pose"].copy(),
                    "image_size": (width, height),
                    "calibration_identity": self.stereo_calibration_identity,
                    "source_calibration_identity": live_source_identity,
                    "live_calibration_identity": live_calibration,
                    "capture_revision": int(self.map.revision),
                    "capture_geometry_revision": int(self.map.geometry_revision),
                    "conflicting_physical_pixels": int(len(conflicting_pixels)),
                }
            return snapshot
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            snapshot["reason"] = "source_history_snapshot_invalid"
            return snapshot

    def _owned_source_history_provider(
        self, capture, arbitration, arbitration_report, info,
        verified_motion, consumed_full_pool,
    ):
        """Supply selected left-only source rows with source-state revalidation."""
        expected = {}

        def reject(reason):
            return (), {"status": "rejected", "reason": reason,
                        "source_frame": capture.get("frame") if isinstance(capture, dict) else None,
                        "captured_rows": (len(capture.get("tracks", []))
                                         if isinstance(capture, dict)
                                         and isinstance(capture.get("tracks", []), (list, tuple)) else 0),
                        "installed_rows": 0, "covariance_claim": False,
                        "heldout_validation_claim": False}

        def provider(payload):
            if not isinstance(payload, dict) or not isinstance(capture, dict):
                return reject("source_history_capture_missing")
            phase = payload.get("source_history_phase", "prepare")
            if phase == "validate":
                if not expected:
                    return reject("source_history_validation_without_prepare")
                with self.map.lock:
                    live_calibration = self._live_owned_bundle_calibration()
                    endpoint = self._source_history_endpoint(expected["frame"])
                    if (live_calibration != expected["live_calibration_identity"]
                            or self._live_source_calibration_identity() != capture.get("calibration_identity")
                            or endpoint is None
                            or endpoint["status"] != expected["source_status"]
                            or endpoint["anchor_keyframe_id"] != expected["source_anchor_keyframe_id"]
                            or not np.array_equal(endpoint["pose"], expected["source_pose"])
                            or not np.array_equal(endpoint["anchor_pose"], expected["source_anchor_pose"])):
                        return reject("source_history_source_changed_before_apply")
                    for landmark_id, point in expected["world_positions"].items():
                        landmark = self.map.landmarks.get(landmark_id)
                        if landmark is None or not np.array_equal(landmark.position, point):
                            return reject("source_history_landmark_changed_before_apply")
                    return (), {"status": "validated", "reason": None,
                                "source_frame": expected["frame"],
                                "installed_rows": len(expected["rows"]),
                                "covariance_claim": False,
                                "heldout_validation_claim": False}
            if phase != "prepare":
                return reject("unknown_source_history_phase")
            if capture.get("status") != "eligible":
                return reject(str(capture.get("reason") or "source_history_capture_ineligible"))
            if (not isinstance(arbitration, dict)
                    or not isinstance(verified_motion, tuple) or len(verified_motion) != 2):
                return reject("reserved_reference_unavailable")
            reserved = arbitration.get("verified")
            previous = arbitration.get("previous")
            capture_frame = capture.get("frame")
            previous_frame = getattr(previous, "frame", None)
            if (not isinstance(capture_frame, (int, np.integer))
                    or isinstance(capture_frame, (bool, np.bool_))
                    or not isinstance(previous_frame, (int, np.integer))
                    or isinstance(previous_frame, (bool, np.bool_))):
                return reject("malformed_source_frame_identity")
            if (not isinstance(reserved, dict) or previous is None
                    or int(previous_frame) != int(capture_frame)
                    or not _reserved_training_selected_as_edge(
                        info.get("pose_source"),
                        arbitration_report.get("choice") if isinstance(arbitration_report, dict) else None,
                        verified_motion[0], previous.frame, verified_motion[1],
                        reserved.get("measurement"), bool(consumed_full_pool))):
                return reject("reserved_reference_not_selected")
            source_frame = int(capture_frame)
            if payload.get("source_history_source_frame") != source_frame:
                return reject("prepared_source_frame_mismatch")
            if (tuple(capture.get("image_size", ())) != tuple(previous.image_size)
                    or capture.get("calibration_identity") != previous.calibration_identity):
                return reject("source_raw_endpoint_mismatch")
            selected = {
                int(value) for value in payload.get("source_history_selected_landmark_ids", [])
                if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            }
            selected_values = payload.get("source_history_selected_landmark_ids", [])
            if (not isinstance(selected_values, (list, tuple))
                    or any(not isinstance(value, (int, np.integer))
                           or isinstance(value, (bool, np.bool_)) for value in selected_values)):
                return reject("malformed_selected_landmark_ids")
            if not selected:
                return reject("no_selected_multiview_landmarks")
            held_ids = {int(value) for value in arbitration.get("excluded_landmarks", set())}
            held_pixels = set()
            evidence = arbitration.get("evidence")
            if evidence is None:
                return reject("reserved_holdout_evidence_missing")
            try:
                raw_source_indices = np.asarray(evidence.source_ids)
                if (raw_source_indices.ndim != 1
                        or not np.issubdtype(raw_source_indices.dtype, np.integer)
                        or np.issubdtype(raw_source_indices.dtype, np.bool_)):
                    return reject("malformed_held_source_rows")
                source_indices = np.asarray(raw_source_indices, dtype=np.int64)
                if (not len(source_indices) or np.any(source_indices < 0)
                        or np.any(source_indices >= len(previous.pixels))):
                    return reject("malformed_held_source_rows")
                held_pixels = {
                    self._physical_pixel_key(previous.pixels[int(source_index)])
                    for source_index in source_indices
                }
            except (AttributeError, TypeError, ValueError, IndexError):
                return reject("malformed_held_source_rows")
            original_values = payload.get("source_history_original_frame_landmark_ids", [])
            owned_values = payload.get("source_history_owned_frame_landmark_ids", [])
            for values in (original_values, owned_values):
                if (not isinstance(values, (list, tuple))
                        or any(not isinstance(value, (int, np.integer))
                               or isinstance(value, (bool, np.bool_)) for value in values)):
                    return reject("malformed_existing_source_landmark_ids")
            original_ids = {int(value) for value in original_values}
            owned_ids = {int(value) for value in owned_values}
            existing_ids = original_ids | owned_ids
            source_tracks = capture.get("tracks", [])
            rows = []
            world_positions = {}
            try:
                with self.map.lock:
                    endpoint = self._source_history_endpoint(source_frame)
                    live_calibration = self._live_owned_bundle_calibration()
                    if (endpoint is None or endpoint["status"] != capture.get("source_status")
                            or live_calibration != capture.get("live_calibration_identity")):
                        return reject("source_endpoint_changed_since_capture")
                    by_pixel = {}
                    for item in source_tracks:
                        raw_id = item.get("landmark_id")
                        raw_pixel = np.asarray(item.get("pixel"))
                        if (not isinstance(raw_id, (int, np.integer))
                                or isinstance(raw_id, (bool, np.bool_))
                                or np.iscomplexobj(raw_pixel)
                                or not np.issubdtype(raw_pixel.dtype, np.number)):
                            continue
                        landmark_id = int(raw_id)
                        pixel = np.asarray(raw_pixel, dtype=np.float32)
                        if (landmark_id not in selected or landmark_id in held_ids
                                or landmark_id in existing_ids):
                            continue
                        width, height = map(int, capture.get("image_size", (0, 0)))
                        if (pixel.shape != (2,) or not np.isfinite(pixel).all()
                                or width <= 0 or height <= 0
                                or pixel[0] < 0 or pixel[0] >= width
                                or pixel[1] < 0 or pixel[1] >= height):
                            continue
                        key = self._physical_pixel_key(pixel)
                        if key in held_pixels:
                            continue
                        landmark = self.map.landmarks.get(landmark_id)
                        if landmark is None or len(landmark.observations) < 2:
                            continue
                        world = np.asarray(landmark.position)
                        if np.iscomplexobj(world) or world.shape != (3,) or not np.isfinite(world).all():
                            continue
                        rows.append({"frame_id": source_frame,
                                     "landmark_id": landmark_id,
                                     "pixel": pixel.copy()})
                        world_positions[landmark_id] = np.array(world, dtype=float, copy=True)
                        by_pixel.setdefault(key, set()).add(landmark_id)
                    conflicting = {key for key, ids in by_pixel.items() if len(ids) > 1}
                    if conflicting:
                        rows = [row for row in rows
                                if self._physical_pixel_key(row["pixel"]) not in conflicting]
                    if not rows:
                        return (), {"status": "skipped", "reason": "no_eligible_source_history_rows",
                                    "source_frame": source_frame, "candidate_rows": 0,
                                    "captured_rows": len(source_tracks),
                                    "installed_rows": 0, "covariance_claim": False,
                                    "heldout_validation_claim": False,
                                    "tracking_fit_evidence_reused": True,
                                    "exclusions": {
                                        "held_landmarks": len(held_ids),
                                        "held_source_pixels": len(held_pixels),
                                        "already_owned_landmarks": len(existing_ids),
                                    }}
                    expected.update({
                        "frame": source_frame,
                        "source_status": endpoint["status"],
                        "source_pose": endpoint["pose"],
                        "source_anchor_keyframe_id": endpoint["anchor_keyframe_id"],
                        "source_anchor_pose": endpoint["anchor_pose"],
                        "live_calibration_identity": live_calibration,
                        "world_positions": world_positions,
                        "rows": rows,
                    })
            except (AttributeError, TypeError, ValueError, IndexError, OverflowError):
                return reject("source_history_prepare_invalid")
            return tuple(rows), {
                "status": "eligible", "reason": None,
                "source_frame": source_frame,
                "candidate_rows": len(source_tracks),
                "captured_rows": len(source_tracks),
                "installed_rows": len(rows),
                "installed_landmark_ids": sorted({row["landmark_id"] for row in rows}),
                "tracking_fit_evidence_reused": True,
                "covariance_claim": False,
                "heldout_validation_claim": False,
                "exclusions": {
                    "held_landmarks": len(held_ids),
                    "held_source_pixels": len(held_pixels),
                    "already_owned_landmarks": len(existing_ids),
                    "not_selected_multiview": max(0, len(source_tracks) - len(rows)),
                },
            }
        return provider

    def process(self, index, image, right=None):
        if index != len(self.map.poses):
            raise ValueError(
                "Frames must arrive in consecutive order, starting at zero"
            )
        self.loop_worker.poll(self.map)
        source_history_capture = (
            self._capture_source_history_snapshot(index)
            if getattr(self.config, "stereo_source_history_bundle", False) else None
        )
        if getattr(self.config, "stereo_source_history_bundle", False):
            self._source_history_capture = source_history_capture
        self._prepare_frame_images(image, right)
        trace_capture = self._tracking_trace_enabled(index)
        self._tracking_trace_frame = int(index) if trace_capture else -1
        self._tracking_trace_probe_count = 0
        self._tracking_trace_selected_emitted = False
        if trace_capture:
            try:
                self._tracking_trace_cohort_frame = min(self.tracking_diagnostics_writer.frames)
                self._tracking_trace_event(index, "frame_start_state",
                                           self._tracking_trace_frame_header(index))
            except Exception as error:
                self.tracking_diagnostic_errors.append({
                    "frame": int(index), "phase": "begin", "error_type": type(error).__name__})
                trace_capture = False
                self._tracking_trace_frame = -1
        diagnostic_capture = False
        if self.bundle_diagnostic_writer is not None:
            try:
                diagnostic_capture = bool(
                    self.bundle_diagnostic_writer.should_capture(index)
                )
            except Exception as error:
                self.bundle_diagnostic_errors.append({
                    "frame": int(index), "phase": "select",
                    "error_type": type(error).__name__,
                })
        raw_stereo_enabled = (self.config.stereo_pose_arbitration
                              or self.config.stereo_raw_reference_retry)
        diagnostic_training_context = None
        diagnostic_training_report = None
        owned_training_context = None
        if raw_stereo_enabled:
            self._supported_extraction = None
            self._arbitration_context = None
        pixels, desc, points, right_u = self._extract(image, right)
        size = (image.shape[1], image.shape[0])
        arbitration_report = None
        arbitration_measurement = None
        if raw_stereo_enabled:
            self.current_supported_stereo = self._capture_supported_stereo(index, pixels, desc, size)
        if self.config.stereo_pose_arbitration:
            capture_training_rows = bool(
                diagnostic_capture or trace_capture or self.config.stereo_owned_image_bundle
            )
            if capture_training_rows:
                self._arbitration_context, arbitration_report = self._prepare_stereo_arbitration(
                    index, self.current_supported_stereo, capture_rows=True)
                if diagnostic_capture:
                    diagnostic_training_context = self._arbitration_context
                    diagnostic_training_report = (
                        dict(arbitration_report) if isinstance(arbitration_report, dict) else {}
                    )
                if self.config.stereo_owned_image_bundle:
                    owned_training_context = self._arbitration_context
            else:
                self._arbitration_context, arbitration_report = self._prepare_stereo_arbitration(
                    index, self.current_supported_stereo)
        elif diagnostic_capture:
            diagnostic_training_report = {
                "reason": ("raw_stereo_disabled" if not raw_stereo_enabled
                           else "stereo_pose_arbitration_disabled")
            }
        info = {
            "frame": index,
            "features": len(pixels),
            "valid_stereo_depth": int(
                len(self._unique_supported_feature_rows(pixels, points, right_u))
                if self.stereo is not None
                else np.isfinite(points).all(axis=1).sum()
            ),
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
                valid = self._unique_supported_feature_rows(pixels, points, right_u)
                if len(valid) >= 30 and coverage(pixels[valid], size) >= 4:
                    anchor = self._keyframe(
                        index, pose, pixels, desc, points, right_u, {}
                    )
                    status = "tracking"
                    info["tracking_ok"] = True
                    self.accepted_tracks = self._unique_physical_tracks([
                        (
                            int(self.map.keyframes[anchor].landmark_ids[j]),
                            pixels[j].copy(),
                        )
                        for j in valid
                    ])
                    inlier_features = set(valid.tolist())
            elif self.initial is None:
                self.initial = (index, pixels.copy(), desc.copy())
            else:
                start, initial_pixels, initial_desc = self.initial
                pairs = self._match(initial_desc, desc)
                pairs = self._collapse_physical_matches(pairs, initial_pixels, pixels)
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
                    initial_groups, initial_group_ids = self._physical_pixel_groups(initial_pixels)
                    current_groups, current_group_ids = self._physical_pixel_groups(pixels)
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
                        first_rows = initial_groups[initial_group_ids[a]]
                        second_rows = current_groups[current_group_ids[b]]
                        first.landmark_ids[first_rows] = lid
                        second.landmark_ids[second_rows] = lid
                        first.depth_points[first_rows] = position
                        second.depth_points[second_rows] = pose[:3, :3].T @ (
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
                    self.accepted_tracks = self._unique_physical_tracks([
                        (
                            int(second.landmark_ids[pairs[j, 1]]),
                            pixels[pairs[j, 1]].copy(),
                        )
                        for j in valid
                    ])
                    inlier_features = set(pairs[valid, 1].tolist())
        else:
            result, stats = self._track(pixels, desc, size)
            map_inlier_miss_snapshot = self._last_map_track_inlier_misses
            info.update(stats)
            recovered = False
            stereo_reference = False
            full_supported_fallback_attempted = False
            full_supported_fallback_report = None
            raw_reference_retry_attempted = False
            raw_reference_retry_succeeded = False
            raw_reference_retry_report = None
            raw_reference_retry_source_pose = None
            raw_reference_retry_source_revision = None
            raw_reference_retry_pose = None
            raw_reference_retry_final_rejected = False
            raw_reference_retry_pose_selected = False
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
                    else:
                        previous_frame, measured, reference_matcher = (
                            self._physical_stereo_reference_inputs(
                                previous_frame, measured
                            )
                        )
                    verified = (arbitration['verified'] if arbitration is not None else estimate_stereo_reference(
                        previous_frame,
                        measured,
                        self.K,
                        min_inliers=self.config.min_inliers,
                        initial_pose=prior,
                        matcher=reference_matcher,
                    ))
                    if (verified is None and arbitration is None
                            and self.config.stereo_raw_reference_retry):
                        raw_reference_retry_attempted = True
                        reservation_unavailable_report = arbitration_report
                        (raw_verified, raw_reference_retry_report,
                         raw_reference_retry_source_pose,
                         raw_reference_retry_source_revision,
                         raw_reference_retry_pose) = self._estimate_raw_supported_reference_retry(
                            self.previous_supported_stereo,
                            self.current_supported_stereo,
                            previous_index, index, previous_frame, measured, size)
                        raw_reference_retry_report = {
                            **raw_reference_retry_report,
                            'reservation_unavailable_report': reservation_unavailable_report,
                            'held_out_arbitration_used': False,
                        }
                        # A failed reserved-fit diagnostic is not a selector for
                        # the consumed full-pool retry and must not be relabeled
                        # as an existing-reference arbitration choice.
                        arbitration_report = None
                        if raw_verified is not None:
                            try:
                                with self.map.lock:
                                    retry_guard = self._hard_reference_retention_guard(
                                        index, previous_index, previous_frame, measured,
                                        raw_verified, raw_reference_retry_source_pose,
                                        raw_reference_retry_pose,
                                        raw_reference_retry_source_revision, size)
                            except (AttributeError, TypeError, ValueError, IndexError):
                                retry_guard = {'eligible': False,
                                               'reason': 'raw_reference_final_guard_error'}
                            if not retry_guard.get('eligible'):
                                raw_reference_retry_report = {
                                    **raw_reference_retry_report,
                                    'eligible': False,
                                    'reason': retry_guard.get(
                                        'reason', 'raw_reference_final_guard_failed'),
                                    'source_guard': retry_guard,
                                    'reverse_checked': True,
                                }
                                raw_verified = None
                        if raw_verified is not None:
                            verified = raw_verified
                            raw_reference_retry_succeeded = True
                        info['stereo_raw_reference_retry'] = raw_reference_retry_report
                    if verified is not None:
                        if raw_reference_retry_succeeded:
                            source_pose = raw_reference_retry_source_pose
                            source_map_revision = raw_reference_retry_source_revision
                            reference_pose = raw_reference_retry_pose
                        else:
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
                        if raw_reference_retry_succeeded:
                            raw_reference_retry_pose_selected = bool(result is None or conflict)

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
                                    'fit_depth_policy': self._depth_fit_policy_label(),
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
                                    'fit_depth_policy': self._depth_fit_policy_label(),
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
                            if (verified['reverse_checked'] and index-previous_index == 1
                                    and not raw_reference_retry_succeeded):
                                self.verified_stereo_motion = (verified['measurement'].copy(), index)
                            if verified['reverse_checked'] and not raw_reference_retry_succeeded:
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
                                retention_guard = None
                                retention_context = None
                                if self.config.stereo_mapping_observation_retention:
                                    with self.map.lock:
                                        retention_guard = self._hard_reference_retention_guard(
                                            index, previous_index, previous_frame, measured,
                                            verified, source_pose, reference_pose,
                                            source_map_revision, size)
                                        if retention_guard.get('eligible') is True:
                                            retention_context = {
                                                'guard': retention_guard,
                                                'verified': verified,
                                                'source_pose': np.asarray(source_pose, float).copy(),
                                                'reference_pose': np.asarray(reference_pose, float).copy(),
                                                'source_map_revision': int(source_map_revision),
                                                'source_geometry_revision': int(self.map.geometry_revision),
                                                'source_frame': int(previous_index),
                                                'target_frame': int(index),
                                                'previous_frame': previous_frame,
                                                'measured_frame': measured,
                                                'excluded_landmark_ids': (
                                                    arbitration.get('excluded_landmarks', set())
                                                    if arbitration is not None else set()),
                                                'excluded_target_pixels': (
                                                    arbitration.get('excluded_target_pixels', [])
                                                    if arbitration is not None else []),
                                            }
                                        associations, retained_tracks, association_validation = (
                                            self._validate_stereo_associations_at_pose(
                                                reference_pose, map_associations, map_tracks,
                                                pixels, size, info.get('valid_3d', 0),
                                                map_inlier_miss_snapshot,
                                                retention_context=retention_context))
                                else:
                                    associations, retained_tracks, association_validation = (
                                        self._validate_stereo_associations_at_pose(
                                            reference_pose, map_associations, map_tracks,
                                            pixels, size, info.get('valid_3d', 0),
                                            map_inlier_miss_snapshot))
                                if retention_guard is not None:
                                    association_validation = {
                                        **association_validation,
                                        'mapping_retention_guard': retention_guard,
                                    }
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
                                if (conflict and result is not None
                                        and (self.config.stereo_pose_arbitration
                                             or raw_reference_retry_succeeded)):
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
                                            retention_context = None
                                            if self.config.stereo_mapping_observation_retention:
                                                retention_context = {
                                                    'guard': reference_guard,
                                                    'verified': verified,
                                                    'source_pose': np.asarray(source_pose, float).copy(),
                                                    'reference_pose': np.asarray(reference_pose, float).copy(),
                                                    'source_map_revision': int(source_map_revision),
                                                    'source_geometry_revision': int(self.map.geometry_revision),
                                                    'source_frame': int(previous_index),
                                                    'target_frame': int(index),
                                                    'previous_frame': previous_frame,
                                                    'measured_frame': measured,
                                                    'excluded_landmark_ids': (
                                                        arbitration.get('excluded_landmarks', set())
                                                        if arbitration is not None else set()),
                                                    'excluded_target_pixels': (
                                                        arbitration.get('excluded_target_pixels', [])
                                                        if arbitration is not None else []),
                                                }
                                            associations, retained_tracks, connection = (
                                                self._validate_stereo_associations_at_pose(
                                                    reference_pose, map_associations, map_tracks,
                                                    pixels, size, info.get('valid_3d', 0),
                                                    map_inlier_miss_snapshot,
                                                    retention_context=retention_context))
                                            self.accepted_tracks = retained_tracks
                                            retained = bool(connection['eligible'])
                                            mapping_retained = bool(
                                                connection.get('mapping_observation_retention_allowed'))
                                        else:
                                            connection = None
                                            mapping_retained = False
                                            if raw_reference_retry_succeeded:
                                                raw_reference_retry_final_rejected = True
                                    reservation_context = arbitration is not None
                                    try:
                                        live_calibration_identity = self._live_stereo_calibration_identity()
                                    except (AttributeError, TypeError, ValueError):
                                        live_calibration_identity = None
                                    reference_validation = {
                                        **reference_guard,
                                        'eligible': bool(reference_guard['eligible'] and retained),
                                        'pose_support_eligible': bool(reference_guard['eligible'] and retained),
                                        'mapping_observation_retention_allowed': bool(
                                            reference_guard['eligible'] and mapping_retained),
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
                                            else 'raw_supported_reference_retry' if raw_reference_retry_succeeded
                                            else 'reserved_supported_training_rows' if reservation_context
                                            else 'map_coordinate_independent_stereo_reference'),
                                        'independent_fit_depth_policy': (
                                            self._depth_fit_policy_label()
                                            if (raw_reference_retry_succeeded
                                                or reservation_context
                                                or full_supported_fallback_attempted)
                                            else self.config.stereo_depth_policy),
                                        'fit_depth_policy': (
                                            self._depth_fit_policy_label()
                                            if (raw_reference_retry_succeeded
                                                or reservation_context
                                                or full_supported_fallback_attempted)
                                            else self.config.stereo_depth_policy),
                                        'map_fit_depth_policy': self.config.stereo_depth_policy,
                                        'prediction_seed_supplied': (
                                            False if (raw_reference_retry_succeeded
                                                      or full_supported_fallback_attempted)
                                            else not reservation_context),
                                        'current_right_measurement': (
                                            ('verified_right_image_at_actual_observation'
                                             if self.config.stereo_depth_policy == 'verified_all'
                                             else 'supported_only_raw_disparity_at_actual_observation')
                                            if connection is not None else 'not_remeasured_guard_rejected'),
                                        'original_final_solve_positions': int(info.get('valid_3d', 0)),
                                        'association_validation': connection,
                                    }
                                    if not reference_validation['eligible']:
                                        if connection is None and not raw_reference_retry_final_rejected:
                                            # The validator did not run, so reconcile
                                            # the provisional _track reset exactly once.
                                            self._age_provisional_map_inliers(
                                                map_inlier_miss_snapshot)
                                        if (not raw_reference_retry_final_rejected
                                                and not reference_validation[
                                                    'mapping_observation_retention_allowed']):
                                            associations = {}
                                            self.accepted_tracks = []
                                        if not raw_reference_retry_final_rejected:
                                            stereo_reference = True
                                    if raw_reference_retry_succeeded:
                                        raw_reference_retry_report = {
                                            **raw_reference_retry_report,
                                            'association_validation': reference_validation,
                                        }
                                        info['stereo_raw_reference_retry'] = raw_reference_retry_report
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
                            if raw_reference_retry_final_rejected:
                                raw_reference_retry_report = {
                                    **raw_reference_retry_report,
                                    'eligible': False,
                                    'installed': False,
                                    'reason': 'raw_reference_final_source_guard_failed',
                                    'held_out_arbitration_used': False,
                                    'association_validation': reference_validation,
                                }
                                info['stereo_raw_reference_retry'] = raw_reference_retry_report
                                for key in (
                                    'stereo_reference_verified',
                                    'stereo_bidirectional_refinement',
                                    'stereo_reference_verification',
                                    'map_reference_translation_error_m',
                                    'map_reference_rotation_error_deg',
                                    'map_pose_rejected_for_stereo_conflict',
                                ):
                                    info.pop(key, None)
                                for key in ('num_matches', 'num_inliers', 'inlier_ratio',
                                            'reprojection_error', 'pose_source'):
                                    if key in stats:
                                        info[key] = stats[key]
                                    else:
                                        info.pop(key, None)
                                info.pop('reference_inlier_features', None)
                                if self.config.stereo_pose_arbitration:
                                    arbitration_report = {
                                        **(reservation_unavailable_report or {}),
                                        'choice': 'map',
                                        'reason': 'raw_reference_final_source_guard_failed',
                                        'held_out_arbitration_used': False,
                                    }
                                verified = None
                            else:
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
                        if raw_reference_retry_succeeded and not raw_reference_retry_final_rejected:
                            if verified['reverse_checked'] and index-previous_index == 1:
                                self.verified_stereo_motion = (verified['measurement'].copy(), index)
                            if verified['reverse_checked']:
                                verified_motion = (previous_index, verified['measurement'].copy())
                            raw_reference_retry_report = {
                                **raw_reference_retry_report,
                                'installed': True,
                                'pose_selected': raw_reference_retry_pose_selected,
                                'held_out_arbitration_used': False,
                            }
                            info['stereo_raw_reference_retry'] = raw_reference_retry_report
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
                if trace_capture:
                    self._tracking_trace_selected(index, "selected_pre_keyframe", pose,
                        status, associations, pixels, right_u, arbitration_report, info)
                if (
                    stereo_reference
                    or index - last.frame >= self.config.keyframe_interval
                    or (index - last.frame >= 2 and len(tracked_landmarks) < 80)
                ):
                    anchor = self._keyframe(
                        index, pose, pixels, desc, points, right_u, associations
                    )
        if trace_capture and not self._tracking_trace_selected_emitted:
            self._tracking_trace_selected(index, "selected_pre_keyframe", pose,
                status, associations, pixels, right_u, arbitration_report, info)
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
            if getattr(self.config, "stereo_source_history_bundle", False):
                self._previous_tracks_frame = int(index)
        self.map.record(pose, status, anchor)
        if trace_capture:
            self._tracking_trace_selected(index, "accepted_pre_ba", pose,
                status, associations, pixels, right_u, arbitration_report, info)
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
        bundle_called = False
        bundle_skip_reason = None
        if trace_capture and index == self._tracking_trace_cohort_frame:
            self._tracking_trace_event(index, "cohort_map_before_ba",
                self._tracking_trace_frame_state(index, include_landmarks=True,
                                                 stage="cohort_map_before_ba"))
        if (
            info["tracking_ok"]
            and anchor is not None
            and self.map.keyframes[anchor].frame == index
            and len(self.map.keyframes) >= 3
        ):
            with self.profile.measure("local_bundle"):
                if not self.config.bundle_enabled:
                    report = {"applied": False, "reason": "diagnostic_ablation"}
                    bundle_skip_reason = "bundle_disabled"
                else:
                    bundle_kwargs = {
                        "window": self.config.bundle_window,
                        "disparity_offset": self.stereo.disparity_offset if self.stereo is not None else 0.,
                        "optimized": self.performance.cpu_optimizations,
                        "solver_accuracy": self.config.bundle_solver_accuracy,
                    }
                    if getattr(self.config, "stereo_bundle_gauge_mode", "veto") != "veto":
                        bundle_kwargs["gauge_mode"] = self.config.stereo_bundle_gauge_mode
                    if diagnostic_capture:
                        diagnostic_inputs = {
                            "window_keyframe_ids": [],
                            "motion_checks": [],
                        }

                        def diagnostic_sink(phase, payload):
                            owned = _owned_diagnostic_value(payload)
                            if not isinstance(owned, dict):
                                raise ValueError("Bundle diagnostic phase must be an object")
                            selected_diagnostic_frames = tuple(
                                getattr(self.bundle_diagnostic_writer, "frames", ())
                            )
                            cohort_start = min(selected_diagnostic_frames) if selected_diagnostic_frames else None
                            cohort_finish = max(selected_diagnostic_frames) if len(selected_diagnostic_frames) > 1 else None
                            if phase == "prepared":
                                history_table = owned.get("source_history_observations")
                                if isinstance(history_table, dict):
                                    capture = source_history_capture if isinstance(source_history_capture, dict) else {}
                                    source_frame = history_table.get("source_frame")
                                    if source_frame is None:
                                        source_frame = capture.get("frame")
                                    endpoint = None
                                    table_valid = False
                                    table_error = "source_history_capture_unavailable"
                                    try:
                                        if (isinstance(source_frame, int)
                                                and not isinstance(source_frame, bool)):
                                            with self.map.lock:
                                                if (self.map.revision != owned.get("revision")
                                                        or self.map.geometry_revision
                                                        != owned.get("geometry_revision")):
                                                    raise ValueError("prepared_map_epoch_changed")
                                                captured_calibration = capture.get(
                                                    "live_calibration_identity")
                                                if (captured_calibration is None
                                                        or self._live_owned_bundle_calibration()
                                                        != captured_calibration):
                                                    raise ValueError("source_calibration_changed")
                                                endpoint = self._source_history_endpoint(source_frame)
                                                if endpoint is None:
                                                    raise ValueError("source_endpoint_unavailable")
                                                if (capture.get("source_status") != endpoint["status"]
                                                        or capture.get("source_anchor_keyframe_id")
                                                        != endpoint["anchor_keyframe_id"]
                                                        or not np.array_equal(
                                                            capture.get("source_pose"), endpoint["pose"]
                                                        )
                                                        or not np.array_equal(
                                                            capture.get("source_anchor_pose"),
                                                            endpoint["anchor_pose"]
                                                        )):
                                                    raise ValueError("source_endpoint_changed_since_capture")
                                                table_valid = True
                                                table_error = None
                                        history_table.update({
                                            "source_frame": source_frame,
                                            "target_frame": int(index),
                                            "source_calibration_identity": (
                                                capture.get("live_calibration_identity")
                                                or capture.get("calibration_identity")
                                            ),
                                            "revision": owned.get("revision"),
                                            "geometry_revision": owned.get("geometry_revision"),
                                            "initial_source_camera_to_world": (
                                                None if endpoint is None else endpoint["pose"].tolist()
                                            ),
                                            "source_status": (
                                                None if endpoint is None else endpoint["status"]
                                            ),
                                            "source_anchor_id": (
                                                None if endpoint is None else endpoint["anchor_keyframe_id"]
                                            ),
                                            "initial_source_anchor_camera_to_world": (
                                                None if endpoint is None else endpoint["anchor_pose"].tolist()
                                            ),
                                            "source_snapshot_valid": table_valid,
                                            "source_snapshot_error": table_error,
                                        })
                                    except Exception as error:
                                        self.bundle_diagnostic_errors.append({
                                            "frame": int(index), "phase": "source_history_table",
                                            "error_type": type(error).__name__,
                                        })
                                    if index == cohort_start:
                                        self._source_history_diagnostic_cohort = {
                                            "cohort_BA_frame": int(index),
                                            "source_history_observations": _owned_diagnostic_value(history_table),
                                        }
                            if (phase == "prepared" and cohort_finish is not None
                                    and index == cohort_finish
                                    and self._source_history_diagnostic_cohort is not None):
                                owned["source_history_cohort_state"] = (
                                    self._source_history_cohort_diagnostic_state(index)
                                )
                            if phase in ("prepared", "solved"):
                                owned["shared_slam_context"] = (
                                    self._bundle_diagnostic_tracking_context(index, size)
                                )
                            if phase == "prepared":
                                training_snapshot = self._bundle_training_observations_context(
                                    index, size, diagnostic_training_context,
                                    diagnostic_training_report, owned,
                                )
                                training_snapshot["tracking_reference_status"] = (
                                    info.get("stereo_reference_verification")
                                )
                                full_fallback_report = info.get(
                                    "full_supported_reference_fallback"
                                )
                                training_snapshot["full_supported_reference_used"] = bool(
                                    isinstance(full_fallback_report, dict)
                                    and full_fallback_report.get("eligible", False)
                                )
                                final_reference_selected = False
                                if (isinstance(diagnostic_training_context, dict)
                                        and verified_motion is not None):
                                    saved_verified = diagnostic_training_context.get("verified")
                                    if isinstance(saved_verified, dict):
                                        final_reference_selected = _reserved_training_selected_as_edge(
                                            info.get("pose_source"),
                                            arbitration_report.get("choice")
                                            if isinstance(arbitration_report, dict) else None,
                                            verified_motion[0],
                                            diagnostic_training_context["previous"].frame,
                                            verified_motion[1],
                                            saved_verified.get("measurement"),
                                            full_supported_fallback_attempted,
                                        )
                                training_snapshot["reserved_training_selected_as_edge"] = (
                                    final_reference_selected
                                )
                                training_snapshot["consumed_full_pool"] = bool(
                                    full_supported_fallback_attempted
                                )
                                training_snapshot["tracking_decision"] = {
                                    "verification": info.get("stereo_reference_verification"),
                                    "pose_source": info.get("pose_source"),
                                    "arbitration_choice": (
                                        arbitration_report.get("choice")
                                        if isinstance(arbitration_report, dict) else None
                                    ),
                                    "final_reference_measurement": (
                                        _owned_diagnostic_value(verified_motion[1])
                                        if verified_motion is not None else None
                                    ),
                                    "final_reference_source_frame": (
                                        int(verified_motion[0])
                                        if verified_motion is not None else None
                                    ),
                                }
                                if (training_snapshot.get("status") == "captured"
                                        and not final_reference_selected):
                                    training_snapshot["eligible"] = False
                                    training_snapshot["reason"] = (
                                        "full_supported_pool_replaced_reserved_reference"
                                        if training_snapshot["consumed_full_pool"] else
                                        "reserved_reference_not_selected"
                                    )
                                owned["shared_slam_context"]["stereo_training_observations"] = (
                                    training_snapshot
                                )
                                selection = owned.get("selection", {})
                                diagnostic_inputs["window_keyframe_ids"] = (
                                    selection.get("window_keyframe_ids", [])
                                    if isinstance(selection, dict) else []
                                )
                                diagnostic_inputs["motion_checks"] = owned.get(
                                    "motion_checks", []
                                )
                                diagnostic_inputs["window_keyframe_ids"] = (
                                    _owned_diagnostic_value(
                                        diagnostic_inputs["window_keyframe_ids"]
                                    )
                                )
                                diagnostic_inputs["motion_checks"] = (
                                    _owned_diagnostic_value(
                                        diagnostic_inputs["motion_checks"]
                                    )
                                )
                            self._emit_bundle_diagnostic(index, phase, owned, size)

                        bundle_kwargs["diagnostic_sink"] = diagnostic_sink
                    if self.config.stereo_owned_image_bundle:
                        bundle_kwargs["training_factor_provider"] = (
                            self._owned_stereo_training_provider(
                                index, size, owned_training_context, arbitration_report,
                                info, verified_motion, full_supported_fallback_attempted,
                            )
                        )
                    if getattr(self.config, "stereo_source_history_bundle", False):
                        bundle_kwargs["source_history_provider"] = (
                            self._owned_source_history_provider(
                                source_history_capture, owned_training_context,
                                arbitration_report, info, verified_motion,
                                full_supported_fallback_attempted,
                            )
                        )
                    if getattr(self.config, "stereo_retained_source_observations", False):
                        retained_exclusions = self._retained_source_exclusion_snapshot(
                            index, size
                        )
                        retained_calibration = self._live_owned_bundle_calibration()
                        if not isinstance(retained_calibration, str) or not retained_calibration:
                            retained_exclusions = {
                                "valid": False, "reason": "retained_calibration_invalid"
                            }
                        bundle_kwargs.update({
                            "retain_source_observations": True,
                            "retained_source_calibration_identity": retained_calibration,
                            "retained_source_exclusions": retained_exclusions,
                        })
                    report = local_bundle_adjustment(
                        self.map,
                        self.K,
                        self.stereo.baseline if self.stereo is not None else 0.0,
                        **bundle_kwargs,
                    )
                    bundle_called = True
                    if diagnostic_capture:
                        try:
                            finished = {
                                "schema": "local_bundle_finished_v1",
                                "frame": int(index),
                                "report": _owned_diagnostic_value(report),
                                **self._bundle_diagnostic_finish_context(
                                    index, size, report,
                                    diagnostic_inputs["window_keyframe_ids"],
                                    diagnostic_inputs["motion_checks"],
                                ),
                            }
                            selected_diagnostic_frames = tuple(
                                getattr(self.bundle_diagnostic_writer, "frames", ())
                            )
                            if (len(selected_diagnostic_frames) > 1
                                    and index == max(selected_diagnostic_frames)
                                    and self._source_history_diagnostic_cohort is not None):
                                finished["source_history_cohort_state"] = (
                                    self._source_history_cohort_diagnostic_state(index)
                                )
                            self._emit_bundle_diagnostic(index, "finished", finished, size)
                        except Exception as error:
                            self.bundle_diagnostic_errors.append({
                                "frame": int(index), "phase": "finished",
                                "error_type": type(error).__name__,
                            })
                        selected_diagnostic_frames = tuple(
                            getattr(self.bundle_diagnostic_writer, "frames", ())
                        )
                        if (len(selected_diagnostic_frames) > 1
                                and index == max(selected_diagnostic_frames)):
                            self._source_history_diagnostic_cohort = None
                        try:
                            writer_frame = next(
                                item for item in self.bundle_diagnostic_writer.manifest().get("frames", [])
                                if item.get("frame") == index
                            )
                            phases = set(writer_frame.get("phases", {}))
                            if "prepared" not in phases:
                                self._skip_bundle_diagnostic(
                                    index, report.get("reason", "prepared_phase_missing")
                                )
                            elif phases != {"prepared", "solved", "finished"}:
                                self.bundle_diagnostic_errors.append({
                                    "frame": int(index), "phase": "post_report_check",
                                    "error_type": "IncompleteCapture",
                                    "missing_phases": sorted(
                                        {"prepared", "solved", "finished"} - phases
                                    ),
                                })
                        except Exception as error:
                            self.bundle_diagnostic_errors.append({
                                "frame": int(index), "phase": "post_report_check",
                                "error_type": type(error).__name__,
                            })
                        if self.bundle_diagnostic_errors:
                            report["bundle_diagnostic_errors"] = [
                                item for item in self.bundle_diagnostic_errors
                                if item.get("frame") == int(index)
                            ]
            self.bundle_reports.append({"frame": index, **report})
            pose = self.map.poses[-1].copy()
            self.loop_worker.schedule(self.map)
        elif diagnostic_capture:
            if not info["tracking_ok"]:
                bundle_skip_reason = "tracking_not_accepted"
            elif not self.config.bundle_enabled:
                bundle_skip_reason = "bundle_disabled"
            elif anchor is None or self.map.keyframes.get(anchor) is None \
                    or self.map.keyframes[anchor].frame != index:
                bundle_skip_reason = "not_keyframe"
            elif len(self.map.keyframes) < 3:
                bundle_skip_reason = "insufficient_keyframes"
            else:
                bundle_skip_reason = "bundle_not_invoked"
        if diagnostic_capture and not bundle_called:
            self._skip_bundle_diagnostic(index, bundle_skip_reason or "bundle_not_invoked")
        if trace_capture and index == self._tracking_trace_cohort_frame:
            self._tracking_trace_event(index, "cohort_map_after_ba",
                self._tracking_trace_frame_state(index, include_landmarks=True,
                                                 stage="cohort_map_after_ba"))
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
            if raw_stereo_enabled:
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
        if self.config.stereo_depth_policy in ('verified_fallback', 'verified_all'):
            info['stereo_depth_verification'] = dict(self.stereo_depth_verification)
        if trace_capture:
            try:
                ledger = self._tracking_trace_reference_context(index, arbitration_report, info)
                self._tracking_trace_event(index, "frame_end", {
                    **ledger, "tracking_status": status,
                    "accepted_pose": np.asarray(self.map.poses[index]).copy(),
                    "revision": int(self.map.revision),
                    "geometry_revision": int(self.map.geometry_revision),
                    "unused_evidence_certified": False,
                })
                self.tracking_diagnostics_writer.finish_frame(index, tracking_status=status)
            except Exception as error:
                self.tracking_diagnostic_errors.append({
                    "frame": int(index), "phase": "finish", "error_type": type(error).__name__})
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
