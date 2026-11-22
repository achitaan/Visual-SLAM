"""Snapshot-based asynchronous image-verified SE(3)/Sim(3) loop correction."""

from concurrent.futures import ThreadPoolExecutor
import copy
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from mapping_geometry import match_descriptors, coverage
from metrics import umeyama_alignment
from loop_geometry import StereoLoopFrame, verify_loop
from pose_graph import optimize
from keyframe_retrieval import KeyframeRetrieval
from keyframe_index import KeyframeIndex
from performance import profiled
from stage_profile import StageProfile

_ORIG_VERIFY_LOOP = verify_loop


def similarity(source, target):
    rotation, scale, translation = umeyama_alignment(source, target, with_scale=True)
    if not np.isfinite(scale) or scale <= 0:
        return None
    return rotation, float(scale), translation


def verify_similarity(first, second, matrix, min_inliers=30):
    pairs = match_descriptors(first.descriptors, second.descriptors)
    if not len(pairs):
        return None
    valid = np.isfinite(first.depth_points[pairs[:, 0]]).all(axis=1) & np.isfinite(
        second.depth_points[pairs[:, 1]]
    ).all(axis=1)
    pairs = pairs[valid]
    if len(pairs) < min_inliers:
        return None
    a, b = pairs.T
    target, source = first.depth_points[a], second.depth_points[b]
    rng = np.random.default_rng(0)
    best = np.empty(0, int)
    threshold = 0.02 * np.median(np.linalg.norm(target, axis=1))
    for _ in range(300):
        ids = rng.choice(len(pairs), 3, replace=False)
        if np.linalg.matrix_rank(source[ids] - source[ids].mean(axis=0)) < 2:
            continue
        candidate = similarity(source[ids], target[ids])
        if candidate is None:
            continue
        r, s, t = candidate
        errors = np.linalg.norm(s * (source @ r.T) + t - target, axis=1)
        ids = np.flatnonzero(errors < threshold)
        if len(ids) > len(best):
            best = ids
    if len(best) < min_inliers or len(best) / len(pairs) < 0.35:
        return None
    result = similarity(source[best], target[best])
    if result is None:
        return None
    r, s, t = result
    moved = s * (source[best] @ r.T) + t
    reverse = (target[best] - t) @ r / s

    def pixels(points):
        projected = points @ matrix.T
        return projected[:, :2] / projected[:, 2:]

    errors = np.r_[
        np.linalg.norm(pixels(moved) - first.pixels[a[best]], axis=1),
        np.linalg.norm(pixels(reverse) - second.pixels[b[best]], axis=1),
    ]
    if (
        np.any(moved[:, 2] <= 0)
        or np.any(reverse[:, 2] <= 0)
        or np.median(errors) > 1.5
    ):
        return None
    # Verify support on both cameras, not just descriptor appearance.
    fallback = (int(matrix[0, 2] * 2), int(matrix[1, 2] * 2))
    if (
        coverage(first.pixels[a[best]], first.image_size or fallback) < 3
        or coverage(second.pixels[b[best]], second.image_size or fallback) < 3
    ):
        return None
    return {
        "rotation": r,
        "scale": s,
        "translation": t,
        "inliers": len(best),
        "median_reprojection_px": float(np.median(errors)),
    }


def optimize_similarities(poses, edges, max_evaluations=200):
    """Fix first similarity; return rigid camera poses and camera-to-map scales."""
    count = len(poses)
    if count < 2:
        return [p.copy() for p in poses], np.ones(count)
    initial = np.concatenate(
        [
            np.r_[Rotation.from_matrix(p[:3, :3]).as_rotvec(), p[:3, 3], 0.0]
            for p in poses[1:]
        ]
    )

    def unpack(x):
        values = np.r_[np.zeros(7), x].reshape(-1, 7)
        rotations = np.concatenate(
            [poses[0][None, :3, :3], Rotation.from_rotvec(values[1:, :3]).as_matrix()]
        )
        translations = values[:, 3:6].copy()
        translations[0] = poses[0][:3, 3]
        scales = np.exp(np.clip(values[:, 6], -10, 10))
        return rotations, translations, scales

    def residual(x):
        rotations, translations, scales = unpack(x)
        out = []
        for i, j, r, s, t, weight in edges:
            predicted_r = rotations[i].T @ rotations[j]
            predicted_s = scales[j] / scales[i]
            predicted_t = (
                rotations[i].T @ (translations[j] - translations[i]) / scales[i]
            )
            out.extend(
                np.r_[
                    Rotation.from_matrix(r.T @ predicted_r).as_rotvec() * 10,
                    (r.T @ (predicted_t - t) / s) * 3,
                    np.log(predicted_s / s) * 10,
                ]
                * weight
            )
        return np.array(out)

    from scipy.sparse import lil_matrix

    pattern = lil_matrix((7 * len(edges), len(initial)), dtype=int)
    for n, edge in enumerate(edges):
        for v in edge[:2]:
            if v:
                pattern[7 * n : 7 * n + 7, 7 * (v - 1) : 7 * v] = 1
    result = least_squares(
        residual,
        initial,
        jac_sparsity=pattern.tocsr(),
        tr_solver="lsmr",
        loss="huber",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=max_evaluations,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("Sim(3) graph did not converge")
    rotations, translations, scales = unpack(result.x)
    output = []
    for r, t in zip(rotations, translations):
        p = np.eye(4)
        p[:3, :3] = r
        p[:3, 3] = t
        output.append(p)
    return output, scales


def fuse_loop_landmarks(
    map_state,
    query_kf_id: int,
    match_kf_id: int,
    T_match_query: np.ndarray,
    K: np.ndarray,
    radius_px: float = 4.0,
) -> int:
    """Fuse duplicate landmarks across a verified loop pair (match_kf_id, query_kf_id)."""
    from local_map_tracker import fuse_neighbor_landmarks

    _ = T_match_query
    if query_kf_id not in map_state.keyframes or match_kf_id not in map_state.keyframes:
        return 0
    query_kf = map_state.keyframes[query_kf_id]
    # Project landmarks from match_kf (and its immediate neighbors) into query_kf after pose correction
    neighbors = [k for k in range(max(0, match_kf_id - 2), match_kf_id + 3) if k in map_state.keyframes]
    return fuse_neighbor_landmarks(
        map_state, query_kf, neighbors, K, radius_px=radius_px
    )


class LiveLoopWorker:
    def __init__(
        self,
        matrix,
        metric,
        mode="live",
        performance=None,
        profiler=None,
        matcher=None,
        max_keyframes=300,
        use_weighted_pgo=False,
        use_vlad_retrieval=False,
    ):
        if mode not in ("off", "live", "offline"):
            raise ValueError("Loop mode must be off, live or offline")
        retrieval = getattr(performance, "retrieval", "current")
        if use_vlad_retrieval:
            retrieval = "indexed"
        if retrieval not in ("current", "indexed", "exhaustive"):
            raise ValueError("Retrieval must be current, indexed or exhaustive")
        self.mode = mode
        self.finalizing = False
        self.performance = performance
        self.retrieval_mode = retrieval
        self.max_keyframes = int(max_keyframes)
        self.use_weighted_pgo = bool(use_weighted_pgo)
        self.use_vlad_retrieval = bool(use_vlad_retrieval)
        self.retrieval = KeyframeRetrieval() if retrieval == "current" else None
        self.retrieval_index = KeyframeIndex() if retrieval == "indexed" else None
        self.cpu_optimizations = bool(getattr(performance, "cpu_optimizations", False))
        self.matcher = matcher if matcher is not None else match_descriptors
        profile_enabled = bool(getattr(performance, "profile", False))
        self.profiler = profiler if profiler is not None else StageProfile(
            detailed=profile_enabled, enabled=profile_enabled
        )
        self.matrix = matrix.copy()
        self.metric = metric
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="loop-optimization"
        )
        self.future = None
        self.last_scheduled_keyframe = None
        self.events = []
        self.verified = {}
        self.pending_loops = {}
        self.pending_pairs = set()
        self.last_correction_before = None
        self.last_correction_after = None
        self.snapshot_poses = None
        self.snapshot_geometry_revision = None
        self.snapshot_pose_revision = None
        self._kf_static_cache = {}
        self._verified_timestamps_s = {}

    @staticmethod
    def _snapshot_array(value, always_copy=False):
        if not isinstance(value, np.ndarray):
            return value
        if always_copy or value.flags.writeable:
            return value.copy()
        return value

    @classmethod
    def _snapshot_keyframe(cls, frame):
        """Copy changing geometry and share feature/image arrays only if frozen."""
        saved = copy.copy(frame)
        for name in ("pose", "landmark_ids", "depth_points"):
            if hasattr(frame, name):
                setattr(saved, name, cls._snapshot_array(getattr(frame, name), always_copy=True))
        for name in ("pixels", "descriptors", "image", "retrieval_descriptors"):
            if hasattr(frame, name):
                setattr(saved, name, cls._snapshot_array(getattr(frame, name)))
        return saved

    def _build_static_entry(self, frame):
        dp = frame.depth_points
        dp_copy = None if dp is None else dp.copy()
        pixels = self._snapshot_array(frame.pixels)
        descriptors = self._snapshot_array(frame.descriptors)
        image = self._snapshot_array(frame.image) if hasattr(frame, "image") else None
        landmark_ids = (
            self._snapshot_array(frame.landmark_ids, always_copy=True)
            if hasattr(frame, "landmark_ids")
            else None
        )
        retrieval_desc = (
            self._snapshot_array(frame.retrieval_descriptors)
            if hasattr(frame, "retrieval_descriptors")
            else descriptors
        )
        img_size = frame.image_size or (
            int(self.matrix[0, 2] * 2),
            int(self.matrix[1, 2] * 2),
        )
        if dp_copy is None:
            measured_desc = descriptors[:0]
            loop_frame = StereoLoopFrame(
                pixels[:0], np.empty((0, 3), dtype=np.float64), descriptors[:0], img_size
            )
        else:
            valid = np.isfinite(dp_copy).all(axis=1)
            measured_desc = np.ascontiguousarray(descriptors[valid])
            loop_frame = StereoLoopFrame(
                pixels[valid], dp_copy[valid], measured_desc, img_size
            )
        return (
            frame.frame,
            pixels,
            descriptors,
            retrieval_desc,
            dp_copy,
            img_size,
            measured_desc,
            loop_frame,
            image,
            landmark_ids,
        )

    def _lightweight_snapshot_keyframe(self, ident, frame, refresh_static):
        entry = None if refresh_static else self._kf_static_cache.get(ident)
        if entry is None:
            entry = self._build_static_entry(frame)
            self._kf_static_cache[ident] = entry
        saved = copy.copy(frame)
        saved.pose = frame.pose.copy()
        saved.pixels = entry[1]
        saved.descriptors = entry[2]
        if hasattr(frame, "retrieval_descriptors"):
            saved.retrieval_descriptors = entry[3]
        saved.depth_points = entry[4]
        if hasattr(frame, "image"):
            saved.image = entry[8]
        if hasattr(frame, "landmark_ids"):
            saved.landmark_ids = entry[9]
        saved._cached_measured_descriptors = entry[6]
        saved._cached_stereo_loop_frame = entry[7]
        return saved

    def schedule(self, state):
        if self.mode == "off":
            return
        if self.future is not None or len(state.keyframes) < 3:
            return
        with state.lock:
            if self.max_keyframes > 0 and len(state.keyframes) > self.max_keyframes:
                self.events.append({"type": "loop_skipped", "reason": "graph_size_not_validated"})
                return
            latest_id = max(state.keyframes)
            first_id = min(state.keyframes)
            latest = state.keyframes[latest_id]
            first = state.keyframes[first_id]
            if latest.frame - first.frame < 150:
                return
            last_sched_frame = getattr(self, "_last_scheduled_frame", -10**9)
            if (
                not self.pending_loops
                and not self.finalizing
                and (latest.frame - last_sched_frame) < 10
            ):
                return
            self._last_scheduled_frame = int(latest.frame)
            if self.cpu_optimizations or self.use_vlad_retrieval:
                # Only the newest 2 keyframes can have depth_points updated by triangulation.
                refresh_cutoff = latest_id - 1
                snapshot = {
                    ident: self._lightweight_snapshot_keyframe(
                        ident, frame, ident >= refresh_cutoff
                    )
                    for ident, frame in state.keyframes.items()
                }
            else:
                snapshot = copy.deepcopy(state.keyframes)
            revision = state.revision
            self.snapshot_poses = {i: k.pose.copy() for i, k in snapshot.items()}
            self.snapshot_geometry_revision = state.geometry_revision
            self.snapshot_pose_revision = getattr(state, "pose_revision", None)
        current = max(snapshot)
        if snapshot[current].frame - snapshot[0].frame < 150:
            return
        previous_loops = copy.deepcopy({**self.verified, **self.pending_loops})
        self.last_scheduled_keyframe = current
        self.future = self.executor.submit(
            self._solve,
            snapshot,
            revision,
            previous_loops,
            bool(self.pending_loops) or (self.mode == "offline" and self.finalizing),
            set(self.pending_pairs),
        )

    @profiled("background_loops")
    def _solve(self, keyframes, revision, loops, force=False, pending_pairs=None):
        current = max(keyframes)
        last = keyframes[current]

        def measured_descriptors(frame):
            cached = getattr(frame, "_cached_measured_descriptors", None)
            if cached is not None:
                return cached
            if frame.depth_points is None:
                return frame.descriptors[:0]
            valid = np.isfinite(frame.depth_points).all(axis=1)
            return frame.descriptors[valid]

        query = measured_descriptors(last)
        eligible = [i for i, k in keyframes.items() if last.frame - k.frame >= 150]
        if self.retrieval_mode == "current":
            # Preserve the established descriptor-sketch shortlist exactly.
            self.retrieval.update(keyframes)
            shortlisted = self.retrieval.query(query, 8, allowed=eligible)
        elif self.retrieval_mode == "indexed":
            def appearance_descriptors(frame):
                return getattr(frame, "retrieval_descriptors", frame.descriptors)

            with self.profiler.measure("loop_retrieval_index"):
                for ident, frame in keyframes.items():
                    self.retrieval_index.upsert(
                        ident, frame.frame, appearance_descriptors(frame)
                    )
                shortlisted = self.retrieval_index.query(
                    appearance_descriptors(last),
                    eligible,
                    limit=8 if self.use_vlad_retrieval else 20,
                )
            # Vocabulary startup signals a bounded exhaustive fallback.
            if shortlisted is None:
                shortlisted = eligible
        else:
            shortlisted = eligible
        self.profiler.count("loop_keyframes_matched", len(shortlisted))
        candidates = []
        precomputed_pairs = {}
        for i in shortlisted:
            matched_pairs = self.profiler.call(
                "loop_matching", self.matcher, measured_descriptors(keyframes[i]), query
            )
            candidates.append((len(matched_pairs), i))
            precomputed_pairs[(i, current)] = matched_pairs
        # The indexed proposal is wider, but exact reranking retains the same
        # eight-candidate budget before the existing top-three verification.
        candidates = sorted(candidates, reverse=True)[:8]
        newly_verified = []
        verified_timestamps = {}
        attempts = []
        min_support = 20 if self.use_vlad_retrieval else 0
        pairs = [
            (support, i, current)
            for support, i in candidates[:3]
            if support >= min_support
        ]
        pairs.extend(
            (0, i, j)
            for i, j in (pending_pairs or set())
            if (i, j) not in {(a, b) for _, a, b in pairs}
        )
        for support, i, j in pairs:
            first = keyframes[i]
            query_frame = keyframes[j]
            if self.metric:

                def frame(k):
                    cached_lf = getattr(k, "_cached_stereo_loop_frame", None)
                    if cached_lf is not None:
                        return cached_lf
                    points = (
                        k.depth_points
                        if k.depth_points is not None
                        else np.full((len(k.pixels), 3), np.nan)
                    )
                    good = np.isfinite(points).all(axis=1)
                    return StereoLoopFrame(
                        k.pixels[good],
                        points[good],
                        k.descriptors[good],
                        k.image_size
                        or (int(self.matrix[0, 2] * 2), int(self.matrix[1, 2] * 2)),
                    )

                pair_matches = precomputed_pairs.get((i, j))
                if verify_loop is _ORIG_VERIFY_LOOP and pair_matches is not None:
                    result = self.profiler.call(
                        "loop_verification",
                        verify_loop,
                        frame(first),
                        frame(query_frame),
                        self.matrix,
                        pairs=pair_matches,
                    )
                else:
                    result = self.profiler.call(
                        "loop_verification",
                        verify_loop,
                        frame(first),
                        frame(query_frame),
                        self.matrix,
                    )
            else:
                result = self.profiler.call(
                    "loop_verification", verify_similarity, first, query_frame, self.matrix
                )
            attempts.append(
                {
                    "first_keyframe": i,
                    "second_keyframe": j,
                    "appearance_matches": support,
                    "geometrically_verified": result is not None,
                }
            )
            if result is not None:
                loops[(i, j)] = result
                newly_verified.append((i, j))
                verified_timestamps[(i, j)] = time.perf_counter()
        if (self.mode == "offline" and not self.finalizing) or (not newly_verified and not force):
            return {
                "revision": revision,
                "loops": loops,
                "correction": None,
                "attempts": attempts,
                "verified_timestamps": verified_timestamps,
            }
        pgo_start_s = time.perf_counter()
        poses = [k.pose for k in keyframes.values()]
        if self.metric:
            if self.use_weighted_pgo:
                from pose_graph import optimize_weighted_pose_graph

                odom_edges = [
                    (i, i + 1, np.linalg.inv(poses[i]) @ poses[i + 1])
                    for i in range(len(poses) - 1)
                ]
                loop_edges = [
                    (i, j, result["measurement"], result)
                    for (i, j), result in loops.items()
                ]
                opt_dict = self.profiler.call(
                    "graph_optimization",
                    optimize_weighted_pose_graph,
                    poses,
                    odom_edges,
                    loop_edges,
                )
                corrected = [opt_dict[i] for i in range(len(poses))]
                scales = np.ones(len(poses))
            else:
                edges = []
                for i in range(len(poses) - 1):
                    edges.append(
                        (
                            i,
                            i + 1,
                            np.linalg.inv(poses[i]) @ poses[i + 1],
                            np.diag([1000.0] * 3 + [10.0] * 3),
                            "odometry",
                        )
                    )
                for (i, j), result in loops.items():
                    edges.append(
                        (
                            i,
                            j,
                            result["measurement"],
                            np.diag([2000.0] * 3 + [20.0] * 3),
                            "loop",
                        )
                    )
                corrected = self.profiler.call(
                    "graph_optimization", optimize, poses, edges, max_evaluations=500
                )
                scales = np.ones(len(poses))
        else:
            edges = []
            for i in range(len(poses) - 1):
                z = np.linalg.inv(poses[i]) @ poses[i + 1]
                edges.append((i, i + 1, z[:3, :3], 1.0, z[:3, 3], 1.0))
            for (i, j), result in loops.items():
                edges.append(
                    (
                        i,
                        j,
                        result["rotation"],
                        result["scale"],
                        result["translation"],
                        2.0,
                    )
                )
            corrected, scales = self.profiler.call(
                "graph_optimization", optimize_similarities, poses, edges
            )
        pgo_end_s = time.perf_counter()
        return {
            "revision": revision,
            "loops": loops,
            "correction": dict(enumerate(corrected)),
            "scales": dict(enumerate(scales)),
            "new_loops": newly_verified,
            "attempts": attempts,
            "verified_timestamps": verified_timestamps,
            "pgo_elapsed_s": float(pgo_end_s - pgo_start_s),
        }

    def poll(self, state, wait=False):
        if self.future is None or (not wait and not self.future.done()):
            return False
        future = self.future
        self.future = None
        try:
            result = future.result()
        except Exception as exc:
            self.events.append({"type": "loop_failed", "reason": str(exc)})
            return False
        for pair, ts in (result.get("verified_timestamps") or {}).items():
            self._verified_timestamps_s.setdefault(pair, float(ts))
        if result.get("attempts"):
            self.events.append(
                {
                    "type": "loop_verification",
                    "attempts": result["attempts"],
                    "snapshot_revision": result["revision"],
                }
            )
            self.pending_pairs.difference_update(
                (a["first_keyframe"], a["second_keyframe"]) for a in result["attempts"]
            )
        if result["correction"] is None:
            if self.mode == "offline":
                self.verified.update(result["loops"])
            return False
        before = [pose.copy() for pose in state.poses]
        if self.snapshot_poses is not None:
            if self.snapshot_pose_revision is not None:
                applied = state.apply_snapshot_corrections(
                    result["revision"], self.snapshot_geometry_revision, self.snapshot_poses,
                    result["correction"], result["scales"],
                    pose_revision=self.snapshot_pose_revision,
                )
            else:
                applied = state.apply_snapshot_corrections(
                    result["revision"], self.snapshot_geometry_revision, self.snapshot_poses,
                    result["correction"], result["scales"],
                )
        else:
            applied = state.apply_corrections(
                result["revision"], result["correction"], result["scales"]
            )
        if applied:
            applied_at_s = time.perf_counter()
            if not self.metric:
                for (i, j), measurement in result["loops"].items():
                    measurement["scale"] *= result["scales"][i] / result["scales"][j]
                    measurement["translation"] = (
                        measurement["translation"] * result["scales"][i]
                    )
                self._kf_static_cache.clear()
            elif self.use_weighted_pgo:
                for (match_kf, query_kf) in result.get("new_loops", ()):
                    loop_rec = result["loops"].get((match_kf, query_kf))
                    if loop_rec is not None and "measurement" in loop_rec:
                        fuse_loop_landmarks(
                            state, query_kf, match_kf, loop_rec["measurement"], self.matrix
                        )
            newly_applied = [pair for pair in result["loops"] if pair not in self.verified]
            v2a_latencies = [
                max(0.0, applied_at_s - self._verified_timestamps_s[pair])
                for pair in newly_applied
                if pair in self._verified_timestamps_s
            ]
            self.verified = result["loops"]
            self.last_correction_before = before
            self.last_correction_after = [pose.copy() for pose in state.poses]
            self.pending_loops.clear()
            self.pending_pairs.clear()
            self.events.append(
                {
                    "type": "loop_applied",
                    "loops": len(self.verified),
                    "revision": state.revision,
                    "pgo_elapsed_s": float(result.get("pgo_elapsed_s", 0.0)),
                    "verification_to_apply_s": v2a_latencies,
                }
            )
            return True
        # Stereo camera-space measurements survive rigid map updates. Monocular
        # measurements are reverified against the latest depth units in the worker.
        if self.metric:
            self.pending_loops.update(
                {
                    pair: measurement
                    for pair, measurement in result["loops"].items()
                    if pair not in self.verified
                }
            )
        else:
            self.pending_pairs.update(result.get("new_loops", []))
        self.events.append({"type": "loop_discarded", "reason": "stale_revision"})
        return False

    def close(self, state, finish=True):
        if not finish:
            if self.future is not None:
                self.future.cancel()
            self.executor.shutdown(wait=False, cancel_futures=True)
            return
        self.poll(state, wait=True)
        self.finalizing = True
        # Once tracking stops, a fresh snapshot can finish without becoming stale.
        stale = self.events and self.events[-1]["type"] == "loop_discarded"
        latest = max(state.keyframes) if state.keyframes else None
        if stale or latest != self.last_scheduled_keyframe or (self.mode == "offline" and self.verified):
            self.schedule(state)
            self.poll(state, wait=True)
        self.executor.shutdown(wait=True)
