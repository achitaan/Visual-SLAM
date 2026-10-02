"""Snapshot-based asynchronous image-verified SE(3)/Sim(3) loop correction."""

from concurrent.futures import ThreadPoolExecutor
import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from mapping_geometry import match_descriptors, coverage
from metrics import umeyama_alignment
from loop_geometry import StereoLoopFrame, verify_loop
from pose_graph import optimize
from performance import StageProfiler, profiled
from keyframe_index import KeyframeIndex


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


class LiveLoopWorker:
    def __init__(self, matrix, metric, profiler=None, retrieval="indexed", cpu_optimizations=True, matcher=None):
        self.profiler = profiler or StageProfiler()
        self.retrieval_index = KeyframeIndex() if retrieval == "indexed" else None
        self.cpu_optimizations = cpu_optimizations
        self.matcher = matcher or match_descriptors
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

    def schedule(self, state):
        if self.future is not None or len(state.keyframes) < 3:
            return
        with state.lock:
            if self.cpu_optimizations:
                current = max(state.keyframes)
                if state.keyframes[current].frame - state.keyframes[0].frame < 150:
                    return
                if len(state.keyframes) > 300:
                    self.events.append({"type": "loop_skipped", "reason": "graph_size_not_validated"})
                    return
                # Descriptors/pixels are immutable after keyframe creation. Geometry
                # changes during BA/corrections, so snapshot those arrays separately.
                snapshot = {}
                for ident, frame in state.keyframes.items():
                    saved = copy.copy(frame)
                    saved.pose = frame.pose.copy()
                    saved.landmark_ids = frame.landmark_ids.copy()
                    saved.depth_points = frame.depth_points.copy() if frame.depth_points is not None else None
                    snapshot[ident] = saved
            else:
                snapshot = copy.deepcopy(state.keyframes)
            revision = state.revision
        current = max(snapshot)
        if snapshot[current].frame - snapshot[0].frame < 150:
            return
        if len(snapshot) > 300:
            self.events.append(
                {"type": "loop_skipped", "reason": "graph_size_not_validated"}
            )
            return
        previous_loops = copy.deepcopy({**self.verified, **self.pending_loops})
        self.last_scheduled_keyframe = current
        self.future = self.executor.submit(
            self._solve,
            snapshot,
            revision,
            previous_loops,
            bool(self.pending_loops),
            set(self.pending_pairs),
        )

    @profiled("background_loops")
    def _solve(self, keyframes, revision, loops, force=False, pending_pairs=None):
        current = max(keyframes)
        last = keyframes[current]

        def measured_descriptors(frame):
            if frame.depth_points is None:
                return frame.descriptors[:0]
            valid = np.isfinite(frame.depth_points).all(axis=1)
            return frame.descriptors[valid]

        query = measured_descriptors(last)
        def appearance_descriptors(frame):
            detected = getattr(frame, "retrieval_descriptors", None)
            return frame.descriptors if detected is None else detected
        eligible = [i for i, k in keyframes.items() if last.frame - k.frame >= 150]
        if self.retrieval_index is not None:
            with self.profiler.measure("loop_retrieval_index"):
                for ident, frame in keyframes.items():
                    self.retrieval_index.upsert(ident, frame.frame, appearance_descriptors(frame))
                shortlist = self.retrieval_index.query(appearance_descriptors(last), eligible)
            if shortlist is not None:
                eligible = shortlist
        self.profiler.count("loop_keyframes_matched", len(eligible))
        candidates = [
            (len(self.profiler.call("loop_matching", self.matcher, measured_descriptors(keyframes[i]), query)), i)
            for i in eligible
        ]
        newly_verified = []
        attempts = []
        pairs = [
            (support, i, current) for support, i in sorted(candidates, reverse=True)[:3]
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

                result = self.profiler.call("loop_verification", verify_loop, frame(first), frame(query_frame), self.matrix)
            else:
                result = self.profiler.call("loop_verification", verify_similarity, first, query_frame, self.matrix)
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
        if not newly_verified and not force:
            return {
                "revision": revision,
                "loops": loops,
                "correction": None,
                "attempts": attempts,
            }
        poses = [k.pose for k in keyframes.values()]
        if self.metric:
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
            corrected = self.profiler.call("graph_optimization", optimize, poses, edges, max_evaluations=500)
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
            corrected, scales = self.profiler.call("graph_optimization", optimize_similarities, poses, edges)
        return {
            "revision": revision,
            "loops": loops,
            "correction": dict(enumerate(corrected)),
            "scales": dict(enumerate(scales)),
            "new_loops": newly_verified,
            "attempts": attempts,
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
            return False
        before = [pose.copy() for pose in state.poses]
        if state.apply_corrections(
            result["revision"], result["correction"], result["scales"]
        ):
            if not self.metric:
                for (i, j), measurement in result["loops"].items():
                    measurement["scale"] *= result["scales"][i] / result["scales"][j]
                    measurement["translation"] = (
                        measurement["translation"] * result["scales"][i]
                    )
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

    def close(self, state):
        self.poll(state, wait=True)
        # Once tracking stops, a fresh snapshot can finish without becoming stale.
        stale = self.events and self.events[-1]["type"] == "loop_discarded"
        latest = max(state.keyframes) if state.keyframes else None
        if stale or latest != self.last_scheduled_keyframe:
            self.schedule(state)
            self.poll(state, wait=True)
        self.executor.shutdown(wait=True)
