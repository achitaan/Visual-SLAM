"""Optional, display-only snapshots. No reference inputs or estimator mutations."""

import hashlib
from itertools import islice
import json
from pathlib import Path
import time

import cv2 as cv
import numpy as np
from telemetry import encode_image, make_frame_message, TelemetryState

REPO = Path(__file__).resolve().parents[1]


def snapshot_target(output, explicit=None):
    if explicit is not None:
        return explicit
    settings = REPO / "results/benchmark-dashboard.json"
    try:
        config = json.loads(settings.read_text(encoding="utf-8"))
        root = (REPO / config["batch_root"]).resolve()
        if (
            config.get("enabled")
            and root.is_relative_to(REPO / "results")
            and output.resolve().is_relative_to(root)
        ):
            return root / "dashboard-frame.json"
    except (OSError, ValueError, KeyError):
        pass
    return None


class SnapshotWriter:
    def __init__(self, target, sequence, run_id, total, interval=1.0):
        self.target = Path(target)
        self.sequence = sequence
        self.run_id = run_id
        self.total = total
        self.interval = interval
        self.last = float("-inf")
        self.elapsed_s = 0.0
        self.error = None
        self.source = Path(__file__).read_bytes()

    def publish(self, slam, index, image, info, force=False):
        now = time.monotonic()
        if self.error or (not force and now - self.last < self.interval):
            return
        started = time.perf_counter()
        try:
            with slam.map.lock:
                if not slam.map.poses:
                    return
                pose = slam.map.poses[-1].copy()
                n = len(slam.map.landmarks)
                points = [
                    p.position.tolist()
                    for p in islice(
                        slam.map.landmarks.values(), 0, None, max(1, (n + 1999) // 2000)
                    )
                ]
                indices = np.linspace(
                    0,
                    len(slam.map.poses) - 1,
                    min(5000, len(slam.map.poses)),
                    dtype=int,
                )
                trajectory = [slam.map.poses[i][:3, 3].tolist() for i in indices]
                map_state = slam.map_state()
            scale = min(1.0, 960 / image.shape[1])
            display = cv.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
            pixels = info.get("feature_points", [])
            inliers = info.get("inlier_mask", [])
            features = [
                {
                    "x": float(p[0] * scale),
                    "y": float(p[1] * scale),
                    "inlier": bool(inliers[j]) if j < len(inliers) else False,
                }
                for j, p in enumerate(pixels[:1500])
            ]
            tracking = {
                k: info[k]
                for k in (
                    "state",
                    "tracking_ok",
                    "num_matches",
                    "num_inliers",
                    "inlier_ratio",
                    "reprojection_error",
                )
                if k in info
            }
            payload = make_frame_message(
                frame_index=index,
                timestamp=time.time(),
                pose_T_wc=pose,
                tracking=tracking,
                map_state=map_state,
                map_points=points,
                image_payload=encode_image(display),
                features=features,
                state=TelemetryState(mode="slam", mode_locked=True),
                translation_scale="metric" if slam.map.metric else "arbitrary",
                sequence=self.sequence,
                total_frames=self.total,
                run_id=self.run_id,
            )
            payload["trajectory"] = trajectory
            payload["snapshot_final"] = force
            self.target.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.target.with_suffix(".part")
            temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            temporary.replace(self.target)
            self.last = now
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            print(f"Dashboard observer disabled: {self.error}", flush=True)
        finally:
            self.elapsed_s += time.perf_counter() - started

    def metadata(self):
        return {
            "enabled": True,
            "interval_s": self.interval,
            "observer_elapsed_s": self.elapsed_s,
            "error": self.error,
            "source_sha256": hashlib.sha256(self.source).hexdigest(),
            "timing_includes_observer": True,
        }
