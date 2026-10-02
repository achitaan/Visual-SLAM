"""Declared tracking ablations; pass normal evaluate_shared_slam arguments too."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import evaluate_shared_slam as evaluator
import shared_slam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disable-motion-prediction", action="store_true")
    parser.add_argument("--disable-flow", action="store_true")
    parser.add_argument(
        "--unseeded-flow",
        action="store_true",
        help="Start optical flow at the previous pixel while retaining map/PnP prediction",
    )
    parser.add_argument(
        "--unseeded-pnp",
        action="store_true",
        help="Estimate map PnP without the motion-prior solver initialization",
    )
    parser.add_argument("--disable-bundle", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args, evaluation_arguments = parser.parse_known_args()
    overrides = {
        "disable_motion_prediction": args.disable_motion_prediction,
        "disable_flow": args.disable_flow,
        "unseeded_flow": args.unseeded_flow,
        "unseeded_pnp": args.unseeded_pnp,
        "disable_bundle": args.disable_bundle,
    }
    if not any(overrides.values()):
        parser.error("Select at least one diagnostic ablation")
    declaration = {
        "diagnostic_overrides": overrides,
        "diagnostic_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostic.json").write_text(json.dumps(declaration, indent=2))
    if args.disable_motion_prediction:
        evaluator.SharedSlam._motion_prediction = lambda self: self.map.poses[-1].copy()
    if args.disable_flow:
        original_track = evaluator.SharedSlam._track

        def track_without_flow(self, *inputs, **options):
            previous_gray = self.previous_gray
            self.previous_gray = None
            try:
                return original_track(self, *inputs, **options)
            finally:
                self.previous_gray = previous_gray

        evaluator.SharedSlam._track = track_without_flow
    if args.unseeded_flow:
        original_flow = shared_slam.cv.calcOpticalFlowPyrLK

        def flow_without_prediction(*inputs, **options):
            if options.get("flags", 0) & shared_slam.cv.OPTFLOW_USE_INITIAL_FLOW:
                inputs = list(inputs)
                inputs[3] = None
                options["flags"] &= ~shared_slam.cv.OPTFLOW_USE_INITIAL_FLOW
            return original_flow(*inputs, **options)

        shared_slam.cv.calcOpticalFlowPyrLK = flow_without_prediction
    if args.unseeded_pnp:
        original_pose = shared_slam.estimate_pose

        def pose_without_prediction(*inputs, **options):
            options["initial_pose"] = None
            return original_pose(*inputs, **options)

        shared_slam.estimate_pose = pose_without_prediction
    if args.disable_bundle:
        shared_slam.local_bundle_adjustment = lambda *_, **__: {
            "applied": False,
            "reason": "diagnostic_ablation",
        }
    # Reference isolation is owned by the ordinary evaluator: it opens poses
    # after estimator shutdown, with no changes to estimator acceptance tests.
    sys.argv = [sys.argv[0], *evaluation_arguments, "--output", str(args.output)]
    evaluator.main()
    for name in ("evaluation.json", "run.json"):
        path = args.output / name
        report = json.loads(path.read_text())
        report.update(declaration)
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
