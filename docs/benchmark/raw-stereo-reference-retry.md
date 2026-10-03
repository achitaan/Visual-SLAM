# Raw stereo reference retry

When a configured stereo reference fails, persistent-map tracking can accept a
small motion that disagrees with the independently supported stereo geometry.
The retained KITTI 01 diagnostic contains this behavior at frames 222, 227 and
233. No local bundle adjustment was accepted between frames 216 and 242, so
those immediate stalls cannot be attributed to an applied BA correction.

`--stereo-raw-reference-retry` enables a separate experiment in the shared stereo
pipeline. After an actual configured-reference failure, it fits against immutable
supported stereo observations, matches appearance descriptors and counts exact
pixel aliases as one physical correspondence. It requires an independently
verified reverse fit, existing spatial support and reprojection gates, and the
existing 0.5 m / 1.5 degree round-trip limits. A valid configured reference is
preserved. Full-pool fits do not claim held-out validation.

The retry remains off by default. It requires `--slam --stereo` in the VO command;
the shared evaluator accepts the same retry flag with `--stereo`. For a diagnostic:

```powershell
.\.venv\Scripts\python scripts/evaluate_shared_slam.py --data-root $data --poses-root $poses --sequence 01 --stereo --loop-mode off --stereo-depth-policy verified_fallback --stereo-pose-arbitration --stereo-raw-reference-retry --max-frames 350 --max-wall-seconds 720 --output results/raw-retry01
```

Estimator inputs contain images and calibration only. Reference trajectories
remain evaluator-only; learned depth remains outside tracking. The evaluator
records the retry mode in configuration and provenance. Enabled and disabled
results cannot satisfy each other's resume identities.

The reviewed implementation is retained in commit
`54bf9003798c843a25300021e1c2a186d23bc57c`. Its backend suite passes 356 tests with
four optional GPU skips. Synthetic tests exercise real bidirectional PnP with
finite but inconsistent configured depths, duplicate appearance rows and
rejection after geometry changes. The synthetic recovered translation is 0.5 m;
this is a correctness check, not a KITTI accuracy result.

## KITTI 04 regression check

A fresh uncached replay processes frames 0–79 with stereo, local BA, arbitration
and raw retry enabled, loops off, and one OpenCV worker. All 80 frames are
processed with zero lost frames. ATE is 0.111884 m, translation drift 0.280753%
and rotation drift 0.005653 degrees/m, identical to the preceding physical-identity
revision on the same prefix. This run does not need the new retry; it checks that
ordinary reference tracking remains stable. It does not validate the affected
KITTI 01 cases.

The evaluator takes 64.50 seconds including export and evaluation; the worker
takes 65.81 seconds and the supervised cycle 69.1 seconds. Peak memory is
264.77 MiB. These are diagnostic measurements. Its finite-export and exact
source/input checks pass. The saved source fingerprint is
`fc8b26eed37b87dce49f9ed01dab1ff84514169ca84ede50d3e98570d3ddd11b`.

![Actual KITTI 04 trajectory, position error, tracking support and sparse map](plots/raw-stereo-reference04-overview.png)

Stateful trajectory evaluation remains necessary. Independently supported pair
diagnostics recover motions at 227 and 233, while 222 remains reverse-inconsistent.
The biased relative comparison at frame 308 is a separate unresolved cause.
Previously published accuracy and runtime results remain attached to their
original revisions. Full paired validation, live-loop evidence, monocular/TUM
checks and dense reconstruction remain release prerequisites.
