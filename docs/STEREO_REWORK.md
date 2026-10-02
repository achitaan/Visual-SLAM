# Stereo reliability development

The experimental snapshot is preserved on `codex/shared-slam-reconstruction`.
Reliability work continues on `codex/stereo-reliability-rework`. The original paired
benchmark and its scheduler remain paused while diagnostic gates are established.

## Current evidence

These matched comparisons use the preserved stereo VO and the new shared tracker,
with loops disabled. ATE uses SE(3) alignment with scale fixed to one. Reference
poses are loaded after tracking; learned depth is excluded from tracking.

| KITTI | Coverage | Estimator | ATE RMSE (m) | Translation drift (%) | Lost frames |
| --- | --- | --- | ---: | ---: | ---: |
| 01 | All 1,101 frames | Preserved stereo VO | 72.293 | 9.574 | 23 |
| 01 | All 1,101 frames | Current shared stereo | 88.682 | 9.104 | 2 |
| 04 | First 80 frames | Preserved stereo VO | 0.444 | 1.673 | 0 |
| 04 | First 80 frames | Current shared stereo | 0.357 | 0.691 | 0 |

The full 01 comparison fails the release gate: shared ATE is 22.7% worse. Rotation
drift also worsens from 0.01337 to 0.01641 degrees/m. Translation drift improves by
4.9%; both isolated shared losses, frames 307 and 334, recover. These improvements
do not offset the unexplained trajectory regression.

The shared worker takes 1,400 seconds and peaks at 2,023 MiB, versus 450 seconds
and 238 MiB for preserved stereo. These uncached worker timings
include initialization, tracking, export and evaluation. Source snapshots, full
input fingerprints and reference identity match. An external baseline observer
records actual returned tracking decisions without changing estimator outputs;
its measured overhead is 0.006 seconds.

![Full stereo 01 matched comparison](benchmark/plots/stereo-rework01-full-matched.png)

![Full stereo 01 input, loss and sparse map](benchmark/plots/stereo-rework01-full-conditioned-overview.png)

The 350-frame shared diagnostic scores 12.026 m ATE, 6.167% translation drift and
two lost frames, versus 21.510 m, 8.357% and 18 for preserved stereo. Its ATE is
worse than the preceding descriptor-retry revision's 9.137 m and 5.137%, despite
fewer losses. The short comparison did not predict complete-sequence behavior.

![Stereo 01 diagnostic comparison](benchmark/plots/stereo-rework01-conditioned-ba.png)

The retained full 04 comparison predates the solver fixes: shared stereo scores
0.879 m ATE and 0.908% drift, versus 1.763 m and 1.683% for preserved stereo.
Those scores do not validate the current source. Both runs use no feature cache:
the shared estimator takes 210.8 seconds
and peaks at 321.7 MiB, versus 148.9 seconds and 231.4 MiB for the baseline.
The current 01 shared replay uses cached extraction and takes 201 seconds under its
supervisor; that timing is a diagnostic, not an official performance comparison.
The complete backend suite passes 120 tests. Dashboard socket tests, type checking
and the production build pass.

![Retained full stereo 04 comparison before solver fixes](benchmark/plots/stereo-rework04-full.png)

[Curated results and source fingerprints](benchmark/STEREO_REWORK_DIAGNOSTICS.json)
retain configurations, coverage, input identity, runtime and memory alongside the
older experiments. Finite trajectories and sparse exports were verified. The
saved dashboard gallery preserves earlier failures under separate labels.

Full 01 exports have consistent keyframe/trajectory poses and shared landmark
positions, no keyframes on lost frames, and no negative multiview observation
depths. Final multiview reprojection has a 0.339 px median, 1.015 px 95th percentile
and 0.054% above 2 px. Low internal reprojection error does not establish accurate
global motion. The largest applied BA translation change is 1.526 m. Loops are off,
so this experiment supplies no live pose-graph correction evidence.

## Tracking and mapping fixes

Local bundle adjustment now includes all affected multiview camera observations
in its acceptance objective. Excluded multiview landmarks retain their world
coordinates; single-view stereo landmarks preserve their measured camera
coordinates when their anchor moves. Camera and explicit landmark updates are
validated and applied atomically. Synthetic checks cover disconnected gauges and
reject updates that improve selected observations while worsening the affected
map.

Held-out residual rows now follow the observation order declared by the sparse
Jacobian. The previous implementation appended all right-image errors after all
left-image errors, declaring the wrong camera dependencies. Numeric derivative
checks reproduce the defect and cover mixed stereo and left-only observations.
Fixing the row order alone scores 13.423 m ATE and one recovered lost frame on
the 01 prefix; that intermediate result is retained separately.

Automatic scaling from the Huber-weighted Jacobian also stalled on a noiseless
synthetic scene despite reporting solver success. BA now uses radians for rotation
and a geometric baseline for translations and points. Stereo uses its calibrated
baseline; monocular uses positive local camera separation. Six stereo/monocular
checks across coordinate scales 0.01, 1 and 100 recover the known cameras and reduce
the complete affected objective. Pose acceptance and robust-loss thresholds remain
unchanged. The numerical fixes improve correctness without guaranteeing a lower
trajectory score for every noisy sequence.

Stereo map PnP also checks current right-image reprojection when enough spatially
distributed disparity observations exist. Rectified principal-point offsets are
handled consistently in tracking and BA. Missing depth support is reported rather
than presented as stereo verification. Wrong-depth maps that fit the left image
are rejected by synthetic checks.

The first depth-check revision failed stereo 01 at frame 275: 53.034 m ATE and
75 unresolved lost frames. Disabling BA produced the same interval. An image-only
probe found too few useful matches under the default SIFT contrast setting.
Detecting weaker texture supplied 33 independent PnP inliers across five spatial
cells on the failing pair. Stereo now uses one fixed SIFT contrast threshold of
0.02 across inputs, with the existing feature cap; monocular extraction remains
unchanged. Pose acceptance thresholds are unchanged. Synthetic weak-texture
images initialize and track; blank images still cannot initialize.

![Image-only feature-support diagnosis](benchmark/plots/stereo-rework01-feature-support.png)

The tracker additionally retries mutual descriptor matches when flow-augmented
PnP is rejected. A synthetic scene reproduces coherent flow outliers in two
spatial cells dominating RANSAC despite 70 correct descriptor matches. The retry
recovers the known camera pose with the existing geometric acceptance checks.
It does not accept the rejected flow hypothesis or fill held poses.

The feature-support revision alone scored 12.249 m ATE, 6.121% drift and 18 lost
frames on 01. Adding the descriptor retry improves that separate frozen replay
to the preceding descriptor-retry revision. Earlier BA, bidirectional-refinement
and motion-prior
experiments remain available in the result JSON and gallery.

Disabling BA on the preceding descriptor-retry source scores 105.055 m ATE, 37.172% drift and 127
unresolved lost frames starting at frame 223. The corresponding BA-enabled run
scores 9.137 m, 5.137% and six recovered lost frames. This ablation supports
local BA in that revision; it does not validate the later solver revision or
demonstrate live loop correction, which is disabled in both runs.

![Identical-source BA ablation](benchmark/plots/stereo-rework01-ba-ablation.png)

## Budgeted checks

Run from the repository root with the project environment. `$data` contains
`sequences/<id>/calib.txt` and synchronized `image_0`/`image_1` streams. `$poses`
contains evaluator-only reference trajectories.

```powershell
.\.venv\Scripts\python scripts/run_development_tests.py --profile quick --data-root $data --poses-root $poses
.\.venv\Scripts\python scripts/run_development_tests.py --profile focused --variants baseline map-only bundle live offline --data-root $data --poses-root $poses
```

Quick cycles target five minutes and replay 80 stereo 04 frames. Focused cycles
have a 60-minute total budget and add the first 350 stereo 01 frames. Preparation,
checks, replay and reporting count toward that budget. Estimate cost before the
next replay; defer work that does not fit. Stateful integration begins at frame
zero. Completed matching evidence can be reused without replacing history.

`--feature-cache .datasets/feature-cache` optionally caches deterministic SIFT and
disparity extraction for diagnostics. Identity includes image content, extraction
implementation and parameters, OpenCV build and calibration. Tracking-only edits
can reuse features; evaluation scores still require the complete estimator,
evaluator, configuration, input and coverage identities to match. Release profiles
reject cached performance measurements.

The evaluator supports `--max-wall-seconds`, `--stop-file`, `--disable-bundle` and
`--loop-mode off|live|offline`. Timeouts, input errors and requested stops retain
available partial exports and explicit status. Owned-process supervision enforces
the hard deadline. Original datasets and previous results are preserved.

## Wider acceptance

Repair and reevaluate the full 01 regression before expansion. Fresh current-source
full 04, 00 and 07 comparisons, monocular initialization/recovery and RGB-only TUM
validation remain required. Investigate rotation regressions and resource use;
passing a short ATE comparison alone is insufficient. Require finite consistent
exports, improved affected failures, no worse tracking-loss coverage and no
unexplained ATE or translation-drift regression above 5% against fresh matched
baselines before broader paired validation.

Keep map-only, BA, live and offline ablations separate. Inclusive stage timings
may overlap. Verify loop corrections on identical graphs and report actual applied
updates, stale rejection and recovery. Synthetic correction and anchored solver
checks cover 300 and 1,000 keyframes; the production 300-keyframe guard remains
until representative larger graphs are validated.

A posthoc 350-frame audit finds 49 early keyframes with at least 80 existing
landmark observations. At frame 29, 122 accepted inliers coexist with only 45
descriptor candidates. The insertion policy counts detector associations and
omits additional verified flow support. A regression test and bounded ablation
must establish the effect of correcting this count; it is not yet a validated
explanation for the full trajectory error. Preserve the current failed revision.

Release runs require a reviewed gate JSON containing the exact `revision` and
`passed`, supplied through `--profile release --release-ready`. Estimate complete
sequence cost before starting; improve performance or obtain a specific extended
budget for work expected to exceed an hour. The scheduler stays paused and main
remains unchanged until the agreed wider validation and review are complete.
