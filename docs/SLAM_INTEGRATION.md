# Stereo reliability and performance integration

The integration branch starts from reliability commit `8a057fc` and adapts
performance work from `0fc86c0`. The original branches and their results remain
separate. Faster processing is a development aid; improved tracking and trajectory
accuracy remain the objective.

This is an experimental draft. The complete branch includes the earlier shared
mapping and reliability work; it is not a performance-only change relative to
main. Stereo uses the reliability branch's SIFT contrast threshold of `0.02`,
while monocular uses `0.04`. Neither threshold was adjusted per sequence.

## Coordination

One head orchestrator owns integration, shared tracking interfaces, mathematical
review and replay scheduling. Workers use isolated worktrees with exclusive file
ownership. Initial assignments cover performance foundations, BA allocation
optimization and accuracy diagnosis. Later assignments cover loop snapshots and
GPU equivalence checks. Only the orchestrator starts integration replays.

## Controls

Existing `--slam`, `--stereo`, loop modes and development test profiles remain.
The shared estimator and evaluator add:

- `--matching-backend cpu|cuda|auto`: CPU default; explicit CUDA requires a
  compatible optional PyTorch environment. Auto reports fallback.
- `--retrieval current|indexed|exhaustive`: the existing sketch is the default.
  Indexed retrieval remains experimental and retains geometric verification.
- `--no-cpu-optimizations`: a diagnostic reference path for tracking/BA allocation
  changes; it does not change pose acceptance thresholds.
- `--profile <file>`: optional bounded stage-latency samples and operation counters.
  Existing inclusive stage totals retain their original format.

CUDA dependencies remain in `requirements-performance-gpu.txt`; sparse CPU SLAM
does not require them. Matching near ties and cancellation-prone distances uses
CPU verification. Neither CUDA nor appearance similarity can accept a pose.

## Correctness and accuracy

The adapted BA path preserves the affected-map objective, mixed observation row
order, sparse dependencies, geometric variable scales, gauge constraints,
principal-point offsets and independent stereo-motion acceptance checks.
Equivalence tests include the previous residual implementation, not just two
branches of the new implementation. Landmark caches observe positions and map
revision under one lock and refresh after corrections or map membership changes.

Saved KITTI 04 diagnostics attribute the retained rotation regression to disparity
sampling. The guard-only revision reproduces the preceding trajectory. Features,
descriptors and disparity grids match across all 80 paired cache entries, while
derived depth changes. At frame 27, independent stereo conflict switches the pose
source and inserts an earlier keyframe. This is a diagnosis, not a validated fix.
Controlled sampling ablations must preserve calibration, support masks and pose
thresholds; reference poses remain evaluation-only.

## Validation policy

Quick checks target five minutes; focused cycles are capped at 60 minutes,
including preparation and reporting. One evaluator runs at a time. Runtime
forecasts use measured rates with reserve, and interrupted evidence is retained.
Source, input, configuration and coverage identities must match for result reuse.
Cached extraction and detailed profiling are diagnostic costs, not official
performance comparisons.

Compare runtime against the frozen reliability revision and accuracy against
preserved stereo VO. Current-source full 01/04 evidence remains required before
00/07 expansion. Improved prefixes, passing synthetic tests or the performance
branch's historical speedups do not establish release readiness. Preserve explicit
loss intervals and investigate unexplained regressions above 5% in trajectory or
drift metrics. Learned depth remains reconstruction-only.

Main and scheduled paired validation remain unchanged while this draft is
reviewed and tested. Historical performance evidence remains on
`codex/slam-performance`; its acceptance file explicitly records provisional
timings and incomplete held-out loop validation.

## Remaining release blockers

The official paired runner (`run_shared_benchmark.py`) now uses a versioned resume
contract with named code/dependency hashes, ordered input/calibration/timestamp/
reference identities, resolved configuration, exact frame coverage and finite
artifact checks. An OS lock and owner-liveness checks prevent concurrent resume;
incomplete attempts are retained in separate retry directories. Legacy manifests,
diagnostic caches and mismatched completed reports cannot be reused. This runner
still needs independent evaluator deadlines before scheduled full validation can
restart. Use the bounded development runner, which includes tests in its gate
fingerprint, for current diagnostic work.

Next, repair the demonstrated depth-support regression using independently
verified geometry, profile extraction and descriptor matching, and repeat the
smallest affected cases. Run full 01/04 only after the focused accuracy gates pass
and their estimated cost fits a declared budget. Live correction, 00/07 expansion,
monocular/TUM and dense reconstruction remain later acceptance stages.

## Measured integration diagnostics

### Independent stereo-depth experiment

Commit `a275cef` adds an opt-in `--stereo-depth-policy verified_fallback`.
The default remains `supported`. Missing depths are considered only when the
original disparity is physically valid. Their measurements come from an
independent full-range epipolar image search, with texture, ambiguity, subpixel
correlation and reverse-match checks. Map poses and ground truth do not enter
this search. This release keeps learned depth out of tracking.

Uncached CPU replays used bundle adjustment, current retrieval and loops off,
starting at frame zero. These are diagnostic prefixes, not full-sequence results.

| Sequence / frames | Depth policy | ATE m | Translation % | Rotation °/m | Lost frames | Total wall s | Peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 04 / 80 | Supported | 0.308 | 0.605 | 0.01510 | 0 | 45.39 | 260.38 |
| 04 / 80 | Independently verified | 0.159 | 0.363 | 0.00954 | 0 | 39.41 | 261.98 |
| 01 / 350 | Supported | 9.292 | 5.052 | 0.01059 | 1, recovered | 246.49 | 672.42 |
| 01 / 350 | Independently verified | 8.223 | 4.685 | 0.01381 | 1, recovered | 215.19 | 707.75 |

04 improves in all three accuracy measures. On 01, ATE and translation improve,
but rotation drift worsens 30.4% relative to supported sampling and 27.1% relative
to preserved stereo VO. **The focused accuracy gate fails; wider validation is
stopped.** No acceptance threshold was relaxed. Passing image checks does not
guarantee correct depth: an image-only temporal-flow audit of 04 finds useful
restored geometry and residual outliers. Temporal consistency and the cause of
01's orientation regression require further investigation before promotion.

The [image-only diagnosis](benchmark/verified-depth-orientation-diagnosis.md)
finds roll-biased map increments even when fresh supported stereo observations
favor the independent motion estimate. This appears before bundle adjustment and
with loops disabled. A pure reserved-evidence scorer and synthetic counterexample
are included as **unwired preparation**. The next implementation must reserve
observations before both fits and clear rejected map associations; this scorer
does not yet correct tracking or validate a new trajectory.

Actual plots: [accuracy measures](benchmark/plots/verified-depth-metrics.png),
[01 comparison](benchmark/plots/verified-depth01-comparison.png)
and [04 comparison](benchmark/plots/verified-depth04-comparison.png).
Saved results and source fingerprints remain separate from the earlier revision.

### Bounded descriptor cache

Tracking now materializes descriptors only for the selected landmark set rather
than retaining a duplicate matrix and ID index for the entire map. The supported
04/80 and 01/350 replays export exactly the same poses as the earlier integration,
with unchanged loss intervals. On 01, peak process memory decreases from 672.42
to 611.70 MiB and total wall time from 246.49 to 221.28 s (10.2%). Compared with
the fresh uncached reliability revision, this run is 6.0% faster with 0.7% more
peak memory. These single shared-host diagnostics do not meet the 20% CPU speed
target. The 01 replay used a 600-second supervisor budget and finished in
222.61 seconds. [Exact evidence](benchmark/BOUNDED_CACHE_RESULTS.json) records
the independently frozen revisions; new code does not inherit old validation.

Frozen commit `7cc6191` was replayed from frame zero with CPU matching, current
retrieval, bundle adjustment and loops off. Ground truth was read only after
tracking. Source archives and finite trajectory/sparse exports were verified.
The subsequent harness and matcher-provenance changes are documented separately;
these scores do not validate a different estimator revision.

| Sequence / frames | Implementation | ATE m | Translation % | Rotation °/m | Lost frames | Total wall s | Peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 / 350 | Integrated CPU | 9.292 | 5.052 | 0.01059 | 1, recovered | 246.49 | 672.42 |
| 01 / 350 | Reliability `8a057fc`, uncached | 9.292 | 5.052 | 0.01059 | 1, recovered | 235.41 | 607.57 |
| 01 / 350 | Preserved stereo VO | 21.510 | 8.357 | 0.01086 | 18 | — | 235.45 |
| 04 / 80 | Integrated CPU | 0.308 | 0.605 | 0.01510 | 0 | 45.39 | 260.38 |
| 04 / 80 | Reliability `8a057fc`, uncached | 0.308 | 0.605 | 0.01510 | 0 | 44.50 | 254.00 |
| 04 / 80 | Preserved stereo VO | 0.444 | 1.673 | 0.01241 | 0 | — | 229.86 |

The integrated CPU trajectories are identical to the reliability revision in both
prefixes. Serial uncached total wall time increased 4.7% on 01 and 2.0% on 04;
memory increased 10.7% and 2.5%. These shared-host diagnostics do not meet the
20% performance target and are not repeated, isolated full-sequence measurements.
The 04 rotation error remains 21.6% worse than preserved VO despite lower ATE and
translation error. Release readiness remains false.

Two controlled 04 sampling ablations used a separate frozen archive and diagnostic
cache. Replacing supported bilinear sampling with nearest samples yielded
0.361 m ATE, 1.104% translation and 0.00686°/m rotation: a mixed result.
Keeping supported bilinear samples and restoring omitted depths with valid nearest
samples yielded 0.189 m, 0.514% and 0.00880°/m, with no lost frames. This promising
candidate is **not enabled**: held-out matches include severe outliers, and restored
points need independent temporal verification before map admission. Neither
experiment changed thresholds or used ground truth to estimate poses.

[Machine-readable evidence](benchmark/SLAM_INTEGRATION_RESULTS.json) records exact
source and input fingerprints. Actual plots include
[01 comparison](benchmark/plots/integration01-comparison.png),
[04 sampling comparison](benchmark/plots/integration04-sampling.png),
[01 input/loss/sparse map](benchmark/plots/integration01-overview.png) and
[04 input/sparse map](benchmark/plots/integration04-overview.png).

A separate current-source optional CUDA smoke replay compared both backends in
the same Python 3.12.14/OpenCV 5.0.0/NumPy 2.5.3 environment. Both processed 04's first
80 frames with exactly equal exported trajectories and no lost frames. CPU total
wall time was 42.90 s; CUDA was 30.95 s, a 27.9% reduction. Peak process memory rose
from 261.26 MiB to 956.08 MiB; peak reserved GPU memory was 44 MiB. This used a
GTX 1660 SUPER, PyTorch 2.7.1+cu126 and CUDA 12.6. All 158 matcher calls used CUDA,
with 6,245 ambiguous rows verified on CPU. This is a single short diagnostic, not
the required full 01/04 performance gate or validation of indexed retrieval/live
loops. Its separate source fingerprint and environment are in the evidence JSON.

Repository checks: 245 backend tests passed, with four optional CUDA skips in the
CPU environment. All seven focused matcher/cleanup tests passed in the GPU
environment. Four frontend tests, type checking, production build and diff checks
passed. Passing software checks does not remove the accuracy/release blockers.

For a bounded diagnostic with a local KITTI root and separate evaluator poses:

```powershell
python scripts/run_development_tests.py --profile quick --variants bundle --data-root .datasets/kitti --poses-root .datasets/kitti-poses --output results/integration-check --budget-seconds 300
```

To test optional matching, install the separately pinned performance requirements
in an isolated environment and pass `--matching-backend cuda`. Retain the CPU
comparison in that same environment. Input preparation, checks, replay and reports
share the budget. Do not restart full paired validation while release blockers
remain. [Runtime/memory plots](benchmark/plots/integration-runtime.png) distinguish
the uncached CPU comparison from the separate GPU diagnostic.

`--timing-history path/to/evaluation.json` accepts explicitly selected completed
reports only as scheduling evidence. Sensor, coverage, input/reference hashes,
configuration, depth policy, backend, threads and cache/profiling category must
match. The slowest compatible rate gets a 1.25 safety factor; rejected evidence
uses conservative defaults. A different source revision can inform cost but
cannot supply accuracy or satisfy a validation gate. Estimates and evidence
hashes are retained in the cycle manifest.

A real two-frame paired runner smoke test completed stereo tracking and retained
monocular initialization failure as a completed experiment with an explicit failed
tracking outcome. Failed initialization is reusable evidence under exact identity
checks, never successful tracking. Its overview now labels accuracy unavailable
instead of attempting a scale fit. A separate plotting regression test covers this
case. This short runner check is not monocular accuracy validation.

Opt-in stereo arbitration now preserves connections that pass geometric checks
at the selected pose. The first disconnected version is retained as a rejected
experiment. [The 04 diagnostic report](benchmark/stereo-arbitration-results.md)
includes the repair, its matched BA ablation and the remaining translation-drift
tradeoff. Neither the short prefix nor passing software checks authorizes release.

The official benchmark runner now supervises preparation, data reads, evaluation
and plotting under one owned process-tree deadline. `--budget-seconds` accepts
100–3600 seconds, reserving time for export and cleanup. Work estimated to exceed
the remaining budget is deferred with a resumable manifest; interrupted results
remain incomplete. Compatible timing history supplies cost estimates only.

The current backend suite passes 279 tests, with four optional GPU skips. A real
two-frame paired deadline-runner smoke completes stereo and explicitly retains
monocular initialization failure. These checks do not replace accuracy coverage.
