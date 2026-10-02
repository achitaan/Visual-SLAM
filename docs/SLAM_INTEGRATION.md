# Stereo reliability and performance integration

The integration branch starts from reliability commit `8a057fc` and adapts
performance work from `0fc86c0`. The original branches and their results remain
separate. Faster processing is a development aid; improved tracking and trajectory
accuracy remain the objective.

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

Saved KITTI04 diagnostics attribute the retained rotation regression to disparity
sampling. The guard-only revision reproduces the preceding trajectory. Features,
descriptors and disparity grids match across all 80 paired cache entries, while
derived depth changes. At frame27, independent stereo conflict switches the pose
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
preserved stereo VO. Current-source full01/04 evidence remains required before
00/07 expansion. Improved prefixes, passing synthetic tests or the performance
branch's historical speedups do not establish release readiness. Preserve explicit
loss intervals and investigate unexplained regressions above5% in trajectory or
drift metrics. Learned depth remains reconstruction-only.

Main and scheduled paired validation remain unchanged while this draft is
reviewed and tested. Historical performance evidence remains on
`codex/slam-performance`; its acceptance file explicitly records provisional
timings and incomplete held-out loop validation.
