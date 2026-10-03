# Target-relative stereo bundle experiment

This branch tests a numerical correction to the
[rejected free-source experiment](stereo-free-source-bundle.md). It changes
coordinates and inner solver accuracy, preserving measurements, robust loss and acceptance limits.
The owned-image bundle remains off by default. Previous experiments and datasets
are preserved; this branch is not approved for main.

For an actual target keyframe T and a free intermediate source camera S, the
solver uses C = inverse(T) S. An eligible singleton landmark uses q = inverse(T) P.
Its two image projections are evaluated directly in those coordinates. Shared
landmarks keep their world coordinates and their genuine camera dependencies.
Accepted candidates reconstruct S = T C and P = T q once, preserving map anchors
and applying the existing depth, motion, ownership and revision checks.

The earlier frame-14 diagnostic demonstrated a feasible lower solution to the
same image objective. Equivalent coordinates can improve numerical convergence,
but a shared trust region can still couple steps. Correctness tests and fresh
trajectory comparisons are required before claiming an accuracy benefit.

## GPU matching component check

The unchanged matching module was checked on real KITTI 04 and 01 image pairs
(frames 0 and 1, approximately 1,500 SIFT descriptors per image). Eight alternating
CPU/GPU repetitions returned exactly the same correspondences in every case.
Median matching times were 79.44 ms versus 7.68 ms on 04 (10.34x), and 84.57 ms
versus 7.26 ms on 01 (11.66x). Transfers, synchronization and CPU tie rechecks
are included. Runtime initialization, extraction and bundle adjustment are
excluded; host background load was uncontrolled. These are warmed component
timings, not a pipeline speedup or a trajectory benchmark.

Source, input and runtime identities are retained in
[the component record](GPU_MATCHING_COMPONENT_CHECK.json). Upcoming diagnostic
replays use explicit CUDA matching with a separately validated environment;
CPU remains available for equivalence controls.

## Estimator validation

The first chart-only synthetic solve still failed the existing camera-recovery
gate. A sparse-Jacobian audit found no missing derivatives, but loose inner LSMR
steps stopped at 0.099666 m error from an initial 0.1 m. Tight inner solves
recovered 1.62e-6 m; an independent dense solver agreed. The original recovery
threshold was retained and passes. Relative translation and rotation checks,
shared-landmark derivatives, ownership, stale-state rejection and atomic
world reconstruction also pass.

Only valid active augmentation uses LSMR atol/btol 1e-12 with a size-aware
iteration ceiling of max(500, variable_count). The outer limit stays at 30.
OFF, empty and rejected augmentation keep their original solver options.
This intentionally increases inner precision and can increase runtime; it is
not a change to observation weights or geometric acceptance thresholds.

The frozen source/runtime fingerprint is
`4c2071864e762480618faa13c1044cde9e4038b609664af6cc1167f453e74062`.
The full CUDA-environment suite passed 435 tests with no skips in 75.57 seconds;
24 focused chart/commit/recovery tests passed. Structured synthetic evidence is
in [the checkpoint](TARGET_RELATIVE_BUNDLE_CHECKPOINT.json). No dataset accuracy
result is claimed yet; the next step is a fresh matched ON/OFF comparison.
Full validation remains paused. The source checkpoint was pushed as
`9046493772c6131e242cb9e076aa6303e03a35e3`. The supervisor deferred the first
paired replay before launching an evaluator: its conservative estimate, input
preparation and reporting reserve no longer fit the current 60-minute cycle.
No new trajectory scores were produced or reused. The next cycle starts with
fresh CUDA-matched stereo 04/80 ON/OFF runs, then expands only if the correctness,
accuracy and time-budget gates pass.
