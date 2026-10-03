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
[the component record](GPU_MATCHING_COMPONENT_CHECK.json). These diagnostic
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
in [the checkpoint](TARGET_RELATIVE_BUNDLE_CHECKPOINT.json). The checkpoint
preserves this synthetic evidence and the earlier budget deferral. The fresh
comparison below supplies actual partial 04/80 accuracy results; full validation
remains paused. The reviewed source checkpoint was pushed as
`9046493772c6131e242cb9e076aa6303e03a35e3`.

## Fresh matched 04/80 result — accuracy gate failed

The frozen target-relative experiment was evaluated on a fresh, matched CUDA
pair for the first 80 KITTI 04 frames. This is a partial diagnostic, not full
validation. The ON run regressed ATE RMSE and translation drift against OFF
while improving rotation drift: ATE RMSE +46.52%, translation drift +46.17%, and rotation drift -14.83%. The declared accuracy gate failed
because ATE RMSE and translation drift each exceeded the 5% regression limit.
Both runs tracked all 80 frames with zero losses. The ON run applied
9 bundle adjustments, including
9 accepted intermediate-camera corrections and
1040 training factors. Temporary source rows were not
persisted.

| Measure | ON | OFF | ON vs OFF |
| --- | ---: | ---: | ---: |
| ATE RMSE (m) | 0.163932 | 0.111884 | +46.52% |
| Translation drift (%) | 0.410376 | 0.280753 | +46.17% |
| Rotation drift (deg/m) | 0.004814 | 0.005653 | -14.83% |
| Runtime (s) | 76.684 | 46.943 | — |
| Peak host memory (MiB) | 972.75 | 970.34 | — |

The comparison uses each run's own saved evaluator output and estimated poses;
the trajectory panel does not overlay ground truth. Ground truth was evaluator
only and was not an estimator input. Source, runtime, input, reference and
coverage identities matched; only the opt-in experiment flag differed. No
metrics were reused from another revision. KITTI 01/350 and 01/129 were not run
after the 04 gate failed. The feature remains opt-in with its default disabled;
this result does not support a general accuracy improvement or release claim.
The pair does not isolate the effect of added measurements from the tighter
inner solver precision used when augmentation is active. A fresh three-arm
comparison—default OFF, precise OFF, precise ON—is the next causal control;
the measurement tradeoff is not established by this pair. The prior frame-14
transport certificate is not treated as proof of lower original cost in this
fresh run.

The saved first correction has identical original inputs. Its original image
cost falls to 10.7797 with augmentation versus 13.6921 without it. Chart
reconstruction, recomputed costs and incident motion checks agree with the
export. Both solves reach the 30-evaluation cap. A transported control candidate
has augmented cost 26.0007, above the saved ON cost of 23.0612; the earlier
lower-cost transport explanation does not recur. These checks establish
transaction consistency, while the trajectory comparison still fails.

Stage timings are inclusive and may overlap: matching takes 1.66/1.64 seconds
(ON/OFF), extraction 21.49/21.84, and local bundle adjustment 36.55/6.13. CUDA
matching was active in both runs. The comparison is not a CPU/GPU speed test.

Fresh source fingerprint: `4c2071864e762480618faa13c1044cde9e4038b609664af6cc1167f453e74062`. See the structured
[checkpoint](TARGET_RELATIVE_BUNDLE_CHECKPOINT.json), the
[ON/OFF trajectory and error plot](plots/target-relative-04-80-on-off-failed-gate.png), and the actual
[ON overview](plots/target-relative-04-80-on-overview.png) and [OFF overview](plots/target-relative-04-80-off-overview.png).
The prior budget-deferral history remains in the checkpoint.
