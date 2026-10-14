# Development roadmap

The objective is reliable SLAM across environments with consistent settings. Evaluate accuracy alongside tracking continuity, failure recovery, runtime and memory. Use ground truth only for evaluation; avoid per-sequence tuning and ground-truth-based selection of trajectories.

## Current foundation

- Metric stereo odometry with validated calibration, disparity/depth filtering and PnP inlier checks.
- Offline image-verified loops, a fixed-origin rigid pose graph and full-frame corrected exports.
- Explicit KITTI trajectory validation, complete/partial coverage and official evaluator parity checks.
- A live dashboard with equal trajectory scales, connection state and saved benchmark comparisons.
- Automated frontend, geometry, graph, telemetry, CLI and streaming regressions.

See the [benchmark report](benchmark/REPORT.md) for measured results and limitations.

## Experimental shared pipeline

The shared branch adds persistent stereo and monocular mapping, geometric
monocular initialization, local bundle adjustment, asynchronous image-verified
SE(3)/Sim(3) corrections, geometric relocalization and optional offline learned
depth reconstruction. These features have implementation and synthetic test
coverage, but are not yet established as generally reliable.

The current backend passes 78 tests. Full paired KITTI evaluation is in progress;
only sequence 04 has completed both modes on the current frozen revision. Its
monocular scale is arbitrary and fitted only during evaluation. Current TUM desk
tracking still loses 155 of 613 frames. Retained shared-pipeline KITTI 01 revisions
also remain less accurate than the original stereo baseline. See the
[sensor coverage report](benchmark/SENSOR_COMPARISON.md) and
[tracking diagnosis](benchmark/TRACKING_DIAGNOSIS.md) for separate revision groups.

Before accepting the shared pipeline, finish the fixed-configuration stereo and
monocular 00–10 batch, diagnose regressions without selecting poses using reference
data, verify recovery across unresolved intervals, reconcile map observations
after loop corrections, and validate larger graphs before lifting the current
300-keyframe limit. Dense reconstruction also needs broader quality evaluation
and validation of GPU execution. Passing synthetic tests or obtaining a lower
ATE on one sequence does not complete these requirements.

## Priorities

1. **Tracking and recovery:** investigate depth uncertainty, nearby static-feature support and motion prediction. Add recovery tests for low texture, rapid motion, dynamic objects and insufficient stereo depth. Do not improve apparent coverage by silently accepting weak estimates.
2. **Persistent mapping:** associate stereo landmarks across frames, track against a local map, manage landmark lifetime and add local bundle adjustment. Check reprojection error and geometry independently of trajectory ground truth.
3. **Live loop correction:** integrate the geometric verifier, update tracking poses and map landmarks consistently, and estimate measurement uncertainty. Test hard-negative loop candidates and correction propagation.
4. **Scalable optimization:** validate larger sparse graphs, bound latency and memory, and move expensive optimization off the live tracking path. The largest graph in the published batch has 234 keyframes.
5. **Relocalization and monocular scale:** implement geometrically verified map-based recovery. Monocular loop handling needs a scale-aware formulation.
6. **Broader validation:** use multiple environments and datasets with fixed configurations, record failures and input conditions, and test fresh installations across supported platforms. Keep results separate by sensor configuration and evaluation protocol.

## Acceptance criteria

- Export one finite rigid pose per input frame and report every held/lost estimate.
- Never construct motion or optimization constraints from ground truth.
- Preserve the fixed graph origin and reproduce saved-constraint optimization within declared tolerances.
- Compare local evaluation metrics against an independent reference implementation.
- Report regressions as well as improvements; do not select corrected output using ground truth.
- Demonstrate tracking recovery and live map consistency before claiming a complete SLAM system.
