# Source-image correction diagnostics

This development branch adds bounded capture for the experimental source-history
bundle. It does not change correspondence selection, residuals, solver settings,
map observations or correction acceptance. The estimator remains experimental.

The prepared packet records the exact installed source-image rows, in objective
order, with stable landmark IDs, selected-point indices and float32 pixels.
Source pose, calibration and map epochs are checked before labeling the snapshot
valid. The calibration identity includes disparity offset.

When two bundle frames are selected, the first packet's landmark cohort is
retained through the last selected frame. Prepared and finished packets copy
the same IDs, current world positions, source camera and actual stored
observations under the map lock. Missing landmarks remain explicit unavailable
rows. The diagnostic does not keep landmarks alive or install observations.

The current source-pose guard also compares the pre-target tracking snapshot.
An otherwise valid correction between that snapshot and bundle preparation can
make the diagnostic abstain. This conservative limitation affects capture,
not estimator acceptance. Invalid capture must not support a numerical claim.

## Validation

The revision passed 492 backend tests, including six new checks for installed-row
identity and order, detached pixels, skipped rows, unavailable cohort members,
calibration changes, disabled capture and writer failure isolation.

Fresh capture/plain replay equivalence remains unverified. Backend checks took
119.55 seconds; the remaining inclusive development budget was insufficient for
the next declared replay estimate plus reporting reserve. No evaluator was
launched for this diagnostic revision, and older accuracy scores are not its
validation.

## Next comparison

Use separate output directories for two fresh KITTI 04 runs from frame zero
through frame 79. Keep the same source, inputs and configuration for both:
stereo, CUDA matching, current retrieval, one OpenCV thread, verified-fallback
depth, pose arbitration, raw-reference retry, owned-image bundle, source-history
bundle, precise solver and loops off. Capture bundle frames 69 and 76 and
tracking frames 68, 69, 75 and 76 only in the diagnostic run.

Require byte-identical exported poses, sparse maps and previews; identical
tracking decisions, estimator reports and processing call counts; and complete
finite diagnostic packets. Then reconstruct the exact frame-69 image cost and
evaluate those same pixels and stable IDs after frame 76. Report missing points
and invalid depth explicitly; cost comparisons require compatible calibration
and complete rows. These measurements were consumed by tracking and fitting.
They are not independent accuracy validation.

If later optimization increases the same-row cost, investigate retention of
source-image constraints. If it does not, examine the camera/landmark geometry
and solver convergence instead. Do not change weights or thresholds based on
ground truth. Reevaluate accuracy with fresh controls after any estimator fix.

The parent [accuracy comparison](source-history-connectivity.md) still fails
the translation regression gate. Wider validation and merging into main remain
blocked by that accuracy result, independently of passing software tests.
