# Retaining stereo source observations

This is an opt-in experiment. The initial matched KITTI 04 replay failed the
accuracy gate; the feature remains disabled by default. See the
[results and limitations](benchmark/SOURCE_RETENTION_04_80.md).

The source-history bundle uses accepted left-image measurements to constrain
selected landmarks. Those measurements previously disappeared from the next
local solve. This experiment retains their actual frame IDs, landmark IDs and
pixels separately from keyframe observations, and reuses eligible rows within
the local bundle window. It does not use ground truth or reference depth.

Enable it with `--stereo-retained-source-observations` together with
`--stereo-source-history-bundle` and `--stereo-owned-image-bundle`. Their existing
stereo, shared-SLAM and pose-arbitration requirements still apply. The default
is off. The same selection is recorded in evaluation and development-test
identities; results from different selections cannot be reused interchangeably.

The model is `anchored_actual_source_image_retention_v1`. A historical camera
pose is the candidate anchor-keyframe pose multiplied by the saved relative
camera pose. The historical camera-to-anchor transform is held fixed. This is
a geometric approximation, not a joint estimate of an additional free camera.
Reused tracking measurements are training constraints, not independent
validation or covariance evidence.

Retained rows use only the original selected landmark variables; the existing
owned-image variables are unchanged. Retained left measurements add two
reprojection components per row, with the existing robust
loss. Rows are bounded, deduplicated against the other image constraints and
filtered by calibration, frame status, landmark availability and window
membership. Registration and expiry occur only with an accepted atomic map
correction. Stale geometry, invalid depth, inconsistent motion or failure to
reduce either required objective must reject the update.

Validation requires synthetic two-solve evidence, atomic-update and gauge
checks, followed by fresh GPU replays using the same source, inputs and
evaluator. The initial comparison is KITTI 04 from frame zero through frame 79:
default tracking, source-history control, and source-history with retention.
Short-prefix results do not establish full-sequence reliability or live
pose-graph performance. Expand to KITTI 01 and longer sequences only after the
correctness and accuracy gates pass within the declared test budget.
