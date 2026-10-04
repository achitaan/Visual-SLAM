# Evaluation report reliability

The evaluator writes its final report once, through the atomic JSON writer. A brief Windows sharing lock during the final replacement is retried within a configured 0.5-second window, with a retry-count bound. Operating-system call and scheduling delays can add wall time. Persistent locks still fail visibly, preserving the previous report and the pending `.part` report. Other I/O errors are not retried, and nonfinite JSON is rejected before replacing an existing report.

Regression tests cover transient locks, persistent locks, other I/O errors, nonfinite values and successful atomic replacement. This change addresses report delivery; it does not alter tracking, mapping, optimization or benchmark accuracy.

Validation: 54 focused tests and all 674 backend tests pass on the same source fingerprint. The full suite took 98.87 seconds. Static review found no estimator or acceptance-gate changes; `git diff --check` passes. No fresh accuracy replay was performed for this I/O-only patch.

## Current experimental limits

The [map-depth comparison](benchmark/STEREO_MAP_DEPTH.md) remains the latest executed accuracy comparison. On the 128-frame sequence 01 prefix, verified map depth reduces position error by 2.8% against the shared control, but position error remains 15.8% worse than previous stereo VO. No full-sequence reliability conclusion follows from this prefix.

Saved-edge accounting separates raw relative motion from the corrected exported trajectory. It points to a position/rotation tradeoff during mapping corrections, alongside an earlier raw-motion rotation difference. Reconstructing those saved edges is a diagnostic, not a newly executed tracker benchmark.

A strict nonworsening gate on fixed-depth reserved-image cost was rejected: it would reject all nine accepted bundle updates in a retained beneficial sequence 04 run. That conditional prediction cost is not the same objective as joint camera/landmark optimization.

The next proposed accuracy experiment is opt-in two-view stereo refinement over the pose and full XYZ landmark variables using the original measured image endpoints. It requires retained forward/reverse geometric verification, unchanged arbitration holdouts, observability checks and a fresh frozen comparison against both controls. It has not been implemented or evaluated.

Speed, accuracy and streaming remain separate acceptance gates. CUDA matching has an earlier short matched speed result with identical trajectories; the latest shared pipeline's diagnostic 1.58 FPS does not establish sustained 10 FPS operation, bounded latency or dropout handling. Main remains unchanged, and validation scheduling stays paused.
