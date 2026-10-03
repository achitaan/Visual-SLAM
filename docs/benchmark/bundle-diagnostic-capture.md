# Local bundle diagnostic capture

Bundle diagnostics are an opt-in record of the inputs and result for selected
local bundle-adjustment calls. They are intended for debugging one solver
window, not for measuring performance or changing estimator behavior.

## Reproduce a stateful KITTI prefix capture

The frame allowlist is zero-based. This example captures frame 127 during a
129-frame prefix of sequence 01, with loop closure disabled so the run focuses
on the sequential tracking and local bundle path. Replace the example roots
with local dataset locations:

```powershell
$DATA_ROOT = "D:\datasets\KITTI"
$POSES_ROOT = "D:\datasets\KITTI\poses"
$OUT = "results\bundle-diagnostics\01-prefix-129"

python scripts/evaluate_shared_slam.py `
  --dataset kitti --data-root $DATA_ROOT --poses-root $POSES_ROOT `
  --sequence 01 --stereo --max-frames 129 --max-wall-seconds 600 `
  --loop-mode off --stereo-depth-policy verified_fallback `
  --stereo-pose-arbitration --stereo-raw-reference-retry `
  --matching-backend cpu --retrieval current --opencv-threads 1 `
  --output $OUT `
  --bundle-diagnostics-dir "$OUT\bundle-diagnostics" `
  --bundle-diagnostics-frames 127
```

For a same-configuration prefix control, use a separate output directory and
omit only `--bundle-diagnostics-dir` and `--bundle-diagnostics-frames`.

The output directory for snapshots must be inside the run output directory.
Both diagnostic options are required together. The requested frame IDs must be
unique and nonnegative. The main application exposes the same options when run
with `--slam --stereo`.

## Capture contents and status

The allowlist is explicitly bounded to 32 frames and 96 snapshots. For each
selected frame that reaches local bundle adjustment, the writer records three
phases:

- `prepared`: the selected window, gauge and solver-input layout immediately
  before optimization;
- `solved`: the optimizer result before the estimator accepts or rejects it;
- `finished`: the returned bundle report and the affected frame and anchor
  poses after the call.

Each phase is a finite JSON file written atomically. The run manifest records a
relative path and SHA-256 digest for each file. A frame is complete only when
all three phases were written. Frames that were requested but did not require
a bundle call (for example, because the frame was not accepted as a keyframe)
are marked skipped with a reason. Write or capture errors are recorded and
prevent the capture from being treated as complete. The manifest is stored in
the run and evaluation reports; it does not contain absolute snapshot paths.

## Interpretation limits

Capture is disabled by default. Enabling it does not add fields to
`MappingConfig` or change the default estimator settings, but serialization and
file writes add runtime overhead. Do not use diagnostic-capture timings as
performance measurements.

The 01/129 frame-127 selection is a late point in a stateful prefix, useful for
examining one local bundle window after earlier frames have shaped the map. It
is not a substitute for the complete 350-frame evaluation and should not be
read as a final-trajectory result. Ground truth, when supplied to the evaluator
for scoring, remains evaluator-only and is not passed into tracking or bundle
adjustment.
