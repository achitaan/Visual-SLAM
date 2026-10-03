# Free intermediate cameras in stereo bundle adjustment

This experiment changes the camera state used by `--stereo-owned-image-bundle`.
It remains off by default. The earlier implementation and its failed accuracy
comparison are retained on `codex/stereo-owned-image-bundle` and in
[the previous report](stereo-owned-image-bundle.md).

Previously, an intermediate image pose was its keyframe pose multiplied by a
fixed recorded transform. That treats accumulated motion between the keyframe
and image as exact. The new model jointly optimizes one camera pose for each
actual intermediate image and reuses the existing camera variable for a
keyframe image. Shared landmarks remain shared point variables. There is no
source-to-anchor pose prior.

The original metric gauge, image residuals, robust loss, solver budget,
calibration and acceptance limits are unchanged. New camera/point components
must connect to the original visual graph. Both the original affected objective
and the augmented objective must decrease. All retained motion measurements
touching an optimized intermediate camera are checked, including measurements
whose endpoints share an anchor.

Validated camera and landmark updates are applied in one map transaction. The
intermediate pose overrides ordinary anchor propagation, and its relative pose
is recomputed against the corrected anchor. Stale snapshots, invalid cameras,
conflicting keyframe overrides and inconsistent frame arrays reject application.

Temporary source-image rows are not added to historical keyframe observations.
Their influence is therefore limited to the current transaction. This correction
does not address coherent disparity bias or establish a calibrated covariance.
The independent depth-acquisition experiment also
[failed its short comparison](stereo-verified-measurements.md).

## Validation

The frozen source `07f1952c04834c5c993ecc31b7d0d80b752aa6eb6f003dac1db9256e1b95a1e3`
passes 424 backend tests, with four optional GPU tests skipped. The ten new
invariants cover nonlinear gauge freedom, numerical elimination of source and
point variables, sparse dependencies, actual SciPy source-pose correction,
atomic application, stale frames and same-anchor motion rejection. The motion
veto uses a controlled candidate; it is an acceptance-path check.

Fresh matched replay results are pending. Tests alone do not establish an
accuracy improvement or release readiness. Evaluation uses reference poses only
after estimation.

Compare separate outputs using the same frozen source, inputs, dependencies
and configuration. Start at frame zero; change only the owned-image flag:

```powershell
python scripts/evaluate_shared_slam.py --stereo --sequence 04 --max-frames 80 `
  --max-wall-seconds 240 --data-root <DATA_ROOT> --poses-root <POSES_ROOT> `
  --output results/free-source04-on --stereo-depth-policy verified_fallback `
  --stereo-pose-arbitration --stereo-raw-reference-retry `
  --stereo-owned-image-bundle --loop-mode off `
  --matching-backend cpu --retrieval current --opencv-threads 1
```

Omit `--stereo-owned-image-bundle` for the control. Preserve finite exports,
explicit tracking loss, factor provenance, actual candidate intermediate poses
and the map revision. Expand to sequence 01 only after the short regression
check passes and the entire comparison fits the declared time budget.
