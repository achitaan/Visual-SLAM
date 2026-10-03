# Owned stereo image-bundle experiment

`stereo_owned_image_bundle` is an opt-in local bundle-adjustment experiment. It
defaults to off in `MappingConfig`. Enable it with
`--stereo-owned-image-bundle`; the evaluator requires calibrated stereo and
`--stereo-pose-arbitration`. Existing runs keep the original call path and
bundle residuals when the option is off.

When enabled, the estimator may add correlated left/right image measurements
from the already-run, reverse-verified reserved stereo fit to local bundle
adjustment. It reuses points already selected by the bundle or current
single-view map points and optimizes those measurements jointly with their
shared camera poses and 3D points. Ambiguous or mismatched map observations,
held-out rows, stale geometry, and any attempt to consume the full supported
match pool make the factors ineligible. The feature creates no new map points.

The model is explicitly a correlated physical stereo image-row model. Left and
right measurements share sensor evidence, so these rows are not an independent
validation set and the implementation makes no covariance claim. Eligibility
requires that the final tracking decision selected the reserved stereo source;
the feature does not fit a new pose or rematch descriptors.

Before applying a candidate, the bundle retains its existing affected-cost,
positive-depth, finite-value, and stereo-motion checks. Both the original
affected objective and the augmented objective, which also includes the new
image rows, must strictly decrease. Immediately before map updates, the
provider rechecks the captured map epoch, endpoint and anchor poses, live
calibration, reused landmark positions, and exact keyframe-feature/Observation
pixel and right-coordinate ownership. A stale or changed input rejects the
augmentation.

## Fresh sequence-04 acceptance check

Run two separately budgeted, matched, uncached 80-frame sequence-04 prefixes
with the same source, runtime, input archive, reference, CPU settings, and
output procedure. Change only the owned-image flag between the two runs. Keep
loops off so the test isolates local bundle adjustment. Ground truth is
evaluator-only. This smoke check covers this short prefix and configuration; it
does not validate the full sequence or a broader configuration matrix.

```powershell
python scripts/evaluate_shared_slam.py --stereo --sequence 04 --max-frames 80 `
  --max-wall-seconds 180 `
  --data-root <DATA_ROOT> --poses-root <POSES_ROOT> --output results/owned-image-04-off `
  --loop-mode off --stereo-depth-policy verified_fallback --stereo-pose-arbitration `
  --stereo-raw-reference-retry `
  --matching-backend cpu --retrieval current --opencv-threads 1

python scripts/evaluate_shared_slam.py --stereo --sequence 04 --max-frames 80 `
  --max-wall-seconds 180 `
  --data-root <DATA_ROOT> --poses-root <POSES_ROOT> --output results/owned-image-04-on `
  --loop-mode off --stereo-depth-policy verified_fallback --stereo-pose-arbitration `
  --stereo-raw-reference-retry `
  --matching-backend cpu --retrieval current --opencv-threads 1 `
  --stereo-owned-image-bundle
```

For the off run, verify the flag is false in the saved configuration and no
owned factors are active. For the on run, verify factor provenance and exact
ownership, absence of held-out or full-pool evidence, final-validation success,
finite exports, and strict reduction of both objective gates. Compare evaluator
metrics and tracking coverage only after both runs complete under matching
identities. A passing smoke check is not evidence of a benchmark improvement;
no improvement has been established, and this experiment is not approved for
the main release.

The earlier capture-only evidence and its limits are documented in
[`stereo-training-observations.md`](stereo-training-observations.md). That
report does not test this optimizer integration and must not be treated as an
accuracy result for it.

## Verification checkpoint

The backend suite passes **411 tests**, with four optional GPU tests skipped.
The frozen implementation identity is recorded in
[`OWNED_IMAGE_BUNDLE_CHECKPOINT.json`](OWNED_IMAGE_BUNDLE_CHECKPOINT.json).
No new dataset replay has completed for this revision; the matched accuracy
comparison remains pending. Earlier capture-only scores are not validation of
this optimizer.
