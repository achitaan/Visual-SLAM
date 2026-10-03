# Independently verified stereo measurements

`--stereo-depth-policy verified_all` is an opt-in measurement experiment. It
searches the rectified right image for every eligible left feature using the
existing texture, correlation, uniqueness and forward/reverse consistency
checks. Accepted right coordinates produce calibrated metric depth. Dense
disparity values neither seed the search nor rescue a rejected match.

Acquisition precedes landmarks and raw-reference snapshots. Flow observations
and cache-hit snapshots use the same verified acquisition. Measurement
provenance identifies right-image correspondence; cache identities distinguish
this policy from the existing policies. The default remains `supported`, and
`verified_fallback` retains its previous behavior. The failed owned-image bundle
augmentation cannot be combined with this policy yet.

The motivation is a limitation of deriving right coordinates from dense
disparity and then treating them as independent pixel measurements. A coherent
disparity bias can produce a low bundle objective at an incorrect metric scale.
Independent correspondence supplies a directly checked image measurement;
it does not guarantee unbiased depth or trajectory accuracy.

The backend suite passes **414 tests**, with four optional GPU tests skipped.
Synthetic textured stereo images verify independence from biased or invalid
dense disparity, calibrated nonzero disparity offset, photometric rejection,
snapshot consistency and cache separation. Malformed calibration fails closed.
The full source identity is recorded in
[the checkpoint](VERIFIED_STEREO_MEASUREMENT_CHECKPOINT.json).

**No dataset replay has validated this revision.** Its real-data accuracy and
runtime remain pending. The previous joint image-bundle experiment failed its
[matched 04/80 gate](stereo-owned-image-bundle.md) and is preserved separately;
those scores do not validate this change.

## Next controlled check

Compare fresh frames 0–79 of KITTI 04 from the same frozen source, inputs,
calibration, dependencies and CPU settings. Change only `--stereo-depth-policy`
between `verified_all` and `verified_fallback`. Keep pose arbitration and raw
reference retry enabled, owned-image augmentation disabled and loops off.
Ground truth remains evaluator-only.

```powershell
python scripts/evaluate_shared_slam.py --stereo --sequence 04 --max-frames 80 `
  --max-wall-seconds 320 --data-root <DATA_ROOT> --poses-root <POSES_ROOT> `
  --output results/verified-stereo04 --stereo-depth-policy verified_all `
  --stereo-pose-arbitration --stereo-raw-reference-retry --loop-mode off `
  --matching-backend cpu --retrieval current --opencv-threads 1
```

Use separate output folders for the control and experiment. Preserve explicit
loss and incomplete runs. Check finite exports and identities before comparing
accuracy, reprojection quality, runtime and memory. The 80-frame check is a
regression gate, not full-sequence or general release validation. Modeling the
uncertainty of non-keyframe source poses remains separate work.
