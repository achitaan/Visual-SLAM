# Stereo motion regularizer experiment

The saved KITTI 01 audit shows local bundle adjustment reducing image error while
changing some previously verified stereo rotations. This branch tests a motion
term inside the local bundle objective. It remains disabled by default.

The term uses `Log(Z^-1 T_source^-1 T_target)` in right SE(3) translation/rotation
coordinates. Its fixed metric comes from raw stereo training observations with
point variables eliminated by QR projection. The declared one-pixel model is
independent in left u, v and disparity; computed right-u errors are correlated
with left-u errors. Rank and conditioning are checked in baseline-scaled units.

This deliberately reweights observations already used by tracking and mapping.
It is a **correlated regularizer**, not independent sensor fusion or a calibrated
pose covariance. A consistently biased raw motion can receive more influence.
Ground truth remains evaluator-only. No sequence-specific weights or relaxed
tracking thresholds are introduced.

Only exact raw fit-row geometry and pixel identities are eligible. Factors own
immutable measurement, information and provenance snapshots. Missing rows,
changed calibration or source geometry, duplicate physical training pixels and
singular information fail closed. Historical recorded poses follow their actual
keyframe anchors during optimization; both image and augmented objectives must
strictly improve before correction. Existing 0.5 m / 1.5 degree agreement gates,
map revision checks and the fixed origin remain in effect.

Enable this experiment with `--stereo-motion-regularizer` alongside `--slam --stereo` in the normal command, or `--stereo` in the shared evaluator. Saved
runs include active/skipped factors, objectives, fit-row IDs and frozen factor
provenance. Invalid unused training depth is explicitly null with validity masks;
information rows, poses and map exports must remain finite.

Synthetic checks cover finite-difference projection/SE(3) Jacobians, independent
point-block Schur elimination, correlated whitening, unit scaling, immutable
arrays, invalid inputs, delayed anchor corrections, unchanged hard gates, stale
calibration, and an actual SciPy solve with competing image/motion evidence.
These checks do not establish KITTI accuracy. The affected 01 prefix and wider
stereo/monocular, loop and reconstruction acceptance checks remain required.

## Validation checkpoint

The frozen estimator fingerprint is
`d620886b6cb4e01651b02595b8bf28ce7e6a51c5658cc8f533413c9cb02dc295`.
The full backend suite passes: 381 tests, with four optional GPU skips. The run
completed in 56.62 seconds and source/runtime fingerprints remained unchanged.

No KITTI score is attributed to this revision yet. The first 04/80 diagnostic
was deferred: its conservative preparation, replay and reporting estimate was
246 seconds, exceeding the remaining bounded cycle allowance. Begin the next
cycle with that smoke gate, then the stateful 01/350 diagnostic if the correctness
and runtime gates permit. Do not use scores from the previous raw-reference
revision as validation of this experiment. The known rotation regression,
coherent raw-motion bias and full-sequence reliability requirements remain open.
