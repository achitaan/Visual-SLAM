# Future landmark observation diagnostics

This experimental checkpoint records how bundle-adjusted stereo landmarks are
used by later tracking. It adds optional evidence capture, without changing
matching, stereo sampling, pose fitting, bundle objectives or acceptance limits.
It does not resolve the accuracy regression reported in
[the solver controls](bundle-solver-accuracy-controls.md).

The evaluator accepts `--tracking-diagnostics-dir` and
`--tracking-diagnostics-frames`. Both must be supplied together. At most seven
distinct nonnegative frame IDs may be selected, and the diagnostics directory
must be inside the run output directory. Each frame has a hashed JSON record.
Snapshots include ordered solve inputs, inlier IDs, accepted image measurements,
map positions and anchors, reference context, and pre/post-optimization state.
They contain no ground-truth poses, raw images or descriptors.

Expected missing stereo measurements use null values with validity masks.
Unexpected invalid geometry, incomplete provenance, truncation and write errors
are recorded explicitly. Capture failures do not change tracking. Missing
reference provenance remains unknown; a measurement used to choose a pose is
not independent validation evidence.

Stored rows and events are capped. Map snapshots currently copy the selected
map state before applying those storage caps; transient memory on large maps
has not been validated. Use this capture for short diagnostic replays.

## Fresh capture parity check

Source fingerprint:
`2f5feb2daf678c2b883a4f2ad46cf9b254d7662042390b06e45d76c0db96885e`.
Four fresh KITTI 04 replays processed frames 0–20 with precise bundle solving,
CUDA descriptor matching and loops off. The two estimator configurations differ
only in the experimental owned image factors. Each configuration was run once
with diagnostics for frames 14–20 and once without diagnostics.

| Image factors | Capture | Frames | Lost | Estimator runtime (s) |
| --- | --- | ---: | ---: | ---: |
| On | Off | 21 | 0 | 15.083 |
| On | On | 21 | 0 | 22.779 |
| Off | Off | 21 | 0 | 13.106 |
| Off | On | 21 | 0 | 20.801 |

For both pairs, exported trajectories, sparse PLY files and previews were
byte-identical. Persistent maps, tracking decisions, matcher counters and depth
sampling counters also matched. All fourteen selected frame records were
complete and passed their file-hash checks. Ground truth remained evaluator-only.

The full backend suite passed **467 tests** in **81.89 seconds**. The focused
capture module passed **17 tests**, including invalid stereo holes, failed pose
attempts, stage ordering, immutable snapshots and write-failure isolation.

These short runs validate capture parity, not full-sequence accuracy or an
end-to-end GPU speedup. Detailed capture adds serialization overhead. Feature
extraction, tracking and bundle adjustment still run on CPU. Historical CPU/GPU
matching comparisons measure only the matching component.

## Reproduction

Run `scripts/evaluate_shared_slam.py` against calibrated KITTI images, with
`--stereo --sequence 04 --max-frames 21 --loop-mode off
--bundle-solver-accuracy precise --matching-backend cuda --retrieval current
--opencv-threads 1 --stereo-depth-policy verified_fallback
--stereo-pose-arbitration --stereo-raw-reference-retry`. Add
`--stereo-owned-image-bundle` for the experimental factor configuration.
Use a separate output directory and a wall-clock deadline for every run.

For captured runs, also supply a diagnostics directory beneath that run and
`--tracking-diagnostics-frames 14 15 16 17 18 19 20`. Bundle input capture at
frames 14 and 20 can be enabled separately with the existing
`--bundle-diagnostics-dir` and `--bundle-diagnostics-frames` options.

The next correctness check compares actual later measurements against the old
and corrected landmark positions at the same accepted camera pose. Results
must retain measurement-consumption labels and unavailable rows. A lower
conditional reprojection residual alone does not establish better tracking or
trajectory accuracy. Full validation remains paused until the focused accuracy
gates pass.

## Saved future-observation audit

The first augmented solve contained 108 added factors: 106 singleton landmarks
and two shared landmarks. The saved input checks matched exactly between the
factor-on and factor-off runs. At accepted pre-BA camera poses for frames 15–20,
221 actual map-input rows from 77 cohort landmarks had summed robust
reprojection cost **492.983** using the old positions and **289.129** using the
corrected positions. Weaker detector-reference claims are reported separately.
No additional matching, pose fitting or ground-truth fitting was performed.

These rows were consumed by estimation or pose selection, with some other
reference paths unknown. They do not provide an independent accuracy test.
The source-frame anchor transport at the second solve, frame 20, agreed to
numerical precision. This short capture does not support a forgotten first
correction or broken point transport as the cause of the later 80-frame
regression. The next bounded experiment should capture the first later event
where the trajectories or keyframe schedules materially diverge, before
changing the estimator again. Start with paired 27-frame replays, retaining the
first solve at frame 14 and tracing frames 21–26 around the next solve. Keep the
same declared configuration and make no estimator changes between the pair.

After these captures, a trailing blank line in the new test module was removed.
The final publication fingerprint is
`b89f7c452546a43b319c71df93ea7341becd2bbd70f09ea484ed4a4b76e932fa`.
All estimator file hashes are unchanged. The final source passed the full
**467-test** backend suite in **74.85 seconds**. The capture group retains its
original fingerprint above; it is not relabeled as a new estimator evaluation.
