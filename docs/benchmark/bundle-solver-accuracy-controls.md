# Bundle solver accuracy controls

This diagnostic separates inner solver accuracy from the added stereo image
rows in the [failed target-relative experiment](stereo-target-relative-bundle.md).
It does not change image measurements, Huber loss, variable scales, map gauge,
the 30-evaluation outer limit, or geometric acceptance limits. Ground truth
remains evaluator-only.

`--bundle-solver-accuracy default` preserves the previous behavior: ordinary
bundle adjustment uses the existing inner defaults, while valid owned image
augmentation already requests precise LSMR. `precise` applies the same
`atol=btol=1e-12` and `maxiter=max(500, variable_count)` to ordinary bundle
adjustment too. More precise steps can change poses and increase runtime.

The requested policy is saved in configuration and result identities. Each
bundle report declares its effective policy and inner options; a skipped solve
reports that it did not run. Precise results cannot reuse older results with
missing policy metadata. Existing historical results remain separate.

The first comparison uses fresh KITTI 04 frames 0–79 for three declared cases:

| Case | Owned image rows | Requested solver policy |
| --- | --- | --- |
| Default control | Off | default |
| Precision control | Off | precise |
| Augmentation control | On | precise |

All cases use stereo, CUDA matching, current retrieval, one OpenCV thread,
verified-fallback depth, stereo pose arbitration and raw-reference retry,
with loops off. No diagnostic cache or per-sequence parameter selection is
used. Compare the precision control with the default control, then the
augmentation control with the precision control. Later map states differ;
their bundle objectives are not interchangeable accuracy scores.

Use `scripts/evaluate_shared_slam.py` with calibrated KITTI image and pose roots,
`--stereo --sequence 04 --max-frames 80 --loop-mode off --matching-backend cuda
--retrieval current --opencv-threads 1 --stereo-depth-policy verified_fallback
--stereo-pose-arbitration --stereo-raw-reference-retry`, and a separate output
directory per case. Set the requested solver policy above and add
`--stereo-owned-image-bundle` only for the augmentation case. Include a
wall-clock deadline. Reference poses are opened after estimator shutdown.

The fresh partial control comparison is recorded below. Full release validation remains paused.

## Fresh matched 04/80 solver-accuracy controls

This is a fresh, matched, partial KITTI 04 comparison from source fingerprint
`67dac6c9a49c807b2a39eaa428821c5ade6f38a0968d66161d2c308d7ba3cc16` with CUDA matching and evaluator-only ground truth. The
three requested arms are default-off, precise-off, and precise-on. The default-off versus precise-off comparison isolates solver accuracy. The precise-on versus precise-off comparison isolates the owned image-factor change under the same precise policy. Later map states differ, so BA objective values are not compared across runs.

| Case | Policy | Image factors | ATE RMSE (m) | Translation (%) | Rotation (deg/m) | Lost | Runtime (s) | Peak memory (MiB) | Applied BA |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default control | default | off | 0.11188 | 0.28075 | 0.0056528 | 0 | 42.447 | 969.99 | 9/9 |
| Precise control | precise | off | 0.11084 | 0.31493 | 0.0051844 | 0 | 55.19 | 962.81 | 9/9 |
| Precise + image factors | precise | on | 0.16393 | 0.41038 | 0.0048145 | 0 | 69.967 | 982.51 | 9/9 |

The precision control reduced ATE by 0.93% and rotation drift by 8.29%, but
increased translation drift by 12.17%. Adding the image factors under the same
precise policy increased ATE by 47.90% and translation drift by 30.31%, while
reducing rotation drift by 7.14%. Both comparisons failed the declared 5%
regression gate. These are short diagnostic runs, with only one eligible
trajectory segment; they do not establish full-sequence accuracy.

All nine bundle solves in each arm reached the 30-evaluation cap. A saved-input
audit of the first solve confirmed identical original inputs and reconstructed
the image objective without ground truth or another solve. Matching solver
precision greatly reduced the first target-pose discrepancy, but the image
augmentation regression persisted over the replay. A small feasible objective
improvement remains; it does not establish the cause of the trajectory error.
Later keyframe schedules differ, so their bundle costs cannot isolate a cause.

The next diagnostic should locate the first divergence between the two precise
arms and inspect observation consistency, conditioning and corrections on the
same saved inputs before changing the measurement model. Full-sequence
expansion remains stopped until the focused accuracy gate passes.

The plot shows estimated trajectories without overlaying the reference path;
metric values above are copied from each actual evaluation export. No metrics
were reused across revisions. The source, runtime, input, reference, and frame
coverage identities match after removing only the declared solver-policy and
image-factor mode fields. Solver reports record requested and effective LSMR
policies. KITTI 01 was not run. This checkpoint is diagnostic only, does not
claim release readiness, and does not change the default-off setting.


See the [control comparison](plots/bundle-solver-controls-04-80-trajectory-and-metrics.png) and actual run overviews:
[Default control](plots/bundle-solver-controls-04-80-default-off-overview.png), [Precise control](plots/bundle-solver-controls-04-80-precise-off-overview.png), [Precise + image factors](plots/bundle-solver-controls-04-80-precise-on-overview.png).
The earlier [target-relative failed partial gate](stereo-target-relative-bundle.md)
remains a separate experiment and is not replaced by these controls.

## Validation and checkpoint

The frozen source checkpoint is `ebcfec12cc9952d739019d9f68a75914e9db270f`
on `codex/stereo-bundle-solver-controls`. The full backend suite passed
450 tests with no skips in 77.81 seconds in the CUDA-enabled environment.
The 15 focused control tests passed in 7.83 seconds. Passing implementation
tests does not override the failed accuracy gate.

Preparation, tests, replays and reporting shared a 60-minute cycle. A
conservative three-arm launch was rejected before reading inputs. Two controls
were scheduled first; their measured completion left enough budget for the
third arm. Each replay used fresh results, preserved prior history, and
completed without timeout or source changes. The scheduler remains paused;
this experimental checkpoint is not approved for main.
