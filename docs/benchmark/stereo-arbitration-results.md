# Stereo pose arbitration diagnostics

These are uncached CPU diagnostics on KITTI 04, frames 0–79. Bundle adjustment
is enabled except in the explicitly labeled ablation; loops are off. Ground-truth
poses are used only for evaluation.
The short prefix contains one eligible trajectory segment; its drift scores do
not establish full-sequence accuracy or release readiness.

| Configuration | ATE, m | Translation, % | Rotation, degrees/m | Lost frames | Keyframes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supported stereo depth | 0.307855 | 0.605204 | 0.015096 | 0 | 9 |
| Verified depth restoration | 0.158714 | 0.363235 | 0.009537 | 0 | 9 |
| Initial arbitration, disconnected observations | 0.271424 | 1.368635 | 0.015600 | 0 | 44 |
| Revalidated arbitration, BA enabled | 0.163960 | 0.630748 | 0.006577 | 0 | 10 |
| Revalidated arbitration, BA disabled | 0.298345 | 1.355426 | 0.013582 | 0 | 10 |

The initial arbitration experiment at `f8ac74e` selected the independent pose on
43 frames. Its fallback path cleared all map associations and flow tracks,
creating new single-view landmarks at every selection. All 42 bundle-adjustment
attempts then reported insufficient observations. Accuracy regressed, so testing
did not expand to 01. This rejected revision and its results are retained.

The repair validates existing connections at the already selected pose, without
refitting that pose. Connections must pass the existing stereo reprojection and
support gates. Passing connections use normal keyframe cadence; insufficient
support retains the established fallback. Reattached observations are subsequent
mapping evidence, not held-out evidence for the earlier arbitration decision.

The repaired estimator at `53af112` retained connections on all 63 independent
selections. All eight BA attempts were accepted. It exported 80 finite poses and
11,684 finite sparse points, with no tracking loss. Total wall time was 53.46 s
and peak process memory was 264.43 MiB. The same-source BA-disabled ablation took
40.61 s and 261.56 MiB. Disabling BA worsened all three accuracy measures; it is
not a proposed production configuration. Tracking and map chronology naturally
change in this pipeline ablation, so it does not isolate an identical graph solve.

Every accepted BA update worsened the current raw stereo residual diagnostic.
At frame 44, its cost rose from 0.396 to 3.626, while inliers fell from 114 to 92.
These rows are not withheld from subsequent mapping/BA, and that change alone
does not prove BA caused benchmark drift. The ablation shows that BA helps this
prefix overall. A blanket residual rollback or disabling BA is not justified.

Relative to the supported-depth baseline, repaired ATE and rotation improve,
while translation increases by 4.2%. Relative to verified depth without
arbitration, translation increases by 73.6%, ATE by 3.3%, and rotation decreases
by 31.0%. This remains an unresolved tradeoff, not a release pass. The next
bounded cycle must examine uncertainty and optimization consistency, then
retest 01 before expanding full-sequence coverage. Repeated association rejection
also needs persistent miss bookkeeping before enabling arbitration by default.

Arbitration remains opt-in through `--stereo-pose-arbitration`; monocular use is
rejected. The default depth policy remains `supported`. Before/after bundle
residual diagnostics do not establish protection against future loop corrections.

[Saved measurements and source hashes](STEREO_ARBITRATION_RESULTS.json) and
[accuracy, keyframe, runtime and memory graphs](plots/stereo-arbitration04-comparison.png)
include the rejected experiment.
[Trajectory, input image, errors and sparse map](plots/stereo-arbitration04-overview.png)
show the repaired run.
