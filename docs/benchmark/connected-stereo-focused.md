# Connected stereo tracking: focused diagnostics

These are saved uncached CPU results for KITTI 04 frames 0–79 and KITTI 01
frames 0–349. Each variant uses one configuration across both sequences. Loops
are off; local bundle adjustment is enabled. Reference poses enter evaluation
after tracking. Stereo alignment is SE(3), with scale fixed to one.

| Sequence / frames | Variant | ATE, m | Translation, % | Rotation, degrees/m | Lost frames |
| --- | --- | ---: | ---: | ---: | ---: |
| 04 / 80 | Supported mapping | 0.307855 | 0.605204 | 0.015096 | 0 |
| 04 / 80 | Verified depth | 0.158714 | 0.363235 | 0.009537 | 0 |
| 04 / 80 | Connected arbitration | 0.163960 | 0.630748 | 0.006577 | 0 |
| 01 / 350 | Supported mapping | 9.291663 | 5.051622 | 0.010592 | 1 |
| 01 / 350 | Verified depth | 8.222911 | 4.684565 | 0.013808 | 1 |
| 01 / 350 | Connected arbitration | 7.297773 | 4.165512 | 0.013341 | 0 |

Connected arbitration was measured at `b01684a`, development fingerprint
`f18bd9ea4df7b17bbc74abba5b8f88bb755b2cee33fe79a18d57468152dc1da1`.
The focused cycle completed in 387.53 seconds, including 203 passing synthetic
and integration tests, four optional GPU skips, input preparation and plots.
It reproduced the preceding repaired 04 trajectory exactly. The older variants
are separate source groups, not validation of later code.

On 01, ATE improves 21.46% and translation drift improves 17.54% against supported
mapping. Rotation drift regresses 25.95%, exceeding the 5% gate. Estimator time
increases from 240.94 to 286.61 seconds and peak memory from 672.42 to 736.41 MiB.
Total evaluator wall time is 296.07 seconds; the supervising subprocess takes
297.92 seconds. These timings are single shared-host diagnostics, not isolated
full-sequence performance measurements. The run is not ready for wider release
validation.

The former frame-124 orientation conflict is reduced from 1.094 to 0.030 degrees.
The new run tracks through frame 334, where both earlier mapping variants lost
tracking for one frame. However, its accepted stereo reference pose has no map
connections. Of 186 arbitration selections, 181 retain geometrically validated
connections. Separately, all 109 hard-conflict reference fallbacks clear
connections and create keyframes; 107 have bidirectional verification. Only 62
have an immutable reserved fitting/holdout context. The other reference fits
use the active depth policy and must not be described as supported-only or
held-out estimation.

This produces 155 keyframes and 133,623 sparse landmarks. Of 153 local BA attempts,
51 apply, 92 lack sufficient connected observations and ten fail independent
motion consistency. Post-BA raw stereo cost worsens on 41 inspected rows and
improves on nine. Those measurements are not withheld from subsequent mapping;
they do not establish that BA caused the rotation regression. The preceding
[matched 04 BA ablation](stereo-arbitration-results.md) also found that disabling
BA worsened all three accuracy measures.

The next diagnostic should validate map connections at an already-selected
bidirectionally verified reference pose, without refitting it. Require matching
accepted frame endpoints and calibration, current supported stereo measurements,
and the existing reprojection, retention and spatial-support gates. Failed
validation must retain the existing fallback. This is a general connectivity
repair, not a sequence-specific acceptance change. Persistent rejection counters
and dependency-aware result reuse also need correction before arbitration is
enabled by default.

Saved source and evaluator hashes, input/reference identities and finite exports
were checked. All 350 poses, keyframe observations and 133,623 PLY points are
finite. The independent stereo-motion audit contains 345 measurements, with
maximum disagreement of 0.436 m and 0.412 degrees, below the existing 0.5 m and
1.5 degree limits. Passing these consistency guards does not waive the rotation
accuracy regression. The development fingerprint still lacks dependency/runtime
identity; artifact inspection does not waive that harness gap.

[Machine-readable results](CONNECTED_STEREO_FOCUSED_RESULTS.json),
[accuracy comparison](plots/connected-stereo-focused-comparison.png), and
[01 trajectory, errors, input and sparse map](plots/connected-stereo01-overview.png)
retain the regressions. Full-sequence stereo validation, live/offline loop
correction, monocular recovery and RGB-only TUM remain unproven by these prefixes.
