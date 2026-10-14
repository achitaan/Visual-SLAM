# Native online SLAM benchmark evidence

This directory publishes saved results for the frozen `orbslam3-origin-serialized` candidate. It does not install the engine or promote it to main. Real-time throughput is outside this implementation's release requirements; tracking, optimization and mapping occur during processing, and emitted poses cannot be rewritten by later work.

## Results

There are 19 paired comparisons: 17 original strict passes and two user-accepted one-second translation-RPE tradeoffs. Full01/04 and short01/350+04/80 use seeds 0, 1 and 2. Full00/03/05/06/07/09/10 use seed 0. Every candidate run has lower ATE, lower whole runtime and zero tracking losses.

| KITTI sequence | Seeds | VO ATE (m) | Native ATE (m) | VO runtime (s) | Native runtime (s) |
|---|---|---:|---:|---:|---:|
| 00† | 0 | 20.610 | 3.947 | 2642.8 | 1029.8 |
| 01 | 0,1,2 | 77.732 | 15.105 | 449.2 | 155.1 |
| 03 | 0 | 2.327 | 1.273 | 449.6 | 94.1 |
| 04 | 0,1,2 | 1.743 | 0.288 | 139.3 | 34.6 |
| 05 | 0 | 9.555 | 2.212 | 1460.0 | 376.6 |
| 06 | 0 | 2.815 | 2.561 | 511.5 | 133.3 |
| 07 | 0 | 3.913 | 1.363 | 581.6 | 120.6 |
| 09 | 0 | 5.133 | 3.299 | 790.2 | 205.0 |
| 10† | 0 | 2.564 | 1.363 | 644.2 | 154.6 |

† Original strict RPE failures on00/10 are preserved. Full00 translation RPE is 0.191209660m versus 0.187450638m (+2.005340%); full10's regression is +0.227636%. Human acceptance is a release tradeoff, not a retroactive strict pass.

[All individual runs](results.csv) include raw/aligned ATE, translation and rotation RPE, drift, losses, runtime, end-to-end FPS, available memory and frame latency. [Results JSON](results.json) also contains all11 grouped results, source identities and case/proof hashes.

## Method and limitations

- Ground truth is evaluator-only. Original held/lost/dropped frame IDs stay in coverage. These measurements use immutable online poses, not shutdown-corrected trajectories.
- Aligned ATE uses SE(3) without scale fitting. RPE uses one-second endpoint pairs. Drift uses metric translation percent and degrees per meter.
- Comparisons use fresh serial processes, fixed configurations and recorded seeds. Configuration, input, source, runtime and command records are preserved in the38 case JSONs under `cases/`; verification records are under `evidence/`.
- Linux C++/OpenCV4.8.1 versus Windows Python/OpenCV5.0 is a whole-system comparison. It does not isolate a language, operating-system or kernel speedup.
- Grouped entries are medians computed independently per arm/metric. No statistical significance or confidence interval is claimed. Most full sequences have only one paired seed.
- Full00's baseline and first actual candidate arm were separated by a documented bootstrap/storage pause. Its native execution completed; the only failed numerical gate was translation RPE. No replay was selected for better timing.
- These public KITTI development sequences informed engineering decisions; they are not an untouched or hidden test set.02/08, other datasets, arbitrary camera inputs and portable installation remain unqualified.
- Memory has the process scope recorded in the export; it is not total machine/WSL memory. Frame latency differs from whole-run throughput. Missing measurements remain missing.

## Evidence and reproducibility

The [source manifest](source-manifest.json) identifies exactly the measured source. [manifest.json](manifest.json) hashes every included original result, case, acceptance and verification record. Original files are copied byte-for-byte; machine-local paths in them document the actual runs, not portable commands. The source-manifest SHA256 is `8fc711ec8d8c629e126c588b2ef78cbe24507f7a3f8aee89cb1c041a33aae4b6`.

The38 `cases/*.json` records retain the exact commands and environments used. Large input images, maps, trajectories, binaries and raw logs remain in the preserved local campaign and are not included in this compact evidence package. Hashes allow identity checks; they do not replace access to those artifacts or constitute independent code review. Reproducing estimator runs requires the pinned native source/runtime and prepared inputs; a portable engine package is still pending.

To check this package without running SLAM:

```sh
python docs/benchmark/online-native/verify.py
```

## Release notes — benchmark publication

- Published19 completed comparisons, all11 grouped scopes,38 case records, source identity and original verification/acceptance evidence.
- Preserved17 strict passes and the accepted00/10 RPE exceptions explicitly.
- Added current full-sequence results to the project README.
- No estimator, default backend or main-branch change; no benchmark replay. Real-time performance is not required for the intended release.
