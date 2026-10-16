# Verified original Python SLAM measurements

Historical completed measurements of this repository’s Python SharedSlam engine, restored on the development branch. The estimator modules match the frozen verified source; application configuration passthrough is tested. These are original Python results.

The table shows paired full-sequence seed0 runs. All original frame IDs, including lost/held poses, remain in coverage. ATE uses SE(3) alignment with scale fixed to one; raw ATE is also shown. Ground truth was evaluator-only.

| KITTI | Frames | ATE m: ours / VO | Raw ATE m: ours / VO | Translation %: ours / VO | Rotation deg/m: ours / VO | Lost: ours / VO | Estimator seconds: ours / VO |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00 | 4541 | 7.422 / 20.610 | 21.113 / 50.200 | 1.291 / 2.147 | 0.004469 / 0.009378 | 0 / 0 | 8167.1 / 2092.2 |
| 01 | 1101 | 71.972 / 72.293 | 292.754 / 170.982 | 7.814 / 9.574 | 0.012369 / 0.013375 | 1 / 23 | 1838.5 / 432.4 |
| 03 | 801 | 1.679 / 2.327 | 2.925 / 5.395 | 1.115 / 1.705 | 0.003120 / 0.007541 | 0 / 0 | 723.8 / 480.0 |
| 04 | 271 | 0.536 / 1.763 | 1.294 / 4.896 | 0.634 / 1.683 | 0.003622 / 0.011001 | 0 / 0 | 218.7 / 134.7 |
| 07 | 1101 | 2.403 / 3.913 | 5.974 / 8.071 | 1.380 / 1.976 | 0.007799 / 0.013475 | 0 / 0 | 1116.2 / 518.6 |
| 09 | 1591 | 3.958 / 5.133 | 10.915 / 17.330 | 1.132 / 1.705 | 0.002859 / 0.007053 | 0 / 0 | 1817.3 / 794.5 |
| 10 | 1201 | 1.602 / 2.564 | 5.958 / 18.695 | 0.688 / 1.368 | 0.004266 / 0.007471 | 0 / 0 | 1361.5 / 655.9 |

Full01 raw error is worse than VO, despite lower aligned error and drift. Runtime is slower on several sequences. These tradeoffs remain visible; wider reliability validation is incomplete. Full02/05/06/08 and real EuRoC are pending. Real-time execution is not a requirement for this implementation.

The operator optimization was separately confirmed on full01/04/07 and three seeds of both short sequences with exact trajectories/state/metrics and lower runtime. This restores previously verified work; it does not claim a new accuracy gain.

[Exact values, configuration, hashes and case identifiers](original-python-verified.json).

## Run the verified configuration

Use the existing CUDA-capable Python environment for the recorded matching backend. Supply local rectified KITTI images and calibration.

```powershell
python src/main.py --data-root .datasets/shared-kitti --sequence 04 --max-frames 80 `
  --slam --stereo --loop-mode off --features 1500 --matching-backend cuda --opencv-threads 1 `
  --stereo-depth-policy verified_fallback --stereo-pose-arbitration `
  --stereo-raw-reference-retry --stereo-verified-reference-retry `
  --stereo-descriptor-association-retry --stereo-map-depth-policy verified `
  --bundle-solver-accuracy precise --bundle-landmark-policy bounded_fixed `
  --stereo-bundle-gauge-mode veto --frame-delay-ms 0 --output results/own-stereo/poses.txt
```

The original dashboard uses localhost:3000 and WebSocket port8765. Add `--no-telemetry` for file-only execution. Changing to CPU matching is supported, but the recorded CUDA timings are not CPU measurements. Preserve previous run folders.

For evaluation, use the same flags with `scripts/evaluate_shared_slam.py` (omit `--slam` and `--frame-delay-ms`, use an output directory). Add `--poses-root <local-reference-directory>`; references are read after estimator shutdown. Record seed, source/runtime/input identities and complete coverage.
