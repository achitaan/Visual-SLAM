# SLAM performance pilot

Performance branch based on shared-SLAM commit `63454a0`. Accuracy settings and geometric checks are unchanged. Shared-SLAM and main were not merged.

Full KITTI01 stereo completed in **21.0 minutes versus 70.8 minutes frozen (3.37×)**. ATE is identical at 87.755872 m; lost frames are 30 versus 30 frozen. Peak RAM rises from 1.43 to 1.98 GiB (+38.4%). This combined CPU/index/CUDA result does not attribute the full gain to indexing alone.

Host: Windows, 12 logical CPU processors, 15.8 GiB RAM; NVIDIA GeForce GTX 1660 SUPER (6.0 GiB). Python 3.12.14, OpenCV 5.0.0, SciPy 1.18.1, NumPy 2.5.3; isolated PyTorch 2.7.1+cu126/CUDA 12.6. OpenCV/BLAS use one worker unless explicitly stated.

Timings are provisional: retrieval audits and validation work overlapped on this shared host, and background workloads were not controlled. Each controller ran its cases serially; the final repeat overlapped the promoted full run. These single-run comparisons do not establish an uncontended throughput guarantee.

## Pilot and final repeat

KITTI 04 is full (271 frames); small KITTI 01 and TUM fr1 desk cases are 300-frame prefixes. The promoted KITTI 01 run is 1101 frames. Stereo ATE uses SE(3); monocular ATE uses Sim(3) with scale fitting only for evaluation. All table runs use one OpenCV worker.

Elapsed time includes input loading, estimator processing and background finalization. It excludes estimator setup, exports and reference evaluation. FPS is frames divided by that elapsed time; export/setup times are separate. Baseline export/setup times and the saved full baseline's frame latency were not collected, so whole-process speedup is not claimed.

| Case | Stage | Backend | Frozen s | Candidate s | Speedup | FPS | RAM MiB | Torch VRAM MiB | ATE m | Lost | Loops | Quality |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 04-stereo | initial | cpu | 277.1 | 261.9 | 1.06× | 1.03 | 341.1 | 0.0 | 0.7637 | 0 | 0 | pass |
| 04-stereo | initial | cuda | 277.1 | 186.5 | 1.49× | 1.45 | 1017.9 | 30.6 | 0.7637 | 0 | 0 | pass |
| 04-stereo | final | cuda | 277.1 | 157.2 | 1.76× | 1.72 | 1041.7 | 30.6 | 0.7637 | 0 | 0 | pass |
| 04-mono | initial | cpu | 160.8 | 161.8 | 0.99× | 1.67 | 291.8 | 0.0 | 0.4151 | 0 | 0 | pass |
| 04-mono | final | cpu | 160.8 | 137.4 | 1.17× | 1.97 | 292.6 | 0.0 | 0.4151 | 0 | 0 | pass |
| 01-stereo | initial | cpu | 309.6 | 291.3 | 1.06× | 1.03 | 502.7 | 0.0 | 14.4019 | 2 | 0 | pass |
| 01-stereo | initial | cuda | 309.6 | 209.4 | 1.48× | 1.43 | 1212.4 | 30.6 | 14.4019 | 2 | 0 | pass |
| 01-stereo | final | cuda | 309.6 | 207.0 | 1.50× | 1.45 | 1213.1 | 30.6 | 14.4019 | 2 | 0 | pass |
| 01-mono | initial | cpu | 443.8 | 489.6 | 0.91× | 0.61 | 297.4 | 0.0 | 43.9145 | 78 | 0 | pass |
| 01-mono | final | cpu | 443.8 | 462.1 | 0.96× | 0.65 | 298.2 | 0.0 | 43.9145 | 78 | 0 | pass |
| tum-desk | initial | cpu | 369.6 | 457.8 | 0.81× | 0.66 | 235.4 | 0.0 | 0.7993 | 105 | 0 | pass |
| tum-desk | final | cpu | 369.6 | 310.4 | 1.19× | 0.97 | 232.9 | 0.0 | 0.7993 | 105 | 0 | pass |
| 01-full-stereo | promoted | cuda | 4248.5 | 1259.7 | 3.37× | 0.87 | 2023.6 | 30.6 | 87.7559 | 30 | 0 | pass |

## Latency, recovery and drift

Final repeats and promotion only. RAM is process peak working set; GPU values are peak Torch allocation, not total device usage.

| Case | Median / p95 ms | Input / export s | RAM change | Lost intervals | Recoveries | Translation % | Rotation deg/m |
|---|---:|---:|---:|---:|---:|---:|---:|
| 04-stereo | 498.57/1120.85 | 3.50/2.39 | +204.5% | 0 | 0 | 1.00 | 0.0083 |
| 04-mono | 260.34/1464.70 | 1.50/1.73 | -11.8% | 0 | 0 | unavailable | unavailable |
| 01-stereo | 513.85/1405.55 | 3.44/5.82 | +159.5% | 1 | 1 | 6.98 | 0.0092 |
| 01-mono | 334.92/4679.51 | 1.81/1.63 | -11.7% | 2 | 1 | unavailable | unavailable |
| tum-desk | 579.71/3140.59 | 3.65/1.76 | -8.6% | 2 | 2 | unavailable | unavailable |
| 01-full-stereo | 757.05/2991.72 | 13.43/24.55 | +38.4% | 2 | 2 | 10.90 | 0.0132 |

JSON and CSV include median/p95 frame latency, drift metrics where available, recovery counts, memory changes, and per-file source hashes. Prefix results are not substitutes for full-sequence scores.

## Attribution and limits

- The original CPU profile identified SIFT, disparity and matching as the largest costs. Bundle adjustment also rebuilt observation arrays inside every residual evaluation.
- Indexed retrieval proposes at most 20 keyframes, followed by exact reranking and unchanged verification. Failed recovery retains exhaustive fallback, so difficult lost-tracking periods can still be expensive.
- CPU changes cache map arrays, batch stereo measurements, avoid global cleanup scans, and precompute bundle observation arrays. Solver budgets, losses, gauges and geometric checks are unchanged.
- CPU matching remains the default. CUDA is optional; transfers and synchronization are included in matching timings. Near ties, ratio boundaries and cancellation-prone distances are resolved on CPU.
- GPU process RAM includes PyTorch/CUDA runtime overhead and can substantially exceed the CPU process. Torch allocation statistics exclude driver/display/context memory, so the VRAM column is not total device usage.
- The 300-keyframe live optimization guard remains unchanged. No accuracy fixes or faster presets were included.

## Explicit worker-count trial

Full KITTI 04 stereo with CUDA and 4 OpenCV workers: **2.08×** versus the one-worker frozen baseline (133.5 s). ATE 0.7637 m; lost frames 0; quality gate passed. This changes CPU parallelism, not feature counts or geometry settings. The default remains one worker; this setting was checked on this case only.

## Retrieval and graph evidence

Saved KITTI 00 input-image audit: **15/15 known loop pairs retrieved**, with 519 retained keyframes. This is a SIFT-image proxy, not fresh live geometry verification.

Graph replay used saved independent loop measurements and odometry reconstructed from exported adjacent poses. It does not reconstruct the original pre-correction solver snapshot or demonstrate long-sequence tracking.

Replay: 12.18 s; finite poses and fixed origin: True. The graph solver source is unchanged.

The initial histogram-only index recalled 11/15 known pairs; residual summaries improved this to 13/15, and reserving neighboring views reached 15/15 within the 20-candidate budget. These choices were tuned on this audit; held-out long-sequence loop validation remains necessary before merging.

## Separate profiling and component checks

| Profile | Stage | Calls | Total s | Median ms | p95 ms |
|---|---|---:|---:|---:|---:|
| profile-final01-stereo | input_loading | 100 | 1.12 | 10.96 | 14.30 |
| profile-final01-stereo | features | 100 | 15.15 | 153.71 | 202.67 |
| profile-final01-stereo | disparity | 100 | 14.42 | 138.53 | 197.48 |
| profile-final01-stereo | extraction | 100 | 29.89 | 300.30 | 399.81 |
| profile-final01-stereo | mapping | 16 | 0.31 | 20.11 | 30.77 |
| profile-final01-stereo | frame | 100 | 65.64 | 623.44 | 1365.52 |
| profile-final01-stereo | matching | 198 | 20.16 | 106.87 | 165.34 |
| profile-final01-stereo | optical_flow | 198 | 2.85 | 10.22 | 42.90 |
| profile-final01-stereo | pnp | 99 | 0.36 | 3.42 | 6.87 |
| profile-final01-stereo | tracking | 99 | 16.37 | 153.96 | 333.43 |
| profile-final01-stereo | bundle_adjustment | 14 | 7.94 | 629.82 | 822.42 |
| profile-final01-stereo | finalization | 1 | 0.00 | 0.03 | 0.03 |
| profile-final01-stereo | export | 1 | 0.97 | 972.28 | 972.28 |
| profile-background01-stereo | input_loading | 180 | 2.03 | 10.86 | 14.19 |
| profile-background01-stereo | features | 180 | 34.82 | 191.32 | 234.92 |
| profile-background01-stereo | disparity | 180 | 35.65 | 193.12 | 254.84 |
| profile-background01-stereo | extraction | 180 | 71.22 | 393.08 | 477.53 |
| profile-background01-stereo | mapping | 31 | 0.97 | 24.81 | 80.10 |
| profile-background01-stereo | frame | 180 | 117.99 | 542.47 | 1408.20 |
| profile-background01-stereo | matching | 358 | 4.00 | 10.40 | 15.82 |
| profile-background01-stereo | optical_flow | 358 | 6.96 | 13.62 | 51.62 |
| profile-background01-stereo | pnp | 179 | 0.91 | 4.91 | 7.80 |
| profile-background01-stereo | tracking | 179 | 19.07 | 85.33 | 247.36 |
| profile-background01-stereo | bundle_adjustment | 29 | 21.90 | 813.30 | 1147.69 |
| profile-background01-stereo | loop_retrieval_index | 2 | 11.12 | 5558.99 | 10538.86 |
| profile-background01-stereo | loop_matching | 5 | 0.06 | 11.86 | 13.12 |
| profile-background01-stereo | loop_verification | 4 | 0.36 | 89.72 | 98.11 |
| profile-background01-stereo | background_loops | 2 | 11.58 | 5791.29 | 10672.78 |
| profile-background01-stereo | finalization | 1 | 0.00 | 0.16 | 0.16 |
| profile-background01-stereo | export | 1 | 2.37 | 2366.28 | 2366.28 |
| profile-recovery-tum | input_loading | 100 | 1.48 | 14.48 | 18.31 |
| profile-recovery-tum | features | 100 | 10.99 | 107.30 | 142.64 |
| profile-recovery-tum | extraction | 100 | 11.10 | 108.10 | 144.32 |
| profile-recovery-tum | frame | 100 | 51.05 | 229.26 | 1476.00 |
| profile-recovery-tum | matching | 448 | 19.26 | 39.06 | 96.83 |
| profile-recovery-tum | optical_flow | 190 | 0.78 | 3.89 | 7.42 |
| profile-recovery-tum | pnp | 105 | 0.54 | 3.98 | 13.63 |
| profile-recovery-tum | tracking | 130 | 5.85 | 38.56 | 117.82 |
| profile-recovery-tum | mapping | 29 | 8.53 | 213.33 | 635.68 |
| profile-recovery-tum | bundle_adjustment | 29 | 14.20 | 385.06 | 907.24 |
| profile-recovery-tum | retrieval_index | 4 | 3.65 | 5.29 | 3091.19 |
| profile-recovery-tum | recovery | 4 | 11.16 | 2316.51 | 4054.30 |
| profile-recovery-tum | finalization | 1 | 0.00 | 0.02 | 0.02 |
| profile-recovery-tum | export | 1 | 1.19 | 1188.48 | 1188.48 |

Profiling runs are excluded from the timing tables. Stages are nested and background work overlaps, so their totals must not be summed. The TUM recovery profile preceded the final fix that reuses shortlist scores during exhaustive fallback.

Real KITTI 01 1500×1500 descriptor pair: CPU 141.64 ms versus CUDA 8.63 ms (16.42×), exact match-pair agreement True. This warmed component check includes transfers and synchronization, excludes extraction/runtime initialization, and is not pipeline speedup.

Early full KITTI04 ablations: histogram indexing alone took 302.5 s (0.92× frozen); adding CPU allocation optimizations took 227.4 s (1.22×). The final index uses SciPy instead of importing scikit-learn, avoiding roughly 40 MiB of unnecessary runtime overhead. These early runs used earlier index revisions and uncontrolled host load; they do not isolate an additive contribution. The initial monocular/indoor regression led to the separate fallback-score reuse fix; both earlier and final measurements remain above.

## Acceptance and review

Quality gate requires identical frame counts, no additional lost frame indices or lost intervals, no fewer verified loops, no additional initialization failures, and ATE ≤ frozen×1.05+0.05 m. The pilot does not prove improved accuracy. CPU matching remains default; GPU requires opt-in.

GPU RAM increases exceed the 10% investigation threshold. Setup peak working set and Torch allocated/reserved VRAM are retained in JSON: the CUDA runtime alone raises setup RAM to roughly 581 MiB on this host, before map growth. Total driver/context VRAM was unavailable to this collector. The memory cost remains a tradeoff; no memory gate is waived silently.

Backend validation: 93 tests passed in the isolated CUDA environment, covering index startup/update/removal, temporal eligibility, exhaustive fallback and score reuse, correction cache refresh, bundle equivalence, matching ties/ratio/cancellation, and unavailable CUDA. Graph solver, accuracy configuration, feature counts, budgets, schedules and the 300-keyframe guard were preserved. Optimization commits are separate from accuracy work.

Review only: leave shared-SLAM and main unchanged. Repeat promising cases without competing workloads and validate held-out live loop closure before merging. The original frozen benchmark checkout, environment, datasets/caches and results were read only; experiment outputs and CUDA installation are isolated in this worktree.

CPU performance gains are not uniform. Final CPU regressions: 01-mono 0.96× frozen throughput. These cases preserve accuracy but do not meet the 20% improvement target.

The persistent cache retains float32 SIFT descriptors (512 bytes per landmark), float64 positions (24 bytes) and IDs (8 bytes). At 264,925 landmarks this is about 137 MiB of array payload. CPU setup was about 110 MiB in the final monocular case versus 583 MiB for the full CUDA stereo run. Runtime initialization plus the retained cache broadly explains the 562 MiB full-run process RAM increase; these working-set peaks are not additive allocation accounting. The cache estimate excludes Python containers.
