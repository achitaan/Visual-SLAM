# Visual SLAM

Stereo and monocular visual odometry with a live dashboard and offline pose graph optimization.

The project aims to build reliable visual SLAM across environments. KITTI is used to diagnose accuracy and tracking failures; benchmark scores are not the sole development objective.

## Pipeline status

The original Python shared mapper is restored on the development branch, with
verified-reference recovery, validated descriptor associations, physical
landmark ownership and precise sparse bundle adjustment. It uses this project's
own tracking and mapping code with OpenCV and SciPy. The current branch passes
384 Python tests, including launcher smoke checks, plus dashboard tests and a
production dashboard build. The earlier four-image real KITTI application check
preserves exact frozen-reference poses, sparse map and configuration.

- [Original Python benchmark table and verified run command](docs/benchmark/original-python-verified.md)
- [Exact configuration, measured values and source hashes](docs/benchmark/original-python-verified.json)

Those benchmark values are prior completed measurements, not new integration
results. Seven full KITTI seed0 comparisons are recorded. Full01 raw error and
runtime remain weaknesses; broader coverage is incomplete. Real-time execution
is not a requirement for this implementation.

### Owned observation value optimization

The original Python mapper now uses immutable internal observation values to
avoid repeated physical-identity work. Six short seed comparisons and six full
seed0 comparisons preserve the control trajectories, maps, states and accuracy,
with observed full whole-case wall reductions of **3.53–26.32%**. Full09 timing
spans resource stops; the report preserves that comparability caveat. Candidate
runtime remains above VO and full01 raw trajectory error remains worse than VO.

- [Measured benchmarks, exact scope and reproduction](docs/benchmark/owned-observation-value.md)
- [Machine-readable measurements and source identity](docs/benchmark/owned-observation-value.json)
- [Installation/review scope and release notes](docs/benchmark/owned-observation-value-release-notes.md)
- [Branch verification and launcher checks](docs/benchmark/owned-observation-value-branch-checks.md)

### Legacy VO and offline graph results

The stereo pipeline uses SIFT features, SGBM disparity and PnP motion estimation. Offline loop verification uses stereo geometry, followed by a SciPy pose graph optimizer and correction propagation to the full trajectory. Ground truth is used only for evaluation and display, never to initialize motion, recover scale or construct graph constraints.

Full runs on KITTI 00–10 cover 23,201 frames and 61 verified loops. Segment-weighted translation drift is **2.287% before optimization and 1.773% afterward**. These are public training-set results, not hidden test-set leaderboard results.

- [Benchmark report and graphs](docs/benchmark/REPORT.md)
- [Machine-readable results](docs/benchmark/results.json)
- [Development roadmap](docs/ROADMAP.md)

The legacy VO mode lacks persistent mapping and optimized-pose feedback. The
shared pipeline implements persistent landmarks, local bundle
adjustment, relocalization and live correction, but wider reliability validation
remains incomplete. Monocular translation has arbitrary scale. In the original
benchmark, sequence 01 has 23 tracking failures and sequence 09's translation
drift worsens after graph correction.

## Setup

Use Python 3.12 and Node.js 22. Run backend commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python src/main.py --max-frames 51
```

On Linux or macOS, use `.venv/bin/python`. The default input is a bundled monocular sample; it is a smoke test without ground-truth accuracy measurements.

### Native kernels on Windows

Build the optional native DLLs with a recent x64 MinGW-w64 compiler, such as
GCC from the MSYS2 UCRT64 environment:

```powershell
.\.venv\Scripts\python scripts/build_native.py --cxx C:/msys64/ucrt64/bin/g++.exe
```

The build writes DLLs into `src/native_kernels`. Windows checks the required
AVX-512 features at build and load time. On an AVX2-only host, SIFT, matching,
LK, and ZNCC use their existing fallbacks. The bundle and pose-graph kernels
can still run natively on x64.

For the throughput-tuned Windows stereo harness, use native-resolution SIFT,
one pending scheduled keyframe prefetch, and CUDA descriptor matching:

```powershell
$env:VISUAL_SLAM_SIFT_FIRST_OCTAVE = '0'
.\.venv\Scripts\python src/loop_harness.py --tier custom --sequences 00 --data-root .datasets/kitti --output-dir results/win_native_cuda_00 --openblas-threads 1 --opencv-threads 2 --pipeline-extraction --pipeline-buffer 4 --coalesce-prefetch --matching-backend cuda --no-profile
Remove-Item Env:VISUAL_SLAM_SIFT_FIRST_OCTAVE
```

Starting at octave 0 changes native SIFT detections. Compare the harness's
trajectory metrics when applying this configuration to other sequences.
CUDA matching can rank both descriptor directions with one kernel on Windows.
The `cuda_fused_rank_calls` counter in `run.json` records calls that used it.
Small loop-closure batches can use the native CPU matcher to avoid GPU
synchronization while tracking uses CUDA. The `cpu_native_loop_calls` counter
records matches routed this way.

For sparse depth on Windows, add `--stereo-depth-policy verified_all
--stereo-verification-backend cuda` to that command. This policy computes
stereo depth at feature coordinates. The CUDA verifier uses NVRTC from
CUDA PyTorch for standard 11 × 11 grayscale batches of at least 256
points. Smaller batches, other search settings, and unavailable CUDA
use the CPU verifier. Check `stereo_verification_backend` in `run.json`
for GPU batch counts and any fallback reason.

Add `--optical-flow-backend cuda` to use fused forward-backward Lucas–Kanade
for stereo tracking with `--pipeline-extraction`. The CUDA backend uses
NVRTC from the same PyTorch installation. Unavailable CUDA falls back to
the existing CPU path. `optical_flow_backend` in `run.json` records fused
pair counts and any fallback reason.

Add `--sift-pyramid-backend cuda` to build the native SIFT scale space on the
GPU. Rebuild `fast_sift` with the command above and use the CUDA-enabled
PyTorch installation on Windows. The AVX-512 kernel reads the fixed-point
pyramid and extracts the same keypoints and descriptors as the CPU path at
the selected octave. If the DLL or CUDA is unavailable, extraction uses the
usual CPU path. Check `sift_pyramid_backend` in `run.json` for the number of
GPU extractions and any fallback reason.

For the 900-feature KITTI 00 benchmark, use the native CPU SIFT pyramid with
CUDA matching, sparse stereo verification, and optical flow.

```powershell
$env:VISUAL_SLAM_SIFT_FIRST_OCTAVE = '0'
.\.venv\Scripts\python src/loop_harness.py --tier custom --sequences 00 --data-root .datasets/kitti --output-dir results/win_native_cuda_00_900 --openblas-threads 1 --opencv-threads 2 --pipeline-extraction --pipeline-buffer 4 --coalesce-prefetch --no-profile --matching-backend cuda --stereo-depth-policy verified_all --stereo-verification-backend cuda --optical-flow-backend cuda --sift-pyramid-backend cpu --features 900
Remove-Item Env:VISUAL_SLAM_SIFT_FIRST_OCTAVE
```

On a Core i5-11400F with a GTX 1660 SUPER, the paired runs processed all 4,541
frames with live loop closure and zero lost frames.

| Feature budget | FPS | SE(3) ATE | Translation drift |
| ---: | ---: | ---: | ---: |
| 1,500 | 41.06 | 2.42 m | 1.028% |
| 900 | 44.52 | 2.43 m | 1.024% |

To use a smaller covisible bundle adjustment window, append
`--config-overrides '{"covisible_bundle_max_keyframes":5}'` to the 900-feature
command. In a matched full KITTI 00 trial, five keyframes reached 47.31 FPS
with 2.50 m ATE and 1.042% translation drift. Eight reached 44.54 FPS with
2.43 m ATE and 1.024% drift. A regular harness rerun with five keyframes reached
47.67 FPS, 2.33 m ATE, and 0.986% drift. All three runs lost zero frames.

With native routing for small loop matches, a five-keyframe window reached
48.38 FPS on KITTI 00. Two full runs with a three-keyframe window reached
49.91–50.10 FPS with 2.23–2.34 m ATE, 0.970–1.001% translation drift, and zero
lost frames. To reproduce that configuration, append
`--config-overrides '{"covisible_bundle_max_keyframes":3}'`
to the 900-feature command.

For faster SIFT detection on KITTI 00, use `--sift-pyramid-backend cuda` and
`--config-overrides '{"covisible_bundle_max_keyframes":3,"sift_detection_scale":0.78}'`
with the 900-feature command. The scale applies to the grid-bucketed SIFT
detector. The frontend maps detections back to the original pixel grid before
stereo verification and pose estimation. It retries detection on the original
image when the reduced image has too few features. Compare trajectory metrics
when applying this setting to other sequences.

Two full KITTI 00 runs at this scale reached 55.90–55.97 FPS with 2.25–2.26 m
ATE, 0.967–0.983% translation drift, and zero lost frames. Over the first 1,200
frames, ATE was 1.45 m at this scale and 1.23 m at full resolution.

To use the native CPU refiner for seeded PnP, keep the same CUDA backends and pass
`--config-overrides '{"covisible_bundle_max_keyframes":3,"sift_detection_scale":0.78,"native_seeded_pnp":true}'`.
The `fast_ba` kernel refines eligible pose estimates using image-coordinate residuals.
OpenCV handles the refinement if the kernel is unavailable or declines a fit.
In a paired full KITTI 00 benchmark, the option raised throughput from 55.23 to
56.52 FPS. ATE went from 2.34 m to 2.23 m, translation drift went from 0.995% to
0.971%, and both runs lost zero frames.

For compact SIFT on Windows, rebuild `fast_sift` and replace
`--sift-pyramid-backend cuda` with `--sift-pyramid-backend cuda-compact` in the
900-feature command with seeded PnP. This backend finds extrema and builds
descriptors on CUDA. The CPU selects grid-bucketed keypoints, and the pipelined
frontend overlaps descriptor readback with stereo verification. Use the same
AVX-512 CPU and CUDA-enabled PyTorch requirements as for `cuda`. Descriptor
rounding can differ between CUDA and the native CPU extractor.

Check `sift_pyramid_backend.compact_frames` in `run.json` to verify that the
compact path ran. Frames that exceed the compact extrema buffer use the
regular CUDA pyramid. If the compact backend is unavailable or fails, feature
extraction falls back to the CPU.

In a full KITTI 00 comparison on the GTX 1660 SUPER, compact SIFT reached
57.31 FPS versus 54.34 FPS with the regular CUDA pyramid. Both runs processed
all 4,541 frames with zero lost frames. ATE was 2.29 m with compact SIFT and
2.34 m with the regular pyramid.

Start the dashboard in a second terminal:

```sh
cd dashboard
npm ci
npm run dev
```

Open http://localhost:3000. Telemetry uses `ws://localhost:8765`; set `NEXT_PUBLIC_WS_URL` to override it. The dashboard provides live camera/feature views, trajectories, tracking statistics, saved benchmarks and events. Pause stops telemetry streaming while processing continues. Clear view resets displayed history, not the estimator.

## KITTI input

Provide rectified stereo images and calibration in this layout:

```text
<KITTI_ROOT>/
  sequences/00/
    image_0/*.png
    image_1/*.png
    calib.txt
    times.txt
  poses/00.txt
```

Replace `./data/kitti` with your dataset directory:

```powershell
$data = './data/kitti'
.\.venv\Scripts\python scripts/check_kitti_data.py --data-root $data --sequences 00 04
.\.venv\Scripts\python src/main.py --data-root $data --sequence 00 --stereo --output results/00.txt
```

Optional flags: `--slam` enables experimental keyframes/local mapping, `--poses-root` supplies ground truth for display, `--no-telemetry` disables streaming, `--max-frames` limits processing, and `--plot` opens trajectory plots. Invalid input paths fail explicitly. OpenCV uses one worker by default; `--opencv-threads` changes this limit.

## Benchmark and tests

```powershell
.\.venv\Scripts\python src/eval_kitti.py --data-root $data --poses-root "$data/poses" --sequences 00 01 02 03 04 05 06 07 08 09 10 --stereo --output-root results/stereo
.\.venv\Scripts\python -m pytest -q
```

```sh
cd dashboard
npm test
npm run build
```

Run the backend suite with `python -m pytest -q`. The dashboard checks include socket tests, type checking and a production build. GitHub Actions runs backend tests on Windows/Linux and dashboard checks on Linux; inspect the exact commit's workflow for remote results.

The evaluator writes KITTI-format poses and JSON metrics. ATE uses explicitly labeled SE(3) alignment without scale fitting. Translation and rotation drift use the official 100–800 m segments in metric coordinates. Partial runs are labeled; insufficient segment coverage produces null drift. `--estimates-root` evaluates saved trajectories without rerunning tracking.

### Offline pose graph

```powershell
.\.venv\Scripts\python src/eval_pose_graph.py --data-root $data --poses-root "$data/poses" --estimates-root results/stereo --sequence 00 --output-root results/pose-graph
```

Visual-word retrieval proposes temporally separated candidates. Mutual matching, bidirectional stereo PnP, depth, reprojection and coverage checks verify loops. The first graph pose is fixed. For camera-to-world poses, an edge measurement is `Z_ij = inverse(T_wi) @ T_wj`.

Corrected exports and the orange dashboard curve are **after optimization**, with keyframe corrections propagated to intermediate frames. Zero verified loops leave the raw trajectory unchanged. Information weights are fixed experimental parameters, not calibrated covariances. Graphs up to 300 poses use dense SVD when the Jacobian fits the memory bound; larger graphs use a sparse solver. The latter branch is not covered by the published full-sequence benchmark.

### Batch processing with limited disk space

Copy public KITTI pose files into `results/benchmark-batch/reference/poses` before running:

```powershell
.\scripts\run_kitti_batch.ps1
.\scripts\run_kitti_batch.ps1 -Sequences @('07') -ForceRetest
.\.venv\Scripts\python scripts/summarize_kitti_batch.py
.\.venv\Scripts\python scripts/plot_kitti_batch.py
.\.venv\Scripts\python scripts/verify_kitti_ground_truth.py
.\.venv\Scripts\python scripts/verify_kitti_batch.py
.\.venv\Scripts\python scripts/run_kitti_stream.py --sequence 07 --replay-graph
```

The PowerShell controller downloads images from the official KITTI archive using HTTP ranges and ZIP CRC checks. It caches one sequence or streams through bounded caches, then deletes only its owned temporary image/geometry files. Trajectories, measured constraints and reports remain in `results/benchmark-batch`. `-ForceRetest` repeats image-based tracking and loop verification; `--replay-graph` repeats optimization from saved measurements. Dataset files and runtime outputs are excluded from Git.

Official evaluator parity is checked with `scripts/check_kitti_devkit.py`, which requires a locally compiled wrapper around the official `calcSequenceErrors`. The benchmark report records the tolerance and validation evidence; the executable is not bundled.

Refresh the dashboard's saved results after a new batch:

```powershell
.\.venv\Scripts\python scripts/update_dashboard_benchmarks.py --data-root results/benchmark-batch/reference
```

## Configuration and telemetry

`src/config.py` contains feature flags, keyframe thresholds, vocabulary parameters, local-map limits and telemetry defaults. Live appearance candidates are not treated as verified loop closures or successful relocalization.

Telemetry schema v1 publishes `pose_T_wc`, frame index, timestamp, mode, matches/inliers, keyframe/map-point counts and optional camera image, features, events, FPS and run metadata.

Original mathematical notes are available in [VisualOdometry.tex](VisualOdometry.tex). They describe the early project background rather than the current stereo implementation.

## Local NVIDIA GPU frontend (experimental)

Use a Windows or Linux Python environment with a CUDA-capable NVIDIA GPU and a
compatible driver. Install the normal project dependencies first, then install
the CUDA extras in the same environment:

```sh
python -m pip install -r requirement.txt
python -m pip install -r requirements-performance-gpu.txt
python -m pip install -r requirements-local-gpu.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The pinned example uses PyTorch 2.7.1 with CUDA 12.6. If that wheel does not
support your Python version or GPU, use the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
to choose a compatible CUDA build, then install the Kornia extras. Allow several
GB of free disk space for the download and installation. The
verification command should end in `True`. CuPy, CUDA-enabled OpenCV, and a
separate CUDA toolkit are not needed with the prebuilt PyTorch wheel. The CPU
path needs neither PyTorch nor Kornia.

```sh
# Synthetic rectified stereo at KITTI 00 resolution (1241 x 376), no download
python scripts/bench_local_gpu.py

# Run both backends on identical decoded KITTI pairs and emit JSON
python scripts/bench_local_gpu.py --backend both --kitti-root /path/to/KITTI/sequences/00 --json

# Opt into the local CUDA frontend in SharedSlam
python src/main.py --slam --stereo --data-root /path/to/KITTI --sequence 00 --frontend-backend cuda --matching-backend auto

# Benchmark CUDA full extraction with the lazy SLAM pipeline
python src/loop_harness.py --tier custom --sequences 00 --data-root /path/to/KITTI --frontend-backend cuda --pipeline-extraction --pipeline-buffer 4 --no-profile

# Run frontend checks (CUDA cases skip if PyTorch cannot access a CUDA device)
python -m pytest -q tests/test_local_gpu_frontend.py
```

The benchmark detects the CUDA device name, compute capability, and VRAM.
Its default `auto` mode uses the OpenCV CPU path if CUDA or Kornia cannot be
initialized; `--backend cpu` skips GPU detection, and explicit `cuda` or
`both` modes fail if CUDA is unavailable. The timings include the gradient
pyramid, SIFT extraction, matching against the previous left frame, disparity,
and CPU/GPU transfers for the feature, match, and disparity outputs. They exclude
file decoding and warm-up, synchronize CUDA before recording each sample, and
report mean, median, and p95 latency and effective sequential FPS. They do not
measure SLAM trajectory accuracy or full-pipeline FPS.

When running SLAM, `--frontend-backend cuda` falls back to the unchanged CPU
frontend with a message if CUDA cannot initialize. Once active, it uses Kornia
scale-space SIFT and rectified SAD block disparity in place of OpenCV SIFT and
StereoSGBM. The CUDA disparity uses a uniqueness check but not SGBM's
smoothness model. Features and disparities can differ from the CPU versions,
so trajectories can differ too. `--matching-backend auto` uses the
existing CUDA descriptor matcher only when it is both faster and agrees with
the CPU matcher on its initial comparison. The separate `verified_all` stereo
policy still uses its original CPU correspondence verifier instead of a dense
disparity map. Tracking, optimization, and other back-end work stay on CPU.

## Shared mapping pipeline (experimental)

Performance integration and bounded stereo diagnostics are documented in
[the integration report](docs/SLAM_INTEGRATION.md), including actual graphs,
runtime regressions and remaining release blockers. CPU matching and current
retrieval remain the defaults; optional CUDA matching and indexed retrieval need
separate validation. The original benchmark above is retained historical evidence.

The source-specific experiments below are historical development records. Their
paused or failed gates apply to those snapshots; the current original Python
results and installed optimization are documented above. Broader reliability
validation remains incomplete. Earlier failures and coverage remain documented
in [the sensor comparison](docs/benchmark/SENSOR_COMPARISON.md).

The [stereo reliability workflow](docs/STEREO_REWORK.md) provides bounded quick,
focused and release profiles, controlled ablations and exact-fingerprint resume
checks. Full validation and a release review remain pending.

[Historical reserved stereo validation](docs/benchmark/reserved-stereo-validation.md) includes
fresh VO comparisons, trajectories and sparse maps for 04/80 and 01/350. The
then-current diagnostic removes the lost frame on 01 but fails the rotation regression
gate; full-sequence testing remains paused.

[Same-source bundle adjustment ablation](docs/benchmark/same-source-bundle-ablation.md) isolates the position and rotation trade-off without changing inputs or acceptance thresholds.

[Historical physical landmark identity validation](docs/benchmark/physical-identity-validation.md) reports that snapshot's correctness repair, improved 04 accuracy, and retained 01 accuracy and runtime regressions.

[Raw stereo reference retry](docs/benchmark/raw-stereo-reference-retry.md) describes
an opt-in experiment for configured-reference failures. It restores two KITTI 01
motions and improves prefix position accuracy, while the rotation regression
still fails the release gate. It retains bidirectional geometric verification.

Use `--slam` to select persistent landmark tracking and local bundle adjustment;
add `--stereo` for calibrated stereo input. Commands without `--slam` retain the
original VO implementations for comparison. Monocular initialization uses two-view
geometry and tracks in arbitrary map units. It can defer initialization or lose
tracking; held poses are explicitly flagged and cannot add geometry.

```powershell
.\.venv\Scripts\python src/main.py --slam --data-root $data --sequence 04 --output results/mono04.txt
.\.venv\Scripts\python src/main.py --slam --stereo --data-root $data --sequence 04 --output results/stereo04.txt
.\.venv\Scripts\python scripts/evaluate_shared_slam.py --data-root $data --poses-root "$data/poses" --sequence 04 --output results/slam04
```

Saved runs contain camera-to-world trajectories, `sparse.ply`, keyframe images,
`run.json` and a sampled `preview.json`. The dashboard Reconstruction tab loads
previews locally and supports orbit, zoom, sparse/dense toggles and preview exports.
Monocular accuracy uses Sim(3) alignment in evaluation only; it does not establish
metric scale for the estimator.

Optional dense reconstruction uses Depth Anything V2 Metric Small after tracking.
Install `requirements-depth.txt` separately and obtain the official metric model
source and checkpoint from the [model documentation](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md).
Select the Hypersim checkpoint for indoor input or Virtual KITTI checkpoint for
outdoor input. Predicted meters remain model estimates. Sparse landmark depths
provide the conversion to map units; ground truth does not participate.

```powershell
.\.venv\Scripts\python scripts/reconstruct_run.py --run results/slam04 --model-source $modelSource --checkpoint $checkpoint --environment outdoor --output results/dense04
```

Dense exports contain `dense.ply`, selected depth predictions, a bounded preview
and a provenance manifest. Fusion starts from saved final poses and rebuilds from
scratch; process each sequence separately. GPU use depends on the installed PyTorch
runtime, with CPU available through `--device cpu`. This is offline processing.
Live loop graphs above 300 keyframes are disabled pending validation.

See the [tracking diagnosis](docs/benchmark/TRACKING_DIAGNOSIS.md) for retained
failure evidence, stereo correspondence checks and reproducible optimizer ablations.

Shared-pipeline evaluation reserves free disk space for map exports. A low-space
interruption saves processed frames with partial coverage; it is not a completed
full-sequence benchmark. The shared batch runner stops at that interruption.

To monitor a shared benchmark batch on the dashboard, start
`python scripts/benchmark_dashboard.py` alongside the frontend. The bridge serves
`ws://localhost:8765` and reads the batch selected by
`results/shared-kitti-comparison/current.json`. It displays checkpoint progress,
the queue and completed results separately from pose telemetry.

For live images and maps, create `results/benchmark-dashboard.json` with
`{"enabled": true, "batch_root": "results/shared-kitti-comparison/<revision>/full"}`
before the evaluator starts. Subsequent evaluators in that batch publish bounded
snapshots of actual images, features, landmarks and the corrected trajectory.
An already-running evaluator continues with checkpoint progress until the next
run. Snapshot failures disable the observer without interrupting tracking.
Reports record observer overhead; reported runtime includes it. Ground truth is
read only after estimator shutdown and never enters the snapshot observer.

The shared tracking and mapping code is implemented in Python using OpenCV for
features, stereo disparity and PnP, and SciPy for bundle adjustment and graph
optimization. It uses standard SLAM methods such as persistent keyframes and
landmarks, geometric loop verification and SE(3)/Sim(3) correction. These methods
also appear in systems such as [ORB-SLAM](https://arxiv.org/abs/1502.00956);
the shared pipeline does not embed or depend on the ORB-SLAM implementation,
DBoW2, g2o or Pangolin. Depth Anything V2 is a separately obtained third-party
model for optional offline reconstruction, with checkpoint provenance recorded
in the reconstruction manifest.
