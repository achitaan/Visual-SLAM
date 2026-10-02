# Visual SLAM

Stereo and monocular visual odometry with a live dashboard and offline pose graph optimization.

The project aims to build reliable visual SLAM across environments. KITTI is used to diagnose accuracy and tracking failures; benchmark scores are not the sole development objective.

## Status

The stereo pipeline uses SIFT features, SGBM disparity and PnP motion estimation. Offline loop verification uses stereo geometry, followed by a SciPy pose graph optimizer and correction propagation to the full trajectory. Ground truth is used only for evaluation and display, never to initialize motion, recover scale or construct graph constraints.

Full runs on KITTI 00–10 cover 23,201 frames and 61 verified loops. Segment-weighted translation drift is **2.287% before optimization and 1.773% afterward**. These are public training-set results, not hidden test-set leaderboard results.

- [Benchmark report and graphs](docs/benchmark/REPORT.md)
- [Machine-readable results](docs/benchmark/results.json)
- [Development roadmap](docs/ROADMAP.md)

Live SLAM is incomplete: persistent landmarks, local bundle adjustment, relocalization and optimized-pose feedback into tracking and mapping remain development priorities. Monocular translation has arbitrary scale. Sequence 01 has 23 repeatable tracking failures; sequence 09's translation drift worsens after graph correction.

## Setup

Use Python 3.12 and Node.js 22. Run backend commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python src/main.py --max-frames 51
```

On Linux or macOS, use `.venv/bin/python`. The default input is a bundled monocular sample; it is a smoke test without ground-truth accuracy measurements.

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

Local backend validation passes 136 tests. The dashboard passed its 4 socket tests, type checking and production build. GitHub Actions runs backend tests on Windows/Linux and dashboard tests/build on Linux; inspect the workflow for remote results.

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

## Shared mapping pipeline (experimental)

This development branch is an experimental snapshot, not a validated replacement for the original pipeline. Full paired validation is paused for stereo reliability rework; retained failures and coverage are documented in [the sensor comparison](docs/benchmark/SENSOR_COMPARISON.md).

The [stereo reliability workflow](docs/STEREO_REWORK.md) provides bounded quick,
focused and release profiles, controlled ablations and exact-fingerprint resume
checks. Full validation and a release review remain pending.

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
