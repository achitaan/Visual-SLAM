# Visual SLAM completion plan

The target is a reproducible full KITTI benchmark and a SLAM pipeline with measured accuracy. The initial audit found working odometry components, but the original pose graph and dashboard could produce misleading results. The current changes establish a tested foundation. Full SLAM accuracy is still unfinished.

Audit date: September 30, 2026; updated October 1, 2026. Baseline commit: `ddf941b`. Development branch: `codex/finish-and-test`. Changes are local and have not been pushed.

## Dataset and environment

The full dataset catalog was found at `C:\Users\Achita\OneDrive\Documents\Programming\Visual Odometry`. Sequences 00–21 are listed, and ground-truth poses 00–10 are present. All 11 ground-truth sequence catalogs have matching left/right filenames and matching pose counts. This validates the catalog; it does not prove every image is readable or decoded correctly. All sequence 04 images and the first 120 stereo frames of 00 are locally readable. Longer 00 reads and calibration/images for 03 and 07 still fail on cloud-only files, including outside the sandbox. The latest Windows error is “The cloud operation is invalid.” Requesting local availability for 07 calibration and its first stereo pair did not restore reads.

To continue the requested graph test, complete sequence 07 was downloaded directly from the official KITTI grayscale ZIP using verified HTTP ranges and ZIP CRC checks. Its independent local copy worked. The temporary image copy was subsequently removed to recover disk space; download provenance and pose files remain. The full batch uses `.datasets/batch-scratch/NN` for a single sequence's images and keyframe geometry, with bounded streaming when a full sequence does not fit. The PowerShell controller checks resolved paths and cache ownership before deleting those temporary files. Raw/corrected trajectories, measured loop transforms and reports remain in `results/benchmark-batch`. This workaround does not demonstrate that OneDrive hydration has recovered.

Only two 51-frame monocular samples are tracked in the GitHub repository. They are useful for smoke tests, but contain neither stereo right images nor ground-truth poses.

Python 3.12, a project virtual environment, and Node 22 are working. Python dependencies are captured in `requirements-lock.txt`; dashboard dependencies are captured in `package-lock.json`. The dashboard was updated to Next 16.3.8 and npm audit reported zero vulnerabilities. Both disk and system memory have constrained testing. The runner defaults to one OpenCV worker; graph retrieval caps retained features at 1,500 per keyframe and uses bounded numerical-library threads. Download checks reserve disk space before extraction.

## Why the previous result could be far off

1. **Graph vertices and keyframes disagreed.** Graph poses were appended only after BoW vocabulary startup, while edges already used all keyframe IDs. Edges could reference nonexistent or unrelated vertices.
2. **Loop constraints used the wrong source and direction.** A transform derived from the estimated trajectory was inserted as a loop measurement, in the opposite edge direction. Such a measurement cannot independently correct drift.
3. **Optimization did not update the estimator.** Optimized poses were sent to telemetry but did not correct the tracked trajectory or map points. Pose export therefore remained raw odometry.
4. **Monocular scale was not observable.** Comparing distances between points related by a rigid transform within one image pair cannot recover motion scale. Unit translation per frame is not meters, especially at variable vehicle speed.
5. **The dashboard changed the apparent result.** It automatically applied a changing 2D similarity alignment, while its error panel used raw coordinates. Rotation direction, sparse graph/frame correspondence, and unequal screen axis scaling also made the visual comparison unreliable.
6. **Stereo and mapping had correctness gaps.** ORB binary descriptors used a Euclidean KD-tree, invalid disparities could contribute to PnP, the Q matrix principal-point offset had the wrong sign, and map point filtering could lose descriptor correspondence.

For camera-to-world poses, the graph measurement for an edge from i to j must be `Z_ij = inverse(T_wi) @ T_wj`. A loop measurement must come from independent image/map geometry. Monocular loops additionally need scale handling, usually a Sim(3) formulation.

## Completed foundation

- Portable dataset paths, explicit ground-truth paths, strict frame pairing, bounded image caching, and a maximum frame count that limits loading.
- Ground truth is never used to initialize the estimator or set its translation scale.
- Monocular pose recovery uses essential-matrix RANSAC and cheirality inliers. Low-feature and stationary inputs hold the last pose with a tracking failure flag.
- Stereo uses Hamming distance for ORB, validates disparity and 3D points, and requires sufficient PnP inliers. SIFT is the common default for the runner and dashboard backend.
- KITTI trajectory I/O uses one row of 12 values per frame, with finite rigid-pose validation.
- Map points retain the correct descriptor indices after triangulation filtering; local point storage is bounded.
- A portable SciPy graph optimizer fixes the first pose and validates vertices and measurements. Tests confirm that an independent loop constraint reduces known synthetic drift and that rotated origins obey the edge convention.
- Real graph tests exposed optimizer convergence failures. Vectorized rigid residuals and fixed rotation/translation variable scales help, but the real seven-loop 06 graph still stalled after 5,000 approximate sparse steps. A dense SVD solve of the same objective converged in 33 evaluations. Graphs up to 300 poses use this solver when their Jacobian fits a 64 MiB bound; larger graphs retain the sparse solver. The retained real 06 fixture, mixed-scale synthetic graph and rotated-origin checks cover these failure modes. A failed solve raises an error instead of exporting an apparently successful correction.
- The 228-vertex, 30-loop 00 graph exposed expensive dense numerical differentiation. Independent vertices now share finite-difference evaluations while the linear step still uses SVD. A coupled rotation/translation regression checks the grouped Jacobian against independent column differences. Replays retain objective, optimality, iteration/evaluation counts and timing without changing graph weights or measurements.
- A separate batch graph evaluator retrieves temporally separated visual candidates, requires mutual descriptor matches, metric bidirectional PnP, reprojection/depth/coverage checks, and adds independently measured loop edges. It propagates optimized anchor corrections to every intermediate frame and exports corrected KITTI poses. Ground truth is loaded only for evaluation, after optimization.
- Graph vertices now include keyframes before vocabulary startup. Appearance retrieval reports candidates without inserting fabricated loop edges. Appearance alone no longer reports a successful relocalization.
- Telemetry reports startup failures, handles malformed controls, bounds its queue, supports late clients, acknowledges pause/resume, and shuts down cleanly.
- The dashboard tracks actual socket connection state, reconnects, bounds history, displays raw trajectories with equal axis scale, and distinguishes metric from arbitrary translation scale. Metric error is hidden for monocular output. Clear view resets the viewer, not the backend; Pause stops telemetry streaming, not odometry computation.
- The dashboard now has a responsive professional layout with Live, Benchmarks and Events views, saved accuracy coverage, raw/corrected graph comparisons, feature toggles and final-frame run completion. Pause preserves the last camera image. Ground-truth comparisons remain explicitly labeled.
- Automated Python and socket tests and GitHub Actions configuration are added. The workflow is configured but has not run remotely.

## Validation results

| Run | Coverage | Result | Limit |
| --- | --- | --- | --- |
| Python regressions | 45 tests | Passed | Includes the real seven-loop graph, grouped Jacobian numerical reference, streaming/CRC handling, cache reuse status, geometry, I/O and telemetry |
| Dashboard socket tests | 3 tests | Passed | Mock socket lifecycle, messages and reconnect behavior |
| Dashboard production build | Next 16.3.8 | Passed with the complete benchmark snapshot | Includes TypeScript checking |
| Redesigned live dashboard browser check | KITTI stereo 04, 40 frames | Camera, Pause/Resume, feature toggles, map controls, narrow-screen layout, and final-frame completion verified | New saved graph panel compiles; subsequent browser reopening was rejected by URL policy |
| Bundled sample with SLAM enabled | All 51 frames | Finite KITTI export and clean shutdown | No ground-truth accuracy claim |
| Full stereo and graph benchmark | All 00–10; 23,201 frames | 11/11 raw and corrected runs complete; 61 verified loops; 23 lost pairs, all on 01 | Offline graph correction; live map/tracking feedback remains unfinished |
| Official reference poses | All 11 sequences | Byte-for-byte identical to the official public pose archive | Reference data checked independently of the OneDrive catalog |
| Official KITTI devkit parity | Every raw and corrected drift segment | Segment keys match; per-segment differences below 0.0001 percentage points and 0.0001 degrees/m | IEEE float32 distance endpoints; MinGW reference compiled with `-O2 -msse2 -mfpmath=sse` |
| Raw tracking reproducibility | Full 01, 04 and 07 repeated | Every saved pose element identical; 01 repeats all 23 losses | Additional raw routes have not been fully retracked twice |
| Graph reproducibility | All 11 saved image-constraint replays | Passed declared tolerances; largest position difference 0.16 mm | Original in-memory poses versus serialized raw pose input; same measured transforms and weights |

Each arrow below compares raw stereo VO with offline graph correction. ATE uses SE(3) alignment without scale fitting. Drift is unscaled; plotted trajectories and position-error curves retain the original metric coordinates.

| Sequence | Frames | ATE m, raw → graph | Translation %, raw → graph | Rotation degrees/m, raw → graph | Verified loops | Lost pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 4541 | 20.610 → 1.935 | 2.147 → 1.079 | 0.00938 → 0.00556 | 30 | 0 |
| 01 | 1101 | 72.293 → 72.293 | 9.574 → 9.574 | 0.01337 → 0.01337 | 0 | 23 |
| 02 | 4661 | 34.113 → 8.181 | 2.081 → 1.458 | 0.00801 → 0.00600 | 9 | 0 |
| 03 | 801 | 2.327 → 2.327 | 1.705 → 1.705 | 0.00754 → 0.00754 | 0 | 0 |
| 04 | 271 | 1.763 → 1.763 | 1.683 → 1.683 | 0.01100 → 0.01100 | 0 | 0 |
| 05 | 2761 | 9.555 → 2.038 | 1.751 → 0.973 | 0.00858 → 0.00475 | 11 | 0 |
| 06 | 1101 | 2.815 → 2.152 | 1.622 → 1.313 | 0.00848 → 0.00462 | 7 | 0 |
| 07 | 1101 | 3.913 → 0.610 | 1.976 → 0.868 | 0.01347 → 0.00667 | 2 | 0 |
| 08 | 4071 | 10.120 → 10.120 | 1.861 → 1.861 | 0.00766 → 0.00766 | 0 | 0 |
| 09 | 1591 | 5.133 → 3.630 | 1.705 → 1.825 | 0.00705 → 0.00649 | 2 | 0 |
| 10 | 1201 | 2.564 → 2.564 | 1.368 → 1.368 | 0.00747 → 0.00747 | 0 | 0 |

Across 14,567 evaluated segments, segment-weighted translation drift is 2.2866% → 1.7735%, and rotation drift is 0.008630 → 0.006481 degrees/m. The loop-containing routes improve aligned ATE; 09's translation drift regresses despite its improved ATE. No corrected route was selected or rejected using ground truth.

Sequence 01 remains the strongest frontend failure: two independent full tracking runs reproduce the same 23 losses and 72.293 m ATE. The retained 271→272 pair produces 211 image matches, 34 usable stereo points, and only 12 PnP inliers against a minimum of 15. Investigate depth uncertainty, static-feature support, motion prediction and map-based recovery before lowering acceptance thresholds. Diagnostics and four example images remain in `results/benchmark-batch/failures/01`.

Weights are experimental fixed values, rather than calibrated covariances. Calibrating measurement uncertainty and testing hard negatives are priorities, particularly for 09. Reports, trajectories, measured constraints, evaluator parity, reproducibility checks and every full-sequence figure are retained under `results/benchmark-batch`; the dashboard snapshot uses these actual reports. Download and image I/O, concurrent workers, and recovered timing metadata limit CPU speed comparisons.

## Remaining work in order

### 1 Maintain the completed full stereo baseline

Full 00–10 coverage, accuracy, tracking counts, graph replays, official pose provenance and devkit parity are now measured. Keep this fixed SIFT/PnP configuration as the reference for subsequent changes. Retain both successful and failed attempts, source checksums and the segment-weighted summary. Full raw tracking has been independently repeated on 01, 04 and 07; repeat other raw routes when changes or unexplained differences require it.

Acceptance: all 11 sequences finish, each output has exactly the ground-truth frame count, every pose is finite and rigid, tracking failures are counted, and results reproduce within a declared tolerance. Cross-check the new drift evaluator against the official KITTI devkit on saved trajectories before treating it as authoritative.

### 2 Build persistent metric mapping and local optimization

Stereo currently provides metric odometry but does not insert stereo landmarks into the local map. Store landmarks with keyframe observations, descriptors, depth quality and reprojection error. Track against the local map with PnP; use a motion model for temporary failures. Add local bundle adjustment, keyframe culling, landmark deduplication and outlier rejection.

Acceptance: synthetic reprojections recover known poses; local adjustment reduces reprojection error; held-out KITTI sequences show no regression against the baseline; memory stays bounded.

### 3 Verify loop candidates geometrically

Integrate the tested batch verifier into live SLAM. Its temporal exclusion, multiple visual candidates, mutual matching, stereo PnP/RANSAC, bidirectional consistency, depth and coverage gates are now implemented for the offline experiment. Add persistence over candidate detections and hard-negative validation. Replace fixed experimental information weights with measurement uncertainty estimates. Preserve raw and corrected trajectories separately.

Acceptance: known loops are accepted, hard negative image pairs are rejected, and each graph loop edge has an independently measured transform. No loop is added from the current estimated poses alone.

### 4 Apply graph corrections to the trajectory and map

The batch experiment now exports corrected full-frame trajectories using rigid correction interpolation. Extend this to live SLAM: maintain stable keyframe IDs and map anchors, correct keyframe poses and associated landmarks, and update the active tracking reference after optimization. Update telemetry with explicit frame IDs and graph constraint counts. Benchmark the sparse solver branch above 300 keyframes and an asynchronous or incremental solve before putting large graph optimization in the live tracking loop.

Acceptance: a synthetic closed trajectory corrects drift in poses and landmarks, the first pose stays fixed, reprojection checks remain consistent, and real loop sequences improve relative to VO without corrupting non-loop sequences.

### 5 Add reliable relocalization and monocular scale handling

Relocalize only with verified 2D-to-3D geometry. Test occlusion, low texture, large viewpoint change and recovery. For monocular SLAM, initialize a persistent map, track across multiple views, and use scale-aware local mapping and Sim(3) loop correction. Report aligned monocular ATE separately from stereo metric drift.

Acceptance: false retrievals never teleport the camera; lost tracking is visible; recovery is geometrically consistent; monocular scale drift is measured rather than hidden by dashboard alignment.

### 6 Complete evaluation and delivery

Choose tuning sequences and reserve others for validation before changing thresholds. Compare VO, local mapping and full SLAM using the same frontend. After measuring the baseline, agree on explicit accuracy and tracking-loss targets. Add benchmark regression artifacts, end-of-sequence status, replay/restart controls, and setup instructions verified on a fresh Windows and Linux installation. Verify the CI runs on GitHub.

Definition of done: a clean checkout can run the documented demo; tests and CI pass; all 00–10 sequences have reproducible reports; validated loops improve measured accuracy; pose/map corrections reach both export and display; known limitations are documented. Live camera support is a later milestone, after calibration and dataset accuracy are established.

## Commands

From the repository root, in PowerShell:

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.mpl-cache'
$data = 'C:\Users\Achita\OneDrive\Documents\Programming\Visual Odometry'
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python src/eval_kitti.py --data-root "$data" --poses-root "$data\poses" --sequences 00 01 02 03 04 05 06 07 08 09 10 --stereo --output-root results/stereo
.\.venv\Scripts\python src/main.py --data-root "$data" --poses-root "$data\poses" --sequence 04 --stereo --slam --output results/slam/04.txt
.\.venv\Scripts\python src/eval_kitti.py --poses-root "$data\poses" --estimates-root results/slam --sequence 04 --stereo --output-root results/slam-metrics
```

`--max-frames` selects a partial run. Omit it for a complete benchmark. For saved metric trajectories, use `--stereo` or explicit `--alignment se3`. Monocular `--alignment sim3` affects ATE only; the drift metrics are deliberately unscaled.

The [official KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) provides ground truth for 00–10 and evaluates translation/rotation drift over 100–800 m segments. Use its devkit as the reference for metric parity. Benchmark artifacts are saved under `results/` and ignored by Git.
