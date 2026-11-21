# Project: Loop-Engineered Stereo Visual-SLAM for KITTI Odometry

## Architecture

### 1. System Overview & Objective
Upgrade `achitaan/Visual-SLAM` on dedicated feature branch `feat/loop-engineered-stereo-slam` (branched from `main` @ `91a8a25`) to achieve a **significant, authentic, multi-sequence improvement over all repository baselines on the KITTI Visual Odometry benchmark** using disciplined **loop-engineering principles** (hypothesis → modular implementation → automated airgapped benchmark → quantitative gate → ledger logging).

All estimation is **100% authentic and ground-truth-free**:
- Only rectified stereo grayscale image pairs (`image_0`, `image_1`) and static camera calibration (`calib.txt`: `P0`, `P1`) enter the SLAM pipeline.
- Ground-truth poses (`poses/<seq>.txt`) are opened **strictly post-shutdown** by evaluation scripts using `src/metrics.py` (`alignment="se3"`, exact metric scale $s = 1.0$, and official KITTI C++ devkit-parity `float32` 100–800 m segment errors).
- **Single Unified Configuration**: One parameter preset (`--preset improved_stereo`, promoted to default stereo SLAM behavior) must be used identically across all KITTI sequences—zero per-sequence hyperparameter lookup or sequence-ID branching inside `src/`.

### 2. Baselines to Beat (`main` @ `91a8a25`)
1. **Legacy Stereo VO (`src/StereoVisualOdometry.py`, 11 seqs `00–10`, 23,201 frames, 14,567 segments)**:
   - Segment-weighted `t_rel`: **2.2866%**, `r_rel`: **0.8630 deg/100m** (`0.008630 deg/m`), `23` lost pairs on `01` (`9.574%` `t_rel`, `72.29 m` ATE).
2. **Legacy Stereo VO + Offline Pose Graph (`src/eval_pose_graph.py`, 11 seqs `00–10`)**:
   - Segment-weighted `t_rel`: **1.7735%** (`2.2041%` on the 7-seq subset), `r_rel`: **0.6481 deg/100m** (`0.006481 deg/m`; `0.6909 deg/100m` on 7-seq subset). Regresses `09` `t_rel` (`1.705% → 1.825%`).
3. **Verified Python `SharedSlam` (`src/shared_slam.py`, `--loop-mode off` + 7 opt-in stereo flags, 7 seqs `00,01,03,04,07,09,10`, 10,607 frames, 5,925 segments)**:
   - Segment-weighted `t_rel`: **1.9570%**, `r_rel`: **0.5224 deg/100m** (`0.005224 deg/m`). Highway `01` suffers `7.814%` `t_rel` / `71.972 m` SE(3) ATE / `292.754 m` raw ATE; loop sequences `00` (`7.422 m` ATE) and `07` (`2.403 m` ATE) lack loop closure; default `MappingConfig()` leaves the 7 verified stereo flags disabled.

### 3. Data Flow & Modular Subsystem Boundaries
```text
Stereo Frames (image_0, image_1) + calib.txt (P0, P1)
   │
   ▼
[Stage 1: Grid-Bucketed Frontend & Fast Epipolar Stereo] (src/frontend_features.py, src/stereo_depth.py)
   • 16×6 spatial grid-bucketed SIFT extraction retaining octave/angle/response metadata
   • Subpixel SGBM + vectorized/batched epipolar NCC verification & disparity-variance model σ_Z(d)
   • Classification into close metric stereo points (Z < 40B) vs far/bearing candidates (Z >= 40B or low d)
   │
   ▼
[Stage 2: Incremental Map Index & Projective Local-Map Tracker] (src/slam_state.py, src/map_index.py, src/local_map_tracker.py)
   • O(1) incremental pixel-identity conflict index & vectorized frustum culling (no O(|M|) Python scans)
   • SearchByProjection in local spatial cells around motion-predicted pose + octave consistency + LK maxLevel=4
   • Neighbor landmark fusion & off-frustum stale single-view landmark pruning
   │
   ▼
[Stage 3: Heteroscedastic Stereo + Far-Bearing Pose Estimation & Arbitration] (src/mapping_geometry.py, src/stereo_pose_arbitration.py)
   • Inverse-variance weighted [u_L, v_L, u_R] pose estimation + analytic 3N×6 Gauss-Newton/LM refinement
   • Far-field 2D bearing constraints ([u_L, v_L], mask=[1,1,0]) + multi-KF temporal far-point triangulation
   • Holdout-validated stereo pose arbitration & adaptive baseline-parallax keyframe insertion
   │
   ▼
[Stage 4: Analytic-Jacobian Covisible Sliding-Window BA] (src/analytic_ba.py, src/covisibility.py, src/local_bundle.py)
   • Top covisible keyframes (8–10 KFs, up to 400 multi-view landmarks) via reverse keyframe-landmark index
   • Vectorized analytic sparse CSR Jacobians [u_L, v_L, u_R] + post-BA chi-square outlier observation culling
   │
   ▼
[Stage 5: Scalable >300-KF VLAD/BoW Loop Closure & Covariance-Weighted PGO] (src/keyframe_index.py, src/live_loops.py, src/pose_graph.py)
   • Incremental/L2-normalized VLAD / TF-IDF retrieval + bidirectional metric stereo PnP verification
   • Uncapped (>300 KFs) sparse LSMR SE(3) pose graph with distance-scaled odometry & Schur-scaled loop weights
   • Duplicate loop-landmark fusion + post-loop local window BA reconciliation
   │
   ▼  (Estimator closes; airgap boundary)
[Stage 6: Loop-Engineering Benchmark Harness & Metrics] (src/loop_harness.py, scripts/run_loop_engineering.py, src/metrics.py)
   • Post-hoc SE(3) Umeyama ATE, raw ATE, KITTI float32 100–800m t_rel (%) & r_rel (deg/100m), FPS, RSS
   • Append-only experiment ledger (results/loop_ledger.jsonl) & multi-baseline comparison report
```

### 4. Loop-Engineering Iterative Protocol (2-vCPU, 7.8 GiB RAM Linux VM)
Every algorithmic milestone (`M2`, `M3`, `M4`) executes iterative hypothesis-test-gate loops:
- **Tier 1 (Unit & Mathematical Invariants, `< 60 s`)**: `pytest` backend suite (384 existing + new analytic Jacobian, index parity, airgap, and geometry tests).
- **Tier 2 (Multi-Regime KITTI Canary, `< 5–8 min` post-M2 speedup)**:
  - `04` full (`271` frames — straight metric scale & pitch/roll stability)
  - `01` prefix (`350` frames — 100 km/h highway depth noise & far-point rotation stability)
  - `07` full (`1,101` frames — urban turns + end-of-sequence loop closure)
  - Every candidate change is logged to `results/loop_ledger.jsonl` with config, per-sequence metrics, stage profile, RSS, and `ACCEPT` / `REJECT` verdict.
- **Tier 3 (Multi-Sequence Generalization Gate)**:
  - Full sequences `00, 01, 03, 04, 06, 07, 09, 10` (covering all 7 Verified `SharedSlam` baseline sequences plus multi-loop `06`, `11,708` frames total; plus `05` if within runtime budget).

### 5. Git Branch & Shared-Workspace Coordination Rules (MANDATORY)
- **Feature Branch**: `feat/loop-engineered-stereo-slam` (created from `main` @ `91a8a25` in `M1`).
- **Metadata Exclusion**: `.agents/`, `.venv/`, `.datasets/`, and `results/` MUST be listed in `.git/info/exclude` (and/or `.gitignore` for `.datasets/`/`.venv/`/`results/` where already present, keeping `.agents/` strictly in `.git/info/exclude` so `.agents/` is **NEVER** staged or committed).
- **Explicit Staging Only**: Implementation workers MUST use `git add <explicit_owned_files>` — **NEVER** `git add .`, `git add -A`, `git clean`, `git reset --hard`, or `git checkout -- .` (because the parallel E2E Testing Track writes untracked files in `tests/e2e/`, `TEST_INFRA.md`, and `TEST_READY.md`).
- **Git Commit Ownership**: During `M1–M4`, ONLY Implementation Track workers run `git checkout` / `git commit`. E2E Testing Track workers MUST NOT run mutating `git` commands (`M5` Final Milestone commits verified E2E tests and `PROJECT.md`/`TEST_INFRA.md`/`TEST_READY.md` onto the branch).
- **Virtualenv Isolation**: `M1` provisions `/home/antigravity/achitaan/Visual-SLAM/.venv` (`Python 3.12` + `requirements-lock.txt`). E2E Testing Track agents use `/home/antigravity/achitaan/Visual-SLAM/.venv` read-only if already fully installed, or create `/home/antigravity/.gemini/jetski/scratch/e2e_venv` via `uv` if `.venv` is not yet ready (never modifying `.venv`).
- **Dataset Storage**: Store downloaded KITTI sequences at `/mnt/kitti_odometry` (`42 GiB` free) with symlink `/home/antigravity/achitaan/Visual-SLAM/.datasets/kitti -> /mnt/kitti_odometry` so `/home/antigravity` (`19 GiB` free) preserves ample disk space. Never store datasets or virtualenvs in `/tmp`.

---

## Code Layout

- **Repository Root**: `/home/antigravity/achitaan/Visual-SLAM`
- **Implementation Source (`src/`)**:
  - Existing core modules: `shared_slam.py`, `slam_state.py`, `owned_observation.py`, `local_bundle.py`, `bundle_solver.py`, `pose_observability.py`, `bundle_gauge.py`, `mapping_geometry.py`, `stereo_depth.py`, `stereo_pose_arbitration.py`, `stereo_two_view_bundle.py`, `stereo_training_factors.py`, `stereo_bundle_validation.py`, `descriptor_matching.py`, `live_loops.py`, `loop_geometry.py`, `pose_graph.py`, `keyframe_retrieval.py`, `keyframe_index.py`, `StereoVisualOdometry.py`, `VisualOdometry.py`, `kitti.py`, `metrics.py`, `eval_kitti.py`, `eval_pose_graph.py`, `main.py`, `telemetry.py`, `config.py`, `CameraCalib.py`, `stage_profile.py`.
  - New modular additions:
    - `src/loop_harness.py` (`M1` — airgapped loop-engineering runner, baseline comparator & JSONL ledger)
    - `src/map_index.py` (`M2` — incremental O(1) pixel-identity conflict & spatial frustum index)
    - `src/analytic_ba.py` (`M2` — vectorized analytic `[u_L, v_L, u_R]` sparse CSR BA & 6-DoF pose Jacobians)
    - `src/frontend_features.py` (`M3` — spatial grid-bucketed keypoint extraction & octave metadata container)
    - `src/local_map_tracker.py` (`M3` — projective search-radius matching, octave gating & neighbor landmark fusion)
    - `src/covisibility.py` (`M4` — keyframe covisibility graph & covisible BA window selection)
- **CLI & Benchmark Scripts (`scripts/`)**:
  - Existing: `evaluate_shared_slam.py`, `run_shared_benchmark.py`, `run_development_tests.py`, `download_kitti_sequence.py`, `summarize_kitti_batch.py`, `run_kitti_stream.py`.
  - New: `scripts/run_loop_engineering.py` (`M1` — CLI entry point for Tier 1/2/3 loop-engineering benchmark, ledger logging & baseline comparison).
- **Unit & Integration Tests (`tests/test_*.py`)**: Owned by Implementation Track (`M1–M4`, `M5`).
- **Opaque-Box E2E Test Suite (`tests/e2e/`, `TEST_INFRA.md`, `TEST_READY.md`)**: Exclusively owned by **E2E Testing Track** during `E2E` build; integrated/verified in `M5`.
- **Benchmark Documentation (`docs/benchmark/`)**:
  - `docs/benchmark/loop_baseline.json` (`M1` — live VM baseline verification)
  - `docs/benchmark/loop_engineering_results.json` & `docs/benchmark/LOOP_ENGINEERING_REPORT.md` (`M4` — full multi-sequence benchmark & ablation results).

### Exclusive Write Ownership Matrix
| Track / Milestone | Exclusively Owned Paths |
|---|---|
| **E2E Testing Track** | `tests/e2e/**`, `TEST_INFRA.md`, `TEST_READY.md` |
| **M1** | `.git/info/exclude`, `.venv/**`, `/mnt/kitti_odometry/**`, `.datasets/**`, `src/loop_harness.py`, `scripts/run_loop_engineering.py`, `src/CameraCalib.py`, `src/eval_kitti.py`, `src/StereoVisualOdometry.py` *(calib fallback)*, `tests/test_loop_harness.py`, `docs/benchmark/loop_baseline.json` |
| **M2** | `src/map_index.py`, `src/slam_state.py`, `src/analytic_ba.py`, `src/pose_observability.py`, `src/local_bundle.py`, `src/bundle_solver.py`, `src/shared_slam.py` *(conflict/index/telemetry paths)*, `src/main.py` *(telemetry idle gate)*, `src/telemetry.py`, `tests/test_map_index.py`, `tests/test_analytic_ba.py` |
| **M3** | `src/frontend_features.py`, `src/stereo_depth.py`, `src/local_map_tracker.py`, `src/descriptor_matching.py`, `src/mapping_geometry.py`, `src/stereo_pose_arbitration.py`, `src/shared_slam.py` *(frontend/tracking/far-point paths)*, `src/StereoVisualOdometry.py` *(mutual/subpixel VO)*, `tests/test_frontend_features.py`, `tests/test_local_map_tracker.py`, `tests/test_heteroscedastic_pose.py` |
| **M4** | `src/covisibility.py`, `src/local_bundle.py` *(covisible window + chi2 culling)*, `src/live_loops.py`, `src/pose_graph.py`, `src/loop_geometry.py`, `src/keyframe_index.py`, `src/keyframe_retrieval.py`, `src/eval_pose_graph.py`, `src/LoopClosure.py`, `src/shared_slam.py` *(preset/keyframe/loop integration)*, `src/main.py`, `scripts/evaluate_shared_slam.py`, `scripts/run_shared_benchmark.py`, `tests/test_covisibility_and_loops.py`, `docs/benchmark/LOOP_ENGINEERING_REPORT.md`, `docs/benchmark/loop_engineering_results.json` |
| **M5 (Final Milestone)** | Whole-repository E2E debug/fixes (sequential Tiers 1–4) + Tier 5 adversarial test additions (`tests/`) + final branch commit hygiene |

---

## Feature Inventory

Every feature identified across `ORIGINAL_REQUEST.md` and Survey Explorers 1, 2, and 3 is assigned a milestone (none unassigned):

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Feature Branch & Git Metadata Hygiene | Create branch `feat/loop-engineered-stereo-slam` off `main` (`91a8a25`); exclude `.agents/`, `.venv/`, `.datasets/`, `results/` in `.git/info/exclude` so `.agents/` is never committed. | M1 |ORIGINAL_REQUEST, survey_1 §1.1 |
| F2 | Reproducible Python 3.12 Runtime & Test Baseline | Provision `.venv` (`Python 3.12` + `requirements-lock.txt`) via `uv` and pass all 384 existing `pytest` backend tests. | M1 | survey_1 §1.1, survey_2 §1.5–1.6 |
| F3 | Official KITTI Odometry Dataset & Pose Cache | Fetch & SHA-256/CRC32-verify `data_odometry_poses.zip` (`00–10`) and stereo grayscale sequences `00, 01, 03, 04, 06, 07, 09, 10` via HTTP-Range to `/mnt/kitti_odometry` (`.datasets/kitti`). | M1 | survey_2 §1.4 |
| F4 | Airgapped 3-Tier Loop-Engineering Benchmark Harness & Ledger | Implement `src/loop_harness.py` & `scripts/run_loop_engineering.py` with strict estimator/evaluator process airgap, Tier 1/2/3 suites, `results/loop_ledger.jsonl`, and multi-baseline comparison (`VO`, `VO+PGO`, `Verified SharedSlam`). | M1 | ORIGINAL_REQUEST, survey_2 §1.3, survey_3 §4.2 |
| F5 | Legacy Evaluator & Import Safety Fixes | Guard `CameraCalib.py` import side-effect, add global segment-weighted aggregate & RNG seeding to `src/eval_kitti.py`, keep scale-invariant `rotation_deg_per_m` consistent, and support unlabeled 2-line `calib.txt` in `StereoVisualOdometry.py`. | M1 | survey_2 §1.3 #3–#7 |
| F6 | Incremental Map Conflict/Spatial Index & Stale Landmark Pruning | Maintain O(1) `(keyframe_id, pixel_key)` conflict index & vectorized world-point frustum buffer on `SlamMapState` (`src/map_index.py`, `src/slam_state.py`), eliminating $O(|M|+|K|)$ per-frame scans/rebuilds and properly aging/pruning off-frustum single-view landmarks. | M2 | survey_1 §1.3 #5, #10; survey_3 §4.1 P0 |
| F7 | Vectorized Analytic Sparse BA & Stereo Pose Jacobians | Replace 3 full-map scans in `local_bundle.py` with reverse-indexed keyframe lookup and implement vectorized closed-form `[u_L, v_L, u_R]` CSR Jacobians (`src/analytic_ba.py`) for `local_bundle_adjustment` and `refine_stereo_map_pose` ($<10^{-7}$ finite-diff parity, $\ge 2.5\times$ pipeline speedup). | M2 | survey_1 §1.3 #11; survey_3 §4.1 P1 |
| F8 | Zero-Copy Strided Telemetry & Idle-Client Fast Path | Replace `SharedSlam.map_points_sample` full-map list allocation with bounded strided sampling and skip JPEG/base64 encoding + frame sleep when zero WebSocket clients are attached. | M2 | survey_1 §1.3 #14 |
| F9 | Spatial Grid-Bucketed SIFT, Octave Retention & Fast Epipolar Stereo | Select features via $16\times 6$ spatial grid quotas (`src/frontend_features.py`) retaining octave/angle/response, batch/vectorize sparse epipolar patch checks (`src/stereo_depth.py`), and upgrade `StereoVisualOdometry.py` with mutual matching + subpixel disparity interpolation. | M3 | survey_1 §1.3 #2, #8, #12; survey_3 §4.1 P2 |
| F10 | Projective Search-Radius Tracking, Octave Gate & Neighbor Fusion | Match local map landmarks within predicted `(u, v)` radius gated by SIFT octave $|\Delta\ell|\le 1$ (`src/local_map_tracker.py`), widen LK pyramid (`maxLevel=4`), and fuse duplicate neighbor landmarks to raise multi-view track ratio above `15%`. | M3 | survey_1 §1.3 #4; survey_3 §4.1 P3 |
| F11 | Heteroscedastic Stereo Pose Estimation & Far-Field Bearing/Triangulation | Weight `[u_L, v_L, u_R]` residuals by octave scale and quadratic stereo depth uncertainty $\sigma_Z = \frac{Z^2}{f_x B}\sigma_d$, incorporate low-disparity far points ($Z\ge 40B$) as 2D `[u_L, v_L]` bearing rotation constraints (`mask=[1,1,0]`), and triangulate far stereo landmarks across keyframes to tame highway `01` drift. | M3 | survey_1 §1.3 #3; survey_3 §4.1 P4 |
| F12 | Covisible Sliding-Window Bundle Adjustment & Chi-Square Culling | Select top covisible keyframes (`8–10` KFs, up to `400` landmarks via `src/covisibility.py`) in `local_bundle.py` with octave/disparity-weighted Huber residuals and post-BA chi-square outlier observation culling. | M4 | survey_1 §4; survey_3 §4.1 P5 |
| F13 | Scalable `>300`-KF VLAD/BoW Loop Closure, Covariance-Weighted PGO & Loop Landmark Fusion | Remove the 300-KF cap in `live_loops.py` via sparse LSMR `pose_graph.py` with analytic Jacobians, upgrade `keyframe_index.py`/`keyframe_retrieval.py` to incremental VLAD/TF-IDF, scale PGO odometry/loop information by segment length & inlier covariance (fixing `09` drift regression), and fuse duplicate loop landmarks + post-loop BA. | M4 | survey_1 §1.3 #6, #7; survey_3 §4.1 P6 |
| F14 | Unified `improved_stereo` Default Preset, Adaptive Parallax Keyframes & Multi-Sequence KITTI Report | Promote verified + loop-engineered stereo SLAM settings (`preset='improved_stereo'`) as default stereo SLAM behavior across `MappingConfig`, `main.py`, `evaluate_shared_slam.py`, and `run_shared_benchmark.py`, add adaptive parallax keyframe gating, clean `LoopClosure.py` stub, and publish full Tier 3 (`00,01,03,04,06,07,09,10`) benchmark evidence beating all baselines. | M4 | survey_1 §1.3 #1, #9; survey_2 §1.3 #1; survey_3 §4.1 P7 |
| F15 | Full E2E Suite Pass (Tiers 1–4) & Adversarial Hardening (Tier 5) | Pass 100% of requirement-driven opaque-box E2E tests (`TEST_READY.md` Tiers 1–4) and complete white-box Challenger → Worker → Reviewer adversarial coverage hardening (Tier 5) with clean git branch state. | M5 | Project Pattern Dual Track |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Branch, Env, KITTI Cache & Loop-Engineering Harness | `F1–F5`: Create branch `feat/loop-engineered-stereo-slam`, `.git/info/exclude`, Python 3.12 `.venv`, KITTI S3 poses + 8 sequences (`00,01,03,04,06,07,09,10`), airgapped `run_loop_engineering.py` + ledger, legacy import/eval fixes, and live Tier 2 VM baseline run. | none | IN_PROGRESS (`ec2ae60fbf8c2682b01ed6c55f3c4db7`) |
| 2 | M2: CPU Throughput, Incremental Map Index & Analytic Sparse BA | `F6–F8`: Incremental `SlamMapState` conflict/spatial index, off-frustum single-view pruning, vectorized analytic `[u_L,v_L,u_R]` sparse CSR BA & 6-DoF pose Jacobians, zero-copy telemetry fast path; loop-engineered Tier 2 verification ($\ge 2.5\times$ speedup, accuracy parity/gain). | M1 | PLANNED |
| 3 | M3: Grid-Bucketed Frontend, Projective Tracking & Heteroscedastic/Far Stereo Geometry | `F9–F11`: $16\times 6$ SIFT grid bucketing + octave metadata, fast sparse epipolar verification, projective search-radius + octave tracking + neighbor fusion, heteroscedastic stereo pose + far-field bearing/triangulated points; iterative Tier 2 ablation loops crushing `01` drift. | M2 | PLANNED |
| 4 | M4: Covisible BA, Scalable >300-KF VLAD Loop Closure, Weighted PGO & Multi-Sequence KITTI Validation | `F12–F14`: Covisible 8–10 KF / 400-pt BA + chi2 culling, uncapped `>300`-KF VLAD/TF-IDF loop closure + distance/covariance-weighted PGO + loop landmark fusion, unified `improved_stereo` default preset + adaptive parallax KFs, and full Tier 3 (`00,01,03,04,06,07,09,10`) KITTI evaluation & report. | M3 | PLANNED |
| 5 | M5: Final Milestone — E2E Pass (Tiers 1–4) & Adversarial Hardening (Tier 5) | `F15`: Phase 1 sequential E2E Tier 1 → 2 → 3 → 4 pass against `TEST_READY.md`; Phase 2 white-box Tier 5 Challenger → Worker → Reviewer hardening loop; final clean branch verification. | M4, E2E | PLANNED |
| E2E | E2E Testing Track (Parallel Dual Track) | Opaque-box requirement-driven E2E test infrastructure (`TEST_INFRA.md`) and 4-tier test suite (`tests/e2e/`) covering `F1–F14`; publishes `TEST_READY.md` upon completion. | none | IN_PROGRESS (`eb43e8c9a03abfb36e40ff931d3e3dd7`) |

---

## Interface Contracts

### 1. CLI & User-Facing Entry Points (Shared by Implementation & E2E Tracks)

#### A. `scripts/run_loop_engineering.py` (Loop-Engineering Benchmark & Ledger CLI)
```bash
python scripts/run_loop_engineering.py \
  --tier {tier1,tier2,tier3,custom} \
  [--sequences 04 01 07] \
  [--max-frames-map '{"01": 350}'] \
  [--preset {improved_stereo,verified_baseline,legacy_default}] \
  [--config-overrides '{"grid_bucketing": true, ...}'] \
  [--data-root /home/antigravity/achitaan/Visual-SLAM/.datasets/kitti] \
  [--poses-root /home/antigravity/achitaan/Visual-SLAM/.datasets/kitti/poses] \
  [--output-dir results/loop_runs/<run_id>] \
  [--ledger-path results/loop_ledger.jsonl] \
  [--hypothesis "short description"]
```
- **Airgap Contract**: Spawns/runs estimator (`SharedSlam`) with **no access to `--poses-root`**, writes `<output-dir>/<seq>/poses.txt` and `<output-dir>/<seq>/run.json` (including `stage_profile` and `peak_rss_mb`), then invokes post-hoc `src/metrics.py:evaluate_kitti(gt_poses, est_poses, alignment="se3")` in a separate evaluation phase, writes `<output-dir>/summary.json` and `<output-dir>/comparison.md`, and appends one JSON object to `--ledger-path`.
- **Exit Code**: `0` when all requested sequences complete with valid `SE(3)` poses (`len(est) == len(gt)`), `non-zero` on invalid args, missing data, crash, or corrupted poses.
- **`summary.json` & `loop_ledger.jsonl` Record Schema**:
  ```json
  {
    "run_id": "string",
    "timestamp_utc": "ISO-8601",
    "git_commit": "sha",
    "tier": "tier1|tier2|tier3|custom",
    "preset": "improved_stereo|verified_baseline|legacy_default",
    "hypothesis": "string",
    "ground_truth_used_for_estimation": false,
    "aggregate": {
      "sequence_count": 3,
      "total_frames": 1722,
      "total_segments": 500,
      "segment_weighted_t_rel_pct": 1.15,
      "segment_weighted_r_rel_deg_per_m": 0.0042,
      "segment_weighted_r_rel_deg_per_100m": 0.42,
      "mean_ate_se3_m": 1.85,
      "total_lost_frames": 0,
      "total_elapsed_s": 180.0,
      "overall_fps": 9.56,
      "peak_rss_mb": 520.0
    },
    "baseline_deltas": {
      "vs_legacy_vo_t_rel_pct": -1.10,
      "vs_legacy_vo_pgo_t_rel_pct": -0.60,
      "vs_verified_shared_slam_t_rel_pct": -0.80
    },
    "sequences": {
      "04": {
        "frames": 271,
        "segments": 43,
        "ate_rmse_m": 0.42,
        "raw_ate_rmse_m": 0.95,
        "translation_percent": 0.51,
        "rotation_deg_per_m": 0.0031,
        "rotation_deg_per_100m": 0.31,
        "lost_frames": 0,
        "verified_loops": 0,
        "applied_loops": 0,
        "elapsed_s": 28.0,
        "processing_fps": 9.68
      }
    },
    "verdict": "ACCEPT|REJECT|BASELINE"
  }
  ```

#### B. `MappingConfig` Presets & Ablation Flags (`src/shared_slam.py`, `src/main.py`, `scripts/evaluate_shared_slam.py`, `scripts/run_shared_benchmark.py`)
- `MappingConfig.from_preset(preset: str = "improved_stereo", *, stereo: bool = True, **overrides) -> MappingConfig`:
  - `"legacy_default"`: Preserves pre-existing `MappingConfig()` defaults for backward-compatible unit tests when explicitly requested.
  - `"verified_baseline"`: Exact 7-flag `original-python-verified` stereo config (`loop_mode="off"`, `stereo_depth_policy="verified_fallback"`, `stereo_map_depth_policy="verified"`, `stereo_pose_arbitration=True`, `stereo_raw_reference_retry=True`, `stereo_verified_reference_retry=True`, `stereo_descriptor_association_retry=True`, `bundle_solver_accuracy="precise"`, `bundle_landmark_policy="bounded_fixed"`, `stereo_bundle_gauge_mode="veto"`).
  - `"improved_stereo"` (default when `--preset improved_stereo` is passed, and promoted to default stereo CLI profile in `M4` for `main.py --slam --stereo`, `evaluate_shared_slam.py --stereo`, and `run_shared_benchmark.py` unless legacy flags explicitly override): enables verified stereo guards + `use_incremental_map_index=True`, `use_analytic_ba_jacobians=True`, `use_grid_bucketing=True`, `use_projective_tracking=True`, `use_heteroscedastic_stereo_pose=True`, `use_far_stereo_points=True`, `use_covisible_ba=True`, `use_vlad_loop_retrieval=True`, `use_weighted_pgo=True`, `use_adaptive_keyframe_parallax=True`, `max_live_loop_keyframes=0` (`0` = uncapped).
- CLI flag `--preset {improved_stereo,verified_baseline,legacy_default}` accepted on `src/main.py`, `scripts/evaluate_shared_slam.py`, `scripts/run_shared_benchmark.py`, and `scripts/run_loop_engineering.py`.

#### C. `src/eval_kitti.py` & `src/eval_pose_graph.py`
- `src/eval_kitti.py` sets both `cv.setRNGSeed(seed)` and `np.random.seed(seed)` (`--seed`, default `0`), writes `summary.json` containing both per-sequence metrics and `"aggregate"` (`segment_weighted_translation_percent`, `segment_weighted_rotation_deg_per_m`, `segment_weighted_rotation_deg_per_100m`, `total_segments`), and marks `translation_percent = None` while retaining scale-invariant `rotation_deg_per_m` when `alignment == "sim3"`.

---

### 2. Internal Cross-Module Contracts

#### `src/map_index.py` ↔ `src/slam_state.py` & `src/shared_slam.py` (`M2`)
- `IncrementalMapIndex`:
  - `on_landmark_added(landmark: Landmark) -> None`
  - `on_observation_added(landmark_id: int, obs: Observation) -> None`
  - `on_landmark_removed(landmark_id: int) -> None`
  - `conflicted_landmark_ids() -> frozenset[int]` (O(1) read; exact parity with `_landmark_pixel_identity_conflicts`)
  - `query_frustum(T_cw: np.ndarray, K: np.ndarray, image_size: tuple[int, int], max_depth: float = 80.0, margin_px: float = 32.0, include_far_bearings: bool = False) -> FrustumQueryResult(landmark_ids: np.ndarray, projected_uv: np.ndarray, depths: np.ndarray)`
  - `prune_stale_landmarks(map_state: SlamMapState, current_kf_id: int, min_age_kfs: int = 3, min_found_ratio: float = 0.25) -> int`

#### `src/analytic_ba.py` ↔ `src/local_bundle.py`, `src/pose_observability.py`, `src/mapping_geometry.py` (`M2`)
- `build_analytic_ba_jacobian(poses_cw, points_w, obs_kf_idx, obs_pt_idx, K, baseline, disparity_offset, component_mask, huber_weights=None) -> scipy.sparse.csr_matrix`
  - Must match numerical finite-difference Jacobian of `local_bundle` residual vector to `max_abs_err < 1e-6` (`rtol < 1e-5`) across randomized & real KITTI stereo configurations, including masked `[True, True, False]` bearing-only observations.
- `refine_stereo_pose_analytic(T_wc_init, points_w, obs_uv_uR, weights_uv_uR, component_mask, K, baseline, disparity_offset, f_scale=1.5, max_iter=15) -> PoseRefineResult(T_wc, converged, initial_cost, final_cost, inlier_mask, hessian_6x6)`

#### `src/frontend_features.py` & `src/stereo_depth.py` ↔ `src/shared_slam.py` (`M3`)
- `StereoFrameFeatures` dataclass:
  - `pixels: np.ndarray` (`float32 [N, 2]`), `descriptors: np.ndarray` (`float32 [N, 128]`), `octaves: np.ndarray` (`int16 [N]`), `angles: np.ndarray` (`float32 [N]`), `responses: np.ndarray` (`float32 [N]`), `disparities: np.ndarray` (`float32 [N]`, `-1.0` if unmatched), `right_u: np.ndarray` (`float32 [N]`), `depths_m: np.ndarray` (`float32 [N]`), `sigma_disparity_px: np.ndarray` (`float32 [N]`), `is_close_stereo: np.ndarray` (`bool [N]`, `0.1 < Z < 40 * baseline`), `is_far_bearing: np.ndarray` (`bool [N]`).
- `extract_grid_bucketed_sift(image_gray, target_features=1500, grid_cols=16, grid_rows=6, contrast_threshold=0.02) -> BucketedKeypoints`

#### `src/local_map_tracker.py` & `src/mapping_geometry.py` ↔ `src/shared_slam.py` (`M3`)
- `match_landmarks_by_projection(frustum_candidates, frame_features, landmark_descriptors, landmark_octaves, radius_px=36.0, ratio=0.75, max_octave_diff=1) -> dict[int, int]`
- `fuse_neighbor_landmarks(map_state, keyframe, neighbor_kf_ids, K, baseline, radius_px=4.0) -> int`
- `triangulate_far_stereo_landmarks(prev_kf, curr_kf, K, baseline, min_parallax_deg=0.35, min_depth_m=20.0, max_depth_m=400.0) -> list[TriangulatedLandmark]`

#### `src/covisibility.py`, `src/keyframe_index.py`, `src/live_loops.py`, `src/pose_graph.py` ↔ `src/slam_state.py` (`M4`)
- `CovisibilityGraph.top_covisible_keyframes(anchor_kf_id: int, max_keyframes: int = 8, min_shared: int = 8) -> list[int]`
- `optimize_weighted_pose_graph(poses_wc, odometry_edges, loop_edges, huber_scale=1.0) -> dict[int, np.ndarray]`:
  - Uses sparse CSR analytic/structured Jacobians + `lsmr` with **no 300-keyframe cap**, scaling $\Omega_{\text{odom},k} \propto (\max(\Delta s_k, 0.5))^{-1/2}$ (residual weight) and $\Omega_{\text{loop}}$ by verified PnP inlier support / Hessian conditioning.
- `fuse_loop_landmarks(map_state: SlamMapState, query_kf_id: int, match_kf_id: int, T_match_query: np.ndarray, K: np.ndarray, radius_px: float = 4.0) -> int`
