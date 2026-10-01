# Visual SLAM: A TLDR

**Author:** Achita  
**Date:** July 2024

## Tested project status — October 1, 2026

Metric stereo odometry and **offline pose graph optimization** have completed
all 11 public-ground-truth KITTI odometry sequences (00–10): **23,201 frames**
and **61 geometrically verified loops**. Segment-weighted translation drift
improved from **2.287% to 1.773%**, and rotation drift from **0.00863 to 0.00648
degrees/m**. These are public training-set measurements, not a submission to
the hidden KITTI test leaderboard.

See the [complete benchmark report](docs/benchmark/REPORT.md) for all sequence
metrics, trajectory plots, position-error curves, methodology, repeatability
checks and limitations. The [machine-readable results](docs/benchmark/results.json)
and [completion plan](PLAN.md) are included in the repository. The dashboard's
saved benchmark snapshot contains all 11 raw and corrected runs.

The orange **Pose graph** curve is the trajectory **after optimization** and
propagation of the keyframe corrections to every frame. Blue is the original
stereo odometry. With no verified loops, the corrected trajectory remains
identical to the original. Ground truth is used only to evaluate the result.

The work fixes inconsistent graph vertex IDs, incorrect loop-edge conventions,
unmeasured loop constraints, invalid stereo depth handling, and an optimizer
convergence failure on a real loop graph. The dashboard now uses equal trajectory
axis scales, explicit metric/alignment labels, actual socket state and saved
benchmark comparisons.

Local validation: **45 Python tests**, **3 dashboard socket tests**, and the
dashboard production build passed. Every raw and corrected sequence agrees
with the official KITTI devkit evaluator within the declared tolerance.
Windows/Linux CI is configured; a remote CI result is not yet claimed.

Full live SLAM remains under development. Persistent landmarks, local bundle
adjustment, relocalization, and feedback of optimized poses into live tracking
and mapping still need work. Sequence 01 has 23 repeatable lost tracking pairs;
sequence 09 improves ATE but worsens translation drift after optimization.
The sparse optimizer branch for graphs above 300 keyframes has not been validated
by this batch. Monocular output has arbitrary scale.

## Quick Start (VO + Dashboard)

### Backend (VO + telemetry)

Use Python 3.12 and the tested dependency lock for a reproducible environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python src/main.py --max-frames 51
```

The default input is the bundled monocular sample. Full KITTI input must be
specified with `--data-root`; invalid paths fail instead of silently falling
back to a different dataset. Add `--slam` for keyframes and the local map,
`--no-telemetry` for batch processing, and `--output results/00.txt` for KITTI
trajectory export. Plot windows are opened only with `--plot`.
OpenCV defaults to one worker to bound memory. Use `--opencv-threads N` to
increase workers when sufficient RAM is available. The runner uses a fixed
OpenCV seed, matching the benchmark configuration.

See [the completion plan](PLAN.md) for the audit, benchmark protocol, known
accuracy gaps, and acceptance checks.
WebSocket telemetry publishes at `ws://localhost:8765` by default.

### Dashboard (Next.js)
1. Install dashboard dependencies:
   ```bash
   cd dashboard
   npm ci
   ```
2. Start the dashboard:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:3000` to view the live dashboard.

You can override the telemetry URL with:
```
NEXT_PUBLIC_WS_URL=ws://localhost:8765
```

## Feature Flags (incremental SLAM)
Configure in `src/config.py`:
- `ENABLE_KEYFRAMES`, `ENABLE_LOCAL_MAP`, `ENABLE_POSE_GRAPH`, `ENABLE_LOOP_CLOSURE`
- `USE_RELATIVE_SCALE_FIX` (optional scale computation update)
- `KEYFRAME_INTERVAL`, `MIN_KEYFRAME_TRANSLATION`
- `VOCAB_BUILD_MIN_FRAMES`, `VOCAB_NUM_CLUSTERS`, `LOOP_CLOSURE_THRESHOLD`, `POSE_GRAPH_OPT_EVERY`

### Optional Dependencies
Pose graph optimization uses SciPy and does not require g2o. Loop detection
in the live runner reports appearance candidates; it does not insert unverified
loop constraints. The offline evaluator verifies metric loops using stereo
geometry. Integrating that verifier and correction feedback into live tracking
and mapping remains unfinished. Monocular translation currently has arbitrary
unit scale; use stereo for metric accuracy evaluation.

### Benchmark and tests

```powershell
$data = 'C:\path\to\KITTI\dataset'
.\.venv\Scripts\python src/eval_kitti.py --data-root "$data" --poses-root "$data\poses" --sequences 00 01 02 03 04 05 06 07 08 09 10 --stereo --output-root results/stereo
.\.venv\Scripts\python -m pytest -q
cd dashboard
npm ci
npm test
npm run build
```

The evaluator writes poses, per-sequence JSON, and `summary.json`. It reports
ATE with explicitly labeled alignment and KITTI drift over 100–800 m segments.
Drift is computed without scale alignment. A sequence too short for these
segments reports null drift metrics. `--max-frames` allows a labeled partial
run; omit it for the full sequence. `--estimates-root` evaluates saved poses
without rerunning odometry. FPS includes image I/O, including OneDrive hydration.
Keep datasets local before comparing speed.

For limited disk space or unavailable OneDrive images, the batch runner reads the
official grayscale archive using HTTP ranges. It caches one sequence when space
permits, otherwise streams CRC-checked images through bounded caches. Ground-truth
pose files must first be copied into `results/benchmark-batch/reference/poses`.

```powershell
.\scripts\run_kitti_batch.ps1
# Run missing selected sequences, with a separate log if another batch is active:
.\scripts\run_kitti_batch.ps1 -Sequences @('03','06','05') -LogName retry-batch.log
# Deliberately repeat tracking and geometric verification for a complete run:
.\scripts\run_kitti_batch.ps1 -Sequences @('07') -ForceRetest -LogName repeat-07.log
.\.venv\Scripts\python scripts/summarize_kitti_batch.py
.\.venv\Scripts\python scripts/plot_kitti_batch.py
# Verify source poses, official evaluator parity, retained graphs and cleanup:
.\.venv\Scripts\python scripts/verify_kitti_ground_truth.py
.\.venv\Scripts\python scripts/recompute_kitti_metrics.py
.\.venv\Scripts\python scripts/check_kitti_devkit.py
.\.venv\Scripts\python scripts/check_kitti_devkit.py --reports-root results/benchmark-batch/graph --output results/benchmark-batch/devkit-parity-graph.json
.\.venv\Scripts\python scripts/verify_kitti_batch.py
# Reoptimize saved, verified image constraints after image cleanup:
.\.venv\Scripts\python scripts/run_kitti_stream.py --sequence 07 --replay-graph
```

Each sequence exports raw and corrected poses, full accuracy metrics, verified
loop transforms, source checksums, runtime versions and explicit failure status.
Only owned `.datasets/batch-scratch/NN` image/keyframe caches are deleted after
testing; the original dataset and benchmark reports are retained. The summary
accounts for all 11 ground-truth sequences, including pending/failed runs, and
uses segment-count weighting for aggregate drift. `REPORT.md` and `plots/` hold
every sequence's trajectory and position-error graphs. Batch graph correction is
separate from live tracking and landmark feedback.

Completed full runs are reused only when pose counts, rigid poses and the saved
constraint checksum match. `-ForceRetest` requests a new image-based run;
`--replay-graph` repeats only optimization from the retained measurements.
KITTI segment endpoints follow the official devkit's float32 distance arithmetic.
`scripts/check_kitti_devkit.py` compares every saved segment with the compiled
official evaluator; the declared tolerance is 0.0001 percentage points and
0.0001 degrees/m. Small pose graphs use a bounded dense SVD solve because the
approximate sparse solve stalled on a real seven-loop graph.
The SVD path groups independent vertices when computing forward differences;
a regression test checks this Jacobian against independent column differences.
Graph reports retain solver success, evaluation counts, initial/final objective,
optimality and solve time. Image-constraint replay can reproduce these diagnostics
after the temporary dataset images have been removed.
The official devkit comparison needs a compiled `calcSequenceErrors` wrapper;
the measured Windows reference and its source are retained under
`results/benchmark-batch/devkit`. That reference uses MinGW flags
`-O2 -msse2 -mfpmath=sse` so distance arithmetic follows IEEE float32.

Dashboard `Clear view` clears the displayed history and reconnects; it does
not restart the backend. Pause/Resume controls telemetry streaming while the
odometry pipeline continues processing.

The dashboard has Live, Benchmarks and Events views. Saved benchmark snapshots
are built from actual JSON reports and pose files, with complete/partial coverage
and separate raw error and aligned ATE. Refresh the saved snapshot before building:

```powershell
.\.venv\Scripts\python scripts/update_dashboard_benchmarks.py --data-root "$data"
```

### Loop-containing pose graph experiment

Check image availability before a long OneDrive run:

```powershell
.\.venv\Scripts\python scripts/check_kitti_data.py --data-root "$data" --sequences 00 03 04 --max-frames 300
```

An alternative is to download one sequence from the official grayscale KITTI
archive into `.datasets/kitti`. This uses HTTP ranges, verifies each ZIP member's
CRC, and checks disk space before extracting. It does not download the full 22 GB
archive. Ground-truth poses must be supplied separately.

```powershell
.\.venv\Scripts\python scripts/download_kitti_sequence.py --sequence 07 --inspect
.\.venv\Scripts\python scripts/download_kitti_sequence.py --sequence 07
.\.venv\Scripts\python src/eval_kitti.py --data-root .datasets/kitti --poses-root "$data\poses" --sequence 07 --stereo --output-root results/stereo-loop-07
.\.venv\Scripts\python src/eval_pose_graph.py --data-root .datasets/kitti --poses-root "$data\poses" --estimates-root results/stereo-loop-07 --sequence 07 --output-root results/pose-graph
```

This is a **batch graph experiment**, separate from live SLAM. It retrieves loop
candidates using visual words and temporal exclusion; requires mutual descriptor
matches, bidirectional PnP, positive depth, image coverage and reprojection checks;
and passes independently measured metric transforms to the same `SLAM` optimizer.
Ground truth is loaded only after optimization, for accuracy evaluation. Rigid
corrections propagate to every intermediate frame, and raw/corrected ATE and drift
are reported together. Information weights are fixed experimental parameters,
not estimated measurement covariances. A run with zero verified loops leaves the
trajectory unchanged. Live tracking/landmark correction feedback is still missing.
Loop features are capped at 1,500 per keyframe and graph evaluation uses one
OpenCV worker. Saved `07-constraints.json` records the image-derived transforms
and raw trajectory checksum. Add `--resume-graph` to retry optimization with
those same constraints and configuration, without rerunning image retrieval.

## Telemetry Schema (v1)
Fields published per frame:
- `pose_T_wc` (4x4), `frame_index`, `timestamp`, `mode`
- `tracking` (`num_matches`, `num_inliers`, `inlier_ratio`)
- `map` (`keyframes`, `map_points`)
- optional: `image`, `features`, `events`, `fps`, `sequence`, `total_frames`, `run_id`, `overlay_enabled`

## Original learning notes

The notes below describe the original project and mathematical background.
They are not the current stereo benchmark's implementation specification:
the tested stereo frontend uses SIFT/SGBM/PnP, ORB matching uses Hamming distance,
and the original relative-scale heuristic does not establish monocular metric scale.

This document provides a comprehensive summary of Visual SLAM,
currently under development, synthesizing information from a variety of
reputable sources, including An Invitation to 3-D Vision, lecture slides
from Carnegie Mellon University and the University of Toronto, IEEE
Robotics & Automation Magazine, and CS231A from Stanford University,
among others. It focuses on the most critical aspects necessary for
coding the program and understanding the underlying mathematical
concepts.
---

## Table of Contents

- [Introduction](#introduction)
- [Image Sequencing](#image-sequencing)
  - [Step 1 — Capture Frame \(I_k\)](#step-1----capture-frame-ik)
  - [Step 2 — Extract and Match Features](#step-2----extract-and-match-the-feature-between-ik-1-and-ik)
- [Exploring Epipolar Geometry](#exploring-epipolar-geometry)
  - [The Epipolar Constraint](#the-epipolar-constraint)
  - [Properties of the Essential Matrix and Pose Recovery](#properties-of-the-essential-matrix-and-pose-recovery)
  - [Possible Pose Solution Pairs](#possible-pose-solution-pairs)
  - [Eight Point Algorithm](#eight-point-algorithm)
  - [Projection Into The Essential Space](#projection-into-the-essential-space)
  - [Triangulation](#triangulation)
- [Theorems and Definitions](#theorems-and-definitions)
- [Implementation of Epipolar Geometry](#implementation-of-epipolar-geometry)

---

# Introduction

Visual SLAM builds off of Visual Odometry (VO), so first we need to understand that. VO is the process of estimating the position and
orientation of a camera from a sequence of images. The path estimation
is done sequentially by a new frame $I_k$, only providing local or
relative estimates. The program of VO can be broken down into two steps
the front end and back end. The front end extracts the data via the
camera while the back end calculates the pose. Furthermore, the process
of VO can be broken down into a few steps.
> **Process of Visual Odometry**
> 
> 1. Capture a frame \( I_k \)
> 2. Extract and match the feature between \( I_{k-1} \) and its subsequent term \( I_k \)
> 3. Compute the essential matrix, \( E \), using the 8-point theorem between the image pair \( I_{k-1} \) and \( I_k \).
>
>    \[
>    WE = 0
>    \]
> 4. Decompose \( E_k \) into \( R_k \) and \( t_k \) into four pairs using Singular Value Decomposition:
>
>    \[
>    E = U \Sigma V^T
>    \]
>
>    where \( U \) and \( V^T \) are rotation matrices.
> 5. Find the correct pose via triangulating the key points to form the transformation matrix \( T \):
>
>    \[
>    T_k = \begin{bmatrix}
>    R_k & t_k \\
>    0 & 1 \\
>    \end{bmatrix}
>    \]
> 6. Compute the relative scale and rescale \( t_k \) accordingly:
>
>    \[
>    r = \frac{||x_{k-1, i} - x_{k-1,j}||}{||x_{k, i} - x_{k,j}||}
>    \]
> 7. Concatenate the transformation by computing:
>
>    \[
>    C_k = T_k C_{k-1}
>    \]

---

## Image Sequencing

### Step 1 — Capture Frame \(I_k\)

We define
\[
I_{0:n} = \{I_0, I_1, \ldots, I_{n-1}, I_n\}
\]

Trivially, using OpenCV, we can capture an instance of the sequence \( I_k \) and analyze at least eight key features of the image.

Import OpenCV and run a video loop that captures a single frame every iteration:

```python
import cv2 as cv

video = cv.VideoCapture(self.captureIndex)
assert video.isOpened()

video.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

while True: 
    ret, frame = video.read()  # Captures frame by frame 
    # video.read() -> (bool of frame is read correctly, frame)
    if not ret: 
        break  # breaks if read incorrectly

    if cv.waitKey(1) & 0xFF == ord(" "): 
        break

# Release the capture
video.release()
cv.destroyAllWindows()

```

## **Step 2 — Extract and Match Features**

> **Step 2**: Extract and match the feature between \(I_{k-1}\) and its subsequent term \(I_k\).  
> This step links the **frontend (image capture)** and **backend (pose estimation)**.  
> It takes a key feature (point) from frame \(I_{k-1}\) and matches it to the corresponding point in the subsequent image \(I_k\).

### **Feature Extraction using ORB**
To detect key features, we use **Oriented FAST and Rotated BRIEF (ORB)**.

```python
import cv2 as cv

orb = cv.ORB_create(nfeatures=1500)

# Keypoints and descriptors for both frames
kp1, desc1 = orb.detectAndCompute(I[i-1], None)
kp2, desc2 = orb.detectAndCompute(I[i], None)
```

### **Feature Matching using FLANN**
To estimate the corresponding key features between the two frames, we use **Fast Library for Approximate Nearest Neighbor (FLANN)**.

```python
FLANN_INDEX_KDTREE = 0
idxParams = dict(algorithm=FLANN_INDEX_KDTREE, 
                 table_number=6, 
                 key_size=12, 
                 multi_probe_level=1)
searchParams = dict(checks=50)

flann = cv.FlannBasedMatcher(idxParams, searchParams) 

matches = flann.knnMatch(desc1, desc2, k=2)

thresh = 0.7
goodMatches = []

# Filters the matches using Lowe's ratio test
for m, n in matches:
    if m.distance < thresh * n.distance:
        goodMatches.append(m)
```
# Exploring Epipolar Geometry

Consider two images taken at two distinct vantage points, assuming a calibrated camera (when $K = I$, where $K$ is the calibration matrix). Let the image coordinates be $x$ and the spatial coordinates of some point $p$ be $X$, both with respect to the camera frame. Then:

\[
\lambda \, x \;=\; \Pi_0 X,
\]

where:

- $\lambda$ is the scale factor (depth),
- $\Pi_0$ is the projection $\mathbb{R}^3 \to \mathbb{R}^2$.

---

## The Epipolar Constraint

> **Theorem 3.1 — (Epipolar Constraint)**  
> For two images $x_1, x_2$ of a point $P$, seen from two vantage points, the following constraint holds:
>
> \[
> x_2^\top \,\hat{t}\,R \, x_1
> \;=\;
> x_2^\top E \, x_1
> \;=\;
> 0,
> \]
>
> where $(R, t)$ is the relative pose (position and orientation) between the two camera frames, and the **essential matrix** $E$ is defined by
>
> \[
> E \;=\; \hat{t}\,R.
> \]

### Proof

Let $X_1, X_2 \in \mathbb{R}^3$ be the 3D coordinates of a point $P$ relative to two camera frames. Then:

\[
X_2 \;=\; R\,X_1 \;+\; t.
\]

Let $x_1, x_2 \in \mathbb{R}^2$ be the projections of $P$ onto the two image planes. Hence:

\[
\lambda_2 \, x_2 \;=\; R \bigl(\lambda_1 \, x_1\bigr) + t.
\]

Multiplying both sides by $\hat{t}$ (the skew-symmetric matrix of $t$):

\[
\begin{aligned}
\lambda_2 \, x_2 
&=\; R \bigl(\lambda_1 \, x_1\bigr) \;+\; t,\\
\lambda_2 \,\hat{t}\, x_2 
&=\; \hat{t}\,R \bigl(\lambda_1 \, x_1\bigr) 
    \;+\; \hat{t}\,t,\\
\lambda_2 \,\bigl(t \times x_2\bigr) 
&=\; t \times \bigl(R\,(\lambda_1 \, x_1)\bigr) 
    \;+\; \bigl(t \times t\bigr),\\
\lambda_2 \,\bigl(t \times x_2\bigr) 
&=\; \hat{t}\,R\,\bigl(\lambda_1 \, x_1\bigr).
\end{aligned}
\]

Next, multiply on the left by $x_2^\top$:

\[
\begin{aligned}
\lambda_2 \; x_2^\top \,\bigl(t \times x_2\bigr) 
&=\; x_2^\top\,\hat{t}\,R\,\bigl(\lambda_1 \, x_1\bigr),\\
\lambda_2 \;\bigl[x_2 \cdot \bigl(t \times x_2\bigr)\bigr] 
&=\; x_2^\top \,\hat{t}\,R\,\bigl(\lambda_1 \, x_1\bigr).
\end{aligned}
\]

But $\,x_2 \cdot (t \times x_2) = 0\,$ (orthogonality of cross product), so

\[
x_2^\top \,\hat{t}\,R\, x_1 
\;=\;
x_2^\top \,( \hat{t}\,R )\, x_1
\;=\;
0,
\]

which completes the proof.

---

## Properties of the Essential Matrix and Pose Recovery

The **essential matrix** is
\[
E \;=\; \hat{t}\,R,
\]
which encodes the relative pose between two cameras (the translation $t$ and a rotation $R \in SO(3)$). We define the **essential space**:

\[
\mathcal{E}
\;=\;
\{\,\hat{t}\,R \;\mid\; R \in SO(3),\; t \in \mathbb{R}^3\}.
\]

> **Claim.**  
> A non-zero matrix $E \in \mathbb{R}^{3 \times 3}$ is an essential matrix if and only if $E$ has a singular value decomposition (SVD) of the form
> \[
> E
> \;=\;
> U\,\Sigma\,V^\top,
> \]
> where
> \[
> \Sigma
> \;=\;
> \mathrm{diag}\{\sigma,\sigma,0\}
> \;=\;
> \begin{bmatrix}
> \sigma & 0 & 0\\[6pt]
> 0 & \sigma & 0\\[6pt]
> 0 & 0 & 0
> \end{bmatrix},
> \]
> for some $\sigma > 0$ and $U, V \in SO(3)$.

### Sketch of Proof

1. $E = \hat{t}\,R$ implies certain rank and skew properties in $E$.  
2. By applying a suitable rotation, we can place $t$ in a canonical form $(0,0,\|t\|)^\top$, revealing that $E$ has two equal non-zero singular values and one zero singular value.  
3. Reversing this process shows that any matrix with exactly two identical non-zero singular values and one zero singular value can be written as $\hat{t}\,R$ with $R \in SO(3)$.

---

## Possible Pose Solution Pairs

> **Claim — (Pose recovery from $E$)**  
> There are only two possible relative poses $(R, t)$ with $R \in SO(3)$ and $t \in \mathbb{R}^3$ corresponding to a non-zero essential matrix $E \in \mathcal{E}$.

### Idea of Proof

- Suppose $E = \hat{t}_1 R_1 = \hat{t}_2 R_2$. Then

  \[
  \hat{t}_1
  \;=\;
  \hat{t}_2 \, R_2 \, R_1^\top.
  \]

- Because $R_2 R_1^\top$ is itself a rotation, a lemma about skew-symmetric transformations shows that $R_2 R_1^\top$ is either the identity or a rotation by $\pi$.  
- Hence, there are exactly two distinct $(R,t)$ pairs that give the same essential matrix $E$.

> **Remark — (Pose Recovery)**  
> Both $E$ and $-E$ satisfy the same epipolar constraints. Hence, in total, there are $2 \times 2 = 4$ possible $(R,t)$ solutions. However, only one of these yields positive depth for all points in both cameras, so the other three are physically invalid.

---

## The Eight-Point Algorithm

To solve for $E$, we collect point correspondences $(x_1, x_2)$ in normalized coordinates. Let

\[
E
\;=\;
\begin{bmatrix}
  e_1 & e_2 & e_3 \\
  e_4 & e_5 & e_6 \\
  e_7 & e_8 & e_9
\end{bmatrix}.
\]

This can be stored as a 9-vector $\,e = (e_1,\ldots,e_9)^\top$. The epipolar constraint for each pair is

\[
x_2^\top \, E \, x_1 
\;=\;
0.
\]

If we have $n$ such correspondences, we form an $n \times 9$ matrix $A$ so that

\[
A \, e
\;=\;
0.
\]

With at least 8 well-chosen correspondences (hence “eight-point algorithm”), $A$ usually has rank 8 and we solve $Ae=0$ for $e$ (up to scale). In practice, more than 8 points are used with an SVD or least-squares approach.

---

## Projection Into the Essential Space

Real-world data is noisy, so the solution $e$ from $A e=0$ often does **not** reshape into a perfect essential matrix (two identical non-zero singular values plus one zero). Therefore, we **project** $E'$ onto $\mathcal{E}$:

> **Claim — (Projection)**  
> Let $E' \in \mathbb{R}^{3 \times 3}$ have SVD
> \[
> E'
> \;=\;
> U \,\mathrm{diag}(\lambda_1,\lambda_2,\lambda_3)\,V^\top,
> \]
> with $U, V \in SO(3)$ and $\lambda_1 \ge \lambda_2 \ge \lambda_3$. The matrix $E \in \mathcal{E}$ that **minimizes** $\|\,E - E'\|_F^2$ is
> \[
> E
> \;=\;
> U \,\mathrm{diag}\!\bigl(\sigma,\sigma,0\bigr)\,V^\top,
> \quad
> \text{where}
> \quad
> \sigma 
> \;=\;
> \frac{\lambda_1 + \lambda_2}{2}.
> \]

### Proof Sketch

1. We want to minimize $\|E - E'\|_F^2$ subject to $E$ having the structure $(\hat{t}\,R)$.  
2. The best choice keeps the same $U,V$ and clamps the singular values to $(\sigma,\sigma,0)$.  
3. One checks via trace arguments that this leads to the smallest Frobenius norm difference.

---

## Triangulation and Disambiguation

After recovering $E$ and decomposing it into $(R,t)$, recall there are four possible $(R,t)$ solutions (because $E$ and $-E$ each allow two decompositions). We **triangulate** a 3D point from each solution. Only one solution yields points with positive depth in both cameras, thus identifying the physically correct $(R,t)$.

---

## Lemma 3.5 (A Skew-Symmetric Matrix Lemma)

> **Lemma 3.5.**  
> Let $\hat{T} \in so(3)$ be non-zero (thus $T \in \mathbb{R}^3$). If for some $R \in SO(3)$ the product $\hat{T}\,R$ is also skew-symmetric, then $R$ must be either the identity $I$ or a rotation by $\pi$ about the axis $u = T/\|T\|$. Furthermore,
>
> \[
> \hat{T}\,\bigl(e^{\hat{u}\,\pi}\bigr) 
> \;=\;
> -\,\hat{T}.
> \]

**Proof Idea**:

1. Assume $\hat{T}\,R$ is skew. Then $(\hat{T}\,R)^\top = -\,\hat{T}\,R$.  
2. Use the orthonormality of $R$ to see how it commutes with $\hat{T}$.  
3. Show $R$ must be a rotation by $0$ or $\pi$ around the direction of $T$.  
4. The relation $\hat{T}\,e^{\hat{u}\,\pi} = -\hat{T}$ follows.

