# Visual SLAM: A TLDR

**Author:** Achita  
**Date:** July 2024

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

$$
\lambda x = \Pi_0 X,
$$

where:
- $\lambda$ is the scale factor (depth),
- $\Pi_0$ is the projection $\mathbb{R}^3 \to \mathbb{R}^2$.

---

## The Epipolar Constraint

> **Theorem 3.1 — (Epipolar Constraint)**  
> For two images $x_1, x_2$ of the point $P$, seen from two vantage points, the following constraint holds:
> $$
> x_2^\top \hat{t} R \, x_1 
> \;=\;
> x_2^\top E \, x_1
> \;=\; 0,
> $$
> where $(R, t)$ is the relative pose (position and orientation) between the two camera frames and the **essential matrix** $E$ is defined by
> $$
> E = \hat{t} R.
> $$

### Proof

Let $X_1, X_2 \in \mathbb{R}^3$ be the coordinates of a point $P$ relative to two camera frames, so

$$
X_2 = R X_1 + t.
$$

Let $x_1, x_2 \in \mathbb{R}^2$ be the projections of $P$ on the two image planes. Hence:

$$
\lambda_2 x_2 = R(\lambda_1 x_1) + t.
$$

Multiplying both sides by the skew-symmetric matrix $\hat{t}$:

\[
\begin{aligned}
\lambda_2 x_2 &= R(\lambda_1 x_1) + t \\
\lambda_2 \,\hat{t}\, x_2 
  &= \hat{t}\,R(\lambda_1 x_1) + \hat{t}\,t \\
\lambda_2 (\,t \times x_2\,) 
  &= t \times (\,R (\lambda_1 x_1)\,) + (\,t \times t\,) \\
\lambda_2 (\,t \times x_2\,) 
  &= \hat{t}\,R\,(\lambda_1 x_1),
\end{aligned}
\]

where we used $\,\hat{t}\,x = t \times x\,$. Next, multiply on the left by $\,x_2^\top\,$:

\[
\begin{aligned}
\lambda_2 \, x_2^\top \,(t \times x_2) 
  &= x_2^\top \,\hat{t}\, R\,(\lambda_1 x_1),\\
\lambda_2\,\bigl[x_2 \cdot (\,t \times x_2\,)\bigr] 
  &= x_2^\top \hat{t}\, R\,(\lambda_1 x_1).
\end{aligned}
\]

But $\,x_2 \cdot (\,t \times x_2\,) = 0\,$ (orthogonality of cross products). Hence,

$$
x_2^\top \,\hat{t}\, R\, x_1 
\;=\; x_2^\top\, (\hat{t} R)\, x_1
\;=\; 0,
$$

which completes the proof.

---

## Properties of the Essential Matrix and Pose Recovery

The **essential matrix** is $\,E = \hat{t}\,R,\,$ which encodes the relative pose between two cameras (the translation $t$ and a rotation $R \in SO(3)$). We define the **essential space**:

$$
\mathcal{E} 
\;=\; 
\{\;\hat{t}\,R \;\mid\; R \in SO(3),\; t \in \mathbb{R}^3\}.
$$

> **Claim.**  
> A non-zero matrix $\,E \in \mathbb{R}^{3\times 3}\,$ is an essential matrix if and only if $\,E\,$ has a singular value decomposition (SVD) of the form
> \[
> E 
> \;=\; 
> U\;\Sigma\;V^\top,
> \]
> where
> \[
> \Sigma 
> \;=\; 
> \mathrm{diag}\{\sigma, \sigma, 0\} 
> \;=\; 
> \begin{bmatrix}
> \sigma & 0 & 0\\
> 0 & \sigma & 0\\
> 0 & 0 & 0
> \end{bmatrix},
> \]
> for some $\sigma > 0$ and $U,\,V \in SO(3)$.

### Sketch of Proof

1. We start by noting that $E = \hat{t}\,R$ implies certain rank and skew-symmetric properties in $E$.  
2. By appropriate rotations, we can transform $t$ to a canonical form $(\,0,\,0,\,\|t\|)^\top$, showing that $E$ has exactly two equal non-zero singular values and a zero singular value.  
3. Reversing this process shows that any matrix with that singular-value pattern can be decomposed in the form $\,\hat{t}\,R\,$ with $\,R \in SO(3)\,$.

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
- Because $R_2\,R_1^\top$ is itself a rotation, from a lemma about skew-symmetric transformations we get that $R_2\,R_1^\top$ is either the identity or $\,e^{\hat{u}\,\pi}\,$ (rotation by $\pi$).  
- Consequently, there are exactly two distinct $(R,t)$ pairs that yield the same essential matrix $\,E$.

> **Remark — (Pose Recovery)**  
> Both $\,E\,$ and $-E$ satisfy the same epipolar constraints. Hence, there are $2\times 2 = 4$ possible $(R,\,t)$ solutions. However, only one of these 4 solutions places all scene points **in front** of both cameras (positive depth), making the other three physically invalid.

---

## The Eight-Point Algorithm

To find $\,E\,$ from point correspondences, let us define:

$$
E 
\;=\; 
\begin{bmatrix}
  e_1 & e_2 & e_3 \\
  e_4 & e_5 & e_6 \\
  e_7 & e_8 & e_9
\end{bmatrix}.
$$

We can store it as a 9-vector $\,e = (e_1, e_2, \dots, e_9)^\top\,$.  
The **epipolar constraint** for normalized image points $x_1, x_2$ is:

\[
  x_2^\top \,E\, x_1
  \;=\;
  0.
\]

Expanding yields a linear equation in the elements of $E$. If we have $\,n\,$ correspondences, we obtain

\[
  A\,e
  \;=\; 
  0,
\]

where $A \in \mathbb{R}^{n \times 9}$ is constructed from the coordinates of the matched points $(x_1, x_2)$. With at least 8 correspondences (hence the “eight-point” name), $A$ typically has rank 8, and we solve $A e = 0$ for $\,e\,$ (up to a scale factor). In practice, we use more than 8 points and solve via SVD or least squares.

---

## Projection Into the Essential Space

In reality, $e$ found by $A e=0$ might not perfectly reshape into a valid essential matrix (two identical non-zero singular values, plus one zero). We **project** onto $\mathcal{E}$:

> **Claim — (Projection)**  
> Given a real matrix $\,E' \in \mathbb{R}^{3 \times 3}\,$ with SVD $E' = U\,\mathrm{diag}(\lambda_1,\lambda_2,\lambda_3)\,V^\top,$ where $U, V \in SO(3)$ and $\lambda_1 \ge \lambda_2 \ge \lambda_3$, the essential matrix $\,E \in \mathcal{E}\,$ minimizing $\,\|E - E'\|_F^2\,$ is
> \[
> E
> \;=\;
> U 
> \,\mathrm{diag}\!\bigl(\sigma,\sigma,0\bigr)
> \,V^\top,
> \quad
> \text{where}
> \quad
> \sigma = \dfrac{\lambda_1 + \lambda_2}{2}.
> \]

### Proof Sketch

1. We want to minimize $\,\|E - E'\|_F^2\,$ subject to $\,E\,$ being in the essential space.  
2. The best choice keeps the same $U, V$ and forces the singular values to $(\sigma, \sigma, 0)$, with $\sigma = (\lambda_1 + \lambda_2)/2$.  
3. One checks that this results in the smallest Frobenius-norm difference by ensuring the middle two singular values match and the last is zero.

---

## Triangulation and Disambiguation

After recovering $E$ and decomposing it into $(R,t)$ (recall there are four possible solutions), we **triangulate** a 3D point from each image pair. Only one of these four will yield consistent, positive-depth points in front of both cameras, thus identifying the physically correct $(R,t)$.

---

## Lemma 3.5 (A Skew-Symmetric Matrix Lemma)

> **Lemma 3.5.**  
> Let $\hat{T}\in so(3)$ be non-zero (thus $T\in\mathbb{R}^3$). If for a rotation $R \in SO(3)$ the product $\hat{T}R$ is also skew-symmetric, then $R$ must be either the identity matrix $I$ or $e^{\hat{u}\,\pi}$, a rotation by $\pi$ about the axis $u = T / \|T\|$. Furthermore, $\hat{T}\,e^{\hat{u}\,\pi} = -\,\hat{T}$.

**Proof Idea**:  

1. Assume $\hat{T}R$ is skew. Then $(\hat{T}R)^\top = -\hat{T}R$.  
2. Use the fact that $R$ is orthonormal and see how $R$ commutes with $\hat{T}$.  
3. Show $R$ must be a rotation by $0$ or $\pi$ around the direction of $T$.  
4. The relation $\hat{T}\,e^{\hat{u}\,\pi} = -\hat{T}$ follows similarly.


