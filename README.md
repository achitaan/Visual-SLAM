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

## The Epipolar Constraint

Consider two images taken at two distinct vantage points, assuming a calibrated camera (When $K=I$, where $K$ is the calibration matrix), the image coordinates $x$ and the spatial coordinates $X$ of some point $p$, with respect to the camera frame is:

$$\lambda x=\Pi_0 X.$$  

Where:
- $\lambda$: is the scale factor of depth  
- $\Pi_0$: is the projection from $\mathbb{R}^3 \to \mathbb{R}^2$

### The Epipolar Constraint

> **Theorem 3.1—(Epipolar Constraint):**  
> For two images $x_1, x_2$ of the point $P$, seen from two vantage points, satisfy the following constraint:
>
> $$x_2^\top\hat{t}Rx_1 = x_2^\top Ex_1=0$$
>
> Where $(R, t)$ is the relative pose (position and orientation) between the two camera frames and the essential matrix:
>
> $$E = \hat{t}R$$

#### Proof
Define $X_1, X_2 \in \mathbb{R}^3$, the 3-D coordinates of a point $P$ relative to the two camera frames respectively, then they are related by:

$$X_2 = RX_1+t$$

Let $x_1, x_2 \in \mathbb{R}^2$ be the projection of the same two points $P$ in the image planes, then:

$$\lambda_2 x_2 = R\lambda_1x_1+t$$

Multiplying both sides by $\hat{t}$:

\[\begin{aligned}
\lambda_2 x_2 &= R\lambda_1x_1+t \\
\lambda_2 \hat{t} x_2 &= \hat{t}R\lambda_1x_1+\hat{t}t \\
\lambda_2 (t \times x_2) &= \hat{t}R\lambda_1x_1+t\times t \\
\lambda_2 (t \times x_2) &= \hat{t}R\lambda_1x_1
\end{aligned}\]

Multiplying both sides by $x_2^\top$:

\[\begin{aligned}
\lambda_2 x_2^\top(t \times x_2) &= x_2^\top\hat{t}R\lambda_1x_1 \\
\lambda_2 x_2\cdot(t \times x_2) &= x_2^\top\hat{t}R\lambda_1x_1
\end{aligned}\]

Since $t \times x_2$ produces a vector orthogonal to both operands:

$$x_2^\top\hat{t}R\lambda_1x_1 = x_2^\top\hat{t}Rx_1 = 0$$

## Properties of the Essential Matrix and Pose Recovery

The essential matrix $E = \hat{t}R$ encodes the relative pose between the two cameras defined by the relative position $t$ and the orientation $R \in SO(3)$. Matrices of this belong to a set of matrices in $\mathbb{R}^{3\times 3}$ called the essential space and denoted by $\mathcal{E}$:

$$\mathcal{E} = \{\hat{t}R | R \in SO(3), t\in\mathbb{R}^3\}$$

> **Claim:**  
> A non-zero matrix $E \in \mathbb{R}^{3\times 3}$ is an Essential matrix iff $E$ has a singular value decomposition (SVD) of:
>
> $$E = U\Sigma V^\top$$
>
> Where:
>
> $$\Sigma = diag\{\sigma, \sigma, 0\} = \begin{bmatrix} \sigma & 0 & 0\\ 0 & \sigma & 0 \\ 0 & 0 & 0 \end{bmatrix}$$
>
> For some $\sigma \in \mathbb{R}^+$ and $U, V \in SO(3)$.

## Possible Pose Solution Pairs

> **Claim — (Pose recovery from the essential matrix):**  
> There exist only two relative poses of $(R, t)$ with $R \in SO(3)$ and $t\in \mathbb{R}^3$ corresponding to a non-zero essential matrix in $E\in \mathcal{E}$.

#### Proof
Assume there exists more than one solution pair for the decomposition of $E=tR$. Define the solution set:

$$S_n = \{(R_1, t_1), (R_2, t_2), \ldots (R_n, t_n)\}$$

where $(R_k, t_k) \in SE(3), \forall k \in \mathbb{N} $ s.t. $k \leq n$.

To prove that no such $n$ other than $n=2$ can exist, consider:

$$E = \hat{t_1}R_1 = \hat{t_i}R_i$$

where $0 < i \leq n$.

From here, we can manipulate the equality to get:

$$\hat{t_1} = \hat{t_i}R_iR_1^\top$$

Since $R_iR_1^\top$ is a rotation matrix, then $\hat{t_i}R_iR_1^\top \in SE(3)$. From **Lemma 3.5**, we know:

$$R_iR_1^\top = I \text{ or } e^{\hat{u}\pi}$$

where $u=\frac{T}{||T||}$. This means the only two distinct solutions are:

> $$(\hat{t}_1, R_1) = (UR_z(\frac{\pi}{2})\Sigma U^\top, UR_z^\top(\frac{\pi}{2})V^\top)$$  
> $$(\hat{t}_2, R_2) = (UR_z(-\frac{\pi}{2})\Sigma U^\top, UR_z^\top(-\frac{\pi}{2})V^\top)$$

Thus, it is easy to verify that $E$ is an essential matrix.

> **Remark — (Pose Recovery):**  
> Since both $E$ and $-E$ satisfy the same Epipolar constraints, there are $2 \times 2 = 4$ possible pairs of poses. However, only one guarantees that all depths of each point are positive, meaning the other three poses are physically impossible.

## The Eight-Point Algorithm

The essential matrix $E$ is the matrix associated with the epipolar constraint. We can define $E\in SE(3)$ as:

$$E = \begin{bmatrix} e_1& e_2& e_3\\ e_4& e_5& e_6\\ e_7& e_8& e_9 \end{bmatrix}$$

From this, we can define a vector $e \in \mathbb{R}^9$ containing the elements of $E$:

$$e = [e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8, e_9]^\top$$


