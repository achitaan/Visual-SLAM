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

