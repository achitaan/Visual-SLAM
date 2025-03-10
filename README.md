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
