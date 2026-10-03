# Stereo bundle observation-model experiment

This branch tests a uniform approximate image-noise model inside the existing
persistent-landmark bundle adjustment. It adds no pairwise pose priors and remains
disabled by default. The rejected motion-regularizer branch is separate.

The default `left_right` mode keeps the existing left-u, left-v and right-u
residuals. Select `left_disparity` with `--stereo-bundle-residuals left_disparity`
alongside `--slam --stereo`, or `--stereo` in the shared evaluator. Its third row
is `fx * baseline / z + disparity_offset - (measured_left_u - measured_right_u)`.
The first two image residuals, row count, point variables and componentwise
two-pixel Huber loss remain the same.

This intentionally changes the objective by assuming independent equal one-pixel
errors in left-u, left-v and disparity. It is **not a calibrated covariance
correction**. Supported depth samples derive right-u from interpolated disparity;
verified fallback samples use a separate image search. Bilinear sampling and
fallback correlations are not modeled by this uniform assumption. Objective
values from the two modes therefore cannot be compared as the same likelihood.

Both selected landmarks and affected unoptimized observations use the declared
mode. The fixed origin, component scale gauges, positive depth, finite solution,
revision checks and existing stereo-motion agreement limits remain unchanged.
No ground-truth or reference-depth input reaches estimation. The default path
and sensor interfaces remain available for comparisons.

Run records declare the mode and approximate-noise provenance. Synthetic and
same-input replay evidence must establish behavior before wider validation. No
KITTI accuracy improvement is claimed by this implementation alone.
