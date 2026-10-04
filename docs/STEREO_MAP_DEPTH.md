# Verified stereo depth for map geometry

This is an experimental, opt-in mapping policy. The default `inherit` setting
preserves the existing map-depth behavior. Enable the verified policy only for
stereo shared-SLAM runs:

```text
python src/main.py --slam --stereo --stereo-map-depth-policy verified ...
python scripts/evaluate_shared_slam.py --stereo --stereo-map-depth-policy verified ...
```

The verified provider uses the current left and right images to measure depth
for keyframe map-landmark initialization and to measure right-image
coordinates at the actual accepted flow pixels stored in existing landmark
observations. It does not reuse a nearby detector feature's right coordinate.
If a correspondence fails verification or its resulting depth is outside the
calibrated range, it contributes no new stereo map landmark; an existing
landmark can still retain its left-image observation without a right-image
coordinate.

The provider uses the existing full-range NCC verifier and search settings:
11-by-11 patches, 96-pixel maximum disparity, minimum correlation 0.85,
uniqueness ratio 0.7, 0.5-pixel round-trip limit, and the existing texture
check. Its disparity search interval is derived from the stereo calibration
for depths from 0.1 m to 100 m, with accepted disparity below 96 pixels.
These settings and depth bounds are unchanged.

This policy is limited to map geometry and map-observation right coordinates.
Frontend depth extraction, pose acceptance, reference fitting, the original
keyframe depth-point inputs, bundle objective, and gauge handling keep their
configured behavior. Run configuration records `stereo_map_depth_policy`; a
verified run labels the map provider as
`verified_right_image_correspondence` and records its per-frame verification
counts.

The 16 focused tests and 669 backend tests passed with frozen source. On KITTI
01's first 128 frames, verified map depth reduced position error 2.8%, translation
drift 4.2% and rotation drift 24.0% against the shared control, with zero lost
frames. Position error remained 15.8% and translation drift 10.2% worse than
previous VO, so the acceptance gate failed. The option remains experimental and
no wider run was admitted. [Results and plots](benchmark/STEREO_MAP_DEPTH.md)
include the final-report save failure and validated control-artifact reuse.

Map geometry and available landmark identities can change subsequent map-based
tracking, even though the frontend acquisition and acceptance rules stay the
same. Right-image matches do not establish sensor truth or independent noise.
Single-run timings do not establish a speed improvement or real-time operation.
