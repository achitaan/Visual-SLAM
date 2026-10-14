# Verified stereo depth orientation diagnosis

Compared `a275cef` verified-depth run `01-bundle-b47cacc69dca` with supported run
`01-bundle-931927b4a8fa`, using saved source archives, diagnostics, and original
stereo images. Both runs were uncached with loops disabled. This audit did not
replay the estimator or select parameters from reference trajectories.

## Independent geometric evidence

At frames 123→124, the independent stereo increments differ between policies by
approximately 0.005 degrees. The verified run's map increment instead disagrees
with its independent stereo increment by 1.094054 degrees, versus 0.084767 degrees
for the supported run. The verified map hypothesis remains inside the existing
0.5 m / 1.5 degree hard disagreement gates.

Original supported stereo depths and mutual descriptor matches give an independent
camera-frame geometry check. PnP was fitted only to even supported matches and
evaluated on odd matches. Restored depths were excluded. The original map estimator
was not rerun with these observations withheld, so this is a diagnostic comparison,
not a prospective arbitration experiment with both fits held out.

| Frames | Hypothesis | Odd-match median left error | Within 2 px |
| --- | --- | --- | --- |
| 123→124 | Exported map increment | 1.014602 px | 206/280 |
| 123→124 | Independent stereo increment | 0.590237 px | 269/280 |
| 123→124 | Supported PnP fit | 0.549436 px | 266/280 |
| 33→34 | Exported map increment | 4.823741 px | 60/235 |
| 33→34 | Independent stereo increment | 0.595463 px | 208/235 |
| 33→34 | Supported PnP fit | 0.501281 px | 210/235 |

For 123→124, the relative Rodrigues roll component is 1.193234 degrees in the map,
0.100108 degrees in independent stereo, and 0.087164 degrees in supported PnP.
For 33→34, map roll is −0.649467 degrees versus −0.083337 degrees in independent
stereo. The frame 34 angular inconsistency already exists in tracking diagnostics
before its bundle adjustment. That adjustment subsequently accepts a 0.668407
degree independent motion residual.

Recovered support changes keyframe chronology: supported begins at frames
0/10/19/29, verified at 0/9/19/25/34. At frame 9 a stereo map refinement rejection
creates a reference keyframe; frame 25 uses descriptor fallback. At 121 a map/reference
conflict creates another reference keyframe. At 124 flow-assisted map tracking
retains 498 inliers across four cells despite the roll excursion. Frame 126 finally
triggers reference fallback. Composition uses the preceding accepted world pose,
so an accurate new increment cannot remove earlier accumulated orientation error.
All final stored independent motion edges pass the existing hard gates.

Recovered depths are not uniformly harmful. Independently initialized temporal LK
with forward/backward cycle below 1 px, and motion fitted solely to original
supported descriptor observations, improves frame 19 candidate median projection
error from 3.717674 to 1.299384 px and frame 34 from 0.838636 to 0.442362 px.
Admitted points nevertheless retain large temporal outliers. Ambiguous flow or
dynamic objects can also cause these outliers; they do not prove erroneous
disparity alone.

Confidence is high that map tracking produces extra roll inconsistent with fresh
raw stereo geometry. Confidence is moderate that the changed support and keyframe
chronology trigger this path. The evidence does not identify one disparity or
photometric parameter as its cause, and no threshold adjustment is selected.

## Initial preparation

The initial pure scorer, now `src/stereo_pose_arbitration.py`, does not fit poses.
Its synthetic test constructs a 1.1 degree roll corruption whose
map geometry fits current left and right observations exactly, while immutable
previous raw stereo geometry rejects that pose. It therefore reproduces how
current map reprojection checks can miss a coherent local geometry error inside
the existing angular gate.

The scorer compares both hypotheses on the same reserved original-supported stereo
observations using mean three-component Huber cost at the existing 2 px scale.
It does not trim a different set of observations for each candidate. Replacement
requires independent training verification, existing support/ratio/spatial/median
gates, lower common cost, and no reduction in inlier count or occupied cells.
Numerical ties, missing provenance, inadequate support, or fit overlap preserve
the map choice. Nonfinite poses, invalid rotations, and negative depths fail
closed. Calibration offset is retained.

The integration must capture supported measurements before restoration in immutable
camera-frame records, reserve IDs before either solve, and exclude the reserved
observations from forward and reverse independent fits/refinements and map
descriptor/flow fits. Linked flow landmark IDs require exclusion even when no
detector feature index exists. After replacement, rejected map associations and
flow tracks must be cleared or independently revalidated before keyframe or bundle
insertion. The provenance label is a caller contract, not proof that arbitrary
input arrays originated in the supported sampler. Existing disagreement gates
remain unchanged. No production correction or trajectory improvement is claimed
for the initial helper alone. Subsequent opt-in integration and its measured
connectivity regression are documented in
[stereo arbitration diagnostics](stereo-arbitration-results.md).
