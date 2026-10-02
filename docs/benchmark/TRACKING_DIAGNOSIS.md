# Tracking and local optimization diagnosis

The shared SLAM pipeline remains experimental. Component gauge anchoring fixes a
reproduced local bundle-adjustment failure. The subsequent independent-PnP revision
improves TUM desk but causes substantial monocular KITTI 04 scale drift; it is
rejected as a default improvement. Results below distinguish retained development
revisions. Passing synthetic tests does not establish sequence-level reliability.
The current branch preserves geometrically validated seeded PnP and retries
independent PnP only when that candidate fails. Complete regression runs for this
revision are reported separately below.

Before the component-gauge fix,
stereo tracking fixes reduce full
KITTI 01 translation error from 37.63% to 10.66% and SE(3)-aligned ATE from
313.00 m to 85.27 m. Lost frames decrease from 43 to 29, but the last 28 frames
remain unresolved. The original stereo VO baseline still performs better on 01:
9.57% translation error and 72.29 m ATE. Sequence 01 contains no verified loop
closures; its failure occurs in tracking, before loop correction could help.

## Evidence

On the first 350 frames, disabling local bundle adjustment increased ATE from
90.70 m to 102.71 m and lost frames from 14 to 91. Removing bundle adjustment is
not a solution. These diagnostic replays use the same images, calibration,
feature budget and acceptance thresholds; the disabled optimizer is an explicit
ablation, not a proposed operating configuration.

Three correspondence problems were found:

- Keyframe observations retained the accepted optical-flow pixel but often
  omitted its available right-image constraint. Re-measuring disparity at that
  exact pixel increased the fraction of local-window observations with stereo
  constraints from 31.5% to 97.5%. That reduced lost frames, but barely changed
  trajectory error.
- Optical flow could overwrite a disagreeing mutual descriptor match. Searches
  also started at the previous pixel rather than its predicted projection. Both
  behaviors are vulnerable to repeated image patterns. Preserving descriptor
  matches and using the map motion prediction helped slightly, but did not solve
  the failure.
- Map PnP was accepted without checking available independent stereo motion.
  At frame 220 its median left reprojection error was 0.48 pixels, while its pose
  disagreed with independently verified stereo motion by 2.65 m. At frame 240 the
  discrepancy was 2.69 m. Low reprojection error alone does not establish correct
  landmark identity or metric motion.

The independent stereo check originally discarded a correspondence if either
frame lacked depth. Each PnP direction needs source depth and target image
coordinates; it does not need target depth. The corrected verifier evaluates
available geometry separately in each direction.

Frame tracking also reused loop acceptance criteria. In a diagnostic replay of
100 consecutive pairs (target frames 250–349), ordinary tracking PnP accepted 73
motions while the stricter bidirectional verifier accepted 43. The 30 additional
tracking motions had a median evaluator-only translation error of 0.040 m;
24 had 15–19 inliers. Dropping these measurements could leave the system using a
misleading map estimate even when usable independent motion was available.
The new temporal reference accepts 71 of these pairs, with a median translation
error of 0.039 m. Two forward estimates are excluded by the reverse consistency
check. These pair-level measurements explain acceptance behavior; they are not
a substitute for a full trajectory evaluation.

The interfaces now distinguish these purposes:

| Check | Temporal frame tracking | Loop constraint |
| --- | --- | --- |
| Minimum inliers | Existing tracker minimum: 15 | 30 |
| Minimum inlier ratio | Existing tracker minimum: 0.25 | 0.35 |
| RANSAC reprojection limit | 2 pixels | 2 pixels |
| Refined median reprojection limit | 1.5 pixels | 1.5 pixels |
| Spatial support | At least three image grid cells | At least three image grid cells |
| Reverse PnP | Reject contradictions when reverse geometry passes | Both directions must pass |
| Forward/reverse disagreement | At most 0.5 m and 1.5 degrees when checked | At most 0.5 m and 1.5 degrees |

Keyframe recovery retains bidirectional verification with at least 20 inliers.
It is invoked after tracking fails. A temporal tracking measurement cannot enter
the loop graph through the frame-tracking interface.

## Changes and validation

Stereo tracking now checks map poses against independently estimated temporal
stereo geometry. Conflicting map estimates are rejected and replaced with the
accepted reference; a new keyframe records fresh metric geometry. Projected
visibility also restricts optical-flow association. No motion prior is exported
as an accepted pose. Failed frames retain flagged held poses and do not create
map geometry.

The minimum PnP support and reprojection thresholds were not weakened. References
are opened only after estimator shutdown. Monocular estimation still receives
one image stream; stereo checks are restricted to stereo mode.

Regression tests cover exact-pixel stereo observations, conflicting flow versus
descriptor matches, rejection of a misleading map pose, independent PnP
verification with complementary depth availability, source-depth-only frame
tracking, and rejection of contradictory reverse geometry. The full backend
suite passes 78 tests, including disconnected-component gauge regressions for
stereo and monocular bundle adjustment, low-disk evaluation safeguards and an
adversarial PnP motion-prior regression.

The first 350-frame replay with independent stereo checking reduced translation
error from 33.29% to 16.07% and ATE from 90.70 m to 45.17 m. This result precedes
the directional-depth correction and remains inadequate. Intermediate full runs
reached 14.98% translation error and 122.73 m ATE, with an unresolved final lost
interval. These are improvements over the shared pipeline's initial failure,
but remain worse than the retained original stereo VO baseline on sequence 01
(9.57% translation error, 72.29 m ATE).

An intermediate full run using the directional-depth fix alone deteriorated to
63.33% translation error, 525.64 m ATE and 680 lost frames. Its final interval,
frames 429–1100, never recovered. Its acceptable 350-frame replay did not predict
the full failure. This failed run is retained; the later temporal tracking
interface and visibility changes require full-sequence validation independently.

Two experiments were rejected: using descriptor-only poses before flow worsened
the first-350-frame result to 22.77% translation error; checking recent keyframes
on every accepted pose worsened rotational drift. Neither is enabled in the
current pipeline. A lower error on one ablation does not establish general
reliability.

The retained diagnostic replays show the effect of successive changes. They are
development revisions, not a benchmark of one frozen implementation. All rows
use the first 350 stereo frames of sequence 01.

| Replay | Aligned ATE (m) | Translation error (%) | Lost frames |
| --- | ---: | ---: | ---: |
| Initial shared tracker, local BA enabled | 90.70 | 33.29 | 14 |
| Local BA disabled | 102.71 | 37.02 | 91 |
| Exact-pixel right-image observations | 91.23 | 33.04 | 7 |
| Preserve conflicting descriptor associations | 89.61 | 32.55 | 5 |
| Seed flow with predicted projections | 85.91 | 31.07 | 7 |
| Independent stereo cross-check | 45.17 | 16.07 | 12 |
| Separate available depth per PnP direction | 45.22 | 15.99 | 7 |
| Restrict flow to projected visibility | 43.30 | 16.19 | 8 |
| Descriptor-only first (rejected) | 68.14 | 22.77 | 91 |
| Recent-keyframe check on accepted tracking (rejected) | 43.55 | 16.28 | 8 |

## Bundle-adjustment gauge diagnosis

The completed tracking-reference run exposed 27.61 m and 26.43 m jumps near
frames 13 and 20. Keyframe 13 moved roughly 24 m vertically despite a small
reprojection objective. This was local bundle adjustment, not loop optimization.

Selecting at most 200 mature landmarks can leave the selected observation graph
disconnected from its oldest fixed camera. Fixing that camera then does not fix
the world gauge of a separate optimized component. An unconstrained component
can translate or rotate while reducing reprojection error. Unsupported cameras
also appeared as free solver variables with no residuals.

Local BA now determines connected components **after landmark selection**. Each
component retains a fixed camera, and monocular components retain two distinct
fixed camera centers to constrain scale. Existing fixed boundary cameras count
as anchors. Cameras without selected observations remain fixed. Stereo components
without right-image constraints receive monocular gauge handling. Components
without an observable scale baseline cannot free their camera poses.

Both disconnected-component regression tests reproduce with the retained optimizer
and pass with the fix. On the first 80 stereo frames of 01, aligned ATE falls from
7.901 m to 1.871 m and the largest exported step falls from 27.606 m to 1.897 m.
There are zero lost frames. This partial replay has no valid KITTI segment-drift
score; full runs are required before claiming a general accuracy improvement.

![First 80 frames before and after component gauge anchoring](plots/shared-bundle-gauge-check.png)

## Validation after the component-gauge fix

The following complete runs share the same estimator source and default mapping
configuration. Stereo uses SE(3)-aligned ATE and KITTI segment drift. Monocular
uses evaluator-only Sim(3) alignment; its output has arbitrary scale. Reference
poses are opened after tracking and optimizer shutdown. Learned depth is absent.
Machine-readable results and source fingerprints are retained in
[shared-component-gauge-results.json](shared-component-gauge-results.json).

| Dataset / mode | Frames | Aligned ATE (m) | Translation error (%) | Lost frames | Verified loops | Time (s) | Peak memory (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KITTI 01 / stereo | 1,101 | 87.756 | 10.901 | 30 | 0 | 2,581.9 | 1,422.2 |
| KITTI 04 / stereo | 271 | 0.764 | 1.005 | 0 | 0 | 353.3 | 334.9 |
| KITTI 04 / monocular | 271 | 0.415 | Arbitrary scale | 0 | 0 | 240.0 | 329.1 |
| KITTI 07 / stereo | 1,101 | 0.838 | 0.811 | 0 | 11 | 1,568.3 | 698.1 |
| TUM fr1/desk / RGB monocular | 613 | 0.807 | Arbitrary scale | 155 | 0 | 1,047.0 | 298.5 |

Times include concurrent evaluations on the same host. They are not isolated
throughput measurements. TUM's original saved sequence field contains the CLI
default `04`; the dataset field, inputs and case label identify fr1/desk.

The stereo 04 segment-drift regression is resolved: drift falls from 3.641% to
1.005%. Monocular 04 is unchanged because its selected observation components
were already anchored. On 07, the last applied loop correction reduces ATE from
1.883 m to 0.838 m and segment drift from 1.033% to 0.811%. This compares the
saved trajectory immediately before and after that correction; the first
trajectory includes earlier live corrections.

TUM desk remains a failure of general reliability. It loses frames 59–61 and
198–349, then recovers geometrically. Its 0.807 m aligned ATE is worse than the
retained earlier monocular revision's 0.195 m, which also lost 140 frames. No
extra gauge anchors or unsupported cameras occur in its local BA windows, so
the newly added component handling does not explain this regression. Motion
prediction is being tested through a declared diagnostic ablation; it is not a
different benchmark configuration selected for a better score.

The desk regression precedes the long loss interval. On accepted frames 4–197,
separately aligned diagnostic ATE is 0.568 m versus 0.045 m in the retained
earlier revision. The current trajectory also needs substantially different
alignment scales before and after recovery. These evaluator-only fits diagnose
nonuniform scale drift; they never change the estimator or its exported poses.
Disabling motion prediction still loses tracking near frame 200. A separate
ablation retains map/PnP prediction but starts optical flow at the previous
image location, isolating projection-based flow seeding.

The first 200 RGB frames of TUM fr1/room complete with six initializing frames,
one lost frame and one recovery. Sim(3)-aligned ATE is 0.046 m. This is a partial
check under the same estimator configuration, not a full room benchmark.

![Actual RGB views around the desk tracking loss](plots/shared-tumdesk-failure-inputs.png)

Final multi-view map quality also exposes limits beyond trajectory scores:

| Export | Multi-view observations | Median residual (px) | 90th percentile (px) | Residual above 3 px (%) | Behind camera |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 stereo | 46,555 | 0.410 | 1.137 | 1.72 | 0 |
| 04 stereo | 7,864 | 0.369 | 1.126 | 1.58 | 0 |
| 04 monocular | 18,441 | 0.281 | 1.006 | 2.60 | 0 |
| 07 stereo after correction | 65,948 | 0.790 | 3.315 | 11.89 | 0 |
| TUM desk monocular | 31,411 | 0.586 | 1.357 | 0.28 | 0 |

Sequence 07's observation residuals require further reconciliation after loop
correction. TUM's low residuals do not establish correct trajectory or scale.
Observation culling and map refinement after correction remain necessary.

The interrupted 01 replay was repeated successfully. The component-gauge fix
removes the demonstrated early jumps but does not improve the full highway
score: ATE rises from 85.268 m to 87.756 m and drift from 10.660% to 10.901%.
Frames 275–276 recover; frames 1073–1100 remain unresolved. The original stereo
VO baseline remains better at 72.29 m ATE and 9.57% drift. Low local reprojection
residuals do not establish globally accurate motion.

No score is assigned to the interrupted run. Evaluation now
checks free space before starting and reserves room for the growing map's
exports. If space becomes insufficient, it closes the estimator and exports
only processed frames, explicitly labeled partial and interrupted. Concurrent
unrelated disk writes can still exhaust this reserve.

## Independent PnP hypothesis generation

The seeded solver switched from EPNP to iterative PnP when given a motion prior.
A synthetic scene with 25% outliers and a badly rotated prior reproduces a
negative-depth rejection, while the same correspondence geometry solves correctly
without that prior. The new regression fails against the retained implementation.

The retained experimental revision always generates RANSAC hypotheses with EPNP, then refines the consensus
with LM. Motion prediction remains in projection and association. Inlier counts,
ratio, positive-depth, spatial-support and reprojection acceptance requirements
are unchanged. The optional prior argument remains accepted for caller
compatibility but does not initialize geometric verification. Diagnostics record
`pnp_method: epnp_ransac`. That revision passed 77 backend tests, but its complete
monocular regression below prevents acceptance as the default.

Declared ablations distinguish this from removing all prediction or changing
flow initialization:

| Diagnostic | Coverage | Sim(3)-aligned ATE (m) | Lost frames |
| --- | --- | ---: | ---: |
| Component-gauge default | Desk, first 100 frames | 0.176 | 3 |
| Unseeded map PnP only | Desk, first 100 frames | 0.023 | 0 |
| Unseeded map PnP only | Desk, all 613 frames | 0.199 | 141 |
| All motion prediction disabled | Desk, all 613 frames | 0.634 | 142 |
| Unseeded optical flow only | Desk, first 200 frames | 0.530 | 143 |

Rows with different coverage cannot be ranked by ATE. Unseeded flow becomes
unresolved from frame 58 onward despite its aggregate error; it is rejected as
an operating change. The default pipeline independently reproduces the full
unseeded-PnP diagnostic's trajectory and 0.199 m ATE exactly. Its sole lost
interval, frames 199–339, recovers at frame 340. The accuracy regression improves;
recovery across the viewpoint change remains unresolved as a reliability problem.
The complete independent-PnP evaluations below expose a serious regression. Their
source fingerprints are retained separately from the component-gauge revision in
[shared-independent-pnp-results.json](shared-independent-pnp-results.json).

| Input | Sensor | Frames | Aligned ATE (m) | Translation error (%) | Lost frames | Verified loops |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KITTI 01 | Stereo | 1,101 | 89.693 | 10.994 | 30 | 0 |
| KITTI 03 | Stereo | 801 | 2.154 | 1.396 | 0 | 0 |
| KITTI 04 | Stereo | 271 | 0.851 | 0.880 | 0 | 0 |
| KITTI 04 | Monocular | 271 | 14.939 | — | 0 | 0 |
| KITTI 07 | Stereo | 1,101 | 0.653 | 0.818 | 0 | 10 |
| TUM fr1 desk | RGB only | 613 | 0.199 | — | 141 | 0 |
| TUM fr1 room | RGB only | 1,362 | 0.803 | — | 970 | 0 |

Stereo accuracy uses SE(3) alignment; monocular accuracy uses Sim(3) alignment
and retains arbitrary map scale. ATE includes held poses, so it must be read with
lost intervals rather than treated as evidence of complete tracking.

Monocular KITTI 04 worsens from 0.415 m to 14.939 m ATE despite having no held
frames. Its trajectory progressively slows in map units, with three geometric
relocalizations at frames 196, 197 and 209. This is scale drift, not one large
pose jump. Low local reprojection residuals do not constrain global scale drift.
The independent-PnP change therefore cannot be accepted on the desk improvement
alone. Candidate PnP changes require both geometric invariants and complete
sequence regressions; no reference trajectory may select estimator hypotheses.

KITTI 01 still has an unresolved interval at frames 1074–1100. TUM room loses
tracking at 197–201, 204–205, 213–1002 and 1189–1361; the final interval never
recovers. Its earlier 200-frame prefix concealed the long subsequent failure.
Both indoor recovery and highway scale accuracy remain open problems.

The last applied KITTI 07 loop correction reduces ATE from 2.713 m to 0.653 m
and segment translation error from 1.162% to 0.818%. The preceding trajectory
already includes earlier live corrections. The corrected sparse map still has
8.53% of its multi-view observations above 3 pixels; trajectory correction alone
does not finish map reconciliation.

All seven runs share one estimator source and the same declared configuration.
KITTI 03 completed directly from the OneDrive dataset. Runtime includes concurrent
jobs and cloud hydration where applicable, so these timings are not isolated
throughput measurements. Original data and previous benchmark results are retained.

### Validated seeded PnP with independent fallback

The current branch preserves a seeded RANSAC/LM estimate when it passes the
existing geometry checks. If it fails, an independent EPNP RANSAC/LM candidate
must pass those same checks. Invalid or nonfinite priors also fall back to
independent geometry. Diagnostics retain the seeded rejection separately from
the accepted candidate; a successful fallback carries no pose-rejection flag.
All 78 backend tests pass, including the reproduced bad-prior case and a valid
prior/nonfinite-prior regression.

Two further diagnostics explain this conservative choice:

| Diagnostic | Coverage | Sim(3)-aligned ATE (m) | Lost frames |
| --- | --- | ---: | ---: |
| Seeded pose, independent fallback on rejection | KITTI 04, all 271 frames | 0.415 | 0 |
| Seeded pose, independent fallback on rejection | Desk, first 100 frames | 0.176 | 3 |
| Two candidates selected by truncated image reprojection objective | KITTI 04, all 271 frames | 4.711 | 0 |
| Two candidates selected by truncated image reprojection objective | Desk, first 100 frames | 0.028 | 0 |

Neither unconditional independent PnP nor choosing the lower image objective
preserves KITTI 04 accuracy. The latter is also rejected as a default change.
The fallback fixes the synthetic bad-prior failure and preserves the complete
KITTI 04 diagnostic trajectory exactly, but does not resolve desk's early loss.
No reference data enters any candidate selection. Full current-revision runs
use the ordinary evaluator and are kept separate from these declared diagnostics.

The ordinary evaluator has also completed all 271 monocular KITTI 04 frames with
the current fallback implementation: 0.415 m Sim(3)-aligned ATE, initialization
at frame 3 and no lost frames. Its exported trajectory is exactly identical to
the component-gauge reference. Wall time was 264.78 seconds and peak process
memory 328.23 MiB under concurrent testing. This establishes preservation of
that sequence, not general monocular reliability. Stereo KITTI 04 also completes
all 271 frames with zero loss: 0.764 m SE(3)-aligned ATE and 1.005% segment
translation error. Wall time was 387.09 seconds and peak memory 339.58 MiB.
Both runs share the same estimator source fingerprint, retained in
[shared-validated-pnp-results.json](shared-validated-pnp-results.json).

| KITTI 04 | Stereo | Monocular |
| --- | ---: | ---: |
| Frames | 271 | 271 |
| Initialization frame | 0 | 3 |
| Lost frames | 0 | 0 |
| Sparse landmarks | 36,317 | 5,759 |
| Map scale | Metric | Arbitrary |
| Evaluation alignment | SE(3), fixed scale | Sim(3), fitted scale |
| Aligned ATE (m) | 0.764 | 0.415 |
| Segment translation error (%) | 1.005 | Not reported |

The monocular ATE does not establish better metric accuracy: evaluation fits
its scale. More landmarks also do not establish a more accurate reconstruction.
The comparison shows each original exported map in its native units, then
aligned position error under the declared alignment for that sensor mode.

![Stereo versus monocular complete KITTI 04 comparison](plots/shared-sensor04-comparison.png)

![Current validated-PnP monocular KITTI 04 complete run](plots/shared-validated-pnp04-monocular.png)

![Current validated-PnP stereo KITTI 04 complete run](plots/shared-validated-pnp04-stereo.png)

The complete current TUM desk rerun reproduces the component-gauge result:
0.807 m Sim(3)-aligned ATE and 155 lost frames, with recovered intervals at
59–61 and 198–349. Independent fallback preserves KITTI 04 and fixes the
adversarial synthetic prior failure, but has not solved the indoor tracking
regression. This result is retained in the same current-revision JSON.

See [stereo and monocular coverage](SENSOR_COMPARISON.md) for the complete paired
KITTI 04 comparison and the pending 00–10 batch. The older all-sequence stereo
baseline remains a separate result group.

## Tracking-reference validation before the component-gauge fix


Full sequence 01 and 07 runs and stereo/monocular sequence 04 regressions use the
same estimator source and default configuration for each sensor mode. Source
fingerprints agree across all four runs. Runtime is wall time with concurrent
evaluations on the same host, rather than an isolated
throughput benchmark. Stereo ATE uses SE(3) alignment; monocular ATE uses Sim(3)
alignment and cannot establish metric scale.

| Sequence / mode | Frames | Aligned ATE (m) | Translation error (%) | Lost frames | Verified loops | Time (s) | Peak memory (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 / stereo | 1,101 | 85.268 | 10.660 | 29 | 0 | 1,508.5 | 1,586.5 |
| 04 / stereo | 271 | 0.757 | 3.641 | 0 | 0 | 263.6 | 344.3 |
| 04 / monocular | 271 | 0.415 | Arbitrary scale | 0 | 0 | 176.6 | 329.0 |
| 07 / stereo | 1,101 | 0.809 | 0.883 | 0 | 10 | 1,013.0 | 792.1 |

Sequence 04 monocular initialization takes three initializing frames and first
tracks at frame 3. Its alignment scale of 4.033 is computed by the evaluator only.
At this earlier revision, sequence 04 stereo has lower ATE but worse segment drift than the intermediate
independent-stereo run (1.123 m ATE, 1.183% drift). This regression prompted the
component-gauge investigation and is resolved in the newer run above.

Sequence 07's final loop correction reduces ATE from 2.605 m to 0.809 m and
translation error from 1.387% to 0.883%. These values compare the saved trajectory
immediately before and after the **last applied correction**. The first contains
earlier live corrections; it is not the original unoptimized VO trajectory.

Final-map checks use left-image reprojection over landmarks observed in at least
two keyframes. Single-view stereo landmarks are excluded because their creation
pixel has near-zero residual by construction.

| Export | Multi-view observations | Median residual (px) | 90th percentile (px) | Residual above 3 px (%) | Behind camera |
| --- | ---: | ---: | ---: | ---: | ---: |
| 04 stereo | 8,992 | 0.410 | 1.462 | 3.27 | 0 |
| 04 monocular | 18,441 | 0.281 | 1.006 | 2.60 | 0 |
| 07 stereo after correction | 66,239 | 0.738 | 2.794 | 8.65 | 0 |

Persistent observation culling remains necessary, particularly after correction.
The stereo 04 drift regression, remaining 01 failures, and expensive retrieval
while lost prevent treating these changes as a general reliability milestone.

Sequence 01 loses frame 299 and recovers, then remains lost from 1073 through
1100. Independent adjacent-frame checks accept 40 of the last 51 pairs, with
0.098 m median translation error. The onset is nevertheless weak: frame 1073
has insufficient correspondences, frame 1074 only ten PnP inliers, and frame 1075
has 16 inliers confined to two spatial cells. These correctly fail the unchanged
acceptance tests. Later accepted adjacent motion does not by itself reconnect
to the last verified map pose. Recovery must establish that connection
geometrically; integrating through the missing interval would conceal the loss.

At this revision the identified priorities were recovery across these short unsupported intervals, with
better landmark association and depth-uncertainty handling, followed by observation
culling and the stereo 04 drift regression. Full 00–10 and RGB-only TUM validation
must use a frozen configuration before this pipeline replaces the original VO.

## Visual evidence

These views show the completed component-gauge runs, including the unresolved
TUM failure. Accepted reprojection traces break across lost frames rather than
joining measurements across an unsupported interval.

![Sequence 01 stereo after component-gauge correction](plots/shared-gauge01-stereo.png)

![Sequence 04 stereo after component-gauge correction](plots/shared-gauge04-stereo.png)

![Sequence 04 monocular after component-gauge correction](plots/shared-gauge04-monocular.png)

![Sequence 07 stereo after live loop correction](plots/shared-gauge07-stereo.png)

![TUM fr1 desk RGB-only monocular failure](plots/shared-gauge-tumdesk-monocular.png)

### Independent PnP revision

These views use the retained independent-PnP revision, with no diagnostic overrides.
Stereo 04 retains zero tracking loss. Its segment drift improves from 1.005% to
0.880%, while aligned ATE worsens from 0.764 m to 0.851 m. Both changes are retained.

![Stereo 04 independent-PnP revision](plots/shared-pnp04-stereo.png)

![Stereo 01 independent-PnP revision with unresolved final tracking loss](plots/shared-pnp01-stereo.png)

![Stereo 03 complete OneDrive input run](plots/shared-pnp03-stereo.png)

![Monocular 04 scale-drift regression](plots/shared-pnp04-monocular.png)

![Stereo 07 after verified live loop corrections](plots/shared-pnp07-stereo.png)

![TUM desk default independent-PnP revision](plots/shared-pnp-tumdesk-monocular.png)

![TUM room complete run with long tracking losses](plots/shared-pnp-tumroom-monocular.png)

![First 100 desk frames under the declared unseeded-PnP diagnostic](plots/shared-pnp-desk-prefix.png)

### Earlier tracking-reference revision

The plots use actual saved trajectories and maps. The sequence 01 comparison
shows development revisions; the four individual views use the final frozen
source before the subsequent component-gauge fix. No learned depth participates
in these tracking runs.

![Sequence 01 tracking comparison](plots/shared-tracking01-comparison.png)

![Sequence 01 stereo trajectory, error and map](plots/shared-tracking01-stereo.png)

![Sequence 04 stereo trajectory, error and map](plots/shared-tracking04-stereo.png)

![Sequence 04 monocular trajectory, error and map](plots/shared-tracking04-monocular.png)

![Sequence 07 stereo trajectory, error and map after live correction](plots/shared-tracking07-stereo.png)

## Reproduction

Paths below are supplied by the operator; datasets and references are separate
from estimator interfaces.

```powershell
.\.venv\Scripts\python scripts/diagnose_shared_tracking.py --data-root $data --poses-root $reference --sequence 01 --max-frames 350 --output results/diagnosis01
.\.venv\Scripts\python scripts/diagnose_shared_tracking.py --data-root $data --poses-root $reference --sequence 01 --max-frames 350 --disable-bundle --output results/diagnosis01-no-bundle
.\.venv\Scripts\python scripts/evaluate_shared_slam.py --data-root $data --poses-root $reference --sequence 01 --stereo --output results/slam01
.\.venv\Scripts\python scripts/diagnose_stereo_pairs.py --data-root $data --poses-root $reference --sequence 01 --start 250 --stop 350 --loop-min-inliers 20 --output results/pair-diagnosis.json
.\.venv\Scripts\python scripts/plot_shared_slam.py --run results/slam01 --reference "$reference/01.txt"
.\.venv\Scripts\python scripts/evaluate_shared_slam.py --dataset tum --sequence fr1-desk --data-root $desk --poses-root $desk --output results/slam-desk
.\.venv\Scripts\python scripts/diagnose_motion_tracking.py --disable-motion-prediction --dataset tum --sequence fr1-desk --data-root $desk --poses-root $desk --output results/diagnostic-desk-last-pose
.\.venv\Scripts\python scripts/run_shared_benchmark.py --data-root $data --poses-root $reference --sequences 00 01 04 07 --modes stereo mono
```

Diagnostic reports include configuration, source fingerprints, per-frame pose
support, stereo disparity agreement and optimizer corrections. Full evaluation
exports preserve the source snapshot, trajectory, tracking states and sparse map.
Previous failed results are retained for comparison.

## Current full paired KITTI 01

The frozen validated-PnP revision now has full stereo and monocular results for 01. Stereo ATE is 87.756 m with 30 lost frames and the last 28 frames unresolved. Monocular ATE is 536.097 m under evaluation-only Sim(3) alignment; 879 frames are lost, with no recovery from frame 223 onward. The accepted-pose reprojection medians (0.687 px stereo, 0.547 px monocular) exclude failed poses and must not be interpreted as overall reliability.

The monocular trigger at frame 223 is `insufficient_spatial_support`: 50 candidate inliers, 0.439 px median error, two occupied image cells. Frames 224–226 also fail spatial support. Frame 213 previously failed inlier support and recovered at 214. Diagnose spatial distribution and landmark correspondence persistence around 210–230 after the running frozen batch finishes; do not reduce the coverage requirement just to accept these candidates. Preserve this revision as failure evidence and reevaluate any general fix against both the highway failure and the already-working 04 pair and TUM cases.

Current results are a release hold, not a successful acceptance run. KITTI 07 separately failed calibration I/O before estimation; a readable retained local image cache is prepared for a sequential retry after the active batch. See [sensor comparison](SENSOR_COMPARISON.md) and its paired plot for the complete metrics and failure intervals.
