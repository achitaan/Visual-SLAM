# Original Python SLAM: owned observation values

Status: independently reviewed for bounded local installation, installed and verified by 75 ROOT-import checks. Packaged on `codex/original-python-recovery`; main is unchanged. The independent review covers the two-file optimization installation, rather than the entire restored branch. [Branch verification](owned-observation-value-branch-checks.md) separately records the full test suite and dashboard build.

This project’s Python shared mapper stores internally committed pixel observations as immutable values and reuses their physical identity. Each public pixel read receives an independent readonly NumPy snapshot; explicit reassignment replaces the value and identity. Ordinary mutable observations, custom hooks and subclasses keep their original fallback. Geometry, detector hashing, thresholds and bundle adjustment are unchanged.

Six full seed0 comparisons preserve byte-identical control final poses, emitted poses and sparse maps, with identical tracking states, metrics and losses. Accuracy below compares the existing mapper result with VO. Runtime improvement compares the candidate with the same mapper before this optimization.

| KITTI | Frames | VO ATE m | Mapper ATE m | VO wall s | Control wall s | Candidate wall s | Less wall vs control | VO / mapper lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | 4541 | 20.609743 | 7.422161 | 2278.250 | 9234.297 | 6803.422 | 26.32% | 0 / 0 |
| 01 | 1101 | 72.292807 | 71.971679 | 522.047 | 2162.969 | 1757.578 | 18.74% | 23 / 1 |
| 03 | 801 | 2.327133 | 1.679271 | 444.219 | 717.547 | 678.625 | 5.42% | 0 / 0 |
| 04 | 271 | 1.763174 | 0.535873 | 160.109 | 268.437 | 258.969 | 3.53% | 0 / 0 |
| 07 | 1101 | 3.913394 | 2.402844 | 572.813 | 1211.797 | 1055.594 | 12.89% | 0 / 0 |
| 09 | 1591 | 5.133404 | 3.957941 | 804.891 | 1995.094 | 1736.828 | 12.95% | 0 / 0 |

Repeated short tests use seeds0/1/2 on01/350 and04/80: median wall reductions9.262529% and3.523868%, with3/3 improving pairs on each sequence. Full numbers above are individual seed0 comparisons, not repeated full-sequence medians.

**Limits:** Full01 raw ATE is292.754327m versus VO170.982438m, despite the small aligned ATE improvement. Candidate runtime remains above VO on every full sequence. Full09 timings span preserved resource stops; resource-equivalent timing or an isolated causal speedup is not established by source identity. The independent review accepts full09 as correctness and qualified observed timing evidence, not an attributable causal speedup. No new accuracy gain relative to control, real-time claim, unseen-sequence guarantee or EuRoC acceptance is made.

ATE uses SE(3) alignment without scale fitting. Translation and rotation drift, raw ATE, losses, runtimes and peak process RSS are included in [the CSV](owned-observation-value.csv) and [JSON](owned-observation-value.json). Every original frame remains in coverage. GT is evaluator-only.

## Reproduction

Measured environment: Windows64bit, Python3.12.14, NumPy2.5.3, OpenCV5.0.0, SciPy1.18.1, Torch2.7.1+cu126, GTX1660SUPER. Source-file hashes in the JSON identify the tested uncommitted tree; the base Git commit alone does not identify it. The uncached configuration uses1500features, one OpenCV thread, CUDA matching, stereo and precise bounded-fixed BA, loops off, and the frozen reference/depth/gauge options below.

Example equivalent evaluator command after installing the source identified in the JSON. Use a new empty output directory and your own dataset/GT roots. Recorded complete commands, case/runtime/input identities and raw proof hashes remain in the local campaign review archive; datasets and GT are excluded.

```powershell
$data = '<KITTI_ROOT>'
$poses = '<KITTI_POSES_ROOT>'
.\.venv\Scripts\python -B -u scripts/evaluate_shared_slam.py `
  --data-root $data --poses-root $poses --sequence 03 --output results/repro-owned-value-03 `
  --max-wall-seconds 2380 --stereo --loop-mode off --features 1500 `
  --matching-backend cuda --opencv-threads 1 --stereo-depth-policy verified_fallback `
  --stereo-pose-arbitration --stereo-raw-reference-retry --stereo-descriptor-association-retry `
  --stereo-map-depth-policy verified --bundle-landmark-policy bounded_fixed `
  --bundle-solver-accuracy precise --stereo-bundle-gauge-mode veto `
  --stereo-verified-reference-retry --min-free-mb 1536 --seed 0
```

The reported wall times are whole fresh case times from the serial campaign harness; this standalone evaluator example does not reproduce its admission, preparation and process-monitoring envelope. Do not compare internal evaluator time directly with whole-case time. Reruns require adequate commit/output headroom and fresh output paths; no completed comparison was replayed for installation.

## Evidence scope

The repository evidence directory contains the completed short/final/earlier full verification records and the source manifest. Raw run logs, full maps, input images and resource journals remain in the local campaign; their hashes are recorded rather than implying they are bundled. Failed resource runs and the rejected earlier shared-array prototype remain preserved separately. Same-agent metric recomputation is distinct from the independent installation review.

Internal API limitation: inherited dataclass equality is unsupported for `OwnedObservation`; even `o == o` may raise `ValueError` because pixel reads return distinct arrays. No affected caller was found in the reviewed repository. The type is not a general equality-compatible `Observation` replacement; the exact benchmarked source is unchanged.

[Authentic independent review](owned-observation-value-evidence/owned-value-independent-review33.md) · [Installed check record](owned-observation-value-evidence/owned-value-integration.json) · [Installed test log](owned-observation-value-evidence/installed-checks.log).
