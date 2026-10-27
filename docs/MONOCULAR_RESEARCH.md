# Default-off monocular research checkpoint

This branch contains isolated research options, not a validated accuracy or
speed release. All four options default to `False`; enabling any of them with
stereo input is rejected by both the library and evaluator.

| Option | Scope |
| --- | --- |
| `monocular_refined_consensus` | Recheck original RANSAC rows against the final refined pose; at most one extra refinement per hypothesis. |
| `monocular_recovery_fallback` | Rank usable distinct 3D support pools and limit PnP probes to five per pass, ten per call. |
| `monocular_covisible_triangulation` | Preserve recent-reference order, then add at most `bundle_window` older reciprocal-observation references. |
| `monocular_scale_safe_triangulation` | Separate numerical prerequisite: normalize finite nonzero rays without a fixed world-length denominator floor. |

The planned isolated contrasts are consensus versus all-off, recovery versus
all-off, numerical versus all-off, and covisible-plus-numerical versus numerical.
No automatic sweep, combination, or default-on promotion is implied.

## Selected correctness checks

From the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_monocular*.py \
  tests/test_scale_safe_triangulation.py tests/test_shared_slam.py \
  tests/test_physical_landmark_identity.py
```

The October 26 checkpoint passed **196 tests with one strict known xfail**.
This is a selected set, not the full project suite. The xfail documents the
unchanged consensus-off baseline's stale post-LM inlier mask; recovery-only
does not silently enable refined consensus and inherits that limitation.

New mapping tests exercise real keyframe insertion and triangulation with
analytic descriptor pairing: recent-before-older ordering, bounded references,
physical aliases, reciprocal owned observations, one subsequent keyframe and
stale support. Narrow recovery tests cover replacement-object rollback and one
controlled competing real map correction under an `RLock`. These do not prove
general concurrency safety, long-horizon retention or trajectory quality.

## Bounded launcher preparation

`scripts/run_monocular_condition.py` uses the tracked
`scripts/run_monocular_instrumented.py` harness. Instrumentation retains two
clock calls per `SharedSlam.process` invocation, excluding loading, export and
shutdown from process-only latency. Harmless fake-evaluator subprocess tests
cover successful and failed-frame timing export; they do not run SLAM data.

The launcher is **currently interlocked against execution** because prior shared
CPU ownership has not been authoritatively released. A JSON field or local
receipt hash cannot clear that source-level interlock. Only a reviewed follow-up
after actual terminal evidence and explicit resource-owner release may clear it.

After that separate decision, an admission file must match one condition/seed,
the complete desk/room ranges, an owner and UTC deadline, positive scene estimates
and uncertainty, justified shutdown/export and terminal-record reserves, and
hash-bound local resource receipts. Source/data/config/evaluator identities and
the truth and authority of receipts still require owner review; parsing a file
does not establish those facts.

Before each child, the wired budget guard checks actual remaining time against
the full-scene estimate plus uncertainty and both reserves. It also enforces
the evaluator's **5.25 GiB startup floor**: 5 GiB plus its existing 256 MiB export
reserve. Preparation/log/disk I/O finishes before final child-budget allocation;
the exact launched command is recorded after terminal acknowledgement. Failed,
timed-out, incomplete or malformed output stops the batch without starting the
next scene. Completion requires matching dataset/sequence, an integer full frame
count, explicit full coverage with no processing/storage interruption, exhaustive
nonnegative integer state counts, and finite nonnegative peak memory. Complete
initialization failures and tracking-loss outcomes remain valid processing
results, not quality passes. Malformed exports remain on disk while terminal
receipts record rejection without copying unsafe values. Existing results are
never overwritten.

Neither mocked timeout tests nor the owned-child wait establish an upper bound
for platform transport loss, uninterruptible I/O or shutdown. Those remain
external admission concerns. Harness-v2 batch bookkeeping differs from earlier
local runs, so historical aggregate timing is not automatically a compatible
speed control. No enabled full-sequence benefit is claimed by this checkpoint.
