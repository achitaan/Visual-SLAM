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

## Offline scoring supplement

`scripts/supplement_monocular_scores.py` adds versioned diagnostics to a saved
full-fit scoring report without loading an estimator or refitting any trajectory:

```sh
python scripts/supplement_monocular_scores.py --scores /path/to/scores.json \
  --runs /path/to/control /path/to/candidate --output /path/to/new-supplement.json
```

Run directories must match the report's folder identities and order. The caller
must independently verify matched dataset/frame identities, original full-fit
alignment provenance and terminal processing before interpreting a comparison.
The supplement hashes its score and tracking inputs and refuses to overwrite an
existing output. It does not replace historical reports or make a gate verdict.

For each online/corrected export, every run uses the same intersection of accepted
**and associated** input frames. RMSE and p95 reuse saved residuals; p95 uses NumPy's
linear percentile (`index = (n-1)*0.95`). Empty intersections or unavailable fits
are unassessable, never zero-error passes. Malformed residuals are rejected.

For each lost-to-accepted transition, the persistence diagnostic counts accepted
states in event+1 through event+30, excluding the recovery frame. Fewer than 27 of
30 flags unstable continuation, not a false recovery. Short terminal windows are
censored, with no stable/unstable verdict. This does not supply missing per-row
geometric recovery audits, rotation/RPE measurements or real-data benefit evidence.

### Orientation and corrected relative-pose audits

`audit_monocular_rotations.py` and `audit_monocular_rpe.py` accept the same
`--scores`, ordered `--runs` and fresh `--output`, plus `--dataset` (the TUM
folder) and `--sequence` (for example `freiburg1_room`). Both validate sequence
labels, reconstruct the historical nearest-within-20-ms unique ground-truth
association, require agreement with saved associated frame IDs, hash inputs and
refuse output replacement. Caller review of the original fit/dataset provenance
remains necessary; matching labels alone are not proof of that provenance.

The orientation audit left-multiplies camera-to-world orientations by the saved
full-fit alignment rotation. It reports angular errors for all, accepted and
common-accepted frames, plus recovery events; no orientation/subset refit is
performed. Quaternions and proper rotation matrices are validated.

The relative-pose audit is **corrected-export only**. Its prospective endpoint
rule selects the nearest full-input timestamp to start+1.0 seconds, within 20 ms,
with ties going to the earlier frame. Missing associated endpoints are excluded,
not replaced by a different pair. No interpolation is performed. It reuses the
saved full-fit Sim(3) scale; global rotation/translation cancel in the relative
transform. Errors are metres/degrees, not distance-normalized KITTI drift.
Common-accepted comparisons use identical associated pair IDs with both endpoints
accepted in every run. Empty/missing comparisons are unassessable.

Online poses lack common correction-epoch transforms, so online RPE is explicitly
unassessable. These audits neither establish completion/physical correctness nor
supply a false-recovery threshold, promotion verdict or candidate benefit claim.

### Recorded process-latency strata

`summarize_monocular_latency.py --runs /path/to/run [...] --output /path/to/new.json`
joins recorded process timings to tracking rows by exact integer frame ID. It
reports count/median/p95/maximum for each state and recovery/nonrecovery strata,
plus unmatched IDs and input hashes. Timing gaps do not change recovery detection:
the prior **input frame's** tracking state is required; a missing predecessor is
unassessable. Exact joins do not prove full dataset completion. Empty strata have
null metrics, and existing output files cannot be overwritten.

These are `SharedSlam.process` wall times, including synchronous waits and worker
contention but excluding load/export/shutdown—not end-to-end or stage timings.
Process-time map size and lock costs remain unavailable; neither `map_revision`
nor matched `tracked_landmarks` is a substitute. No speed benefit is inferred.
