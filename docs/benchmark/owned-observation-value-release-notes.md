# Original Python SLAM: owned observation value optimization

Installed and checked locally on October 15, 2026 (America/Toronto), then packaged
on `codex/original-python-recovery` for main consideration. Main is unchanged.

- Store internal observation pixels as immutable owned values, with independent readonly NumPy metadata snapshots and identity replacement on explicit reassignment.
- Reuse cached physical identity only for the exact supported owned-value type and original hook; preserve ordinary mutable/custom/subclass fallback, detector hashing, geometry, thresholds and BA.
- Six repeated short comparisons and six full seed0 comparisons preserve exact control outputs/states/metrics/losses. Observed full wall reductions range3.53–26.32%; full09 resource-pause caveat remains explicit.
- Fresh independent review cleared the bounded installation. The actual installed ROOT-import checks passed; logSHA256 `dc73a6a17f85da301db856cc40b7a82c89956132a28946c3323e5a9db5521399`. ReviewSHA256 `11436d47edd69223a4c458a8da0ea0306448e92be688f1c1199c854144fc0fca`.
- Branch verification passed all 384 Python tests, four dashboard tests and a production dashboard build. Launcher smoke checks cover the bundled four-frame sample with and without telemetry; frozen stereo CLI configuration and invalid-option rejection also pass. No benchmark was rerun for this packaging step. [Verification details](owned-observation-value-branch-checks.md).
- Known internal API limit: inherited dataclass equality can raise `ValueError` even for `o == o`; no affected repository caller was found. This internal type is not a general dataclass-equality drop-in replacement. The exact reviewed source was not repaired or changed after benchmarking.
- No new control-relative accuracy, VO runtime parity, real-time, broader generalization or main-release claim. Full01 raw error remains worse than VO.

[Measured benchmarks and reproduction](owned-observation-value.md). Original source and Git status were preserved in the guarded local integration backup; existing uncommitted work and Git history were not replaced.
