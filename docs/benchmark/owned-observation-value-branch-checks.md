# Original Python SLAM: branch verification

The restored Python engine and retained owned-observation optimization are
packaged on `codex/original-python-recovery`. No additional benchmark, trajectory
replay or threshold tuning was performed for this step. Main is unchanged.

| Check | Result |
|---|---|
| Full Python suite | 384 passed in 54.40 seconds |
| Dashboard tests | 4 passed |
| Production dashboard build | Passed, including TypeScript and static page generation |
| Installed optimization checks, completed previously | 75 passed against the frozen original control |
| Source identity | All 43 installed Python modules match the reviewed candidate manifest |

The Python suite includes four-frame bundled launcher smoke checks with and
without a real local telemetry connection, configuration agreement with the
verified stereo profile, and rejection of incompatible stereo options. These
are functional checks; they do not establish new accuracy or timing results.
The previous four-image real stereo application proof remains the stereo
integration evidence. Benchmark measurements and their limitations remain in
[the measured report](owned-observation-value.md).

## Commands and environment

Windows, existing Python 3.12 scientific environment, existing Node.js 22.17.0
and dashboard dependencies; no dependency installation was needed.

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp .test-build/pysuite
```

Dashboard commands were `npm test` and `npm run build` in a disposable copy of
the tracked dashboard, with existing dependencies linked from the repository.
The staged Next configuration used a common Turbopack root for those linked
dependencies and one build worker. Application code and the user's dashboard
server were untouched. This verifies the staged production build; GitHub's
normal Ubuntu/Windows workflow remains a separate check.

The first restricted Python run had four local-socket failures and one Windows
path-length failure. Local socket access and a short temporary path resolved
them without engine changes; the complete corrected run then passed. Original
failed logs remain in the local campaign, alongside the successful records.

Successful logs and their hashes are preserved in
[the branch evidence record](owned-observation-value-evidence/branch-verification.json).
The portable regression tests include a small original-control oracle with its
source hash; datasets, GT, environments and large run artifacts are excluded.

The authentic independent review approved the exact two-file optimization's
bounded installation. It did not independently approve the whole restoration
diff or a main merge. Its documented equality limitation and full09 timing
qualification remain in the release notes and benchmark report.
