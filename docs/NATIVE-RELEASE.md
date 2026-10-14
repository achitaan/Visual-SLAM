# Online native candidate: local setup

The candidate tracks stereo images, optimizes poses and updates its map during processing. Emitted poses remain immutable. Real-time throughput is outside this release scope. The [published comparisons](benchmark/online-native/README.md) cover 19 paired runs on the exact packaged C++ source, including the two explicitly accepted RPE tradeoffs.

This setup targets the existing qualified Windows checkout at `C:\Users\Achita\Documents\ChatGPT\VSLAM`, Ubuntu WSL and the pinned GPU Python at `C:\Users\Achita\.codex\worktrees\e824\VSLAM\.venv-performance-gpu\Scripts\python.exe`. It also requires the already-qualified native binary/libraries/vocabulary, prepared KITTI inputs and the installed root dashboard dependencies. The package restores source/configuration bindings; it does not install those external runtimes or download datasets. Another machine needs its runtime and build qualified before these commands can be treated as tested there.

## Start the dashboard

In one terminal, from the repository root:

```powershell
$nativePython = 'C:\Users\Achita\.codex\worktrees\e824\VSLAM\.venv-performance-gpu\Scripts\python.exe'
& $nativePython -B scripts/start_native_dashboard.py
```

The helper reuses `dashboard/node_modules` through a scoped junction and serves `native/dashboard` on loopback port 4301. It refuses a different existing dependency directory or an occupied port. If the earlier isolated dashboard still owns 4301, use that existing viewer until it has been intentionally shut down; do not start a duplicate. The package keeps the qualified route, components, styles and layout bytes unchanged, and disables persistent webpack caching.

## Check and launch

Check the source/configuration bindings and host before dispatch:

```powershell
& 'C:\Users\Achita\.codex\worktrees\e824\VSLAM\.venv-performance-gpu\Scripts\python.exe' -B scripts/run_native_candidate.py --preset quick04 --setup --check --check-host
```

`--setup` restores only missing files from the packaged bindings, after validating every source hash. Existing differing files are preserved and cause refusal. `--check` starts no estimator or browser.

Run the short demonstration:

```powershell
& 'C:\Users\Achita\.codex\worktrees\e824\VSLAM\.venv-performance-gpu\Scripts\python.exe' -B scripts/run_native_candidate.py --preset quick04
```

Open its printed browser link before the readiness deadline. The application starts native processing only after the browser has connected to its control and telemetry streams. Use **Stop run** or terminal Ctrl+C to request owned cleanup. Cancelled runs retain their emitted pose/map prefix and explicit unprocessed original IDs; they are incomplete runs. **Clear display** changes the display only.

| Preset | Prepared input | Seed | Output forecast | Disk reserve |
|---|---|---:|---:|---:|
| `quick04` | KITTI04, 60 frames | 2 | 70 MB | 1.5 GiB |
| `short04` | KITTI04, 80 frames | 0 | Original guarded forecast | 1.5 GiB |
| `short01` | KITTI01, 350 frames | 0 | Original guarded forecast | 1.5 GiB |
| `full01` | KITTI01, 1,101 frames | 0 | 2.25 GB | 1.5 GiB |

Each command prints the required and current free bytes. A fresh run ID is generated unless `--run-id` supplies a scoped `native_prototype_...` ID. Existing outputs cannot be overwritten or silently replayed.

Outputs remain in `results/recovery-campaign/20261006T003837Z/<run-id>_application_release_v7/`, or `_application_release_v10/` for full01. They include immutable poses, publication and ownership ledgers, strict map exports and truthful completion/cancellation status. Historical protocol suffixes are preserved; the manifest identifies the actual ordered-control launcher bytes.

## Verification scope

- All 371 packaged native files match the benchmark source manifest; the separate vocabulary identity was checked.
- The dashboard's source/layout/style bytes match the actual browser-qualified viewer, and its TypeScript check passes.
- The quick04 actual external Ctrl+C test delivered 40 poses and one strict map, with 41 browser receipts, correct stopping/terminal states, exact saved bytes and whole-owned cleanup. No browser JavaScript or decode errors occurred.
- The full01 original normal application test delivered 1,101 poses and 37 strict maps with immutable bytes, disk reserve and owned cleanup verified. Its ordered-control successor changes only the control import and component manifest; the estimator and other launcher code are AST-identical. This does not claim a fresh full01 execution of the successor.
- Focused package and restoration tests check guarded routing, missing/corrupt files and preservation of existing data. No accuracy reruns are attributed to packaging work.

The [final independent review](benchmark/online-native/application/native-packaged-release-review31-result.json) clears this package for the qualified local Windows/Ubuntu WSL release scope. The connection-order successor preserves the tested launcher behavior and admits a subscriber only after its hello has been sent; 25 focused checks pass. The prior full01 and short actual browser evidence remains applicable as scoped by the review. Main has not been updated.

A new run still requires its documented free-space threshold. At review time, quick04 admission correctly refused 1,675,972,608 available bytes against 1,680,612,736 required. This operational storage guard does not change the completed release checks.

## Release notes

- Added the exact benchmarked native C++ source, original license notices and hash verification.
- Packaged application source/configuration dependencies and safe restoration tooling.
- Added a repository command for the four prepared native presets and a packaged dashboard using existing dependencies.
- Fixed external Stop publication ordering and the new-client hello ordering race, preserving owned cleanup and immutable poses.
- Qualified the short browser normal/Stop paths and completed an independent release review; setup commands now select the tested Python runtime.
- Published all 19 numerical comparisons, with the accepted full00/full10 RPE tradeoffs retained explicitly.
- Supported setup remains the qualified local Windows/Ubuntu WSL environment; cross-machine installation is not yet qualified.
