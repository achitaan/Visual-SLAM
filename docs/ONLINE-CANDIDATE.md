# Online SLAM candidate

The isolated native candidate tracks stereo frames, optimizes poses and updates its map online. Emitted poses are immutable: future frames and shutdown work cannot correct previously emitted history. The original Python pipeline remains the default; this guide does not promote or merge the candidate into main.

## What is usable

Independent review26 cleared the short V7 presets. Review29 additionally cleared the tested full01 V10 application for experimental use:

| KITTI sequence | Frames | Seed | Launcher |
|---|---:|---:|---|
| 04 | 60 | 2 | V7 |
| 04 | 80 | 0 | V7 |
| 01 | 350 | 0 | V7 |
| 01 | 1101 | 0 | V10 |

Actual browser evidence covers immutable received poses, strict maps, normal completion and owned Stop. Stop preserves the emitted prefix and accounts for every unprocessed original frame; cancellation is not an accuracy pass. Clear only changes the display. Arbitrary datasets and live camera inputs are not qualified.

This checkout currently relies on pinned local Windows/Ubuntu WSL runtimes, prepared KITTI inputs and an isolated dashboard. It is not a portable installation package. No OneDrive, Docker or external model API is needed.

## Start a prepared run

Do not launch beside a benchmark or an existing native worker. Reuse the owned dashboard at `http://127.0.0.1:4301`; do not restart the root dashboard. If the isolated dashboard is stopped, its exact start command is in the campaign's [prototype instructions](../results/recovery-campaign/20261006T003837Z/NATIVE-PROTOTYPE.md). Check that backend ports 8779 and 8780 are free. The launcher refuses reused run IDs, unsupported presets and insufficient disk space; preserve its 1.5 GiB physical reserve plus 70 MB short-run margin.

Run from PowerShell on this machine:

```powershell
$slamReleaseRunId = 'native_prototype_release_user_' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ').ToLower()
& 'C:\Users\Achita\.codex\worktrees\e824\VSLAM\.venv-performance-gpu\Scripts\python.exe' -B 'C:\Users\Achita\Documents\ChatGPT\VSLAM\results\recovery-campaign\20261006T003837Z\native_application_launcher_release_v7.py' --run-id $slamReleaseRunId --sequence 04 --frames 60 --seed 2 --telemetry-port 8779 --control-port 8780 --start-delay 1 --serve-after 120
```

Open the printed browser URL before its 120-second readiness deadline. For another V7 preset, change sequence, frames and seed together to one V7 row of the table. A completed run serves its final map for 120 seconds, then exits. A retained browser view does not mean processing is still live; reloading after shutdown does not restart the estimator.

For the independently cleared full01 preset, use the V10 entry point. It requires **3,860,612,736 bytes physically free on C** before dispatch, including the unchanged reserve and measured output forecast. Run one prepared application at a time:

```powershell
$slamFullRunId = 'native_prototype_release_full01_' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ').ToLower()
& 'C:\Users\Achita\.codex\worktrees\e824\VSLAM\.venv-performance-gpu\Scripts\python.exe' -B 'C:\Users\Achita\Documents\ChatGPT\VSLAM\results\recovery-campaign\20261006T003837Z\native_application_launcher_release_v10.py' --run-id $slamFullRunId --sequence 01 --frames 1101 --seed 0 --telemetry-port 8779 --control-port 8780 --start-delay 1 --serve-after 120
```

Outputs are under `results/recovery-campaign/20261006T003837Z/<run-id>_application_release_v7/` or `_application_release_v10/` for the selected launcher, including immutable pose and publication ledgers, portable map exports and lifecycle evidence. Ground truth is evaluator-only and is not loaded by the application launcher.

## Draft repository preflight

The draft repository frontend also supports a check-only preflight:

```powershell
& 'C:\Users\Achita\.codex\worktrees\e824\VSLAM\.venv-performance-gpu\Scripts\python.exe' -B scripts/online_candidate.py --preset quick04 --check --check-host
```

This inspects source bindings, capacity, live Windows/Ubuntu campaign workers, backend ports and dashboard availability without starting an estimator or browser. It refuses when observation is unavailable. The draft frontend has focused component checks, a real host preflight pass and a real isolated Windows console Ctrl+C/Stop test. Independent frontend release review remains pending. Use the reviewed direct commands above for experimental application runs.

The first actual frontend normal run is saved-verified: all 60 original frames, 62 browser receipts, two strict maps and confirmed cleanup. The original mapping Ctrl+C attempt preserved 41 emitted poses but failed intermediate stopping-state delivery; that failure remains archived. Review30 also identified a possible stale-state publication race.

The isolated ordered-control successor now passes 11 socket/ordering/guard checks and one actual quick04 mapping Ctrl+C qualification. The browser received prepared → running → stopping → cancelled with confirmed cleanup, 40 immutable poses and one validated map (18 keyframes, 2,749 landmarks). All 60 original IDs remain accounted for, including 20 unprocessed IDs. Saved verification checks 41 received messages, exact emitted bytes, source/runtime bindings, sealed owned-process exit and the 1.5 GiB disk reserve. There were no browser JavaScript or message-decoding errors. The successor changes only the control publication binding; estimator code and the preserved launcher remain unchanged. This closes the tested Stop-delivery defect, without claiming a new numerical benchmark or main release.

[Actual ordered-control verification](../results/recovery-campaign/20261006T003837Z/native-repo-app-mapping-04-60-seed2-ordered30show-streaming-verification.json) · [Browser capture](../results/recovery-campaign/20261006T003837Z/native-repo-app-mapping-04-60-seed2-ordered30show-browser.png)

## Current numerical evidence

The frozen `orbslam3-origin-serialized` source has **19 completed paired comparisons**: 17 strict passes plus two specifically accepted translation-RPE tradeoffs on full00 and full10. Strict passes cover short/full01 and04 across seeds0/1/2 and full03/05/06/07/09 seed0. Every candidate comparison improved ATE and whole-dispatch runtime and had zero tracking losses. Original full00/full10 strict failures remain preserved; human acceptance does not rewrite their gates. Numerical coverage does not expand the independently reviewed application presets.

| Completed scope | VO ATE (m) | Candidate ATE (m) | VO whole runtime (s) | Candidate whole runtime (s) |
|---|---:|---:|---:|---:|
| Full00, seed0; accepted RPE tradeoff | 20.609743 | 3.947487 | 2642.797 | 1029.828 |
| Full01, three-seed medians | 77.731503 | 15.104658 | 449.250 | 155.063 |
| Full03, seed0 | 2.327133 | 1.272749 | 449.594 | 94.110 |
| Full06, seed0 | 2.815149 | 2.561234 | 511.516 | 133.328 |
| Full05, seed0 | 9.555220 | 2.211941 | 1460.016 | 376.593 |
| Full10, seed0; accepted RPE tradeoff | 2.563907 | 1.363188 | 644.234 | 154.625 |

Full01 worst tracking loss is zero versus 30 for VO. ATE uses SE(3) alignment without scale fitting. This compares Linux C++/OpenCV4.8.1 against Windows Python/OpenCV5.0 as whole systems, not an isolated kernel or language speedup. The accepted one-second translation-RPE regressions are 2.005340% on full00 and 0.227636% on full10. Full00 includes a recorded bootstrap/storage pause between baseline and the first actual native arm; no baseline or native replay was used to select timing.

[All 19 individual runs and measured metrics](benchmark/online-native/results.csv) include raw ATE, both RPE metrics, translation/rotation drift, losses, whole-dispatch FPS, native frame latency and available process memory. Sequences02/08 and native-rate streaming remain unqualified.

The original 14-pair retention record and prior supplements remain unchanged. The [17-pair supplement](../results/recovery-campaign/20261006T003837Z/native-serialized-release-confirmation-summary-plus-full03-full06-full05.json) adds completed full03/full06/full05 evidence without transferring older-source results. [The candidate report](../results/recovery-campaign/20261006T003837Z/NATIVE-ONLINE-CANDIDATE-REPORT.md) provides additional metrics and reproducibility records.

## Before replacing main

- Full01/1101/seed0 application completion and Stop pass on the V10 path:1,101 poses,1,138 browser receipts and37 strict maps, with sampled disk reserve and owned cleanup verified. Independent review29 cleared this tested experimental preset; broader main-release clearance is still pending. The earlier V8 disk failure is preserved.
- Full00 is completed and saved-verified, with its specific RPE tradeoff accepted. Remaining declared representative coverage includes02/08; historical results from another source do not transfer.
- Verify broader application/distribution integration and obtain final independent clearance for that scope.
- Package the frozen native source, build/runtime instructions, qualified adapters and documentation into a reviewable Git change. The native source currently lives in the ignored campaign directory; numerical manifests and backups are not a committed release.

The new compressed audit format is prepared and has saved-byte/component checks, but is unlaunched. Restored physical capacity allowed the existing reviewed V10 path to complete without using it. It is not the reviewed V7 entry point. Real-time throughput and native-rate latency/resource targets are explicitly outside this release scope.

[Current completion audit](../results/recovery-campaign/20261006T003837Z/native-online-goal-current-audit-full00-accepted-review30-pending.json) · [Application qualification and preserved failure](../results/recovery-campaign/20261006T003837Z/NATIVE-FULL-APPLICATION-QUALIFICATION.md) · [Restricted retention record](../results/recovery-campaign/20261006T003837Z/native-restricted-prototype-candidate-retention.json)

The original saved coverage-index audit checks 17 strict passing pair keys and 34 unchanged case bindings. Strict full-sequence coverage is01/03/04/05/06/07/09;01/04 have three seeds and the others seed0. The current 19-comparison export additionally checks both accepted00/10 tradeoffs, for38 case bindings. Original strict failures and historical audits remain unchanged. Exporting these records is not a new benchmark or independent review.

[Full01 experimental retention](../results/recovery-campaign/20261006T003837Z/native-full01-experimental-prototype-retention29.json) · [Independent review29](../results/recovery-campaign/20261006T003837Z/native-fullapp-v10-independent-review29-result.json)

## Draft release notes — unreleased experimental candidate

- Added native stereo tracking, pose optimization and mapping during execution. Emitted poses stay immutable and cannot borrow future or shutdown corrections.
- Fixed duplicate physical observation ownership and serialized map snapshots against mapper mutations; bounded native execution and export retain truthful incomplete outcomes.
- Added a guarded dashboard prototype with map/trajectory exports, normal-completion evidence and owned Stop qualification. The ordered-control repair now delivers intermediate stopping and terminal cleanup states to a browser when Stop originates from the separate command-line client; actual quick04 browser and immutable-output checks pass.
- Verified19 paired comparisons on the exact serialized source:17 strict passes and two explicitly accepted one-second translation-RPE tradeoffs. Every candidate run improved ATE/runtime and had zero tracking losses.
- Known limits: this is a online prototype without a real-time requirement using pinned local Windows/Ubuntu WSL runtimes. Portable installation, broader main-release clearance, sequences02/08 and native-rate streaming are not qualified. Main has not been updated and no release has been published.
