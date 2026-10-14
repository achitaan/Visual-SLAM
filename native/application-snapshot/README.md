# Tested application source snapshot

This preserves the 32 Python files reached by static imports from the ordered-control launcher and its owned native driver. Every copied byte matches the working application used in the quick04 Stop test. `manifest.json` records their original relative paths and SHA256 hashes. Files under `campaign/` intentionally retain their original imports, paths, deadlines and provenance checks.

`local-bindings.json` additionally covers 80 files, including the original protocol, review, calibration and resource records required by all four prepared presets and the full01 ordered-control launcher. Preset selection succeeds using only these packaged records. The packaged record statuses remain historical evidence, rather than new acceptance claims.

Check existing local bindings without writing anything:

```console
python scripts/restore_native_application_bindings.py
```

Add `--apply` to restore missing files into the repository's original campaign layout. The helper first validates every packaged hash, refuses any existing differing file, never overwrites evidence and preserves the 1.5 GiB reserve. Five tests exercise exact restoration of the complete package, idempotence, check-only behavior, existing-file preservation, corrupt input and path traversal rejection.

The repository entry is now `scripts/run_native_candidate.py`, with its packaged viewer under `native/dashboard`. It supports the existing four prepared presets while retaining the pinned Windows checkout, WSL runtime, prepared datasets and dashboard requirements. Restoration installs none of those external requirements. The [setup guide](../../docs/NATIVE-RELEASE.md) describes the supported local layout. Do not bypass the original identity checks to make these copies run.

The exact entrypoint is `campaign/native_application_launcher_ordered_v11.py`. It uses the tested ordered-control class; the legacy protocol/output names retain their V7 suffix for existing saved verifiers. This naming does not make the changed launcher the independently reviewed original V7.

The latest actual browser check recorded prepared → running → stopping → cancelled, with cleanup confirmed, 40 immutable emitted poses, 20 unprocessed original IDs and a strict map containing 18 keyframes and 2,749 landmarks. The browser had no JavaScript or decoding errors. This deliberately cancelled functional check is separate from the completed numerical comparisons.

## Release notes

- Preserved the tested launcher, control, ownership, observation, telemetry and map-export dependency sources in Git.
- Retained exact source bytes and historical module names for comparison with qualification evidence.
- Added the 79-file local binding package and a tested, non-overwriting restore helper; all four prepared preset selectors pass from packaged data.
- Extended the bindings to 80 files for the ordered full01 launcher; added the repository command and packaged dashboard.
- The packaged entry completed an actual 60-frame normal run, with 62 browser receipts, two strict maps, immutable poses and confirmed cleanup. Final focused release review remains pending; no new accuracy claim is made here.

Final release review31 clears the local package after the isolated hello-order repair. `local-bindings.json` preserves all 80 earlier bindings and adds three handshake successors (83 total). The repository entry routes to these successors; the algorithm and original application proofs are preserved. See `docs/benchmark/online-native/application/native-packaged-release-review31-result.json`.
