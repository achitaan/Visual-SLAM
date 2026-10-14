# Tested application source snapshot

This preserves the 32 Python files reached by static imports from the ordered-control launcher and its owned native driver. Every copied byte matches the working application used in the quick04 Stop test. `manifest.json` records their original relative paths and SHA256 hashes. Files under `campaign/` intentionally retain their original imports, paths, deadlines and provenance checks.

This is a source preservation checkpoint, **not yet a standalone application installation**. The launcher still refers to the original campaign location and its protocol/review records, WSL runtime, prepared input data and dashboard. Those bindings must be assembled and tested from the repository before a clean checkout can use it. Do not bypass the original identity checks to make these copies run.

The exact entrypoint is `campaign/native_application_launcher_ordered_v11.py`. It uses the tested ordered-control class; the legacy protocol/output names retain their V7 suffix for existing saved verifiers. This naming does not make the changed launcher the independently reviewed original V7.

The latest actual browser check recorded prepared → running → stopping → cancelled, with cleanup confirmed, 40 immutable emitted poses, 20 unprocessed original IDs and a strict map containing 18 keyframes and 2,749 landmarks. The browser had no JavaScript or decoding errors. This deliberately cancelled functional check is separate from the completed numerical comparisons.

## Release notes

- Preserved the tested launcher, control, ownership, observation, telemetry and map-export dependency sources in Git.
- Retained exact source bytes and historical module names for comparison with qualification evidence.
- Application relocation and clean-checkout launch verification remain outstanding; no new main or accuracy claim is made here.
