# Qualified native source

This directory preserves the exact `orbslam3-origin-serialized` source behind the [19 published comparisons](../../docs/benchmark/online-native/README.md). It adds no estimator changes and does not claim a newly qualified build.

The original manifest describes 372 files. All 371 source, configuration, documentation and license files are copied byte-for-byte under `orbslam3/`. The remaining file, `Vocabulary/ORBvoc.txt.tar.gz`, is a separately stored 42,527,984-byte vocabulary archive. Its required SHA256 is recorded in `package-manifest.json`; it is not included in Git. No datasets, native binaries, compiler outputs or experimental run directories are included.

Verify the source from the repository root with Python 3.11 or later:

```console
python scripts/verify_native_release_source.py
```

To also verify a separately available vocabulary, supply its containing source root:

```console
python scripts/verify_native_release_source.py --assets /path/to/source-root
```

`source-manifest.json` is the original immutable pre-build record. Its historical `unbuilt` status is preserved. `qualified-build.json` records the subsequently tested binary and dependency hashes, including the original build environment and reused objects. It is build provenance, not a portable installation script. Rebuilding in another environment does not establish identical binary behavior solely because these source hashes match.

The supplied upstream CMake files retain their original bytes. The campaign used a headless build and pinned external dependencies; a clean repository build recipe and relocation of the application adapters still need verification before this becomes a self-contained release. The current tested application entry remains documented in [ONLINE-CANDIDATE.md](../../docs/ONLINE-CANDIDATE.md).

## Notices

Preserve the upstream [ORB-SLAM3 license](orbslam3/LICENSE), [dependency notices](orbslam3/Dependencies.md), source headers, and bundled third-party license files when redistributing this source. The snapshot includes local modifications for immutable online emissions, strict map snapshots, observation ownership and serialization. Their exact bytes are covered by the manifest.

## Release notes

- Added the exact benchmarked native source to the release branch, with original notices.
- Added a verifier covering every manifest entry and the separately stored vocabulary identity.
- Preserved the original build provenance without claiming a fresh build or changing SLAM arithmetic.
- Main and the installed runtime are unchanged.
