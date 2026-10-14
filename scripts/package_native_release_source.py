"""Copy the already-qualified native source byte-for-byte; never rebuild or run it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/recovery-campaign/20261006T003837Z'
SOURCE = BASE / 'native-online-build/orbslam3-origin-serialized'
DEST = ROOT / 'native/qualified'
MANIFEST_SHA256 = '8fc711ec8d8c629e126c588b2ef78cbe24507f7a3f8aee89cb1c041a33aae4b6'
EXTERNAL = {'Vocabulary/ORBvoc.txt.tar.gz'}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    manifest = BASE / 'native-origin-serialized-source.json'
    if digest(manifest) != MANIFEST_SHA256:
        raise ValueError('The benchmark source manifest changed')
    files = json.loads(manifest.read_text())['files']
    assert EXTERNAL <= files.keys()
    # Validate the complete original before copying anything.
    for name, expected in files.items():
        path = SOURCE / name
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError('Source escaped the frozen root: ' + name)
        if digest(path) != expected:
            raise ValueError('Frozen source differs from benchmark: ' + name)
    required = sum((SOURCE/n).stat().st_size for n in files if n not in EXTERNAL)
    if shutil.disk_usage(ROOT).free < 1610612736 + 2 * required:
        raise ValueError('Source copy would consume the reserved capacity')
    if DEST.exists():
        raise FileExistsError('Preserve an existing release snapshot; no overwrite')
    DEST.mkdir(parents=True)
    copied = {}
    for name, expected in files.items():
        if name in EXTERNAL:
            continue
        target = DEST / 'orbslam3' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, target)
        if digest(target) != expected:
            raise ValueError('Copied source hash mismatch: ' + name)
        copied[name] = expected
    shutil.copyfile(manifest, DEST/'source-manifest.json')
    shutil.copyfile(BASE/'native-origin-serialized-build.json', DEST/'qualified-build.json')
    record = {
        'scope': 'exact benchmark source snapshot; application relocation remains separate',
        'source_manifest_sha256': MANIFEST_SHA256,
        'copied_files': copied,
        'external_assets': {n: {'sha256': files[n], 'bytes': (SOURCE/n).stat().st_size} for n in EXTERNAL},
        'copied_bytes': required,
        'estimator_changed': False,
        'binaries_included': False,
        'new_build_qualified': False,
    }
    (DEST/'package-manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
    (DEST/'.gitattributes').write_text('* -text\n', encoding='utf8')
    print(json.dumps({'source_files':len(copied), 'bytes':required, 'all_hashes':'pass',
                      'external_assets':len(EXTERNAL)}))


if __name__ == '__main__':
    main()
