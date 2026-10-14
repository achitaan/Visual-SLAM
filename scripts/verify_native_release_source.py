"""Verify packaged native source against its original benchmark manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'native/qualified'
SOURCE_SHA256 = '8fc711ec8d8c629e126c588b2ef78cbe24507f7a3f8aee89cb1c041a33aae4b6'
BUILD_RECORD_SHA256 = 'bc768fb2685f54557ad66e3119f372cb209079a0297c529972763ae330bc37b7'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(package: Path, assets: Path | None = None) -> dict:
    record = json.loads((package/'package-manifest.json').read_text())
    if digest(package/'source-manifest.json') != SOURCE_SHA256 or record['source_manifest_sha256'] != SOURCE_SHA256:
        raise ValueError('Benchmark source manifest hash mismatch')
    original = json.loads((package/'source-manifest.json').read_text())['files']
    included, external = record['copied_files'], record['external_assets']
    if set(included) & set(external) or set(included) | set(external) != set(original):
        raise ValueError('Original source manifest coverage is incomplete or duplicated')
    for name, expected in included.items():
        path = package/'orbslam3'/name
        if not path.resolve().is_relative_to((package/'orbslam3').resolve()) or path.is_symlink():
            raise ValueError('Source path escaped package: ' + name)
        if expected != original[name] or digest(path) != expected:
            raise ValueError('Packaged source hash mismatch: ' + name)
    for name, item in external.items():
        if item['sha256'] != original[name]:
            raise ValueError('External asset identity changed: ' + name)
        if assets is not None:
            path = assets/name
            if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError('External asset content mismatch: ' + name)
    if digest(package/'qualified-build.json') != BUILD_RECORD_SHA256:
        raise ValueError('Qualified build record changed')
    build = json.loads((package/'qualified-build.json').read_text())
    if build['source_manifest_sha256'] != record['source_manifest_sha256']:
        raise ValueError('Qualified build references a different source')
    return {'source_files_verified':len(included), 'external_assets':len(external),
            'external_assets_verified':assets is not None, 'matches_benchmark_source':True,
            'new_build_or_application_claim':False}


def verify_index() -> int:
    record = json.loads((PACKAGE/'package-manifest.json').read_text())
    expected = {'native/qualified/orbslam3/' + name: value
                for name, value in record['copied_files'].items()}
    expected.update({'native/qualified/source-manifest.json': SOURCE_SHA256,
                     'native/qualified/qualified-build.json': BUILD_RECORD_SHA256})
    result = subprocess.run(['git', 'cat-file', '--batch'], cwd=ROOT,
        input=''.join(':' + name + '\n' for name in expected).encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    position = 0
    for name, digest_expected in expected.items():
        end = result.stdout.index(b'\n', position)
        header = result.stdout[position:end].split()
        if len(header) != 3 or header[1] != b'blob':
            raise ValueError('Source absent from Git index: ' + name)
        size = int(header[2]); position = end + 1
        content = result.stdout[position:position + size]; position += size + 1
        if hashlib.sha256(content).hexdigest() != digest_expected:
            raise ValueError('Git index bytes differ from benchmark: ' + name)
    if position != len(result.stdout):
        raise ValueError('Unexpected Git batch output')
    return len(expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, help='Root containing the separately stored Vocabulary asset')
    parser.add_argument('--git-index', action='store_true', help='Also verify the actual staged Git blobs')
    args = parser.parse_args()
    result = verify(PACKAGE, args.assets)
    if args.git_index:
        result['git_index_blobs_verified'] = verify_index()
    print(json.dumps(result))
