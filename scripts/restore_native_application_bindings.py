"""Restore missing qualified local-layout files; refuse all existing differences.

This does not install runtimes, download inputs, start processes or approve a release.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'native/application-snapshot'
BASE = ROOT/'results/recovery-campaign/20261006T003837Z'


def restore(package: Path, destination: Path, *, apply: bool = False) -> dict:
    records = json.loads((package/'local-bindings.json').read_text())['files']
    source_root = (package/'campaign').resolve()
    destination_root = destination.resolve()
    if not source_root.is_relative_to(ROOT.resolve()) or not destination_root.is_relative_to(ROOT.resolve()):
        raise ValueError('Application restoration layout escaped the repository')
    pending = []
    for name, item in records.items():
        source = package/'campaign'/name
        target = destination/name
        if source.is_symlink() or not source.resolve().is_relative_to(source_root):
            raise ValueError('Packaged source escaped root: '+name)
        if not target.resolve().is_relative_to(destination_root) or target.is_symlink():
            raise ValueError('Destination escaped root: '+name)
        data = source.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Packaged binding hash mismatch: '+name)
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise ValueError('Preserving existing differing file: '+name)
        else:
            pending.append((target,data))
    if apply:
        ancestor = destination_root
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < 1610612736 + sum(len(data) for _,data in pending):
            raise ValueError('Disk reserve prevents restoration')
        for target, data in pending:
            target.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive creation also refuses a concurrent writer after preflight.
            with target.open('xb') as stream:
                stream.write(data)
            if target.read_bytes() != data:
                raise ValueError('Restored file differs: '+str(target))
    return {'verified_files':len(records), 'missing_files':len(pending),
            'restored_files':len(pending) if apply else 0, 'runtime_started':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='Create only missing local binding files after complete validation')
    args = parser.parse_args()
    if not BASE.resolve().is_relative_to(ROOT.resolve()) or not PACKAGE.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Local application layout escaped the repository')
    print(json.dumps(restore(PACKAGE, BASE, apply=args.apply)))
