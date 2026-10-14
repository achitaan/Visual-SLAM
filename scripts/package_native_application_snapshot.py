"""Preserve the tested application import closure without relocating or executing it."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
B = ROOT/'results/recovery-campaign/20261006T003837Z'
DEST = ROOT/'native/application-snapshot'
SEARCH = [B, B/'native-app-adapter-origin', B/'retry-bundle-operator-source/src']
ENTRYPOINTS = ['native_application_launcher_ordered_v11.py', 'run_native_serialized_application_case_v5.py']


def main():
    queue = [B/n for n in ENTRYPOINTS]
    selected = {}; unresolved = set()
    while queue:
        path = queue.pop()
        name = path.relative_to(B).as_posix()
        if name in selected:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(B.resolve()):
            raise ValueError('Import path escaped campaign')
        raw = path.read_bytes()
        selected[name] = {'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw)}
        for node in ast.walk(ast.parse(raw)):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module.split('.')[0]]
            if isinstance(node, ast.Import):
                modules = [n.name.split('.')[0] for n in node.names]
            for module in modules:
                candidate = next((r/(module+'.py') for r in SEARCH if (r/(module+'.py')).is_file()), None)
                if candidate is not None:
                    queue.append(candidate)
                elif module.startswith('native_'):
                    unresolved.add(module)
    if unresolved:
        raise ValueError('Unresolved native imports: ' + ', '.join(sorted(unresolved)))
    if DEST.exists():
        raise FileExistsError('Preserve existing application snapshot')
    if shutil.disk_usage(ROOT).free < 1610612736 + sum(x['bytes'] for x in selected.values()) * 2:
        raise ValueError('Insufficient reserve for application snapshot')
    DEST.mkdir(parents=True)
    for name, item in selected.items():
        target = DEST/'campaign'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(B/name, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Copied application differs: ' + name)
    (DEST/'.gitattributes').write_text('* -text\n')
    (DEST/'manifest.json').write_text(json.dumps({
        'scope':'exact Python import closure of tested quick04 launcher; not a relocated executable distribution',
        'entrypoints':ENTRYPOINTS, 'files':selected,
        'remaining':['protocol and review data bindings','dynamic subprocess/data paths','runtime and dashboard setup'],
        'source_changed':False,
    },indent=2)+'\n')
    print(json.dumps({'copied_files':len(selected),'bytes':sum(x['bytes'] for x in selected.values()),
                      'all_hashes':'pass','executable_distribution':False}))


if __name__ == '__main__':
    main()
