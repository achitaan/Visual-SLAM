"""Preserve the exact data and identity dependencies of the tested local entry."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
B = ROOT/'results/recovery-campaign/20261006T003837Z'
PACKAGE = ROOT/'native/application-snapshot'


def main():
    manifest = json.loads((PACKAGE/'manifest.json').read_text())
    names = set(manifest['files'])
    names.update(['native-origin-serialized-development.json', 'native-origin-serialized-build.json',
        'native-origin-serialized-source.json', 'native-component-review18-B1-result.json',
        'native-component-review18-B1-packet.json', 'native-full01-experimental-prototype-retention29.json',
        'native-fullapp-v10-independent-review29-result.json'])
    packet = json.loads((B/'native-component-review18-B1-packet.json').read_text())
    retention = json.loads((B/'native-full01-experimental-prototype-retention29.json').read_text())
    pinned = {**packet['sources'], **packet['evidence'], **retention['source_sha256']}
    for record_name, groups in [
        ('native-application-prepared-presets.json', ('protocol_sha256','evidence_sha256')),
        ('native-restricted-prototype-candidate-retention.json', ('source_evidence_sha256',)),
        ('native-bounded-full01-measured-resource-design.json', ('evidence_sha256',)),
    ]:
        names.add(record_name)
        record = json.loads((B/record_name).read_text())
        for group in groups:
            for name, expected in record[group].items():
                if name in pinned and pinned[name] != expected:
                    raise ValueError('Conflicting provenance for '+name)
                pinned[name] = expected
    names.update(pinned)
    names.update('native-online-build/configurations-real/'+seq+'.yaml' for seq in ('01','04'))
    for name in names:
        path = B/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(B.resolve()):
            raise ValueError('Unscoped or missing application binding: '+name)
        if name in pinned and hashlib.sha256(path.read_bytes()).hexdigest() != pinned[name]:
            raise ValueError('Reviewed binding differs: '+name)
    records = {}
    for name in sorted(names):
        source, dest = B/name, PACKAGE/'campaign'/name
        raw = source.read_bytes()
        if dest.exists():
            if dest.read_bytes() != raw:
                raise ValueError('Existing snapshot differs: '+name)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as stream:
                stream.write(raw)
        records[name] = {'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw)}
    output = PACKAGE/'local-bindings.json'
    if output.exists():
        previous = json.loads(output.read_text())
        if any(name not in records or records[name] != value for name,value in previous['files'].items()):
            raise ValueError('Cannot replace a changed existing binding')
        backup = B/'native-release-local-bindings-before-presets.json'
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(output.read_bytes())
    with output.open('w',encoding='utf8') as stream:
        json.dump({'scope':'data/source dependencies for the original qualified local layout',
            'files':records, 'runtime_installed':False,
            'external_requirements':['pinned Windows GPU Python', 'qualified Ubuntu native binary/libraries/vocabulary',
                                     'prepared KITTI inputs', 'qualified dashboard on loopback 4301']},stream,indent=2)
        stream.write('\n')
    print(json.dumps({'bound_files':len(records),'bytes':sum(r['bytes'] for r in records.values()),'all_original_hashes':'pass'}))


if __name__ == '__main__':
    main()
