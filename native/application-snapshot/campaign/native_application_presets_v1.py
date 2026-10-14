"""Select only exact preverified numerical presets, with no dataset guessing."""
import hashlib
import json
from pathlib import Path

B=Path(__file__).resolve().parent
CATALOG=B/'native-application-prepared-presets.json'
EXPECTED_CATALOG_SHA256='7cba5e55e5274bdf2582fb2a458fabc8055568cedbfb3758b86602382fcd2a79'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def select_preset(sequence,frames,seed):
    if (not isinstance(sequence,str) or type(frames) is not int or type(seed) is not int
            or frames<=0 or seed not in (0,1,2)):
        raise ValueError('exact prepared sequence/count/seed required')
    if sha(CATALOG)!=EXPECTED_CATALOG_SHA256:raise ValueError('preset catalog identity changed')
    catalog=json.loads(CATALOG.read_text(encoding='utf8'))
    if catalog['status']!='frozen_verified_numerical_presets_not_full_application_qualification':
        raise ValueError('unqualified preset catalog')
    key=f'{sequence}/{frames}/{seed}'
    if key not in catalog['presets']:raise ValueError('input preset not verified: '+key)
    for group in ('protocol_sha256','evidence_sha256'):
        for name,expected in catalog[group].items():
            if sha(B/name)!=expected:raise ValueError('prepared evidence identity changed: '+name)
    if sha(B/'native_application_launcher_release_v7.py')!=catalog['restricted_predecessor_sha256']:
        raise ValueError('cleared application predecessor changed')
    retention_path=B/'native-restricted-prototype-candidate-retention.json'
    if sha(retention_path)!=catalog['retention_manifest_sha256']:
        raise ValueError('restricted clearance identity changed')
    retention=json.loads(retention_path.read_text(encoding='utf8'))
    for name,expected in retention['source_evidence_sha256'].items():
        if sha(B/name)!=expected:raise ValueError('cleared component identity changed: '+name)
    value=catalog['presets'][key]
    protocol_path=B/value['source_protocol']
    protocol=json.loads(protocol_path.read_text(encoding='utf8'))
    if protocol['source_manifest_sha256']!=catalog['source_manifest_sha256'] or protocol['active_case'] is not None:
        raise ValueError('native identity or terminal source protocol changed')
    spec=value['spec']
    if spec not in protocol['runs'].values():raise ValueError('spec differs from exact saved protocol')
    config=B/f'native-online-build/configurations-real/{sequence}.yaml'
    if sha(config)!=spec['configuration_sha256']:raise ValueError('configuration changed')
    if type(value['output_forecast_bytes']) is not int or value['output_forecast_bytes']<=0:
        raise ValueError('positive explicit output forecast required')
    return dict(value,source_protocol_sha256=sha(protocol_path),
                configuration_sha256=sha(config),catalog_sha256=sha(CATALOG))
