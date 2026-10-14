"""Start the packaged native dashboard with existing repository dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT/'native/dashboard'
DEPENDENCIES = ROOT/'dashboard/node_modules'
UI_MANIFEST_SHA256 = 'c709944bc2d8324c772eb3ba4f04d9cf6193c0495e4bd09d0dcf55ee26b3b52e'


def prepare(*, apply: bool = False) -> dict:
    if hashlib.sha256((DASHBOARD/'qualified-ui-manifest.json').read_bytes()).hexdigest() != UI_MANIFEST_SHA256:
        raise ValueError('Qualified UI manifest changed')
    manifest = json.loads((DASHBOARD/'qualified-ui-manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        path = DASHBOARD/name
        if not path.resolve().is_relative_to(DASHBOARD.resolve()) or path.is_symlink():
            raise ValueError('Dashboard path escaped package')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Qualified dashboard content changed: '+name)
    if not (DEPENDENCIES/'next/dist/bin/next').is_file():
        raise FileNotFoundError('Install the locked root dashboard dependencies before starting this dashboard')
    link = DASHBOARD/'node_modules'
    if link.exists() or link.is_symlink():
        if link.resolve() != DEPENDENCIES.resolve():
            raise ValueError('Preserving a different native dashboard dependency directory')
    elif apply:
        if os.name != 'nt':
            raise ValueError('The qualified local dashboard setup requires Windows')
        environment = os.environ.copy()
        environment.update(VSLAM_NATIVE_NODE_LINK=str(link),VSLAM_NATIVE_NODE_TARGET=str(DEPENDENCIES))
        subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
            'New-Item -ItemType Junction -Path $env:VSLAM_NATIVE_NODE_LINK -Target $env:VSLAM_NATIVE_NODE_TARGET | Out-Null'],
            env=environment,check=True,timeout=15)
        if link.resolve() != DEPENDENCIES.resolve():
            raise ValueError('Dependency junction did not resolve to the root dashboard')
    node = shutil.which('node')
    if not node:
        raise FileNotFoundError('The existing Node runtime is unavailable')
    return {'dashboard':str(DASHBOARD),'verified_ui_files':len(manifest['files_sha256']),
        'dependency_link_ready':link.exists(),
        'command':[node,str(DEPENDENCIES/'next/dist/bin/next'),'dev',str(DASHBOARD),
                   '--hostname','127.0.0.1','--port','4301','--webpack'],
        'url':'http://127.0.0.1:4301/native-stop-v2'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--setup',action='store_true',help='Create a scoped dependency junction, with no installation')
    args = parser.parse_args()
    value = prepare(apply=args.setup or not args.check)
    print(json.dumps(value,indent=2),flush=True)
    if not args.check:
        with socket.socket() as probe:
            if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
                probe.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            probe.bind(('127.0.0.1',4301))
        raise SystemExit(subprocess.call(value['command'],cwd=ROOT))
