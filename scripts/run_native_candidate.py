"""Run the packaged online native candidate in the qualified local Windows/WSL layout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import online_candidate as original
from restore_native_application_bindings import PACKAGE, restore
from verify_native_release_source import PACKAGE as SOURCE_PACKAGE, verify as verify_source

ROOT = Path(__file__).resolve().parents[1]
BINDINGS_SHA256 = '068e44c97a254b4ea397a5e7a812b6f977674393fdeb7fe15ff2b9c5bbe38bfe'
FRONTEND_SHA256 = {
    'scripts/online_candidate.py':'63e83d8fd8037ef9cf7453454cd15abce9a5a9dc2105de506ffbb26e43c1d25e',
    'scripts/online_candidate_stop.py':'6bc9fa517183cd7fa84077149ac0cc8b5bc88c1bbf0339cd31c8e7af33400886',
}
LAUNCHERS = {
    'quick04': 'native_application_launcher_handshake_v12.py',
    'short04': 'native_application_launcher_handshake_v12.py',
    'short01': 'native_application_launcher_handshake_v12.py',
    'full01': 'native_application_launcher_handshake_full_v10.py',
}


def verify_entry_bindings() -> None:
    for name, expected in FRONTEND_SHA256.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Qualified frontend binding changed: '+name)
    if hashlib.sha256((PACKAGE/'local-bindings.json').read_bytes()).hexdigest() != BINDINGS_SHA256:
        raise ValueError('Packaged application binding manifest changed')


def plan(preset: str, run_id: str) -> dict:
    verify_entry_bindings()
    bindings = restore(PACKAGE, original.BASE)
    if bindings['missing_files']:
        raise ValueError('Packaged bindings are missing; run this command with --setup --check first')
    verify_source(SOURCE_PACKAGE)
    value = original.plan(preset, run_id)
    launcher = LAUNCHERS[preset]
    record = json.loads((PACKAGE/'local-bindings.json').read_text())['files'][launcher]
    actual = hashlib.sha256((original.BASE/launcher).read_bytes()).hexdigest()
    if actual != record['sha256']:
        raise ValueError('Packaged ordered launcher binding changed')
    value['command'][2] = str(original.BASE/launcher)
    value['reviewed_inner_launcher'] = launcher
    value['status'] = 'packaged_online_native_launch_plan'
    value['packaged_binding_files_verified'] = bindings['verified_files']
    value['control_ordering_repair'] = True
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=tuple(LAUNCHERS),required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--check',action='store_true',help='Validate sources and print command; launch nothing')
    parser.add_argument('--check-host',action='store_true',help='With --check, also inspect live workers, ports and dashboard')
    parser.add_argument('--setup',action='store_true',help='Restore only missing packaged local-layout bindings')
    args = parser.parse_args(argv)
    if args.check_host and not args.check:
        parser.error('--check-host requires --check')
    try:
        verify_entry_bindings()
        if args.setup:
            restore(PACKAGE,original.BASE,apply=True)
        run_id = args.run_id or 'native_prototype_repo_'+datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%fz')
        value = plan(args.preset,run_id)
        if args.check:
            if args.check_host:
                value['runtime_admission'] = original.assert_runtime_ready()
            print(json.dumps(value,indent=2,allow_nan=False))
            return 0
        original.assert_runtime_ready()
        value = plan(args.preset,run_id)
        print('Open the printed dashboard link before the launcher readiness deadline.',flush=True)
        return original.run_foreground(value)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print('Native candidate refused: '+str(error))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
