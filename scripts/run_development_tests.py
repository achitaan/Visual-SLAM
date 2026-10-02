"""Sequential, budgeted diagnostic cycles; incomplete work is never reused as a pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from test_budget import Budget, write_json
from data_preflight import preflight

REPO = Path(__file__).resolve().parents[1]
VARIANTS = {
    'baseline': [],
    'map-only': ['--disable-bundle', '--loop-mode', 'off'],
    'bundle': ['--loop-mode', 'off'],
    'live': ['--loop-mode', 'live'],
    'offline': ['--loop-mode', 'offline'],
}


def terminate_owned_tree(process):
    if os.name != 'nt':
        import signal
        os.killpg(process.pid,signal.SIGKILL)
        return
    import ctypes
    from ctypes import wintypes
    class Entry(ctypes.Structure):
        _fields_=[('size',wintypes.DWORD),('usage',wintypes.DWORD),('pid',wintypes.DWORD),('heap',ctypes.c_size_t),('module',wintypes.DWORD),('threads',wintypes.DWORD),('parent',wintypes.DWORD),('priority',wintypes.LONG),('flags',wintypes.DWORD),('name',wintypes.WCHAR*260)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    kernel.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
    snapshot=kernel.CreateToolhelp32Snapshot(2,0);entry=Entry();entry.size=ctypes.sizeof(entry);rows=[]
    if kernel.Process32FirstW(snapshot,ctypes.byref(entry)):
        while True:
            rows.append((entry.pid,entry.parent))
            if not kernel.Process32NextW(snapshot,ctypes.byref(entry)):break
    kernel.CloseHandle(snapshot)
    owned={process.pid};order=[process.pid]
    while True:
        children=[pid for pid,parent in rows if parent in owned and pid not in owned]
        if not children:break
        owned.update(children);order.extend(children)
    # The root PID comes from our Popen handle; only its enumerated descendants qualify.
    for pid in reversed(order):
        handle=kernel.OpenProcess(1,False,pid)
        if handle:
            try:kernel.TerminateProcess(handle,1)
            finally:kernel.CloseHandle(handle)
    if process.poll() is None:process.kill()


def run_owned(command, log, seconds, env, *, on_start=None):
    """Terminate only this supervisor's process tree if a worker misses its deadline."""
    started = time.monotonic()
    with Path(log).open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != 'nt')
        if on_start is not None:
            on_start(process.pid)
        try:
            code = process.wait(timeout=seconds)
            return {'exit_code': code, 'elapsed_s': time.monotonic()-started, 'timed_out': False}
        except subprocess.TimeoutExpired:
            terminate_owned_tree(process)
            process.wait(timeout=5)
            return {'exit_code': process.returncode, 'elapsed_s': time.monotonic()-started, 'timed_out': True}


def reusable(report, identity):
    return (report.get('development_identity') == identity
            and report.get('status') in ('completed','completed_with_tracking_loss')
            and report.get('frames') == identity['frames'])


def source_fingerprint():
    sources = sorted((REPO/'src').glob('*.py')) + [
        REPO/'scripts'/name for name in ('evaluate_shared_slam.py',
        'evaluate_stereo_baseline.py', 'run_development_tests.py',
        'test_budget.py', 'data_preflight.py')]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['quick','focused','release'], default='quick')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--poses-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/stereo-rework'))
    parser.add_argument('--budget-seconds', type=float)
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=['bundle'])
    parser.add_argument('--release-ready', type=Path, help='Exact-revision focused gate assessment required for full runs')
    parser.add_argument('--feature-cache',type=Path,help='Bounded cached diagnostics; unavailable for release performance runs')
    args=parser.parse_args()
    if args.profile=='release' and args.feature_cache:parser.error('Release performance must use uncached extraction')
    seconds=args.budget_seconds if args.budget_seconds is not None else (300 if args.profile=='quick' else 3600)
    if not 0 < seconds <= 3600:parser.error('Cycle budget must be within 1–3600 seconds')
    budget=Budget(seconds)
    fingerprint=source_fingerprint()
    root=args.output/fingerprint[:12]/args.profile;root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'cycle.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'revision':fingerprint,'attempts':[]}
    manifest.update(profile=args.profile,budget_seconds=seconds,status='running',supervisor_pid=os.getpid())
    write_json(manifest_path,manifest)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONIOENCODING':'utf-8','MPLCONFIGDIR':str(REPO/'.mpl-cache')}
    checks=run_owned([sys.executable,'-m','pytest','tests/test_shared_slam.py','tests/test_keyframe_retrieval.py','tests/test_append_only_corrections.py','tests/test_local_bundle_landmarks.py','tests/test_bidirectional_refinement.py','tests/test_stereo_motion_prior.py','tests/test_stereo_map_refinement.py','tests/test_stereo_feature_support.py','tests/test_descriptor_fallback.py','tests/test_feature_cache.py','tests/test_test_budget.py','-q'],root/'checks.log',min(120,budget.remaining),env)
    manifest['checks']=checks;write_json(manifest_path,manifest)
    if checks['exit_code'] != 0:
        manifest['status']='failed_checks';write_json(manifest_path,manifest);return 1
    cases=[('04',80)] if args.profile=='quick' else [('04',80),('01',350)]
    if args.profile=='release':
        if not args.release_ready:parser.error('Release runs require --release-ready')
        gate=json.loads(args.release_ready.read_text(encoding='utf-8'))
        if gate.get('revision')!=fingerprint or not gate.get('passed'):parser.error('Focused gate must pass for this exact revision')
        cases=[('01',1101),('04',271),('00',4541),('07',1101)]
    for seq,frames in cases:
        try:
            inputs=preflight(args.data_root,seq,frames,budget=budget)
        except (OSError,ValueError,TimeoutError) as error:
            manifest['attempts'].append({'sequence':seq,'status':'input_failure','error_type':type(error).__name__})
            manifest['status']='input_failure';write_json(manifest_path,manifest);return 1
        for variant in args.variants:
            if source_fingerprint() != fingerprint:
                manifest['status']='interrupted_source_change';write_json(manifest_path,manifest);return 1
            if (root/'stop.request').exists():
                manifest['status']='interrupted_requested_stop';write_json(manifest_path,manifest);return 1
            identity={'revision':fingerprint,'sequence':seq,'frames':frames,'input':inputs['sha256'],'variant':variant,
                      'cached':args.feature_cache is not None and variant!='baseline',
                      'reference':hashlib.sha256((args.poses_root/f'{seq}.txt').read_bytes()).hexdigest()}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
            folder=root/f'{seq}-{variant}-{key}'
            report_path=folder/'evaluation.json'
            if not report_path.exists():
                for candidate in (args.output/fingerprint[:12]).glob(f'*/{folder.name}/evaluation.json'):
                    if reusable(json.loads(candidate.read_text(encoding='utf-8')),identity):
                        manifest['attempts'].append({'sequence':seq,'variant':variant,'output':str(candidate.parent.relative_to(args.output/fingerprint[:12])),'status':'reused'})
                        write_json(manifest_path,manifest)
                        break
                else:
                    candidate=None
                if candidate is not None:continue
            if report_path.exists() and reusable(json.loads(report_path.read_text(encoding='utf-8')),identity):
                manifest['attempts'].append({'sequence':seq,'variant':variant,'output':folder.name,'status':'reused'})
                write_json(manifest_path,manifest);continue
            # Use the slowest observed same-case rate; budget conservatively until measured.
            samples=[r['elapsed_s']/r['frames'] for r in manifest['attempts'] if r.get('sequence')==seq and r.get('frames',0)>0 and r.get('elapsed_s') and (r.get('variant')=='baseline')==(variant=='baseline')]
            rate=max(samples,default=2.0 if seq=='04' else 4.0)
            estimate=frames*rate*1.25
            if not budget.permits(estimate,reserve=30):
                manifest.update(status='deferred_budget',next_case={'sequence':seq,'variant':variant,'estimated_seconds':estimate})
                write_json(manifest_path,manifest);print(json.dumps(manifest['next_case']),flush=True);return 2
            folder.mkdir(parents=True,exist_ok=True)
            if report_path.exists():
                # Retain interrupted exports instead of overwriting the evidence.
                folder=folder.with_name(folder.name+'-retry-'+str(time.time_ns()));folder.mkdir()
                report_path=folder/'evaluation.json'
            command=[sys.executable,'-u',str(REPO/'scripts/evaluate_shared_slam.py'),'--stereo','--data-root',str(args.data_root),
                     '--poses-root',str(args.poses_root),'--sequence',seq,'--max-frames',str(frames),'--output',str(folder),
                     '--max-wall-seconds',str(max(1,budget.remaining-30)),*VARIANTS[variant]]
            if variant=='baseline':
                command[2]=str(REPO/'scripts/evaluate_stereo_baseline.py');command.remove('--stereo')
            elif args.feature_cache:command.extend(['--feature-cache',str(args.feature_cache)])
            command.extend(['--stop-file',str(root/'stop.request')])
            if args.profile=='release':
                index=command.index('--max-frames');del command[index:index+2]
            print(f'{seq} {variant}: {frames} frames; estimated {estimate:.0f}s; remaining {budget.remaining:.0f}s',flush=True)
            def started(pid):
                manifest['active_case']={'sequence':seq,'variant':variant,'output':folder.name,'worker_pid':pid,'started_unix':time.time()}
                write_json(manifest_path,manifest)
            result=run_owned(command,folder/'runner.log',max(1,budget.remaining-20),env,on_start=started)
            manifest.pop('active_case',None)
            row={'sequence':seq,'variant':variant,'output':folder.name,**result}
            if report_path.exists():
                report=json.loads(report_path.read_text(encoding='utf-8'));report['development_identity']=identity;write_json(report_path,report)
                row.update(frames=report['frames'],status=report['status'],metrics=report.get('metrics'),stage_timings=report.get('stage_timings'),lost_frames=report['lost_frames'])
            else:row['status']='interrupted_time_budget' if result['timed_out'] else 'worker_failure'
            manifest['attempts'].append(row);write_json(manifest_path,manifest)
            if row['status'] not in ('completed','completed_with_tracking_loss') or row.get('frames')!=frames:
                manifest['status']=row['status'];write_json(manifest_path,manifest);return 1
            if variant!='baseline' and budget.permits(15):
                plot=run_owned([sys.executable,str(REPO/'scripts/plot_shared_slam.py'),'--run',str(folder),'--reference',str(args.poses_root/f'{seq}.txt')],folder/'plot.log',min(15,budget.remaining),env)
                row['plot']=plot;write_json(manifest_path,manifest)
    manifest.update(status='completed_diagnostics',elapsed_s=seconds-budget.remaining)
    manifest.pop('next_case',None)
    write_json(manifest_path,manifest)
    print(manifest_path,flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
