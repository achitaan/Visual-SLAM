"""Application-only authenticated Stop beside the unchanged absolute watchdog.

The control channel is the application-owned WSL stdin pipe. It admits no PID,
path, shell command, resume, or replacement deadline from a client. The outer
driver is our direct paused child; its existing nested watchdogs remain armed.
"""
from __future__ import annotations
import hashlib,hmac,json,math,os,re,select,signal,subprocess,sys,time
from pathlib import Path
from native_owned_deadline import still_owned
from native_process_owner_guarded import OwnedPausedProcess as OriginalOwner
STOP_RUN='NATIVE_APPLICATION_STOP_RUN'
STOP_DIGEST='NATIVE_APPLICATION_STOP_DIGEST'
MAX_CONTROL=512

def validate_control(payload,run_id,digest):
    if not isinstance(payload,bytes) or len(payload)>MAX_CONTROL:return False
    def unique(pairs):
        if len({key for key,_ in pairs})!=len(pairs):raise ValueError('duplicate control key')
        return dict(pairs)
    try:value=json.loads(payload.decode('utf8'),object_pairs_hook=unique)
    except (ValueError,UnicodeError):return False
    if not isinstance(value,dict) or set(value)!={'action','run_id','capability'}:return False
    if value['action']!='stop_run' or value['run_id']!=run_id:return False
    capability=value['capability']
    if not isinstance(capability,str) or not re.fullmatch(r'[a-f0-9]{64}',capability):return False
    return hmac.compare_digest(hashlib.sha256(capability.encode('ascii')).hexdigest(),digest)

def watch_stop(pid,ticks,deadline,run_id,digest,output):
    if sys.platform!='linux' or not math.isfinite(deadline) or not re.fullmatch(r'[a-z0-9_]+',run_id) or not re.fullmatch(r'[a-f0-9]{64}',digest):raise ValueError('invalid application Stop ownership')
    expected=(ticks,pid,pid)
    state=dict(schema='native_application_stop_owner_v1',status='armed',pid=pid,pid_start_ticks=ticks,process_group=pid,session=pid,original_deadline_monotonic_s=deadline,run_id=run_id,rejected_controls=0,signals=[],outer_driver_cancel_grace_s=.1)
    with output.open('x',encoding='utf8') as stream:json.dump(state,stream)
    print(json.dumps(dict(status='armed')),flush=True)
    buffer=bytearray();discarding=False
    try:
        while still_owned(pid,expected):
            remaining=deadline-time.monotonic()
            if remaining<=0:
                state['status']='original_deadline_left_to_unchanged_watchdog';return
            ready,_,_=select.select([sys.stdin.buffer],[],[],min(.025,remaining))
            if not ready:continue
            part=os.read(sys.stdin.fileno(),1024)
            if not part:
                reason='application_control_pipe_closed';break
            accepted=False
            for byte in part:
                if byte==10:
                    if not discarding and validate_control(bytes(buffer),run_id,digest):accepted=True;break
                    state['rejected_controls']+=1;buffer.clear();discarding=False
                elif not discarding:
                    buffer.append(byte)
                    if len(buffer)>MAX_CONTROL:buffer.clear();discarding=True
            if accepted:
                reason='authenticated_stop';state['request_observed_monotonic_s']=time.monotonic();break
        else:
            state['status']='owned_driver_exited_without_stop_signal';return
        # Check PID generation/group/session before EACH signal. No descendant
        # discovery or PID-only kill; child watchdogs independently handle EOF.
        for sig in (signal.SIGTERM,signal.SIGKILL):
            if not still_owned(pid,expected):break
            try:os.killpg(pid,sig)
            except ProcessLookupError:break
            state['signals'].append(dict(signal=int(sig),monotonic_s=time.monotonic()))
            if sig==signal.SIGTERM:
                until=time.monotonic()+.1
                while still_owned(pid,expected) and time.monotonic()<until:time.sleep(.02)
        state['status']='owned_driver_stopped_'+reason
    finally:
        state['completed_monotonic_s']=time.monotonic()
        output.write_text(json.dumps(state,indent=2)+'\n',encoding='utf8')

class OwnedPausedProcess(OriginalOwner):
    """New application owner; no changes to benchmark ownership helpers."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.stop_guard=None;self.stop_result=Path(str(self.result_path)+'.application-stop.json')
    def start(self,command,*,env,stdout,**kwargs):
        run_id=env.get(STOP_RUN,'');digest=env.get(STOP_DIGEST,'')
        if not re.fullmatch(r'[a-z0-9_]+',run_id) or not re.fullmatch(r'[a-f0-9]{64}',digest):raise ValueError('application-bound Stop configuration required')
        if self.stop_result.exists():raise FileExistsError(self.stop_result)
        process,guard=super().start(command,env=env,stdout=stdout,**kwargs)
        try:
            self.stop_guard=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--watch-stop',str(process.pid),str(guard.expected[0]),repr(self.deadline),run_id,digest,str(self.stop_result)],stdin=sys.stdin.buffer,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            ready,_,_=select.select([self.stop_guard.stdout],[],[],3.)
            if not ready or json.loads(self.stop_guard.stdout.readline()).get('status')!='armed':raise TimeoutError('application Stop guardian admission failed')
            return process,guard
        except BaseException:
            self.close();raise
    def close(self):
        try:super().close()
        finally:
            watcher=self.stop_guard
            if watcher is not None:
                try:watcher.wait(timeout=14)
                except subprocess.TimeoutExpired:
                    # This exact unreaped Popen child is ours; never an external
                    # process identity or a request-supplied PID.
                    watcher.kill();watcher.wait(timeout=4);raise
                finally:
                    watcher.stdout.close();watcher.stderr.close();self.stop_guard=None
                if watcher.returncode!=0:raise RuntimeError('application Stop guardian failed')

if __name__=='__main__':
    if len(sys.argv)!=8 or sys.argv[1]!='--watch-stop':raise ValueError('application Stop guardian requires its owned launcher')
    watch_stop(int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4]),sys.argv[5],sys.argv[6],Path(sys.argv[7]))
