"""Bounded reads of committed output; never restarts or signals a writer.

Only ENODATA is retryable, based on the preserved live /mnt/c failure. This
does not explain its OS cause. A recovered read is an observation, never proof
of process completion, row coverage, numerical correctness or acceptance.
"""
from pathlib import Path
import errno,math,os,time

class ObservationBudget:
    def __init__(self,*,window_s=.5,retry_s=.02,max_errors=8,clock=time.monotonic,sleep=time.sleep):
        if not math.isfinite(window_s) or window_s<=0 or not math.isfinite(retry_s) or retry_s<=0:
            raise ValueError('finite positive observation bounds required')
        if type(max_errors) is not int or max_errors<0:raise ValueError('nonnegative integer error budget required')
        self.window_s=window_s;self.retry_s=retry_s;self.max_errors=max_errors
        self.clock=clock;self.sleep=sleep;self.transient_errors=0;self.events=[]
    def call(self,operation,*,deadline,poll=lambda:None):
        if not math.isfinite(deadline):raise ValueError('finite absolute deadline required')
        stop=min(deadline,self.clock()+self.window_s)
        while True:
            if self.clock()>=stop or self.transient_errors>self.max_errors:
                raise TimeoutError('observation read budget exhausted')
            try:
                result=operation()
                if self.clock()>=stop:raise TimeoutError('observation read exceeded deadline')
                return result
            except OSError as error:
                if error.errno!=getattr(errno,'ENODATA',61):raise
                self.transient_errors+=1
                if len(self.events)<=self.max_errors:
                    self.events.append(dict(monotonic_s=self.clock(),errno=error.errno,process_exit_code=poll(),attempt=self.transient_errors))
                remaining=stop-self.clock()
                if self.transient_errors>self.max_errors or remaining<=0:
                    raise TimeoutError('transient observation error budget exhausted') from error
                self.sleep(min(self.retry_s,remaining))

def _snapshot(path):
    path=Path(path)
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno());data=stream.read();after=os.fstat(stream.fileno())
        visible=path.stat()
    identity=(before.st_dev,before.st_ino)
    if identity!=(after.st_dev,after.st_ino) or identity!=(visible.st_dev,visible.st_ino):
        raise ValueError('native output replaced during read')
    if after.st_size<before.st_size:raise ValueError('native output truncated during read')
    return data,identity

class CommittedPrefixObserver:
    def __init__(self,*,read_snapshot=_snapshot,**budget_options):
        self.budget=ObservationBudget(**budget_options);self.read_snapshot=read_snapshot
        self.prefix=b'';self.identity=None;self.raw_size=0;self.successful_reads=0
    def read_prefix(self,path,*,deadline,poll=lambda:None):
        data,identity=self.budget.call(lambda:self.read_snapshot(path),deadline=deadline,poll=poll)
        if self.identity is not None and identity!=self.identity:raise ValueError('native output replaced')
        if len(data)<self.raw_size:raise ValueError('native output truncated')
        committed=data[:data.rfind(b'\n')+1]
        if not committed.startswith(self.prefix):raise ValueError('emitted history rewritten')
        # Commit observer state only after every identity/size/prefix check passes.
        self.identity=identity;self.raw_size=len(data);self.prefix=committed;self.successful_reads+=1
        return committed
    def statistics(self):
        return dict(successful_reads=self.successful_reads,transient_errors=self.budget.transient_errors,
                    events=list(self.budget.events),window_s=self.budget.window_s,
                    retry_s=self.budget.retry_s,max_errors=self.budget.max_errors)
