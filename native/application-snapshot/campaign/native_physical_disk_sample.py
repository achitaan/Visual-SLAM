"""Low-cost physical C free-space sampling; no cleanup or estimator authority."""
import shutil
import threading
import time

class PhysicalDiskSample:
    def __init__(self,path,reserve_bytes,*,period_s=.1):
        self.path=path;self.reserve=reserve_bytes;self.period=period_s
        self.stop_event=threading.Event();self.min_free=None;self.count=0;self.error=None
        self.thread=threading.Thread(target=self.observe,daemon=True)
    def observe(self):
        try:
            while not self.stop_event.is_set():
                free=shutil.disk_usage(self.path).free
                self.min_free=free if self.min_free is None else min(self.min_free,free)
                self.count+=1;self.stop_event.wait(self.period)
        except Exception as exc:self.error=repr(exc)
    def start(self):self.thread.start()
    def close(self):
        self.stop_event.set();self.thread.join(timeout=2)
        if self.thread.is_alive():raise RuntimeError('disk sampler failed to stop')
    def summary(self):
        if self.thread.is_alive():raise ValueError('terminal sampler required')
        return dict(minimum_sampled_physical_free_bytes=self.min_free,samples=self.count,
            period_s=self.period,reserve_bytes=self.reserve,error=self.error,
            sampled_reserve_gate_pass=self.error is None and self.count>0 and self.min_free>=self.reserve,
            scope='Whole physical C volume, sampled every0.1s; not attribution to one process or a proof of shorter transient minima')
