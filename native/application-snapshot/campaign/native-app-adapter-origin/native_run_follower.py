"""Follow an owned native process's output directory without changing its history.

The caller owns process creation/termination and supplies its poll method. This
reader never signals arbitrary processes. A stream/lifecycle success is not a
replacement for optimizer-event verification or an accuracy benchmark.
"""
from pathlib import Path
import csv,math,os,time
from native_online_adapter import OnlinePairDecoder

class NativeRunFollower:
    def __init__(self,directory,expected_frames,*,timeout_s=180,poll_s=.01,
                 clock=time.monotonic,sleep=time.sleep):
        if expected_frames<1 or not math.isfinite(timeout_s) or timeout_s<=0:
            raise ValueError('positive frame count and finite deadline required')
        if not math.isfinite(poll_s) or not 0<poll_s<=1:raise ValueError('poll interval must be in(0,1]')
        self.directory=Path(directory);self.expected_frames=expected_frames
        self.timeout_s=timeout_s;self.poll_s=poll_s;self.clock=clock;self.sleep=sleep
        self.status='new';self.lifecycle=None
        self._offset={};self._identity={}

    def _read_new(self,name):
        path=self.directory/name
        try:
            with path.open('rb') as f:
                stat=os.fstat(f.fileno());identity=(stat.st_dev,stat.st_ino)
                if name in self._identity and self._identity[name]!=identity:raise ValueError('native output replaced')
                self._identity[name]=identity;offset=self._offset.get(name,0)
                if stat.st_size<offset:raise ValueError('native output truncated')
                f.seek(offset);data=f.read(65536);self._offset[name]=offset+len(data);return data
        except FileNotFoundError:
            if name in self._identity:raise ValueError('native output disappeared')
            return b''

    def _verify_lifecycle(self,last):
        with (self.directory/'lifecycle.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        fields=('rows','input_complete','workers_joined','shutdown_start_ns','shutdown_end_ns')
        if len(rows)!=1 or any(k not in rows[0] for k in fields):raise ValueError('invalid lifecycle record')
        v={k:int(rows[0][k]) for k in fields}
        if v['rows']!=self.expected_frames or v['input_complete']!=1 or v['workers_joined']!=1:
            raise ValueError('incomplete input or worker shutdown')
        if not last.completion_ns<=v['shutdown_start_ns']<=v['shutdown_end_ns']:
            raise ValueError('invalid shutdown chronology')
        self.lifecycle=v

    def frames(self,poll):
        if self.status!='new':raise ValueError('one owned stream per follower')
        self.status='running';decoder=OnlinePairDecoder();deadline=self.clock()+self.timeout_s;last=None
        try:
            while True:
                if self.clock()>=deadline:raise TimeoutError('native stream deadline exceeded')
                # Sample exit BEFORE reading, then drain remaining final bytes.
                # Sampling exit after reading can miss the child's final write.
                exit_code=poll()
                poses=self._read_new('online_poses.txt');metadata=self._read_new('online_frames.csv')
                decoder.append(poses=poses,metadata=metadata)
                while (frame:=decoder.pop_frame()) is not None:
                    last=frame;yield frame
                if exit_code is not None:
                    if exit_code!=0:raise ValueError(f'native process exited with code{exit_code}')
                    if not poses and not metadata:
                        decoder.finish(self.expected_frames);self._verify_lifecycle(last)
                        self.status='complete_stream_lifecycle_not_accuracy_acceptance';return
                if not poses and not metadata:self.sleep(self.poll_s)
        finally:
            if self.status!='complete_stream_lifecycle_not_accuracy_acceptance':self.status='incomplete'
