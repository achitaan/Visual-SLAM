"""Read committed native pose/metadata pairs into immutable application values.

This module neither starts an estimator nor reconstructs optimized history.
Optimizer-event/lifecycle validation remains mandatory for run acceptance.
"""
from dataclasses import dataclass
import math

HEADER=(b'original_id,timestamp,start_ns,completion_ns,state,map_id,outcome,'
        b'pose_calls,pose_frame_id,native_frame_id,ba_started,ba_finished,mapping_queue,map_generation,anchor_generation\n')
FIELDS=HEADER.decode('ascii').strip().split(',')
IDENTITY=(1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.)

@dataclass(frozen=True)
class EmittedFrame:
    original_id:int
    timestamp:float
    pose_T_wc:tuple
    pose_line:bytes
    start_ns:int
    completion_ns:int
    native_state:int
    map_id:int
    outcome:int
    mapping_queue:int
    map_generation:int
    anchor_generation:int

    @property
    def tracking(self):
        # Unknown correspondence/residual counts must not be invented.
        return dict(state='TRACKING' if self.outcome in (0,1) else 'HELD',
                    tracking_ok=self.outcome in (0,1),native_state=self.native_state,
                    native_outcome=self.outcome,num_matches=None,num_inliers=None,
                    inlier_ratio=None,reprojection_error=None)

class OnlinePairDecoder:
    def __init__(self,*,max_buffer_bytes=1024*1024):
        if max_buffer_bytes<512:raise ValueError('buffer capacity must be at least512bytes')
        self._limit=max_buffer_bytes;self._poses=b'';self._metadata=b''
        self._header=False;self._poisoned=False;self._closed=False
        self._count=0;self._map=None;self._generation=None;self._previous=IDENTITY
        self._last_timestamp=-math.inf;self._last_completion=-1

    def _fail(self,message):
        self._poisoned=True
        raise ValueError(message)

    def _ready(self):
        if self._poisoned or self._closed:raise ValueError('decoder is failed or closed')

    def append(self,*,poses=b'',metadata=b''):
        self._ready()
        if not isinstance(poses,bytes) or not isinstance(metadata,bytes):self._fail('bytes required')
        if len(self._poses)+len(poses)>self._limit or len(self._metadata)+len(metadata)>self._limit:
            self._fail('bounded pending-stream buffer exceeded')
        self._poses+=poses;self._metadata+=metadata

    def _line(self,buffer):
        end=buffer.find(b'\n')
        if end>4096 or (end<0 and len(buffer)>4096):self._fail('line length exceeded')
        return None if end<0 else buffer[:end+1]

    def pop_frame(self):
        self._ready()
        try:
            return self._pop()
        except (ValueError,UnicodeError,OverflowError) as exc:
            self._fail(str(exc))

    def _pop(self):
        if not self._header:
            line=self._line(self._metadata)
            if line is None:return None
            if line!=HEADER:self._fail('unexpected native metadata header')
            self._metadata=self._metadata[len(line):];self._header=True
        pline=self._line(self._poses);mline=self._line(self._metadata)
        if pline is None or mline is None:return None
        parts=mline.decode('ascii').strip().split(',')
        if len(parts)!=len(FIELDS):self._fail('wrong metadata width')
        values={k:(float(v) if k=='timestamp' else int(v)) for k,v in zip(FIELDS,parts)}
        ident=values['original_id'];stamp=values['timestamp']
        start,end=values['start_ns'],values['completion_ns']
        state,map_id,outcome=values['state'],values['map_id'],values['outcome']
        generation=values['map_generation'];anchor=values['anchor_generation']
        if not 0<=generation<2**64 or not 0<=anchor<2**64:self._fail('invalid coordinate generation')
        expected_anchor=generation if self._generation is None and outcome in (0,1) else (self._generation or 0)
        if anchor!=expected_anchor:self._fail('anchored coordinate generation changed')
        if ident!=self._count:self._fail('duplicate, missing or reordered original ID')
        if not math.isfinite(stamp) or stamp<=self._last_timestamp:self._fail('invalid input timestamp')
        if start<0 or start<self._last_completion or end<start:self._fail('invalid processing order')
        if outcome not in range(5) or values['mapping_queue']<0:self._fail('invalid outcome/queue')
        if not 0<=values['native_frame_id']<=ident:self._fail('future native frame')
        pose=tuple(float(v) for v in pline.decode('ascii').split())
        if len(pose)!=12 or not all(math.isfinite(v) for v in pose):self._fail('invalid pose values')
        r=[pose[i*4:i*4+3] for i in range(3)]
        if any(abs(sum(r[i][k]*r[j][k] for k in range(3))-(i==j))>1e-4 for i in range(3) for j in range(3)):
            self._fail('non-orthogonal rotation')
        det=r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])-r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])+r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0])
        if abs(det-1)>1e-4:self._fail('non-SE3 rotation')
        if outcome in (0,1):
            if state!=2 or map_id<0 or generation==0:self._fail('accepted pose lacks native OK/map/generation')
            if self._map is None:
                if outcome!=0:self._fail('first accepted pose lacks initialization')
            elif outcome!=1 or map_id!=self._map or generation!=self._generation:self._fail('disconnected/reinitialized map accepted')
        else:
            bad_map=map_id<0 or generation==0 or (self._map is not None and (map_id!=self._map or generation!=self._generation))
            if (outcome==2 and state==2) or (outcome==3 and (state!=2 or not bad_map)) or (outcome==4 and (state!=2 or bad_map)):
                self._fail('hold outcome inconsistent with native emission contract')
            if pose!=self._previous:self._fail('held pose differs from last emitted pose')
        frame=EmittedFrame(ident,stamp,tuple(pose[i*4:i*4+4] for i in range(3))+((0.,0.,0.,1.),),
                           pline,start,end,state,map_id,outcome,values['mapping_queue'],generation,anchor)
        # Commit only after every field validates. Returned tuples/bytes own their values.
        self._poses=self._poses[len(pline):];self._metadata=self._metadata[len(mline):]
        if outcome in (0,1):self._map=map_id;self._generation=generation
        self._count+=1;self._previous=pose;self._last_timestamp=stamp;self._last_completion=end
        return frame

    def finish(self,expected_frames):
        self._ready()
        if expected_frames<1 or self._count!=expected_frames or not self._header or self._poses or self._metadata:
            self._fail('incomplete or unconsumed original-ID stream')
        self._closed=True
