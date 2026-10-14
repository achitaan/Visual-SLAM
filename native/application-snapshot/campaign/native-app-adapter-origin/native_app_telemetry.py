"""Actual native values for the existing telemetry transport.

Unknown map counts stay null until a validated snapshot exists. Consumers must
support that explicit unavailable state. Display sampling never changes the
full sparse map or the immutable emitted trajectory.
"""
import copy,math
import numpy as np
from telemetry import make_frame_message,TelemetryState
from native_map_export import validate_snapshot

class NativeTelemetry:
    def __init__(self, *, sequence, total_frames, run_id, max_display_points=5000):
        if type(max_display_points) is not int or max_display_points<1:
            raise ValueError('positive display point bound required')
        self.sequence=sequence;self.total_frames=total_frames;self.run_id=run_id
        self.limit=max_display_points;self.frames=[];self.rows=[]
        self._map=None;self._initialized=False

    def push(self, frame):
        if frame.original_id!=len(self.frames) or frame.original_id>=self.total_frames:
            raise ValueError('original emitted frame sequence violated')
        self.frames.append(frame);self.rows.append(frame.metadata_row)
        self._initialized |= frame.outcome in (0,1)
        return self.message()

    def accept_snapshot(self,snapshot):
        valid=validate_snapshot(snapshot,self.rows)
        if self._map is not None and valid['frame']<=self._map['original_id']:
            raise ValueError('duplicate or reordered map snapshot')
        points=list(valid['points'].values());stride=max(1,math.ceil(len(points)/self.limit))
        self._map=dict(original_id=valid['frame'],map_id=valid['map_id'],map_generation=str(valid['map_generation']),
                       request_ns=snapshot['request_ns'],capture_ns=snapshot['capture_ns'],
                       completion_ns=snapshot['completion_ns'],revision=snapshot['map_revision'],
                       keyframes=len(valid['keyframes']),map_points=len(points),
                       display_points=[p.tolist() for p in points[::stride]],
                       display_points_sampled=stride>1)
        return self.message()

    def message(self):
        if not self.frames:raise ValueError('no emitted frame to display')
        frame=self.frames[-1];tracking=frame.tracking
        tracking['state']='tracking' if frame.outcome in (0,1) else ('lost' if self._initialized else 'initializing')
        mapping=(dict(available=False,keyframes=None,map_points=None) if self._map is None
                 else dict(available=True,keyframes=self._map['keyframes'],map_points=self._map['map_points'],
                           revision=self._map['revision'],snapshot_original_id=self._map['original_id']))
        message=make_frame_message(frame_index=frame.original_id,timestamp=frame.timestamp,
            pose_T_wc=np.asarray(frame.pose_T_wc),tracking=tracking,map_state=mapping,
            map_points=None if self._map is None else copy.deepcopy(self._map['display_points']),
            sequence=self.sequence,total_frames=self.total_frames,run_id=self.run_id,
            translation_scale='metric',state=TelemetryState(mode='slam',mode_locked=True))
        message['trajectory']=[[f.pose_T_wc[i][3] for i in range(3)] for f in self.frames]
        message['native_origin']=dict(version=2,map_id=frame.map_id,map_generation=str(frame.map_generation),anchor_generation=str(frame.anchor_generation))
        message['native_snapshot']=None if self._map is None else {k:v for k,v in self._map.items() if k!='display_points'}
        message['native_coverage']=dict(emitted_original_rows=len(self.frames),
            held_rows=sum(f.outcome not in (0,1) for f in self.frames),
            display_delivery_is_not_coverage=True,mapping_queue=frame.mapping_queue)
        return message
