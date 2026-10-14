"""Causal proof/coverage checks. Synthetic tests are never SLAM acceptance."""
import numpy as np

def verify_online_records(rows,events,poses,lifecycle,*,expected_frames,require_local_optimization=True):
    def require(condition,message):
        if not condition:raise ValueError(message)
    require(len(rows)==expected_frames and len(poses)==expected_frames,'incomplete original-ID coverage')
    poses=np.asarray(poses,float)
    require(poses.shape==(expected_frames,3,4) and np.isfinite(poses).all(),'invalid emitted poses')
    require(int(lifecycle['rows'])==expected_frames and int(lifecycle['input_complete'])==1,'incomplete input/lifecycle')
    require(int(lifecycle['workers_joined'])==1,'workers did not join')
    primary=None;held=0;previous=np.eye(4)[:3];last_timestamp=-np.inf;last_complete=-1
    for i,row in enumerate(rows):
        require(int(row['original_id'])==i,'original IDs omitted/reordered')
        start,complete=int(row['start_ns']),int(row['completion_ns'])
        timestamp=float(row['timestamp'])
        require(np.isfinite(timestamp) and timestamp>last_timestamp,'timestamps invalid')
        require(last_complete<=start<=complete,'processing intervals invalid')
        require(int(row['mapping_queue'])>=0,'invalid mapping queue')
        rot=poses[i,:,:3]
        require(np.max(np.abs(rot@rot.T-np.eye(3)))<1e-4 and abs(np.linalg.det(rot)-1)<1e-4,'non-SE3 emission')
        outcome=int(row['outcome']);state=int(row['state']);map_id=int(row['map_id'])
        require(outcome in range(5),'unknown outcome')
        if outcome in (0,1):
            require(state==2 and map_id>=0,'accepted pose without native OK/map')
            if primary is None:
                require(outcome==0,'first accepted pose is not initialization');primary=map_id
            else:require(outcome==1 and map_id==primary,'disconnected map/reinitialization accepted')
        else:
            held+=1
            require(np.array_equal(poses[i],previous),'held pose borrowed a future or corrected pose')
        previous=poses[i].copy();last_timestamp=timestamp;last_complete=complete
    require(primary is not None,'no tracked map')
    require(int(lifecycle['shutdown_start_ns'])>=last_complete and int(lifecycle['shutdown_end_ns'])>=int(lifecycle['shutdown_start_ns']),'invalid shutdown order')
    active={};pose_proof=set();online_local=0;postrun_local=0;last_event=-1
    for event in events:
        t=int(event['monotonic_ns']);admitted=int(event['admitted_original_id']);native=int(event['native_frame_id'])
        require(t>=last_event,'event clock/order invalid');last_event=t
        require(0<=admitted<expected_frames and 0<=native<=admitted,'future/unadmitted frame in optimization')
        kind=event['event']
        if kind.endswith('_solve_begin'):
            key=(kind.removesuffix('_begin'),native)
            require(key not in active,'overlapping same-frame solver event')
            active[key]=(t,admitted)
        elif kind.endswith('_solve_end'):
            key=(kind.removesuffix('_end'),native)
            require(key in active,'solver end without begin');begin,begin_id=active.pop(key)
            require(begin<=t,'negative solver interval')
            if kind=='pose_solve_end':
                row=rows[admitted]
                require(begin_id==admitted and int(row['native_frame_id'])==native,'pose solver frame mismatch')
                require(int(row['start_ns'])<=begin<=t<=int(row['completion_ns']),'pose optimization outside synchronous processing')
                if int(event['iterations'])>0:pose_proof.add(admitted)
            elif int(event['iterations'])>0:
                if t<=last_complete:online_local+=1
                else:postrun_local+=1
    require(not active,'unfinished solver events')
    for i,row in enumerate(rows):
        if int(row['outcome'])==1:require(i in pose_proof,'tracked frame lacks actual before-return optimization')
    if require_local_optimization:require(online_local>0,'no actual local solve completed during image processing')
    return {'frames':expected_frames,'held_frames':held,'tracked_pose_proofs':len(pose_proof),
        'online_local_solves':online_local,'post_emission_local_solves':postrun_local,
        'primary_map_id':primary,'scope':'causal/coverage/lifecycle proof only; accuracy requires evaluator and baseline comparison'}
