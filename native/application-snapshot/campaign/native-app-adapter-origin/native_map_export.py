"""Portable exports from an owned online map; never correct emitted poses."""
import json
from pathlib import Path
import numpy as np

def require(ok,message):
    if not ok:raise ValueError(message)

def integer(v):
    require(isinstance(v,(int,np.integer)) and not isinstance(v,bool),'integer identity required')
    require(v>=0,'negative identity');return int(v)

def rigid(value):
    p=np.asarray(value,dtype=float)
    require(p.size==16 and np.isfinite(p).all(),'finite4x4 pose required');p=p.reshape(4,4)
    require(np.array_equal(p[3],[0,0,0,1]),'homogeneous row invalid')
    require(np.allclose(p[:3,:3]@p[:3,:3].T,np.eye(3),rtol=0,atol=1e-4) and abs(np.linalg.det(p[:3,:3])-1)<=1e-4,'proper rigid pose required')
    return p

def validate_snapshot(s,rows):
    require(s['version']==2,'coordinate-generation snapshot version required')
    generation=integer(s['map_generation']);require(generation>0,'positive map generation required')
    frame=integer(s['original_id']);map_id=integer(s['map_id'])
    require(frame<len(rows),'snapshot frame outside original coverage')
    require([int(r['original_id']) for r in rows]==list(range(len(rows))),'original row gap')
    accepted=[r for r in rows[:frame+1] if int(r['outcome']) in (0,1)]
    require(accepted and int(accepted[0]['map_id'])==map_id,'disconnected snapshot map')
    require(all(int(r['map_generation'])==generation and int(r['anchor_generation'])==generation for r in accepted),'snapshot spans accepted coordinate generations')
    require(int(rows[frame]['map_generation'])==generation and int(rows[frame]['anchor_generation'])==generation,'snapshot current origin is not anchored')
    times=[integer(s[k]) for k in ('request_ns','capture_ns','completion_ns')]
    require(int(rows[frame]['completion_ns'])<=times[0]<=times[1]<=times[2],'snapshot predates emission or clock order invalid')
    if frame+1<len(rows):require(times[2]<=int(rows[frame+1]['start_ns']),'snapshot includes future input interval')
    points={};keys={};original={}
    for k in s['keyframes']:
        ident=integer(k['id']);native=integer(k['native_frame_id']);require(ident not in keys,'duplicate keyframe')
        pose=rigid(k['pose_T_wc']);require(np.isfinite(k['timestamp']),'nonfinite keyframe time')
        matches=[int(r['original_id']) for r in rows[:frame+1] if int(r['map_id'])==map_id and int(r['map_generation'])==generation and int(r['native_frame_id'])==native and float(r['timestamp'])==k['timestamp']]
        require(len(matches)==1,'future/missing/ambiguous native frame association')
        keys[ident]=pose;original[ident]=matches[0]
    for point in s['points']:
        require(len(point)==4,'point shape');ident=integer(point[0]);require(ident not in points,'duplicate point')
        xyz=np.asarray(point[1:],dtype=float);require(np.isfinite(xyz).all(),'nonfinite point');points[ident]=xyz
    claims=set();pairs=set()
    for obs in s['observations']:
        require(len(obs)==5,'observation shape');key,point,feature=map(integer,obs[:3])
        require(key in keys and point in points,'missing observation endpoint')
        require((key,feature) not in claims and (key,point) not in pairs,'conflicting observation claim')
        require(np.isfinite(np.asarray(obs[3:],dtype=float)).all(),'nonfinite pixel')
        claims.add((key,feature));pairs.add((key,point))
    require(keys and points and claims,'empty connected snapshot')
    return dict(original_ids=original,keyframes=keys,points=points,frame=frame,map_id=map_id,map_generation=generation)

def export_snapshot(output,s,rows,pose_bytes,K):
    v=validate_snapshot(s,rows);K=np.asarray(K,dtype=float)
    require(K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 and np.array_equal(K[2],[0,0,1]),'invalid pinhole calibration')
    require(isinstance(pose_bytes,bytes) and pose_bytes.endswith(b'\n'),'complete immutable pose bytes required')
    lines=pose_bytes.splitlines();require(len(lines)==v['frame']+1,'all original emitted poses required')
    poses=[]
    for line in lines:
        numbers=np.asarray([float(x) for x in line.split()]);require(numbers.size==12,'pose row shape')
        p=np.eye(4);p[:3]=numbers.reshape(3,4);poses.append(rigid(p))
    keyframes=[]
    for k in s['keyframes']:
        ident=k['id'];obs=[dict(landmark=o[1],pixel=list(o[3:]),position=v['points'][o[1]].tolist(),native_feature_index=o[2]) for o in s['observations'] if o[0]==ident]
        keyframes.append(dict(id=ident,frame=v['original_ids'][ident],pose=v['keyframes'][ident].tolist(),image=None,observations=obs))
    payload=dict(version=1,revision=s['map_revision'],translation_scale='metric',camera_matrix=K.tolist(),keyframes=keyframes,
        sparse_points=len(v['points']),depth_source='geometry',tracking=None,bundle_adjustment=None,loop_events=[],
        configuration={'backend':'ORB-SLAM3','sensor':'rectified_stereo','loop_mode':'off'},performance_configuration=None,
        matching_backend={'backend':'native_ORB'},independent_stereo_motion=None,verified_loops=[],
        native_snapshot=dict(original_id=s['original_id'],map_id=s['map_id'],map_generation=s['map_generation'],request_ns=s['request_ns'],capture_ns=s['capture_ns'],completion_ns=s['completion_ns'],
            observations_outside_snapshot=s['observations_outside_snapshot'],trajectory='immutable emitted poses; map keyframes optimized at capture'))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'poses.txt').write_bytes(pose_bytes)
    with (output/'sparse.ply').open('x',encoding='ascii') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(v["points"])}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for xyz in v['points'].values():f.write(' '.join(format(x,'.17g') for x in xyz)+' 160 160 160\n')
    (output/'run.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    preview=dict(revision=s['map_revision'],translation_scale='metric',depth_source='geometry',sparse=[p.tolist() for p in v['points'].values()],dense=[],trajectory=[p[:3,3].tolist() for p in poses])
    (output/'preview.json').write_text(json.dumps(preview,allow_nan=False)+'\n')
    return payload
