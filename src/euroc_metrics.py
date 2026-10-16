"""Evaluator-only EuRoC reference conversion and metric one-second RPE.

This module must be called after estimation. No IMU observations enter tracking.
Reference CSV poses are T_R_S; sensor.yaml supplies T_B_S. The evaluated camera
pose is T_R_S @ inverse(T_B_S) @ T_B_rectified_left.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml
from euroc import read_transform
from kitti import validate_pose
from metrics import compute_ate_rmse, umeyama_alignment


def reference_camera_poses(folder, timestamps_ns, body_from_camera, max_gap_ns=20_000_000):
    folder = Path(folder)
    if (folder/'mav0').is_dir():
        folder = folder/'mav0'
    if (folder/'state_groundtruth_estimate0').is_dir():
        folder = folder/'state_groundtruth_estimate0'
    calibration = yaml.safe_load((folder/'sensor.yaml').read_text(encoding='utf-8'))
    sensor_from_body = np.linalg.inv(read_transform(calibration['T_BS']))
    validate_pose(np.asarray(body_from_camera))
    times, positions, quaternions = [], [], []
    expected = ['timestamp','p_RS_R_x','p_RS_R_y','p_RS_R_z','q_RS_w','q_RS_x','q_RS_y','q_RS_z']
    header_valid = False
    with (folder/'data.csv').open(newline='',encoding='utf-8') as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if row[0].lstrip().startswith('#'):
                names = [v.lstrip('#').strip().split()[0] for v in row if v.strip()]
                if names[:8] == expected:
                    header_valid = True
                continue
            if not header_valid or len(row) < 8:
                raise ValueError('Expected EuRoC p_RS_R and scalar-first q_RS reference columns')
            timestamp = int(row[0])
            values = np.asarray(row[1:8],float)
            if (not np.isfinite(values).all() or timestamp < 0
                    or (times and timestamp <= times[-1])):
                raise ValueError('Invalid reference state or timestamp ordering')
            quaternion = values[[4,5,6,3]]  # w,x,y,z -> x,y,z,w
            if not np.isclose(np.linalg.norm(quaternion),1.,atol=1e-3):
                raise ValueError('Reference quaternion must be normalized')
            times.append(timestamp); positions.append(values[:3]); quaternions.append(quaternion)
    if len(times) < 2:
        raise ValueError('Need at least two reference states')
    times = np.asarray(times,np.int64)
    requested = np.asarray(timestamps_ns,np.int64)
    if requested.ndim != 1 or np.any(np.diff(requested) <= 0) or max_gap_ns <= 0:
        raise ValueError('Evaluation timestamps must be strictly increasing')
    upper = np.searchsorted(times,requested)
    clipped = np.minimum(upper,len(times)-1)
    exact = (upper < len(times)) & (times[clipped] == requested)
    lower = np.maximum(upper-1,0)
    available = exact | ((upper > 0) & (upper < len(times))
                         & ((times[clipped]-times[lower]) <= max_gap_ns))
    indices = np.flatnonzero(available)
    seconds = (times-times[0]).astype(float)*1e-9
    query = (requested[indices]-times[0]).astype(float)*1e-9
    if not len(indices):
        return indices, np.empty((0,4,4))
    rotations = Slerp(seconds,Rotation.from_quat(quaternions))(query).as_matrix()
    positions = np.asarray(positions)
    translations = np.column_stack([np.interp(query,seconds,positions[:,axis]) for axis in range(3)])
    poses = np.repeat(np.eye(4)[None],len(indices),axis=0)
    poses[:,:3,:3] = rotations
    poses[:,:3,3] = translations
    return indices, poses @ sensor_from_body @ body_from_camera


def evaluate_euroc(reference, estimated, timestamps_ns, reference_indices, states, rpe_tolerance_ns=20_000_000):
    """Held/dropped estimates remain in ATE/RPE wherever reference is available."""
    estimated, reference = np.asarray(estimated), np.asarray(reference)
    times = np.asarray(timestamps_ns,np.int64)
    indices = np.asarray(reference_indices,dtype=int)
    if (len(estimated) != len(times) or len(states) != len(times) or len(reference) != len(indices)
            or len(indices) < 2 or np.any(np.diff(times) <= 0)
            or np.any(np.diff(indices) <= 0) or np.any(indices < 0) or np.any(indices >= len(times))):
        raise ValueError('Invalid full trajectory or reference association')
    for pose in list(estimated)+list(reference):
        validate_pose(pose)
    paired = estimated[indices]
    xyz, target = paired[:,:3,3], reference[:,:3,3]
    rotation, _, translation = umeyama_alignment(xyz,target,with_scale=False)
    aligned = xyz @ rotation.T + translation
    paired_times = times[indices]
    translation_errors, rotation_errors = [], []
    for i, timestamp in enumerate(paired_times):
        wanted = int(timestamp)+1_000_000_000
        j = int(np.searchsorted(paired_times,wanted))
        options = [k for k in (j-1,j) if i < k < len(paired_times)]
        if not options:
            continue
        j = min(options,key=lambda k:abs(int(paired_times[k])-wanted))
        if abs(int(paired_times[j])-wanted) > rpe_tolerance_ns:
            continue
        truth_delta = np.linalg.inv(reference[i]) @ reference[j]
        estimate_delta = np.linalg.inv(paired[i]) @ paired[j]
        error = np.linalg.inv(truth_delta) @ estimate_delta
        translation_errors.append(float(np.linalg.norm(error[:3,3])))
        rotation_errors.append(float(np.degrees(np.arccos(np.clip((np.trace(error[:3,:3])-1)/2,-1,1)))))
    tracked = [state in ('tracking','relocalized') for state in states]
    intervals = []
    i = 0
    while i < len(tracked):
        if tracked[i]:
            i += 1
            continue
        start = i
        while i < len(tracked) and not tracked[i]:
            i += 1
        end = min(i,len(times)-1)
        intervals.append({'first_frame':start,'last_lost_frame':i-1,
                          'recovered_frame':i if i < len(times) else None,
                          'duration_s':float(times[end]-times[start])*1e-9,
                          'censored':i == len(times)})
    rmse = lambda values: float(np.sqrt(np.mean(np.square(values)))) if values else None
    return {'ate_alignment':'se3','alignment_scale':1.,'frames':len(times),
            'reference_frames':len(indices),'reference_coverage':len(indices)/len(times),
            'ate_rmse_m':compute_ate_rmse(target,aligned),'raw_ate_rmse_m':compute_ate_rmse(target,xyz),
            'rpe_interval_s':1.,'rpe_pairs':len(translation_errors),
            'rpe_translation_rmse_m':rmse(translation_errors),'rpe_rotation_rmse_deg':rmse(rotation_errors),
            'tracking_coverage':sum(tracked)/len(tracked),'lost_or_dropped_frames':len(tracked)-sum(tracked),
            'recovery_intervals':intervals,'reference_indices':indices.tolist(),
            'held_and_dropped_frames_included':True,'kitti_segment_scores_applied':False}
