"""Batch stereo pose-graph experiment: real image loops, raw/corrected full-frame evaluation."""
import argparse
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import hashlib
import json
from pathlib import Path
import time
import cv2 as cv
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from kitti import load_poses_txt, save_poses_txt, validate_sequence
from loop_geometry import extract_loop_frame, verify_loop, propagate_corrections
from metrics import evaluate_trajectory
from SLAM import SLAM
from StereoVisualOdometry import StereoVisualOdometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--poses-root', type=Path, required=True)
    parser.add_argument('--estimates-root', type=Path, required=True)
    parser.add_argument('--sequence', default='07')
    parser.add_argument('--output-root', type=Path, default=Path('results/pose-graph'))
    parser.add_argument('--keyframe-step', type=int, default=20)
    parser.add_argument('--min-frame-gap', type=int, default=150)
    parser.add_argument('--resume-graph', action='store_true', help='Reuse saved image constraints for the unchanged raw trajectory')
    args = parser.parse_args()
    if args.keyframe_step < 1 or args.min_frame_gap < 1:
        parser.error('Keyframe step and temporal exclusion must be positive')
    cv.setRNGSeed(0)
    cv.setNumThreads(1)
    folder = validate_sequence(args.data_root, args.sequence, True)
    raw = load_poses_txt(args.estimates_root / f'{args.sequence}.txt')
    vo = StereoVisualOdometry(str(folder / 'image_'), str(folder / 'calib.txt'), False, draw_matches=False)
    if len(raw) != len(vo.Images_1):
        raise ValueError('Graph experiment requires a complete raw trajectory')
    indices = sorted(set(range(0, len(raw), args.keyframe_step)) | {len(raw) - 1})
    started = time.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_root / f'{args.sequence}-constraints.json'
    source_hash = hashlib.sha256((args.estimates_root / f'{args.sequence}.txt').read_bytes()).hexdigest()
    if args.resume_graph:
        cache = json.loads(cache_path.read_text())
        if cache['raw_trajectory_sha256'] != source_hash or cache['keyframe_indices'] != indices or cache['min_frame_gap'] != args.min_frame_gap or cache.get('loop_feature_limit') != 1500:
            raise ValueError('Saved constraints do not match the raw trajectory/configuration')
        loops, attempted = cache['loops'], cache['candidate_pairs_tested']
    else:
        loops, attempted = find_image_loops(vo, indices, args.min_frame_gap)
        cache_path.write_text(json.dumps({'sequence': args.sequence, 'raw_trajectory_sha256': source_hash,
                                         'keyframe_indices': indices, 'min_frame_gap': args.min_frame_gap,
                                         'loop_feature_limit': 1500,
                                         'candidate_pairs_tested': attempted, 'loops': loops}, indent=2, allow_nan=False))
    corrected = optimize_image_graph(raw, indices, loops)
    # Ground truth is loaded only after image measurements and optimization finish.
    gt = load_poses_txt(args.poses_root / f'{args.sequence}.txt')
    before = evaluate_trajectory(gt, raw, 'se3')
    after = evaluate_trajectory(gt, corrected, 'se3')
    save_poses_txt(args.output_root / f'{args.sequence}.txt', corrected)
    report = {'sequence': args.sequence, 'experiment': 'batch_stereo_pose_graph', 'frames': len(raw),
              'keyframe_count': len(indices), 'keyframe_indices': indices, 'candidate_pairs_tested': attempted,
              'verified_loop_count': len(loops), 'loops': loops, 'ground_truth_used_for_constraints': False,
              'odometry_information_diagonal': [1000.] * 3 + [10.] * 3,
              'loop_information_diagonal': [2000.] * 3 + [20.] * 3,
              'keyframe_step': args.keyframe_step, 'min_frame_gap': args.min_frame_gap,
              'loop_feature_limit': 1500,
              'elapsed_s': time.perf_counter() - started, 'reused_image_constraints': args.resume_graph,
              'before': before, 'after': after,
              'scope': 'Offline graph correction and full-frame propagation; live tracking/map feedback remains separate.'}
    (args.output_root / f'{args.sequence}.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ('before', 'after', 'loops', 'keyframe_indices')}, indent=2))
    print('Before:', {k: v for k, v in before.items() if k != 'segments'})
    print('After:', {k: v for k, v in after.items() if k != 'segments'})


def optimize_image_graph(raw, indices, loops, max_evaluations=500, diagnostics=None):
    graph = SLAM()
    graph.initial_poses = [raw[index] for index in indices]
    odometry_information = np.diag([1000.] * 3 + [10.] * 3)
    loop_information = np.diag([2000.] * 3 + [20.] * 3)
    for k in range(len(indices) - 1):
        graph.add_odometry_edge(k, k + 1, np.linalg.inv(raw[indices[k]]) @ raw[indices[k + 1]], odometry_information)
    for loop in loops:
        graph.add_loop_closure_edge(indices.index(loop['first_frame']), indices.index(loop['second_frame']), np.array(loop['measurement']), loop_information)
    if loops:
        print(f'Optimizing {len(indices)} vertices with {len(loops)} verified loops...', flush=True)
        optimized = graph.optimize_pose_graph(num_iterations=max_evaluations, diagnostics=diagnostics)
        corrected = propagate_corrections(raw, indices, optimized)
    else:
        corrected = [pose.copy() for pose in raw]
        if diagnostics is not None:
            diagnostics.update(solver='not run', reason='No verified image loops', success=True, function_evaluations=0)
    return corrected


def find_image_loops(vo, indices, min_frame_gap, frames=None):
    if frames is None:
        frames = []
        for index in indices:
            frames.append(extract_loop_frame(vo, index))
            if len(frames) % 10 == 0:
                print(f'Extracted stereo geometry {len(frames)}/{len(indices)} keyframes', flush=True)
    training = [f.descriptors for f in frames[:10] if len(f.descriptors)]
    vocabulary = MiniBatchKMeans(n_clusters=64, random_state=0, n_init=3).fit(np.vstack(training))
    histograms = []
    for frame in frames:
        hist = np.bincount(vocabulary.predict(frame.descriptors), minlength=64).astype(float) if len(frame.descriptors) else np.zeros(64)
        histograms.append(hist)
    histograms = np.array(histograms)
    # Inverse document frequency suppresses visual words common to most places.
    histograms *= np.log((len(frames) + 1) / (np.count_nonzero(histograms, axis=0) + 1)) + 1
    histograms /= np.maximum(np.linalg.norm(histograms, axis=1, keepdims=True), 1e-12)
    loops = []
    attempted = 0
    for second in range(len(indices)):
        if second % 10 == 0:
            print(f'Checking image loops {second + 1}/{len(indices)} keyframes', flush=True)
        candidates = [first for first in range(second) if indices[second] - indices[first] >= min_frame_gap]
        candidates.sort(key=lambda first: -float(histograms[first] @ histograms[second]))
        for first in candidates[:3]:
            attempted += 1
            loop = verify_loop(frames[first], frames[second], vo.K1)
            if loop is None:
                continue
            loops.append({**loop, 'measurement': loop['measurement'].tolist(), 'first_frame': indices[first], 'second_frame': indices[second]})
            print(f"Verified image loop {indices[first]} -> {indices[second]} ({loop['inliers']} inliers)", flush=True)
            break
    return loops, attempted


if __name__ == '__main__':
    main()
