"""Run KITTI VO or evaluate saved trajectories without importing OpenCV."""
import argparse
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
from pathlib import Path
import time
from kitti import load_poses_txt, save_poses_txt, validate_sequence
from metrics import compute_ate_rmse, evaluate_trajectory, umeyama_alignment


def run_vo_sequence(image_dir, calib_path, camera_id, max_frames, use_stereo, stereo_root=None, stats=None):
    import cv2 as cv
    cv.setRNGSeed(0)
    cv.setNumThreads(1)
    tracked = 0
    if use_stereo:
        from StereoVisualOdometry import StereoVisualOdometry
        vo = StereoVisualOdometry(stereo_root, calib_path, use_brute_force=False,
                                  draw_matches=False, max_frames=max_frames)
        for i in range(1, len(vo.Images_1)):
            transform, debug = vo.find_transf_pnp_debug(i)
            tracked += int(debug.get("tracking_ok", True))
            vo.poses.append(vo.poses[-1] @ transform)
            if i % 100 == 0:
                print(f"Processed {i + 1}/{len(vo.Images_1)} stereo frames", flush=True)
    else:
        from VisualOdometry import VisualOdometry
        vo = VisualOdometry(image_dir, calib_path, use_brute_force=False, camera_id=camera_id,
                            draw_matches=False, max_frames=max_frames)
        for i in range(1, len(vo.Images)):
            p1, p2 = vo.flann_match_features(i)
            transform, debug = vo.find_transf(p1, p2, return_debug=True)
            tracked += int(debug["tracking_ok"])
            vo.poses.append(vo.poses[-1] @ transform)
            if i % 100 == 0:
                print(f"Processed {i + 1}/{len(vo.Images)} monocular frames", flush=True)
    if stats is not None:
        stats.update({"tracked_pairs": tracked, "lost_pairs": len(vo.poses) - 1 - tracked})
    return vo.poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', '--sequence', nargs='+', default=['00'])
    parser.add_argument('--data-root', type=Path, help='KITTI dataset directory containing sequences/')
    parser.add_argument('--poses-root', type=Path, required=True, help='Ground truth directory containing 00.txt, etc.')
    parser.add_argument('--estimates-root', type=Path, help='Evaluate saved 00.txt, etc. without running VO')
    parser.add_argument('--output-root', type=Path, default=Path('results/kitti'))
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--stereo', action='store_true')
    parser.add_argument('--alignment', choices=['none', 'se3', 'sim3'], help='ATE only; drift is always unscaled')
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 2:
        parser.error('--max-frames must be at least 2')
    if not args.estimates_root and not args.data_root:
        parser.error('provide --data-root to run VO or --estimates-root to evaluate saved poses')
    alignment = args.alignment or ('se3' if args.stereo else 'sim3')
    reports = {}
    for sequence in args.sequences:
        if len(sequence) != 2 or not sequence.isdigit():
            parser.error(f'invalid sequence: {sequence}')
        gt = load_poses_txt(args.poses_root / f'{sequence}.txt')
        started = time.perf_counter()
        tracking = {}
        if args.estimates_root:
            est = load_poses_txt(args.estimates_root / f'{sequence}.txt')
            if args.max_frames is not None:
                est = est[:args.max_frames]
        else:
            seq_dir = validate_sequence(args.data_root, sequence, args.stereo, args.max_frames)
            est = run_vo_sequence(str(seq_dir / 'image_0'), str(seq_dir / 'calib.txt'), 0,
                                  args.max_frames, args.stereo, str(seq_dir / 'image_'), stats=tracking)
        elapsed = time.perf_counter() - started
        if args.max_frames is not None:
            gt = gt[:args.max_frames]
        # A short/failed prediction must not be silently truncated to appear complete.
        report = evaluate_trajectory(gt, est, alignment)
        report.update({'sequence': sequence, 'source': 'saved' if args.estimates_root else 'stereo_vo' if args.stereo else 'mono_vo'})
        report.update(tracking)
        report['translation_scale'] = 'metric' if args.stereo else 'unspecified' if args.estimates_root else 'arbitrary'
        if not args.estimates_root:
            report.update({'elapsed_s': elapsed, 'processing_fps': (len(est) - 1) / elapsed})
        args.output_root.mkdir(parents=True, exist_ok=True)
        save_poses_txt(args.output_root / f'{sequence}.txt', est)
        (args.output_root / f'{sequence}.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        reports[sequence] = {key: value for key, value in report.items() if key != 'segments'}
        print(json.dumps(reports[sequence], allow_nan=False))
    (args.output_root / 'summary.json').write_text(json.dumps(
        {'sequences': reports, 'drift_uses_scale_alignment': False}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
