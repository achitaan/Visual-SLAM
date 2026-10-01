"""Retain and inspect one failed stereo pair without changing benchmark settings."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cv2 as cv
import numpy as np
from StereoVisualOdometry import StereoVisualOdometry
from config import STEREO_MIN_MATCHES, STEREO_MIN_VALID_3D
from json_output import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', required=True, choices=[f'{index:02d}' for index in range(11)])
    parser.add_argument('--scratch-root', type=Path, default=Path('.datasets/batch-scratch'))
    parser.add_argument('--output-root', type=Path, default=Path('results/benchmark-batch'))
    args = parser.parse_args()
    checkpoint = args.output_root / 'stereo' / f'{args.sequence}-checkpoint.txt'
    rows = np.loadtxt(checkpoint, ndmin=2)
    held = np.flatnonzero(np.max(np.abs(np.diff(rows, axis=0)), axis=1) == 0) + 1
    if not len(held):
        raise ValueError('No held-pose pair in the saved checkpoint')
    current = int(held[0])
    source = args.scratch_root / args.sequence
    target = args.output_root / 'failures' / args.sequence
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for camera in ('image_0', 'image_1'):
        (target / camera).mkdir(exist_ok=True)
        for offset, frame in enumerate((current - 1, current)):
            path = source / camera / f'{frame:06d}.png'
            shutil.copy2(path, target / camera / f'{offset:06d}.png')
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(source / 'calib.txt', target / 'calib.txt')
    cv.setNumThreads(1)
    cv.setRNGSeed(0)
    vo = StereoVisualOdometry(str(target / 'image_'), str(target / 'calib.txt'), False, draw_matches=False)
    pnp_details = {}
    original_pnp = cv.solvePnPRansac

    def inspect_pnp(*inputs, **options):
        result = original_pnp(*inputs, **options)
        success, rotation, translation, inliers = result
        pnp_details.update(success=bool(success), candidate_inliers=0 if inliers is None else len(inliers),
                           iterations=options.get('iterationsCount'), reprojection_threshold_px=options.get('reprojectionError'))
        return result

    cv.solvePnPRansac = inspect_pnp
    try:
        transform, debug = vo.find_transf_pnp_debug(1)
    finally:
        cv.solvePnPRansac = original_pnp
    scalars = {name: value for name, value in debug.items() if isinstance(value, (int, float, bool))}
    report = dict(sequence=args.sequence, previous_frame=current - 1, current_frame=current,
                  held_pose_indices_in_checkpoint=held.tolist(), checkpoint_frames=len(rows), pair_debug=scalars, pnp_details=pnp_details,
                  minimum_matches=STEREO_MIN_MATCHES, minimum_valid_3d=STEREO_MIN_VALID_3D,
                  transform=transform.tolist(), retained_input_sha256=hashes,
                  limitation='Single-pair replay resets the random seed; its RNG history differs from the full benchmark. A held pose may also represent a stationary frame.')
    write_json(target / 'diagnostic.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
