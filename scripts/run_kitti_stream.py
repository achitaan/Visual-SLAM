"""Full KITTI stereo/graph test using CRC-checked remote images and bounded caches.

Cache one complete sequence when disk headroom permits; otherwise stream images.
The PowerShell batch driver removes owned image/keyframe scratch after each run.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from collections import OrderedDict
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import cv2 as cv
import numpy as np
import scipy
import sklearn
from download_kitti_sequence import RangeFile, URL
from json_output import write_json
from eval_pose_graph import find_image_loops, optimize_image_graph
from kitti import load_poses_txt, save_poses_txt
from image_sequence import ImageSequence
from loop_geometry import StereoLoopFrame, extract_loop_frame
from metrics import evaluate_trajectory
from StereoVisualOdometry import StereoVisualOdometry


class RemoteImages(Sequence):
    def __init__(self, archive, entries):
        self.archive, self.entries = archive, entries
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self.cache:
            entry = self.entries[index]
            if hasattr(self.archive.fp, 'prefetch'):
                # A local ZIP header may differ slightly from the central extra fields.
                # ZipFile still parses that header and validates the complete member CRC.
                length = 30 + len(entry.filename.encode('utf-8')) + len(entry.extra) + entry.compress_size + 256
                self.archive.fp.prefetch(entry.header_offset, length)
            data = self.archive.read(entry)  # Verifies ZIP member CRC.
            decoded = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_GRAYSCALE)
            if decoded is None:
                raise ValueError(f'Cannot decode {self.entries[index].filename}')
            self.cache[index] = decoded
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


class GeometryStore(Sequence):
    """Spill descriptors to owned scratch; retain at most four frames in RAM."""
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.cache = OrderedDict()

    def append(self, frame):
        if shutil.disk_usage(self.root).free < 150 * 1024 * 1024:
            raise OSError('Less than 150 MiB free; stopping before filling the drive')
        np.savez_compressed(self.root / f'{self.count:06d}.npz', pixels=frame.pixels,
                            points=frame.points, descriptors=frame.descriptors, image_size=frame.image_size)
        self.count += 1

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self.cache:
            with np.load(self.root / f'{index:06d}.npz', allow_pickle=False) as data:
                self.cache[index] = StereoLoopFrame(data['pixels'], data['points'], data['descriptors'], tuple(data['image_size']))
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def graph_result(args, raw, indices, loops, attempted, started, reused):
    optimizer = {}
    corrected = optimize_image_graph(raw, indices, loops, args.graph_evaluations, optimizer)
    full_gt = load_poses_txt(args.poses_root / f'{args.sequence}.txt')
    gt = full_gt[:len(raw)]
    before = evaluate_trajectory(gt, raw, 'se3')
    after = evaluate_trajectory(gt, corrected, 'se3')
    root = args.output_root / 'graph'
    save_poses_txt(root / f'{args.sequence}.txt', corrected)
    report = dict(sequence=args.sequence, experiment='batch_stereo_pose_graph', frames=len(raw),
                  keyframe_count=len(indices), keyframe_indices=indices, candidate_pairs_tested=attempted,
                  verified_loop_count=len(loops), loops=loops, ground_truth_used_for_constraints=False,
                  keyframe_step=20, min_frame_gap=150, loop_feature_limit=1500,
                  graph_max_evaluations=args.graph_evaluations,
                  optimizer=optimizer,
                  odometry_information_diagonal=[1000.] * 3 + [10.] * 3,
                  loop_information_diagonal=[2000.] * 3 + [20.] * 3,
                  raw_trajectory_path=str((args.output_root / 'stereo' / f'{args.sequence}.txt').resolve()),
                  elapsed_s=time.perf_counter() - started, reused_image_constraints=reused,
                  before=before, after=after, status='complete' if len(raw) == len(full_gt) else 'partial',
                  scope='Offline graph correction and full-frame propagation; live tracking/map feedback remains separate.')
    write_json(root / f'{args.sequence}.json', report)
    print(json.dumps(dict(sequence=args.sequence, loops=len(loops), before_ate=before['ate_rmse_m'],
                          after_ate=after['ate_rmse_m'], before_translation=before['translation_percent'],
                          after_translation=after['translation_percent'])), flush=True)


def reuse_complete(args):
    """Reuse only full, rigid trajectories with matching saved image constraints."""
    if args.force_retest or args.max_frames is not None or args.graph_from_raw:
        return False
    root, seq = args.output_root, args.sequence
    paths = [root / kind / f'{seq}.json' for kind in ('stereo', 'graph')]
    cache_path = root / 'graph' / f'{seq}-constraints.json'
    if not all(path.exists() for path in paths + [cache_path]):
        return False
    reports = [json.loads(path.read_text()) for path in paths]
    if any(report.get('status') != 'complete' for report in reports):
        return False
    raw_path = root / 'stereo' / f'{seq}.txt'
    cache = json.loads(cache_path.read_text())
    if cache['raw_trajectory_sha256'] != hashlib.sha256(raw_path.read_bytes()).hexdigest():
        return False
    gt_count = len(load_poses_txt(args.poses_root / f'{seq}.txt'))
    if any(report.get('frames') != gt_count for report in reports):
        return False
    if any(len(load_poses_txt(path.with_suffix('.txt'))) != gt_count for path in paths):
        return False
    print(f'{seq}: reused validated complete raw/graph reports; use --force-retest to rerun images', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', required=True, choices=[f'{i:02d}' for i in range(11)])
    parser.add_argument('--poses-root', type=Path, default=REPO / 'results/benchmark-batch/reference/poses')
    parser.add_argument('--output-root', type=Path, default=REPO / 'results/benchmark-batch')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--replay-graph', action='store_true')
    parser.add_argument('--graph-from-raw', action='store_true', help='Retrieve only keyframe images for an already saved complete raw trajectory')
    parser.add_argument('--tracked-pairs', type=int, help='Verified tracking count from the original run log when recovering a missing raw report')
    parser.add_argument('--graph-evaluations', type=int, default=500)
    parser.add_argument('--force-retest', action='store_true')
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 2:
        parser.error('--max-frames must be at least two')
    if args.graph_evaluations < 1:
        parser.error('--graph-evaluations must be positive')
    cv.setNumThreads(1)
    cv.setRNGSeed(0)
    started = time.perf_counter()
    seq = args.sequence
    stereo_root, graph_root = args.output_root / 'stereo', args.output_root / 'graph'
    status_path = args.output_root / f'{seq}-status.json'
    cache_path = graph_root / f'{seq}-constraints.json'
    phase = 'initializing'
    try:
        if args.replay_graph:
            phase = 'graph'
            raw = load_poses_txt(stereo_root / f'{seq}.txt')
            cache = json.loads(cache_path.read_text())
            if cache['raw_trajectory_sha256'] != hashlib.sha256((stereo_root / f'{seq}.txt').read_bytes()).hexdigest():
                raise ValueError('Raw trajectory no longer matches saved image constraints')
            graph_result(args, raw, cache['keyframe_indices'], cache['loops'], cache['candidate_pairs_tested'], started, True)
        else:
            scratch = REPO / '.datasets/batch-scratch' / seq
            # Another batch may already be testing this sequence. Wait for its
            # controller to remove scratch before returning/reusing its report;
            # otherwise our caller could accidentally remove the active cache.
            waited = 0
            while scratch.exists():
                owner_path = scratch / 'owner.json'
                if owner_path.exists():
                    owner = json.loads(owner_path.read_text())
                    if owner.get('purpose') != 'temporary KITTI benchmark images and geometry' or owner.get('sequence') != seq:
                        raise ValueError('Unrecognized scratch owner; refusing to use this cache')
                if waited % 30 == 0:
                    print(f'{seq}: waiting for the existing owned cache to finish and be cleaned ({waited}s)', flush=True)
                time.sleep(5)
                waited += 5
            if reuse_complete(args):
                write_json(status_path, dict(sequence=seq, status='complete', phase='finished',
                                            reused_validated_reports=True, elapsed_s=time.perf_counter() - started))
                return
            scratch.mkdir(parents=True, exist_ok=True)
            write_json(scratch / 'owner.json', dict(sequence=seq, source=URL, process_id=os.getpid(), purpose='temporary KITTI benchmark images and geometry'))
            tracked = 0
            with RangeFile(URL) as remote, zipfile.ZipFile(remote) as archive:
                prefix = f'dataset/sequences/{seq}/'
                catalogs = [sorted((entry for entry in archive.infolist() if re.fullmatch(prefix + rf'image_{camera}/[0-9]{{6}}\.png', entry.filename)), key=lambda e: e.filename) for camera in range(2)]
                names = [[Path(e.filename).name for e in entries] for entries in catalogs]
                if names[0] != names[1] or names[0] != [f'{i:06d}.png' for i in range(len(names[0]))] or len(names[0]) < 2:
                    raise ValueError('Invalid/mismatched stereo image catalog')
                total = len(names[0])
                count = min(args.max_frames or total, total)
                # Only metadata/count is inspected here; GT never enters tracking or graph measurements.
                gt_count = sum(bool(line.strip()) for line in (args.poses_root / f'{seq}.txt').read_text().splitlines())
                if gt_count != total:
                    raise ValueError('Full image/ground-truth counts differ')
                calib = archive.read(prefix + 'calib.txt')
                (scratch / 'calib.txt').write_bytes(calib)
                required = sum(entry.file_size for entries in catalogs for entry in entries[:count])
                geometry_reserve = (count // 20 + 2) * 600 * 1024
                local_cache = not args.graph_from_raw and shutil.disk_usage(scratch).free > required + geometry_reserve + 500 * 1024 * 1024
                if local_cache:
                    # Physical ZIP order avoids HTTP requests in random frame order.
                    phase = 'download'
                    entries = sorted(catalogs[0][:count] + catalogs[1][:count], key=lambda entry: entry.header_offset)
                    for position, entry in enumerate(entries):
                        target = scratch / Path(entry.filename).parent.name / Path(entry.filename).name
                        target.parent.mkdir(exist_ok=True)
                        target.write_bytes(archive.read(entry))
                        if position % 100 == 0 or position == len(entries) - 1:
                            print(f'{seq}: cached {position + 1}/{len(entries)} image files', flush=True)
                            write_json(status_path, dict(sequence=seq, phase=phase, status='running', files=position + 1, total_files=len(entries)))
                        if shutil.disk_usage(scratch).free < geometry_reserve + 250 * 1024 * 1024:
                            raise OSError('Disk headroom fell below cache reserve')
                    sources = [ImageSequence(scratch / f'image_{camera}') for camera in range(2)]
                else:
                    sources = [RemoteImages(archive, entries[:count]) for entries in catalogs]
                    for camera in range(2):
                        folder = scratch / f'image_{camera}'
                        folder.mkdir(exist_ok=True)
                        for entry in catalogs[camera][:2]:
                            # Encode the CRC-checked pixels for the small constructor seed.
                            decoded = sources[camera][int(Path(entry.filename).stem)]
                            (folder / Path(entry.filename).name).write_bytes(cv.imencode('.png', decoded)[1].tobytes())
                vo = StereoVisualOdometry(str(scratch / 'image_'), str(scratch / 'calib.txt'), False, draw_matches=False)
                vo.Images_1, vo.Images_2 = sources
                indices = sorted(set(range(0, count, 20)) | {count - 1})
                selected = set(indices)
                geometry = GeometryStore(scratch / 'geometry')
                phase = 'stereo'
                tracking_started = time.perf_counter()
                cv.setRNGSeed(0)
                if args.graph_from_raw:
                    vo.poses = load_poses_txt(stereo_root / f'{seq}.txt')
                    if len(vo.poses) != total or count != total:
                        raise ValueError('Graph recovery requires a complete raw trajectory')
                    if args.tracked_pairs is None or not 0 <= args.tracked_pairs <= count - 1:
                        raise ValueError('Recovery needs the verified tracking count from the original log')
                    tracked = args.tracked_pairs
                    phase = 'geometry'
                    write_json(status_path, dict(sequence=seq, phase=phase, status='running', frames=count, total_frames=total))
                    for position, index in enumerate(indices):
                        geometry.append(extract_loop_frame(vo, index))
                        if position % 10 == 0:
                            print(f'{seq}: recovered image geometry {position + 1}/{len(indices)} keyframes', flush=True)
                else:
                    geometry.append(extract_loop_frame(vo, 0))
                    for index in range(1, count):
                        transform, debug = vo.find_transf_pnp_debug(index)
                        tracked += int(debug.get('tracking_ok', True))
                        vo.poses.append(vo.poses[-1] @ transform)
                        if index in selected:
                            geometry.append(extract_loop_frame(vo, index))
                        if index % 50 == 0 or index == count - 1:
                            write_json(status_path, dict(sequence=seq, phase=phase, status='running', frames=index + 1, total_frames=total, tracked_pairs=tracked, lost_pairs=index-tracked, elapsed_s=time.perf_counter() - started))
                            save_poses_txt(stereo_root / f'{seq}-checkpoint.txt', vo.poses)
                            print(f'{seq}: stereo {index + 1}/{count}, tracked {tracked}/{index}, downloaded {remote.bytes_downloaded / 1024**2:.1f} MiB', flush=True)
                elapsed = time.perf_counter() - tracking_started
                raw = vo.poses
                save_poses_txt(stereo_root / f'{seq}.txt', raw)
                gt = load_poses_txt(args.poses_root / f'{seq}.txt')[:count]
                report = evaluate_trajectory(gt, raw, 'se3')
                report.update(sequence=seq, source='stereo_vo', translation_scale='metric', tracked_pairs=tracked,
                              lost_pairs=count - 1 - tracked, elapsed_s=None if args.graph_from_raw else elapsed,
                              processing_fps=None if args.graph_from_raw else (count - 1) / elapsed,
                              dataset_source=URL, input_mode='CRC-verified local sequence cache' if local_cache else 'CRC-verified HTTP-range ZIP stream', total_frames=total,
                              status='complete' if count == total else 'partial', opencv_threads=1, random_seed=0,
                              downloaded_bytes=remote.bytes_downloaded, calibration_sha256=hashlib.sha256(calib).hexdigest())
                source_files = ['src/StereoVisualOdometry.py', 'src/config.py', 'src/loop_geometry.py',
                                'src/pose_graph.py', 'src/metrics.py', 'src/eval_pose_graph.py',
                                'scripts/run_kitti_stream.py', 'scripts/download_kitti_sequence.py']
                report['source_sha256'] = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in source_files}
                report['runtime_versions'] = {'numpy': np.__version__, 'scipy': scipy.__version__, 'opencv': cv.__version__, 'scikit-learn': sklearn.__version__}
                report['recovered_reporting_from_saved_raw'] = args.graph_from_raw
                if args.graph_from_raw:
                    report['input_mode'] = 'Saved raw stereo trajectory; CRC-verified streamed keyframe images'
                    report['tracking_count_source'] = 'Original complete tracking log; explicitly supplied to recovery CLI'
                write_json(stereo_root / f'{seq}.json', report)
                print(f"{seq}: raw ATE {report['ate_rmse_m']:.3f} m, translation {report['translation_percent']}%, lost {report['lost_pairs']}", flush=True)
                phase = 'graph'
                write_json(status_path, dict(sequence=seq, phase=phase, status='running', frames=count, total_frames=total))
                cv.setRNGSeed(0)
                loops, attempted = find_image_loops(vo, indices, 150, geometry)
                write_json(cache_path, dict(sequence=seq, raw_trajectory_sha256=hashlib.sha256((stereo_root / f'{seq}.txt').read_bytes()).hexdigest(),
                                           keyframe_indices=indices, min_frame_gap=150, loop_feature_limit=1500,
                                           candidate_pairs_tested=attempted, loops=loops))
                graph_result(args, raw, indices, loops, attempted, started, False)
        write_json(status_path, dict(sequence=seq, status='complete' if args.max_frames is None else 'partial', phase='finished', elapsed_s=time.perf_counter() - started))
    except Exception as exc:
        write_json(status_path, dict(sequence=seq, status='failed', phase=phase, error=f'{type(exc).__name__}: {exc}', elapsed_s=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    main()
