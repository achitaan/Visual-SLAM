"""Budgeted replay of the preserved frame-to-frame stereo estimator; references are evaluator-only."""
import argparse
import hashlib
import sys
import time
from pathlib import Path
import cv2 as cv
import numpy as np
from test_budget import Budget,write_json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from StereoVisualOdometry import StereoVisualOdometry
from kitti import load_poses_txt,save_poses_txt
from metrics import evaluate_trajectory
from evaluate_shared_slam import peak_memory_mb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--poses-root',type=Path,required=True)
    p.add_argument('--sequence',required=True);p.add_argument('--max-frames',type=int)
    p.add_argument('--stop-file',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-wall-seconds',type=float,required=True)
    args=p.parse_args();budget=Budget(args.max_wall_seconds);started=time.perf_counter()
    source={f.name:f.read_bytes() for f in sorted((Path(__file__).resolve().parents[1]/'src').glob('*.py'))}
    evaluator_source=Path(__file__).read_bytes()
    cv.setNumThreads(1);cv.setRNGSeed(0)
    seq=args.data_root/'sequences'/args.sequence
    vo=StereoVisualOdometry(str(seq/'image_'),str(seq/'calib.txt'),False,draw_matches=False,max_frames=args.max_frames)
    args.output.mkdir(parents=True,exist_ok=True);lost=0;status='completed';states=['tracking']
    for i in range(1,len(vo.Images_1)):
        if args.stop_file and args.stop_file.exists():status='interrupted_requested_stop';break
        if budget.remaining<=5:status='interrupted_time_budget';break
        try:transform,debug=vo.find_transf_pnp_debug(i)
        except (ValueError,OSError):status='interrupted_input_error';break
        ok=bool(debug['tracking_ok']);lost+=not ok;states.append('tracking' if ok else 'lost')
        vo.poses.append(vo.poses[-1]@transform)
        if i%50==0:save_poses_txt(args.output/'checkpoint-poses.txt',vo.poses)
    save_poses_txt(args.output/'poses.txt',vo.poses)
    # Reference poses enter only this evaluator, after motion estimation finishes.
    truth=load_poses_txt(args.poses_root/f'{args.sequence}.txt')[:len(vo.poses)]
    metrics=evaluate_trajectory(truth,vo.poses,'se3')
    metrics={k:v for k,v in metrics.items() if k!='segments'}
    if status=='completed' and lost:status='completed_with_tracking_loss'
    (args.output/'source').mkdir(exist_ok=True)
    for name,data in source.items():(args.output/'source'/name).write_bytes(data)
    (args.output/'evaluator.py').write_bytes(evaluator_source)
    write_json(args.output/'evaluation.json',{'sequence':args.sequence,'stereo':True,'estimator':'preserved_stereo_vo','frames':len(vo.poses),
      'coverage':'partial' if args.max_frames or status.startswith('interrupted') else 'full','status':status,'lost_frames':lost,'metrics':metrics,'elapsed_s':time.perf_counter()-started,
      'peak_memory_mb':peak_memory_mb(),'configuration':{'feature_extractor':'preserved_stereo_defaults','loop_mode':'off'},
      'evaluator_sha256':hashlib.sha256(evaluator_source).hexdigest(),
      'ground_truth_used_for_estimation':False,'source_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in source.items()}})


if __name__=='__main__':main()
