"""Bounded synthetic solver evidence; does not authorize removing the live graph guard."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pose_graph import optimize
from test_budget import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();reports=[]
    for count in [300,1000]:
        truth=[];initial=[]
        for i in range(count):
            pose=np.eye(4);pose[:3,3]=[i*.05,np.sin(i*.03),0];truth.append(pose)
            noisy=pose.copy()
            if i:noisy[:3,3]+=np.array([.01*np.sin(i),.005*np.cos(i),.002])
            initial.append(noisy)
        # Anchored constraints plus temporal edges exercise both solver branches.
        edges=[(0,i,np.linalg.inv(truth[0])@truth[i],np.eye(6),'verified_synthetic') for i in range(1,count)]
        edges.extend((i-1,i,np.linalg.inv(truth[i-1])@truth[i],np.eye(6),'odometry') for i in range(1,count))
        diagnostics={};start=time.perf_counter()
        corrected=optimize(initial,edges,max_evaluations=20,diagnostics=diagnostics)
        assert np.array_equal(corrected[0],initial[0]) and np.isfinite(corrected).all()
        assert diagnostics['final_cost']<diagnostics['initial_cost']
        error=max(np.linalg.norm(a[:3,3]-b[:3,3]) for a,b in zip(corrected,truth))
        assert error<1e-4
        reports.append({'keyframes':count,'max_position_error':error,'elapsed_s':time.perf_counter()-start,'diagnostics':diagnostics})
        write_json(args.output,{'scope':'synthetic anchored and temporal constraints only','production_guard_removed':False,'cases':reports})
        print(count,diagnostics['solver'],error,flush=True)


if __name__=='__main__':main()
