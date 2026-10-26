"""Unmodified evaluator entry with process-only wall latency (two clock calls/frame).

Portable copy of the original measurement harness. It contains no dataset,
account or private evidence paths; paths are supplied by the invoking launcher.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--entry-point', type=Path, required=True)
parser.add_argument('--frame-timings', type=Path, required=True)
args, remaining = parser.parse_known_args()
if remaining and remaining[0] == '--':
    remaining = remaining[1:]
entry = args.entry_point.resolve()
sys.path.insert(0, str(entry.parent))
spec = importlib.util.spec_from_file_location('instrumented_evaluator', entry)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = module.SharedSlam.process
records = []


def timed(self, index, *positional, **keyword):
    start = time.perf_counter()
    try:
        return original(self, index, *positional, **keyword)
    finally:
        records.append({'frame': int(index), 'process_wall_seconds': time.perf_counter() - start})


module.SharedSlam.process = timed
sys.argv = [str(entry), *remaining]
try:
    module.main()
finally:
    args.frame_timings.parent.mkdir(parents=True, exist_ok=True)
    args.frame_timings.write_text(json.dumps({
        'definition': 'SharedSlam.process wall latency; includes synchronous waits and worker contention, excludes loading/export/shutdown; two perf_counter calls per frame',
        'records': records,
    }, indent=2) + '\n')
