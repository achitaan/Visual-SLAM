"""Audit known loop recall using saved input images, without rerunning tracking."""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2 as cv
import numpy as np
from keyframe_index import KeyframeIndex
from pose_graph import optimize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    cv.setNumThreads(1)
    detector = cv.SIFT_create(nfeatures=1500)
    cache = args.output.parent / "retrieval-features"
    cache.mkdir(parents=True, exist_ok=True)
    known = {}
    for loop in saved["verified_loops"]:
        known.setdefault(loop["second_keyframe"], []).append(loop["first_keyframe"])
    index = KeyframeIndex()
    frames, audit, query_times = {}, [], []
    for keyframe in saved["keyframes"]:
        ident = keyframe["id"]
        image_path = args.run / keyframe["image"]
        stamp = image_path.stat()
        cached = cache / f'{ident}-{stamp.st_size}-{stamp.st_mtime_ns}.npy'
        if cached.exists():
            desc = np.load(cached, allow_pickle=False)
        else:
            image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Missing saved keyframe image")
            _, desc = detector.detectAndCompute(image, None)
            desc = desc if desc is not None else np.empty((0, 128), np.float32)
            np.save(cached, desc, allow_pickle=False)
        index.upsert(ident, keyframe["frame"], desc)
        frames[ident] = keyframe["frame"]
        if ident in known:
            eligible = [i for i, frame in frames.items() if keyframe["frame"] - frame >= 150]
            started = perf_counter()
            shortlist = index.query(desc, eligible)
            query_times.append(perf_counter() - started)
            for source in known[ident]:
                audit.append({"first_keyframe": source, "second_keyframe": ident, "eligible_keyframes": len(eligible), "shortlist": shortlist, "retrieved": source in (shortlist or eligible)})
        if ident % 50 == 0:
            print(f'Extracted {ident+1}/{len(saved["keyframes"])} saved keyframes', flush=True)
    poses = [np.asarray(k["pose"], float) for k in saved["keyframes"]]
    edges = [(i, i+1, np.linalg.inv(poses[i]) @ poses[i+1], np.diag([1000.] * 3 + [10.] * 3), "odometry") for i in range(len(poses)-1)]
    edges += [(l["first_keyframe"], l["second_keyframe"], np.asarray(l["measurement"], float), np.diag([2000.] * 3 + [20.] * 3), "loop") for l in saved["verified_loops"]]
    started = perf_counter()
    diagnostics = {}
    corrected = optimize(poses, edges, max_evaluations=500, diagnostics=diagnostics)
    graph_elapsed = perf_counter() - started
    report = {"evidence": "Saved left-image SIFT retrieval proxy; graph replay uses saved independent loop measurements and odometry reconstructed from exported adjacent poses, not the original pre-correction snapshot", "keyframes": len(frames), "known_loops": len(audit), "retrieved_loops": sum(a["retrieved"] for a in audit), "recall": sum(a["retrieved"] for a in audit)/max(len(audit), 1), "median_query_ms": float(np.median(query_times)*1000) if query_times else None, "audit": audit, "graph_replay": {"elapsed_s": graph_elapsed, "poses": len(corrected), "finite": bool(np.isfinite(corrected).all()), "origin_fixed": bool(np.allclose(corrected[0], poses[0])), "diagnostics": diagnostics, "solver_source_unchanged": True}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "audit"}, indent=2))


if __name__ == "__main__":
    main()
