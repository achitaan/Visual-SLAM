"""Compare warmed matching on a real image pair, including CUDA transfers."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2 as cv
import numpy as np
from descriptor_matching import DescriptorMatcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    cv.setNumThreads(1)
    sift = cv.SIFT_create(nfeatures=1500)
    descriptors = []
    hashes = []
    for path in (args.first, args.second):
        image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if image is None:
            parser.error(f"Unreadable image: {path}")
        _, desc = sift.detectAndCompute(image, None)
        if desc is None or len(desc) < 2:
            parser.error("Image pair must have usable descriptors")
        descriptors.append(desc)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    cpu, gpu = DescriptorMatcher("cpu"), DescriptorMatcher("cuda")
    agree = np.array_equal(cpu(*descriptors), gpu(*descriptors))
    samples = {"cpu": [], "cuda": []}
    for iteration in range(args.repetitions):
        # Alternate ordering to reduce bias from changing background load.
        results = {}
        order = [("cpu", cpu), ("cuda", gpu)]
        for label, matcher in order if iteration % 2 == 0 else reversed(order):
            started = perf_counter()
            results[label] = matcher(*descriptors)
            samples[label].append(perf_counter() - started)
        agree = agree and np.array_equal(results["cpu"], results["cuda"])
    medians = {label: float(np.median(values) * 1000) for label, values in samples.items()}
    report = {"shapes": [list(d.shape) for d in descriptors], "input_sha256": hashes,
              "matcher_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "src/descriptor_matching.py").read_bytes()).hexdigest(),
              "pair_agreement": bool(agree), "cpu_ms": medians["cpu"], "cuda_ms": medians["cuda"],
              "matching_speedup": medians["cpu"] / medians["cuda"], "gpu": gpu.metadata(),
              "policy": "Warmed matcher; transfers and synchronization included; excludes extraction and runtime initialization; shared-host timings provisional"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not agree:
        raise SystemExit("CPU/CUDA match disagreement")


if __name__ == "__main__":
    main()
