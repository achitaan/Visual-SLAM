"""Reconstruct a saved image-only SLAM run using optional learned depth."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reconstruction import DepthAnythingProvider, fuse_reconstruction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--environment", choices=["indoor", "outdoor"], default="outdoor"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--keyframe-stride", type=int, default=1)
    args = parser.parse_args()
    provider = DepthAnythingProvider(
        args.model_source,
        args.checkpoint,
        args.environment,
        args.device,
        args.input_size,
    )
    result = fuse_reconstruction(
        args.run,
        provider,
        args.output,
        args.voxel_size,
        keyframe_stride=args.keyframe_stride,
    )
    print(f'Exported {result["points"]} dense points to {args.output}')


if __name__ == "__main__":
    main()
