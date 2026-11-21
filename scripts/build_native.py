#!/usr/bin/env python3
"""Build optional native C++/AVX-512 acceleration kernels for Visual-SLAM.

Each kernel accelerates a single self-contained numerical or vision operation
(descriptor matching, epipolar ZNCC verification, SIFT extraction, local BA
Schur solver, pyramidal LK optical flow) while leaving the Python SLAM engine
architecture untouched. When a host lacks AVX-512 or a C++ compiler, Visual-SLAM
falls back seamlessly to its Python/OpenCV/NumPy implementations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = REPO_ROOT / "src" / "native_kernels"

KERNELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fast_matcher", ("avx512f", "avx512_vnni")),
    ("fast_zncc", ("avx512f",)),
    ("fast_sift", ("avx512f", "avx512bw", "avx512vl")),
    ("fast_ba", ()),
    ("fast_lk", ("avx512f",)),
)


def _cpu_flags() -> set[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return set()
    for line in cpuinfo.read_text(errors="ignore").splitlines():
        if line.lower().startswith("flags"):
            _, _, rhs = line.partition(":")
            return set(rhs.strip().split())
    return set()


def build_kernels(compiler: str | None = None, force: bool = False) -> list[str]:
    if platform.system() != "Linux":
        print(
            f"Skipping native kernel build on {platform.system()}; Python/OpenCV fallback will be used.",
            file=sys.stderr,
        )
        return []

    cxx = compiler or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        print(
            "No C++ compiler (g++/clang++) found; Python/OpenCV fallback will be used.",
            file=sys.stderr,
        )
        return []

    flags = _cpu_flags()
    built: list[str] = []
    for name, required_flags in KERNELS:
        src = KERNEL_DIR / f"{name}.cpp"
        out = KERNEL_DIR / f"lib{name}.so"
        if not src.is_file():
            print(f"[skip] missing source {src}", file=sys.stderr)
            continue
        missing = [f for f in required_flags if f not in flags]
        if missing and not force:
            print(
                f"[skip] {name}: host CPU lacks {', '.join(missing)} (Python/OpenCV fallback active)"
            )
            continue
        cmd = [
            cxx,
            "-O3",
            "-march=native",
            "-ffast-math",
            "-shared",
            "-fPIC",
            str(src),
            "-o",
            str(out),
        ]
        print(f"[build] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        built.append(name)
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=None, help="C++ compiler binary (default: g++ or clang++)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Attempt to build even if /proc/cpuinfo does not list required AVX-512 flags",
    )
    args = parser.parse_args(argv)
    built = build_kernels(compiler=args.cxx, force=args.force)
    print(f"Built {len(built)}/{len(KERNELS)} native kernels: {', '.join(built) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
