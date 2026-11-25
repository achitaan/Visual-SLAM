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
import os
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
    ("fast_sift", ("avx512f", "avx512bw", "avx512vl", "avx512dq")),
    ("fast_ba", ()),
    ("fast_lk", ("avx512f",)),
    ("fast_pgo", ()),
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
    system = platform.system()
    windows = system == "Windows"
    if system not in ("Linux", "Windows"):
        print(
            f"Skipping native kernel build on {system}; Python/OpenCV fallback will be used.",
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
    environment = None
    if windows:
        if platform.machine().lower() not in ("amd64", "x86_64"):
            print("Native Windows kernels require an x64 host.", file=sys.stderr)
            return []
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from native_accel import windows_missing_simd

        # Keep the linker and compiler helper programs from the same MinGW toolchain.
        cxx = shutil.which(cxx) or cxx
        compiler_dir = str(Path(cxx).resolve().parent)
        environment = {**os.environ, "PATH": compiler_dir + os.pathsep + os.environ.get("PATH", "")}
    built: list[str] = []
    for name, required_flags in KERNELS:
        src = KERNEL_DIR / f"{name}.cpp"
        out = KERNEL_DIR / f"lib{name}{'.dll' if windows else '.so'}"
        if not src.is_file():
            print(f"[skip] missing source {src}", file=sys.stderr)
            continue
        missing = windows_missing_simd(name) if windows else [f for f in required_flags if f not in flags]
        if missing and not force:
            print(
                f"[skip] {name}: host CPU lacks {', '.join(missing)} (Python/OpenCV fallback active)"
            )
            continue
        isa = (
            ["-march=x86-64", "-mtune=native", *(f"-m{f.replace('_', '')}" for f in required_flags)]
            if windows else ["-march=native"]
        )
        cmd = [
            cxx,
            "-std=c++17",
            "-O3",
            *isa,
            *([] if name in ("fast_pgo", "fast_ba") else ["-ffast-math"]),
            "-shared",
            "-static" if windows else "-fPIC",
            str(src),
            "-o",
            str(out),
        ]
        print(f"[build] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=environment)
        built.append(name)
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=None, help="C++ compiler binary (use a recent MinGW-w64 g++ on Windows)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Attempt to build even if the host feature check does not list the required AVX-512 flags",
    )
    args = parser.parse_args(argv)
    built = build_kernels(compiler=args.cxx, force=args.force)
    print(f"Built {len(built)}/{len(KERNELS)} native kernels: {', '.join(built) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
