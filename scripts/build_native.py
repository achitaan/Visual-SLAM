#!/usr/bin/env python3
"""Build optional native C++ SIMD acceleration kernels for Visual-SLAM.

Supports Linux (g++/clang++ -> lib<name>.so) and Windows (MSVC cl.exe or MinGW
g++/clang++ -> <name>.dll), automatically selecting the best available ISA
(AVX-512, AVX2, or portable scalar C++). When a C++ compiler is unavailable or
``VISUAL_SLAM_DISABLE_NATIVE=1`` is set, Visual-SLAM falls back seamlessly to
its Python/OpenCV/NumPy implementations.
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

KERNELS: tuple[str, ...] = (
    "fast_matcher",
    "fast_zncc",
    "fast_sift",
    "fast_ba",
    "fast_lk",
    "fast_pgo",
)

_EXACT_FP_KERNELS = frozenset({"fast_pgo", "fast_ba", "fast_zncc"})


def _cpu_flags() -> set[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return set()
    for line in cpuinfo.read_text(errors="ignore").splitlines():
        if line.lower().startswith("flags"):
            _, _, rhs = line.partition(":")
            return set(rhs.strip().split())
    return set()


def _detect_isa(requested: str) -> str:
    if requested != "auto":
        return requested
    sys_name = platform.system()
    if sys_name == "Linux":
        flags = _cpu_flags()
        if {"avx512f", "avx512bw", "avx512vl"}.issubset(flags):
            return "native"
        if "avx2" in flags:
            return "avx2"
        return "portable"
    if sys_name == "Windows":
        machine = platform.machine().lower()
        if machine in ("amd64", "x86_64"):
            return "avx2"
        return "portable"
    return "native"


def _library_filename(name: str, system_name: str) -> str:
    if system_name == "Windows":
        return f"{name}.dll"
    if system_name == "Darwin":
        return f"lib{name}.dylib"
    return f"lib{name}.so"


def _build_command(
    compiler: str,
    src: Path,
    out: Path,
    name: str,
    isa: str,
    system_name: str,
) -> list[str]:
    comp_name = Path(compiler).name.lower()
    is_msvc = comp_name in ("cl", "cl.exe")
    if is_msvc:
        flags = ["/nologo", "/std:c++17", "/O2", "/EHsc", "/LD", "/MD"]
        if isa == "avx512":
            flags.append("/arch:AVX512")
        elif isa in ("avx2", "native"):
            flags.append("/arch:AVX2")
        if name not in _EXACT_FP_KERNELS:
            flags.append("/fp:fast")
        else:
            flags.append("/fp:precise")
        return [compiler, *flags, str(src), f"/Fe:{out}"]

    flags = ["-std=c++17", "-O3"]
    if isa == "native":
        flags.append("-march=native")
    elif isa == "avx512":
        flags.extend(["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512vnni", "-mfma"])
    elif isa == "avx2":
        flags.extend(["-mavx2", "-mfma"])
    if name not in _EXACT_FP_KERNELS:
        flags.append("-ffast-math")
    if system_name == "Darwin":
        flags.append("-dynamiclib")
    else:
        flags.append("-shared")
    if system_name != "Windows":
        flags.append("-fPIC")
    return [compiler, *flags, str(src), "-o", str(out)]


def build_kernels(
    compiler: str | None = None,
    force: bool = False,
    isa: str = "auto",
) -> list[str]:
    _ = force
    system_name = platform.system()
    if compiler:
        cxx = compiler
    elif system_name == "Windows":
        cxx = shutil.which("cl") or shutil.which("clang++") or shutil.which("g++")
    else:
        cxx = shutil.which("g++") or shutil.which("clang++")

    if cxx is None:
        print(
            f"No C++ compiler found on {system_name}; Python/OpenCV fallback will be used.",
            file=sys.stderr,
        )
        return []

    selected_isa = _detect_isa(isa)
    built: list[str] = []
    for name in KERNELS:
        src = KERNEL_DIR / f"{name}.cpp"
        out = KERNEL_DIR / _library_filename(name, system_name)
        if not src.is_file():
            print(f"[skip] missing source {src}", file=sys.stderr)
            continue
        cmd = _build_command(cxx, src, out, name, selected_isa, system_name)
        print(f"[build:{selected_isa}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        built.append(name)
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cxx",
        default=None,
        help="C++ compiler binary (default: cl/clang++/g++ on Windows; g++/clang++ on Linux/macOS)",
    )
    parser.add_argument(
        "--isa",
        choices=("auto", "native", "avx512", "avx2", "portable"),
        default="auto",
        help="Target SIMD instruction set (default: auto)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Kept for CLI backwards compatibility (portable fallbacks always build)",
    )
    args = parser.parse_args(argv)
    built = build_kernels(compiler=args.cxx, force=args.force, isa=args.isa)
    print(f"Built {len(built)}/{len(KERNELS)} native kernels: {', '.join(built) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
