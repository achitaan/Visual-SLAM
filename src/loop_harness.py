"""3-tier loop-engineering benchmark harness, baseline comparator, and JSONL ledger.

Evaluation isolation: estimator workers run in separate subprocesses with sanitized
argv and environment variables (receiving only `sequence_dir`, never `poses_root`),
and ground-truth `poses/*.txt` files are loaded only in the parent evaluator after
the estimator subprocess exits. OS-level filesystem sandboxing is not enforced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
try:
    import resource
except ImportError:  # Windows does not provide the Unix resource module
    resource = None  # type: ignore[assignment]
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from eval_kitti import _segment_weighted_mean
from kitti import image_paths, load_poses_txt, save_poses_txt
from metrics import evaluate_trajectory
from performance import PerformanceConfig
from shared_slam import MAPPING_PRESET_NAMES, MappingConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID_TIERS = ("tier1", "tier2", "tier3", "custom")
VALID_VERDICTS = ("ACCEPT", "REJECT", "BASELINE")

TIER_DEFAULTS: dict[str, dict[str, Any]] = {
    "tier1": {
        "sequences": ["04"],
        "max_frames_map": {"04": 12},
        "run_pytest": True,
    },
    "tier2": {
        "sequences": ["04", "01", "07"],
        "max_frames_map": {"01": 350},
        "run_pytest": False,
    },
    "tier3": {
        "sequences": ["00", "01", "03", "04", "06", "07", "09", "10"],
        "max_frames_map": {},
        "run_pytest": False,
    },
    "custom": {
        "sequences": None,
        "max_frames_map": {},
        "run_pytest": False,
    },
}

_SANITIZED_ENV_ALLOWLIST = (
    "PATH",
    "PATHEXT",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "VIRTUAL_ENV",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "MPLCONFIGDIR",
    "CUDA_VISIBLE_DEVICES",
    "VISUAL_SLAM_DISABLE_NATIVE",
    "VISUAL_SLAM_SIFT_FIRST_OCTAVE",
)


class AirgapViolationError(RuntimeError):
    """Raised when ground-truth paths overlap or leak into estimator inputs."""


def _current_git_commit(repo_root: Path = REPO_ROOT) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except OSError:
        pass
    return "unknown"


def _windows_peak_rss_mb() -> float:
    try:
        import ctypes
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak",
                    "working",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(process), ctypes.byref(counters), counters.cb
        ):
            return float(counters.peak) / (1024.0 * 1024.0)
    except (AttributeError, ImportError, OSError, ValueError):
        pass
    return 0.0


def _self_peak_rss_mb() -> float:
    vm_hwm_mb = 0.0
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        try:
            for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        vm_hwm_mb = float(parts[1]) / 1024.0
                    break
        except (OSError, ValueError):
            vm_hwm_mb = 0.0
    if resource is not None:
        ru_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        ru_mb = ru_kb / (1024.0 if sys.platform != "darwin" else (1024.0 * 1024.0))
    elif os.name == "nt":
        ru_mb = _windows_peak_rss_mb()
    else:
        ru_mb = 0.0
    return float(max(vm_hwm_mb, ru_mb))


def _atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_json_object_arg(raw: str | dict[str, Any] | None, flag_name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{flag_name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag_name} must decode to a JSON object")
    return parsed


def parse_max_frames_map(raw: str | dict[str, Any] | None) -> dict[str, int | None]:
    raw_dict = parse_json_object_arg(raw, "--max-frames-map")
    result: dict[str, int | None] = {}
    for seq, count in raw_dict.items():
        if not isinstance(seq, str) or len(seq) != 2 or not seq.isdigit():
            raise ValueError(f"Invalid sequence key in --max-frames-map: {seq!r}")
        if count is None or count == 0:
            result[seq] = None
            continue
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError(
                f"Invalid frame limit for sequence {seq} in --max-frames-map: {count!r} (must be int >= 2)"
            )
        result[seq] = int(count)
    return result


def resolve_preset_config(
    preset: str = "improved_stereo",
    *,
    stereo: bool = True,
    config_overrides: dict[str, Any] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> tuple[MappingConfig, dict[str, Any]]:
    merged_overrides: dict[str, Any] = {}
    if config_overrides:
        merged_overrides.update(config_overrides)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                merged_overrides[key] = value
    cfg = MappingConfig.from_preset(preset, stereo=stereo, **merged_overrides)
    return cfg, merged_overrides


def build_airgapped_estimator_invocation(
    *,
    sequence: str,
    data_root: Path,
    poses_root: Path,
    seq_out_dir: Path,
    preset: str,
    effective_overrides: dict[str, Any],
    seed: int,
    opencv_threads: int,
    max_frames: int | None,
    pipeline_extraction: bool = False,
    pipeline_buffer: int = 1,
    openblas_threads: int = 1,
    matching_backend: str = "cpu",
    profile: bool = True,
    frontend_backend: str = "cpu",
    coalesce_prefetch: bool = False,
) -> tuple[list[str], dict[str, str], Path]:
    if matching_backend not in ("cpu", "cuda", "auto"):
        raise ValueError(f"Invalid matching_backend: {matching_backend!r} (expected cpu, cuda, or auto)")
    if frontend_backend not in ("cpu", "cuda"):
        raise ValueError(f"Invalid frontend_backend: {frontend_backend!r} (expected cpu or cuda)")
    if len(sequence) != 2 or not sequence.isdigit():
        raise ValueError(f"Invalid KITTI sequence ID: {sequence!r}")
    data_root_resolved = Path(data_root).expanduser().resolve(strict=False)
    sequence_dir = (data_root_resolved / "sequences" / sequence).resolve(strict=False)
    poses_dir = Path(poses_root).expanduser().resolve(strict=False)

    if (
        sequence_dir == poses_dir
        or sequence_dir.is_relative_to(poses_dir)
        or poses_dir.is_relative_to(sequence_dir)
    ):
        raise AirgapViolationError(
            f"Airgap violation: poses_root ({poses_dir}) overlaps sequence_dir ({sequence_dir})"
        )

    if not (sequence_dir / "calib.txt").is_file():
        raise FileNotFoundError(f"Missing calibration file: {sequence_dir / 'calib.txt'}")
    for cam in ("image_0", "image_1"):
        cam_dir = sequence_dir / cam
        if not cam_dir.is_dir():
            raise FileNotFoundError(f"Missing stereo camera directory: {cam_dir}")
        imgs = image_paths(cam_dir, max_frames)
        if len(imgs) < 2:
            raise ValueError(f"Sequence {sequence} {cam} requires at least 2 PNG frames")
        if max_frames is not None and len(imgs) < max_frames:
            raise ValueError(
                f"Sequence {sequence} {cam} has {len(imgs)} frames, fewer than max_frames={max_frames}"
            )

    env: dict[str, str] = {}
    for key in _SANITIZED_ENV_ALLOWLIST:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    blas_t = str(max(1, int(openblas_threads)))
    env["OPENBLAS_NUM_THREADS"] = blas_t
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = blas_t
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))

    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--_estimator-worker",
        "--sequence",
        sequence,
        "--sequence-dir",
        str(sequence_dir),
        "--output-seq-dir",
        str(Path(seq_out_dir).resolve(strict=False)),
        "--preset",
        preset,
        "--config-overrides",
        json.dumps(effective_overrides, separators=(",", ":"), allow_nan=False),
        "--seed",
        str(int(seed)),
        "--opencv-threads",
        str(int(opencv_threads)),
    ]
    if max_frames is not None:
        command.extend(["--max-frames", str(int(max_frames))])
    if pipeline_extraction:
        command.append("--pipeline-extraction")
        command.extend(["--pipeline-buffer", str(max(1, int(pipeline_buffer)))])
        if coalesce_prefetch:
            command.append("--coalesce-prefetch")
    if matching_backend != "cpu":
        command.extend(["--matching-backend", matching_backend])
    if frontend_backend != "cpu":
        command.extend(["--frontend-backend", frontend_backend])
    if not profile:
        command.append("--no-profile")

    forbidden_tokens = {
        str(poses_dir),
        str(Path(poses_root)),
        "--poses-root",
        "--data-root",
    }
    for arg in command:
        if arg in ("--poses-root", "--data-root") or str(poses_dir) in arg:
            raise AirgapViolationError(
                f"Airgap violation: forbidden token leaked into estimator argv: {arg!r}"
            )
    for env_key, env_val in env.items():
        if "POSE" in env_key.upper() or str(poses_dir) in env_val:
            raise AirgapViolationError(
                f"Airgap violation: ground-truth reference leaked into estimator env {env_key!r}"
            )
    _ = forbidden_tokens
    return command, env, sequence_dir


def run_airgapped_estimator_subprocess(
    *,
    sequence: str,
    data_root: Path,
    poses_root: Path,
    seq_out_dir: Path,
    preset: str,
    effective_overrides: dict[str, Any],
    seed: int,
    opencv_threads: int,
    max_frames: int | None,
    pipeline_extraction: bool = False,
    pipeline_buffer: int = 1,
    openblas_threads: int = 1,
    matching_backend: str = "cpu",
    profile: bool = True,
    frontend_backend: str = "cpu",
    coalesce_prefetch: bool = False,
) -> dict[str, Any]:
    seq_out_dir = Path(seq_out_dir).resolve(strict=False)
    seq_out_dir.mkdir(parents=True, exist_ok=True)
    command, env, _ = build_airgapped_estimator_invocation(
        sequence=sequence,
        data_root=data_root,
        poses_root=poses_root,
        seq_out_dir=seq_out_dir,
        preset=preset,
        effective_overrides=effective_overrides,
        seed=seed,
        opencv_threads=opencv_threads,
        max_frames=max_frames,
        pipeline_extraction=pipeline_extraction,
        pipeline_buffer=pipeline_buffer,
        openblas_threads=openblas_threads,
        matching_backend=matching_backend,
        profile=profile,
        frontend_backend=frontend_backend,
        coalesce_prefetch=coalesce_prefetch,
    )
    log_path = seq_out_dir / "estimator.log"
    wait4_rss_mb = 0.0
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            command,
            env=env,
            cwd=str(REPO_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        if hasattr(os, "wait4"):
            _, wait_status, rusage = os.wait4(proc.pid, 0)
            returncode = os.waitstatus_to_exitcode(wait_status)
            ru_kb = float(rusage.ru_maxrss)
            wait4_rss_mb = ru_kb / (1024.0 if sys.platform != "darwin" else (1024.0 * 1024.0))
        else:
            returncode = proc.wait()

    if returncode != 0:
        tail = ""
        if log_path.is_file():
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(
            f"Estimator subprocess for sequence {sequence} failed with exit code {returncode}:\n{tail}"
        )

    run_json_path = seq_out_dir / "run.json"
    poses_txt_path = seq_out_dir / "poses.txt"
    if not run_json_path.is_file() or not poses_txt_path.is_file():
        raise RuntimeError(
            f"Estimator subprocess for sequence {sequence} did not materialize poses.txt and run.json"
        )

    run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    run_payload["peak_rss_mb"] = float(max(float(run_payload.get("peak_rss_mb", 0.0)), wait4_rss_mb))
    run_payload["isolation_mode"] = "subprocess_argv_env_sanitized_posthoc_eval"
    run_payload["argv_env_sanitized"] = True
    run_payload["posthoc_gt_evaluation"] = True
    run_payload["filesystem_sandbox_enforced"] = False
    run_payload["ground_truth_used_for_estimation"] = False
    run_payload["estimator_argv"] = command
    run_payload["estimator_env_keys"] = sorted(env.keys())
    _atomic_write_json(run_json_path, run_payload)
    return run_payload


def _summarize_latencies_ms(values_s: list[float]) -> dict[str, Any]:
    if not values_s:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    arr = np.asarray(values_s, dtype=np.float64) * 1000.0
    return {
        "count": int(len(arr)),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
    }


def _summarize_loop_events(slam: Any) -> dict[str, Any]:
    verified_pairs: set[tuple[int, int]] = set()
    raw_verified = getattr(slam.loop_worker, "verified", ())
    if isinstance(raw_verified, dict):
        for pair in raw_verified.keys():
            if isinstance(pair, (tuple, list)) and len(pair) == 2:
                verified_pairs.add((int(pair[0]), int(pair[1])))
    else:
        for edge in raw_verified:
            i = getattr(edge, "first_keyframe", None)
            j = getattr(edge, "second_keyframe", None)
            if i is not None and j is not None:
                verified_pairs.add((int(i), int(j)))
    applied_loops = 0
    applied_timeline: list[dict[str, Any]] = []
    v2a_latencies_s: list[float] = []
    pgo_durations_s: list[float] = []
    events_by_type: Counter[str] = Counter()
    for event in getattr(slam.loop_worker, "events", ()):
        if not isinstance(event, dict):
            continue
        etype = str(event.get("type", "unknown"))
        events_by_type[etype] += 1
        if etype == "loop_verification":
            for attempt in event.get("attempts", ()):
                if isinstance(attempt, dict) and attempt.get("geometrically_verified"):
                    i = attempt.get("first_keyframe")
                    j = attempt.get("second_keyframe")
                    if i is not None and j is not None:
                        verified_pairs.add((int(i), int(j)))
        elif etype == "loop_applied":
            applied_loops = int(event.get("loops", applied_loops))
            ev_v2a = [float(x) for x in (event.get("verification_to_apply_s") or ())]
            v2a_latencies_s.extend(ev_v2a)
            if "pgo_elapsed_s" in event:
                pgo_durations_s.append(float(event["pgo_elapsed_s"]))
            applied_timeline.append(
                {
                    "loops": int(applied_loops),
                    "revision": int(event.get("revision", -1)),
                    "pgo_elapsed_s": float(event.get("pgo_elapsed_s", 0.0)),
                }
            )
    if v2a_latencies_s:
        arr = np.asarray(v2a_latencies_s, dtype=np.float64)
        v2a_summary = {
            "count": int(len(arr)),
            "mean_s": float(np.mean(arr)),
            "p50_s": float(np.percentile(arr, 50)),
            "p95_s": float(np.percentile(arr, 95)),
            "max_s": float(np.max(arr)),
        }
    else:
        v2a_summary = {"count": 0, "mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "max_s": 0.0}
    return {
        "verified_loops": int(len(verified_pairs)),
        "applied_loops": int(applied_loops),
        "discarded_loops": int(events_by_type.get("loop_discarded", 0)),
        "failed_loops": int(events_by_type.get("loop_failed", 0)),
        "skipped_loops": int(events_by_type.get("loop_skipped", 0)),
        "events_by_type": dict(sorted(events_by_type.items())),
        "verified_pairs": [[int(a), int(b)] for a, b in sorted(verified_pairs)],
        "applied_timeline": applied_timeline,
        "verification_to_apply_latency_s": v2a_summary,
    }


def _count_loop_metrics(slam: Any) -> tuple[int, int]:
    summary = _summarize_loop_events(slam)
    return summary["verified_loops"], summary["applied_loops"]


def _trajectory_translation_delta(
    poses_a: list[np.ndarray],
    poses_b: list[np.ndarray],
) -> dict[str, float]:
    if len(poses_a) != len(poses_b) or not poses_a:
        return {"rmse_m": 0.0, "max_m": 0.0}
    ta = np.asarray([p[:3, 3] for p in poses_a], dtype=np.float64)
    tb = np.asarray([p[:3, 3] for p in poses_b], dtype=np.float64)
    norms = np.linalg.norm(ta - tb, axis=1)
    return {
        "rmse_m": float(np.sqrt(np.mean(norms * norms))),
        "max_m": float(np.max(norms)),
    }


def _estimator_worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="loop_harness_estimator_worker")
    parser.add_argument("--_estimator-worker", action="store_true", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-seq-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=list(MAPPING_PRESET_NAMES), required=True)
    parser.add_argument("--config-overrides", type=str, default="{}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--pipeline-extraction", action="store_true")
    parser.add_argument("--pipeline-buffer", type=int, default=2)
    parser.add_argument("--coalesce-prefetch", action="store_true")
    parser.add_argument(
        "--matching-backend", choices=["cpu", "cuda", "auto"], default="cpu"
    )
    parser.add_argument("--frontend-backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args(argv)

    import gc
    import cv2 as cv
    from StereoVisualOdometry import StereoVisualOdometry
    from shared_slam import SharedSlam, StereoCamera

    gc.set_threshold(50000, 20, 200)
    np.random.seed(args.seed)
    cv.setRNGSeed(args.seed)
    cv.setNumThreads(args.opencv_threads)

    overrides = parse_json_object_arg(args.config_overrides, "--config-overrides")
    cfg = MappingConfig.from_preset(args.preset, stereo=True, **overrides)
    perf = PerformanceConfig(
        retrieval="current",
        matching_backend=str(args.matching_backend),
        cpu_optimizations=True,
        profile=not bool(args.no_profile),
        pipeline_extraction=bool(args.pipeline_extraction),
        frontend_backend=args.frontend_backend,
        coalesce_prefetch=args.coalesce_prefetch,
    )

    seq_dir = args.sequence_dir.resolve()
    vo = StereoVisualOdometry(
        str(seq_dir / "image_"),
        str(seq_dir / "calib.txt"),
        use_brute_force=False,
        poses_path=None,
        draw_matches=False,
        max_frames=args.max_frames,
    )
    total_frames = len(vo.Images_1)
    if total_frames < 2:
        raise ValueError(f"Sequence {args.sequence} requires at least 2 stereo pairs")
    if args.max_frames is not None and total_frames < args.max_frames:
        raise ValueError(
            f"Sequence {args.sequence} only has {total_frames} frames (< max_frames={args.max_frames})"
        )

    stereo_cam = StereoCamera(vo.stereo, vo.Q, vo.baseline)
    slam = SharedSlam(vo.K1, stereo=stereo_cam, config=cfg, performance=perf)
    if args.frontend_backend == "cuda" and slam.cuda_frontend is None:
        slam.close(finish=False)
        raise RuntimeError(f"CUDA frontend is unavailable: {slam.cuda_frontend_reason}")

    online_first_poses: list[np.ndarray] = []
    online_end_poses: list[np.ndarray] = []
    frame_latencies_s: list[float] = []
    service_intervals_s: list[float] = []
    process_latencies_s: list[float] = []
    keyframe_latencies_s: list[float] = []
    non_keyframe_latencies_s: list[float] = []
    io_elapsed_s = 0.0
    slam_process_elapsed_s = 0.0
    worker_extract_elapsed_s = 0.0
    extract_wait_elapsed_s = 0.0
    online_elapsed_s = 0.0
    shutdown_drain_s = 0.0

    pipeline = None
    if args.pipeline_extraction:
        from frontend_pipeline import FrontendExtractionPipeline
        pipeline = FrontendExtractionPipeline(
            slam,
            buffer_size=max(1, int(args.pipeline_buffer)),
            lazy_extraction=True,
        )

    started = time.perf_counter()
    try:
        if pipeline is not None:
            _load_pair = (vo.Images_1.__getitem__, vo.Images_2.__getitem__)

            submit_times: dict[int, float] = {}
            prefetch_depth = min(max(1, int(args.pipeline_buffer)), total_frames)
            next_submit = 0
            while next_submit < prefetch_depth:
                submit_times[next_submit] = time.perf_counter()
                pipeline.submit_loader(next_submit, _load_pair)
                next_submit += 1

            for frame_idx in range(total_frames):
                t_iter_start = time.perf_counter()
                extracted = pipeline.get(frame_idx)
                t_extract_ready = time.perf_counter()
                if next_submit < total_frames:
                    submit_times[next_submit] = time.perf_counter()
                    pipeline.submit_loader(
                        next_submit,
                        _load_pair,
                        eager=slam.should_prefetch_full_extraction(next_submit),
                    )
                    next_submit += 1
                prev_kfs = len(slam.map.keyframes)
                t_proc_start = time.perf_counter()
                pose_emitted, _ = slam.process(frame_idx, extracted=extracted)
                t_frame_end = time.perf_counter()

                online_first_poses.append(np.asarray(pose_emitted, dtype=np.float64).copy())
                t_avail = submit_times.pop(frame_idx, t_iter_start)
                # True first-pose latency from image availability (includes queueing + I/O + extraction + tracking)
                first_pose_dt = t_frame_end - t_avail
                service_dt = t_frame_end - t_iter_start
                proc_dt = t_frame_end - t_proc_start
                io_elapsed_s += extracted.io_elapsed_s
                worker_extract_elapsed_s += extracted.extract_elapsed_s
                extract_wait_elapsed_s += (t_extract_ready - t_iter_start)
                slam_process_elapsed_s += proc_dt
                frame_latencies_s.append(first_pose_dt)
                service_intervals_s.append(service_dt)
                process_latencies_s.append(proc_dt)
                if len(slam.map.keyframes) > prev_kfs:
                    keyframe_latencies_s.append(first_pose_dt)
                else:
                    non_keyframe_latencies_s.append(first_pose_dt)
        else:
            for frame_idx in range(total_frames):
                t_frame_start = time.perf_counter()
                im1 = vo.Images_1[frame_idx]
                im2 = vo.Images_2[frame_idx]
                t_io_end = time.perf_counter()
                prev_kfs = len(slam.map.keyframes)
                pose_emitted, _ = slam.process(frame_idx, im1, im2)
                t_frame_end = time.perf_counter()

                online_first_poses.append(np.asarray(pose_emitted, dtype=np.float64).copy())
                io_dt = t_io_end - t_frame_start
                proc_dt = t_frame_end - t_io_end
                frame_dt = t_frame_end - t_frame_start
                io_elapsed_s += io_dt
                slam_process_elapsed_s += proc_dt
                frame_latencies_s.append(frame_dt)
                service_intervals_s.append(frame_dt)
                process_latencies_s.append(proc_dt)
                if len(slam.map.keyframes) > prev_kfs:
                    keyframe_latencies_s.append(frame_dt)
                else:
                    non_keyframe_latencies_s.append(frame_dt)

        online_elapsed_s = max(time.perf_counter() - started, 1e-9)
        with slam.map.lock:
            online_end_poses = [p.copy() for p in slam.map.poses]
    finally:
        t_close_start = time.perf_counter()
        if pipeline is not None:
            pipeline.close()
        slam.close(finish=True)
        shutdown_drain_s = max(time.perf_counter() - t_close_start, 0.0)
    elapsed_s = max(time.perf_counter() - started, 1e-9)

    poses = slam.map.poses
    if len(poses) != total_frames:
        raise RuntimeError(
            f"Estimator produced {len(poses)} poses for {total_frames} input frames"
        )

    out_dir = args.output_seq_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_poses_txt(out_dir / "poses.txt", poses)
    save_poses_txt(out_dir / "online_first_poses.txt", online_first_poses)
    if len(online_end_poses) == total_frames:
        save_poses_txt(out_dir / "online_end_poses.txt", online_end_poses)

    states = Counter(slam.map.statuses)
    loop_summary = _summarize_loop_events(slam)
    verified_loops = loop_summary["verified_loops"]
    applied_loops = loop_summary["applied_loops"]
    peak_rss_mb = _self_peak_rss_mb()
    bundle_reports = getattr(slam, "bundle_reports", ())

    matcher_meta = (
        slam.matcher.metadata()
        if hasattr(getattr(slam, "matcher", None), "metadata")
        else {"requested": str(args.matching_backend)}
    )
    run_payload = {
        "sequence": args.sequence,
        "frames": total_frames,
        "max_frames": args.max_frames,
        "seed": args.seed,
        "opencv_threads": args.opencv_threads,
        "openblas_threads": int(os.environ.get("OPENBLAS_NUM_THREADS", "1")),
        "native_sift_first_octave": 0 if os.environ.get("VISUAL_SLAM_SIFT_FIRST_OCTAVE", "").strip() == "0" else -1,
        "pipeline_extraction": bool(args.pipeline_extraction),
        "matching_backend": matcher_meta,
        "preset": args.preset,
        "config_overrides": overrides,
        "configuration": asdict(slam.config),
        "performance_configuration": asdict(perf),
        "ground_truth_used_for_estimation": False,
        "status": (
            "completed_with_tracking_loss" if states.get("lost", 0) > 0 else "completed"
        ),
        "states": dict(states),
        "lost_frames": int(states.get("lost", 0)),
        "keyframes": len(slam.map.keyframes),
        "landmarks": len(slam.map.landmarks),
        "verified_loops": int(verified_loops),
        "applied_loops": int(applied_loops),
        "loop_events": loop_summary,
        "bundle_adjustments_called": int(len(bundle_reports)),
        "bundle_adjustments_applied": int(
            sum(1 for r in bundle_reports if r.get("applied"))
        ),
        "elapsed_s": float(elapsed_s),
        "online_elapsed_s": float(online_elapsed_s),
        "slam_process_elapsed_s": float(slam_process_elapsed_s),
        "io_elapsed_s": float(io_elapsed_s),
        "worker_extract_elapsed_s": float(worker_extract_elapsed_s),
        "extract_wait_elapsed_s": float(extract_wait_elapsed_s),
        "shutdown_drain_s": float(shutdown_drain_s),
        "processing_fps": float(total_frames / elapsed_s),
        "online_fps": float(total_frames / online_elapsed_s),
        "frame_latency_ms": {
            "all": _summarize_latencies_ms(frame_latencies_s),
            "service_interval": _summarize_latencies_ms(service_intervals_s),
            "process_only": _summarize_latencies_ms(process_latencies_s),
            "keyframe": _summarize_latencies_ms(keyframe_latencies_s),
            "non_keyframe": _summarize_latencies_ms(non_keyframe_latencies_s),
        },
        "peak_rss_mb": float(peak_rss_mb),
        "stage_timings": slam.profile.report(),
        "stage_profile": slam.profile.detailed_report(),
    }
    _atomic_write_json(out_dir / "run.json", run_payload)
    return 0


def evaluate_sequence_posthoc(
    *,
    sequence: str,
    poses_root: Path,
    seq_out_dir: Path,
    run_payload: dict[str, Any],
    max_frames: int | None,
) -> dict[str, Any]:
    pose_file = Path(poses_root) / f"{sequence}.txt"
    if not pose_file.is_file():
        raise FileNotFoundError(f"Missing ground-truth pose file: {pose_file}")
    gt_full = load_poses_txt(pose_file)
    if max_frames is not None:
        if len(gt_full) < max_frames:
            raise ValueError(
                f"Ground-truth pose file {pose_file} has {len(gt_full)} poses (< max_frames={max_frames})"
            )
        gt = gt_full[:max_frames]
    else:
        gt = gt_full

    seq_dir = Path(seq_out_dir)
    est = load_poses_txt(seq_dir / "poses.txt")
    if len(est) != len(gt):
        raise ValueError(
            f"Trajectory length mismatch on sequence {sequence}: len(est)={len(est)} != len(gt)={len(gt)}"
        )

    traj_metrics = evaluate_trajectory(gt, est, alignment="se3")
    rot_deg_m = traj_metrics.get("rotation_deg_per_m")
    rot_deg_100m = float(100.0 * rot_deg_m) if rot_deg_m is not None else None
    t_rel = traj_metrics.get("translation_percent")

    online_first_metrics = None
    online_vs_final = None
    online_first_path = seq_dir / "online_first_poses.txt"
    if online_first_path.is_file():
        online_first = load_poses_txt(online_first_path)
        if len(online_first) == len(gt):
            om = evaluate_trajectory(gt, online_first, alignment="se3")
            om_rot = om.get("rotation_deg_per_m")
            om_t = om.get("translation_percent")
            online_first_metrics = {
                "ate_rmse_m": float(om["ate_rmse_m"]),
                "raw_ate_rmse_m": float(om["raw_ate_rmse_m"]),
                "translation_percent": float(om_t) if om_t is not None else None,
                "rotation_deg_per_100m": float(100.0 * om_rot) if om_rot is not None else None,
            }
            online_vs_final = _trajectory_translation_delta(online_first, est)

    online_end_vs_final = None
    online_end_path = seq_dir / "online_end_poses.txt"
    if online_end_path.is_file():
        online_end = load_poses_txt(online_end_path)
        if len(online_end) == len(est):
            online_end_vs_final = _trajectory_translation_delta(online_end, est)

    result = {
        "frames": int(traj_metrics["frames"]),
        "segments": int(traj_metrics["segment_count"]),
        "ate_rmse_m": float(traj_metrics["ate_rmse_m"]),
        "raw_ate_rmse_m": float(traj_metrics["raw_ate_rmse_m"]),
        "translation_percent": float(t_rel) if t_rel is not None else None,
        "rotation_deg_per_m": float(rot_deg_m) if rot_deg_m is not None else None,
        "rotation_deg_per_100m": rot_deg_100m,
        "lost_frames": int(run_payload["lost_frames"]),
        "verified_loops": int(run_payload["verified_loops"]),
        "applied_loops": int(run_payload["applied_loops"]),
        "elapsed_s": float(run_payload["elapsed_s"]),
        "online_elapsed_s": float(run_payload.get("online_elapsed_s", run_payload["elapsed_s"])),
        "shutdown_drain_s": float(run_payload.get("shutdown_drain_s", 0.0)),
        "processing_fps": float(run_payload["processing_fps"]),
        "online_fps": float(run_payload.get("online_fps", run_payload["processing_fps"])),
        "peak_rss_mb": float(run_payload["peak_rss_mb"]),
    }
    if "matching_backend" in run_payload:
        result["matching_backend"] = run_payload["matching_backend"]
    if "frame_latency_ms" in run_payload:
        result["frame_latency_ms"] = run_payload["frame_latency_ms"]
    if "loop_events" in run_payload:
        result["loop_events"] = run_payload["loop_events"]
    if online_first_metrics is not None:
        result["online_first_metrics"] = online_first_metrics
    if online_vs_final is not None:
        result["online_first_vs_final_delta"] = online_vs_final
    if online_end_vs_final is not None:
        result["online_end_vs_final_delta"] = online_end_vs_final
    return result


def compute_loop_aggregate(sequences_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(sequences_summary.values())
    if not rows:
        return {
            "sequence_count": 0,
            "total_frames": 0,
            "total_segments": 0,
            "segment_weighted_t_rel_pct": None,
            "segment_weighted_r_rel_deg_per_m": None,
            "segment_weighted_r_rel_deg_per_100m": None,
            "mean_ate_se3_m": None,
            "total_lost_frames": 0,
            "total_elapsed_s": 0.0,
            "overall_fps": 0.0,
            "peak_rss_mb": 0.0,
        }
    normalized_for_weighting = [
        {**r, "segment_count": int(r.get("segments", 0))}
        for r in rows
    ]
    t_rel = _segment_weighted_mean(normalized_for_weighting, "translation_percent")
    r_rel_m = _segment_weighted_mean(normalized_for_weighting, "rotation_deg_per_m")
    r_rel_100m = float(100.0 * r_rel_m) if r_rel_m is not None else None
    total_frames = int(sum(int(r["frames"]) for r in rows))
    total_segments = int(sum(int(r["segments"]) for r in rows))
    total_elapsed_s = float(sum(float(r["elapsed_s"]) for r in rows))
    mean_ate = float(np.mean([float(r["ate_rmse_m"]) for r in rows]))
    total_lost = int(sum(int(r["lost_frames"]) for r in rows))
    peak_rss = float(max(float(r.get("peak_rss_mb", 0.0)) for r in rows))
    overall_fps = float(total_frames / total_elapsed_s) if total_elapsed_s > 0.0 else 0.0
    return {
        "sequence_count": len(rows),
        "total_frames": total_frames,
        "total_segments": total_segments,
        "segment_weighted_t_rel_pct": t_rel,
        "segment_weighted_r_rel_deg_per_m": r_rel_m,
        "segment_weighted_r_rel_deg_per_100m": r_rel_100m,
        "mean_ate_se3_m": mean_ate,
        "total_lost_frames": total_lost,
        "total_elapsed_s": total_elapsed_s,
        "overall_fps": overall_fps,
        "peak_rss_mb": peak_rss,
    }


def load_reference_baselines(
    repo_root: Path = REPO_ROOT,
) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    baselines: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        "legacy_vo": {},
        "legacy_vo_pgo": {},
        "verified_shared_slam": {},
    }
    bench_dir = Path(repo_root) / "docs" / "benchmark"

    results_path = bench_dir / "results.json"
    if results_path.is_file():
        data = json.loads(results_path.read_text(encoding="utf-8"))
        for entry in data.get("sequences", ()):
            seq = entry["sequence"]
            frames = int(entry["frames"])
            raw = entry.get("raw")
            if isinstance(raw, dict):
                baselines["legacy_vo"][(seq, frames)] = {
                    "segments": int(raw["segment_count"]),
                    "translation_percent": float(raw["translation_percent"]),
                    "rotation_deg_per_m": float(raw["rotation_deg_per_m"]),
                    "ate_rmse_m": float(raw["ate_rmse_m"]),
                }
            corr = entry.get("corrected")
            if isinstance(corr, dict):
                baselines["legacy_vo_pgo"][(seq, frames)] = {
                    "segments": int(corr["segment_count"]),
                    "translation_percent": float(corr["translation_percent"]),
                    "rotation_deg_per_m": float(corr["rotation_deg_per_m"]),
                    "ate_rmse_m": float(corr["ate_rmse_m"]),
                }

    fresh_path = bench_dir / "FRESH_STEREO_BASELINE_RESULTS.json"
    if fresh_path.is_file():
        fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
        for row in fresh.get("rows", ()):
            if row.get("label") == "Preserved VO" and row.get("status") == "completed":
                seq = row["sequence"]
                frames = int(row["frames"])
                m = row["metrics"]
                entry_dict = {
                    "segments": int(m["segment_count"]),
                    "translation_percent": float(m["translation_percent"]),
                    "rotation_deg_per_m": float(m["rotation_deg_per_m"]),
                    "ate_rmse_m": float(m["ate_rmse_m"]),
                }
                baselines["legacy_vo"].setdefault((seq, frames), entry_dict)
                if seq in ("01", "04"):
                    baselines["legacy_vo_pgo"].setdefault((seq, frames), dict(entry_dict))

    verified_path = bench_dir / "original-python-verified.json"
    if verified_path.is_file():
        ver = json.loads(verified_path.read_text(encoding="utf-8"))
        short01 = ver.get("short01_seed0")
        if isinstance(short01, dict) and isinstance(short01.get("metrics"), dict):
            m = short01["metrics"]
            baselines["verified_shared_slam"][("01", int(short01["frames"]))] = {
                "segments": int(m["segment_count"]),
                "translation_percent": float(m["translation_percent"]),
                "rotation_deg_per_m": float(m["rotation_deg_per_m"]),
                "ate_rmse_m": float(m["ate_rmse_m"]),
                "elapsed_s": float(short01.get("elapsed_s", 0.0)),
                "lost_frames": int(short01.get("lost_frames", 0)),
            }
        for case in ver.get("full_seed0_cases", ()):
            seq = case["sequence"]
            frames = int(case["frames"])
            cand = case.get("candidate", {})
            m = cand.get("metrics")
            if isinstance(m, dict):
                baselines["verified_shared_slam"][(seq, frames)] = {
                    "segments": int(m["segment_count"]),
                    "translation_percent": float(m["translation_percent"]),
                    "rotation_deg_per_m": float(m["rotation_deg_per_m"]),
                    "ate_rmse_m": float(m["ate_rmse_m"]),
                    "elapsed_s": float(cand.get("elapsed_s", 0.0)),
                    "lost_frames": int(cand.get("lost_frames", 0)),
                }

    loop_base_path = bench_dir / "loop_baseline.json"
    if loop_base_path.is_file():
        try:
            lb = json.loads(loop_base_path.read_text(encoding="utf-8"))
            for seq, srow in lb.get("sequences", {}).items():
                frames = int(srow["frames"])
                if srow.get("translation_percent") is not None:
                    baselines["verified_shared_slam"].setdefault(
                        (seq, frames),
                        {
                            "segments": int(srow["segments"]),
                            "translation_percent": float(srow["translation_percent"]),
                            "rotation_deg_per_m": float(srow["rotation_deg_per_m"]),
                            "ate_rmse_m": float(srow["ate_rmse_m"]),
                            "elapsed_s": float(srow.get("elapsed_s", 0.0)),
                            "lost_frames": int(srow.get("lost_frames", 0)),
                        },
                    )
        except (OSError, ValueError, KeyError):
            pass

    return baselines


def compute_paired_baseline_comparison(
    sequences_summary: dict[str, dict[str, Any]],
    baselines: dict[str, dict[tuple[str, int], dict[str, Any]]],
) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    deltas: dict[str, float | None] = {
        "vs_legacy_vo_t_rel_pct": None,
        "vs_legacy_vo_pgo_t_rel_pct": None,
        "vs_verified_shared_slam_t_rel_pct": None,
    }
    details: dict[str, dict[str, Any]] = {}
    mapping_keys = (
        ("legacy_vo", "vs_legacy_vo_t_rel_pct"),
        ("legacy_vo_pgo", "vs_legacy_vo_pgo_t_rel_pct"),
        ("verified_shared_slam", "vs_verified_shared_slam_t_rel_pct"),
    )
    for family, delta_key in mapping_keys:
        fam_table = baselines.get(family, {})
        paired_cand = []
        paired_base = []
        for seq, cand_row in sequences_summary.items():
            key = (seq, int(cand_row["frames"]))
            base_row = fam_table.get(key)
            if base_row is None:
                seq_candidates = [
                    (f_cnt, row)
                    for (b_seq, f_cnt), row in fam_table.items()
                    if b_seq == seq
                ]
                if seq_candidates:
                    seq_candidates.sort(key=lambda item: item[0], reverse=True)
                    base_row = seq_candidates[0][1]
            if (
                base_row is not None
                and int(cand_row.get("segments", 0)) > 0
                and cand_row.get("translation_percent") is not None
                and int(base_row.get("segments", 0)) > 0
                and base_row.get("translation_percent") is not None
            ):
                paired_cand.append({**cand_row, "segment_count": int(cand_row["segments"])})
                paired_base.append({**base_row, "segment_count": int(base_row["segments"])})
        if paired_cand:
            cand_t = _segment_weighted_mean(paired_cand, "translation_percent")
            base_t = _segment_weighted_mean(paired_base, "translation_percent")
            base_r_m = _segment_weighted_mean(paired_base, "rotation_deg_per_m")
            base_ate = float(np.mean([float(r["ate_rmse_m"]) for r in paired_base]))
            delta_val = (
                float(cand_t - base_t)
                if (cand_t is not None and base_t is not None)
                else None
            )
            deltas[delta_key] = delta_val
            details[family] = {
                "paired_sequences": len(paired_cand),
                "paired_segments": int(sum(r["segment_count"] for r in paired_base)),
                "baseline_t_rel_pct": base_t,
                "baseline_r_rel_deg_per_100m": (
                    float(100.0 * base_r_m) if base_r_m is not None else None
                ),
                "baseline_mean_ate_se3_m": base_ate,
                "candidate_t_rel_pct": cand_t,
                "delta_t_rel_pct": delta_val,
            }
        else:
            details[family] = {
                "paired_sequences": 0,
                "paired_segments": 0,
                "baseline_t_rel_pct": None,
                "baseline_r_rel_deg_per_100m": None,
                "baseline_mean_ate_se3_m": None,
                "candidate_t_rel_pct": None,
                "delta_t_rel_pct": None,
            }
    return deltas, details


def _cohort_signature(sequences_summary: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {seq: int(row["frames"]) for seq, row in sorted(sequences_summary.items())}


def _find_cohort_reference(
    sequences_summary: dict[str, dict[str, Any]],
    *,
    ledger_path: Path | None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any] | None:
    target_sig = _cohort_signature(sequences_summary)
    if not target_sig:
        return None

    if ledger_path is not None and Path(ledger_path).is_file():
        lines = Path(ledger_path).read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("verdict") not in ("ACCEPT", "BASELINE"):
                continue
            rec_seqs = rec.get("sequences")
            if isinstance(rec_seqs, dict) and _cohort_signature(rec_seqs) == target_sig:
                return rec

    loop_base_path = Path(repo_root) / "docs" / "benchmark" / "loop_baseline.json"
    if loop_base_path.is_file():
        try:
            lb = json.loads(loop_base_path.read_text(encoding="utf-8"))
            lb_seqs = lb.get("sequences")
            if isinstance(lb_seqs, dict) and _cohort_signature(lb_seqs) == target_sig:
                return lb
        except (OSError, ValueError):
            pass

    return None


def compute_verdict(
    *,
    aggregate: dict[str, Any],
    sequences_summary: dict[str, dict[str, Any]],
    mark_baseline: bool,
    ledger_path: Path | None,
    repo_root: Path = REPO_ROOT,
) -> tuple[str, str]:
    if mark_baseline:
        return "BASELINE", "Explicitly marked as baseline via --mark-baseline"

    if aggregate["sequence_count"] == 0:
        return "BASELINE", "Tier 1 unit-test-only verification passed"

    ref = _find_cohort_reference(
        sequences_summary,
        ledger_path=ledger_path,
        repo_root=repo_root,
    )
    if ref is None:
        return "BASELINE", "Initial cohort baseline recorded (no prior cohort record found)"

    ref_agg = ref.get("aggregate", {})
    cand_lost = int(aggregate.get("total_lost_frames", 0))
    ref_lost = int(ref_agg.get("total_lost_frames", 0))
    if cand_lost > ref_lost:
        return "REJECT", f"Lost frames regressed ({cand_lost} > {ref_lost})"

    cand_rss = float(aggregate.get("peak_rss_mb", 0.0))
    if cand_rss > 6144.0:
        return "REJECT", f"Peak RSS ({cand_rss:.1f} MiB) exceeded 6144 MiB budget"

    cand_elapsed = float(aggregate.get("total_elapsed_s", 0.0))
    ref_elapsed = float(ref_agg.get("total_elapsed_s", 0.0))
    speedup = (ref_elapsed / max(cand_elapsed, 1e-9)) if ref_elapsed > 0.0 else 1.0

    cand_t = aggregate.get("segment_weighted_t_rel_pct")
    ref_t = ref_agg.get("segment_weighted_t_rel_pct")
    if int(aggregate.get("total_segments", 0)) > 0 and cand_t is not None and ref_t is not None:
        delta_t = float(cand_t) - float(ref_t)
        if delta_t < -1e-4:
            return (
                "ACCEPT",
                f"Segment-weighted t_rel improved by {-delta_t:.4f}% ({ref_t:.4f}% -> {cand_t:.4f}%)",
            )
        if abs(delta_t) <= 0.01 and speedup >= 1.05:
            return (
                "ACCEPT",
                f"Runtime speedup {speedup:.2f}x with preserved accuracy (delta_t={delta_t:+.4f}%)",
            )
        return (
            "REJECT",
            f"No improvement over cohort reference (delta_t={delta_t:+.4f}%, speedup={speedup:.2f}x)",
        )

    cand_ate = aggregate.get("mean_ate_se3_m")
    ref_ate = ref_agg.get("mean_ate_se3_m")
    if cand_ate is not None and ref_ate is not None:
        delta_ate = float(cand_ate) - float(ref_ate)
        if delta_ate < -1e-5:
            return (
                "ACCEPT",
                f"Short-cohort SE(3) ATE improved ({ref_ate:.4f} m -> {cand_ate:.4f} m)",
            )
        if abs(delta_ate) <= 1e-3 and speedup >= 1.05:
            return (
                "ACCEPT",
                f"Short-cohort speedup {speedup:.2f}x at equal SE(3) ATE",
            )
    return "REJECT", f"No measurable accuracy or throughput improvement (speedup={speedup:.2f}x)"


def _fmt_opt(value: float | None, fmt: str = ".4f", suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/A"
    return f"{value:{fmt}}{suffix}"


def format_comparison_markdown(
    summary: dict[str, Any],
    baseline_details: dict[str, dict[str, Any]],
    run_payloads: dict[str, dict[str, Any]],
) -> str:
    agg = summary["aggregate"]
    lines = [
        f"# Loop-Engineering Benchmark Report — `{summary['run_id']}`",
        "",
        f"- **Timestamp (UTC)**: `{summary['timestamp_utc']}`",
        f"- **Git Commit**: `{summary['git_commit']}`",
        f"- **Tier**: `{summary['tier']}`",
        f"- **Preset**: `{summary['preset']}`",
        f"- **Hypothesis**: {summary['hypothesis'] or '(none)'}",
        f"- **Ground Truth Used For Estimation**: `{summary['ground_truth_used_for_estimation']}`",
        f"- **Verdict**: **`{summary['verdict']}`** ({summary.get('verdict_reason', '')})",
        "",
        "## Multi-Baseline Comparison (Paired Cohort)",
        "",
        "| System / Baseline | Paired Seqs | Paired Segs | `t_rel` (%) | `r_rel` (deg/100m) | Mean SE(3) ATE (m) | Candidate `Δt_rel` (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = (
        ("legacy_vo", "Legacy Stereo VO (`VO`)"),
        ("legacy_vo_pgo", "Legacy Stereo VO + PGO (`VO+PGO`)"),
        ("verified_shared_slam", "Verified SharedSlam (`loop_mode=off`)"),
    )
    for fam_key, label in labels:
        det = baseline_details.get(fam_key, {})
        lines.append(
            f"| {label} | {det.get('paired_sequences', 0)} | {det.get('paired_segments', 0)} "
            f"| {_fmt_opt(det.get('baseline_t_rel_pct'))} "
            f"| {_fmt_opt(det.get('baseline_r_rel_deg_per_100m'))} "
            f"| {_fmt_opt(det.get('baseline_mean_ate_se3_m'), '.3f')} "
            f"| {_fmt_opt(det.get('delta_t_rel_pct'), '+.4f')} |"
        )
    lines.append(
        f"| **Current Candidate (`{summary['preset']}`)** | {agg['sequence_count']} | {agg['total_segments']} "
        f"| **{_fmt_opt(agg['segment_weighted_t_rel_pct'])}** "
        f"| **{_fmt_opt(agg['segment_weighted_r_rel_deg_per_100m'])}** "
        f"| **{_fmt_opt(agg['mean_ate_se3_m'], '.3f')}** | — |"
    )
    lines.extend(
        [
            "",
            "## Per-Sequence Metrics",
            "",
            "| Seq | Frames | Segs | `t_rel` (%) | `r_rel` (deg/100m) | SE(3) ATE (m) | Raw ATE (m) | Lost | Verified Loops | Applied Loops | Elapsed (s) | FPS | Peak RSS (MiB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seq, srow in summary["sequences"].items():
        rss_val = run_payloads.get(seq, {}).get("peak_rss_mb", srow.get("peak_rss_mb"))
        lines.append(
            f"| `{seq}` | {srow['frames']} | {srow['segments']} "
            f"| {_fmt_opt(srow['translation_percent'])} "
            f"| {_fmt_opt(srow['rotation_deg_per_100m'])} "
            f"| {_fmt_opt(srow['ate_rmse_m'], '.3f')} "
            f"| {_fmt_opt(srow['raw_ate_rmse_m'], '.3f')} "
            f"| {srow['lost_frames']} | {srow['verified_loops']} | {srow['applied_loops']} "
            f"| {srow['elapsed_s']:.2f} | {srow['processing_fps']:.2f} "
            f"| {_fmt_opt(rss_val, '.1f')} |"
        )

    stage_totals: Counter[str] = Counter()
    for rpay in run_payloads.values():
        stage_map = rpay.get("stage_timings") or {}
        if isinstance(stage_map, dict) and "stages" in stage_map and isinstance(stage_map["stages"], dict):
            stage_map = stage_map["stages"]
        for st_name, st_val in stage_map.items():
            if isinstance(st_val, dict):
                sec = st_val.get("inclusive_seconds", st_val.get("total_s"))
                if sec is not None:
                    stage_totals[st_name] += float(sec)
            elif isinstance(st_val, (int, float)):
                stage_totals[st_name] += float(st_val)
    if stage_totals:
        lines.extend(
            [
                "",
                "## Stage Timing Hotspots (Inclusive Seconds Across Run)",
                "",
                "| Stage | Inclusive Time (s) |",
                "|---|---:|",
            ]
        )
        for st_name, st_sec in stage_totals.most_common(10):
            lines.append(f"| `{st_name}` | {st_sec:.3f} |")

    lines.append("")
    return "\n".join(lines)


def append_ledger_record(ledger_path: Path, record: dict[str, Any]) -> None:
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(serialized)


def run_tier1_pytest_suite(output_dir: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return {
            "executed": False,
            "skipped_reason": "nested_pytest_session_detected",
            "returncode": 0,
        }
    pytest_tmp = (Path(output_dir).resolve() / ".pytest_tmp")
    if pytest_tmp.exists():
        shutil.rmtree(pytest_tmp, ignore_errors=True)
    env = {
        **os.environ,
        "MPLCONFIGDIR": str(Path(repo_root) / ".mpl-cache"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests",
        "--ignore=tests/e2e",
        f"--basetemp={pytest_tmp}",
    ]
    t0 = time.perf_counter()
    res = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    shutil.rmtree(pytest_tmp, ignore_errors=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Tier 1 pytest suite failed with exit code {res.returncode}:\n{res.stdout}\n{res.stderr}"
        )
    return {
        "executed": True,
        "returncode": int(res.returncode),
        "elapsed_s": float(elapsed),
    }


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run airgapped 3-tier loop-engineering SLAM benchmarks and append to JSONL ledger."
    )
    parser.add_argument("--tier", choices=list(VALID_TIERS), default="tier2")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument(
        "--max-frames-map",
        type=str,
        default=None,
        help='JSON object mapping sequence IDs to frame limits, e.g. \'{"01": 350}\'',
    )
    parser.add_argument(
        "--preset",
        choices=list(MAPPING_PRESET_NAMES),
        default="improved_stereo",
    )
    parser.add_argument(
        "--config-overrides",
        type=str,
        default=None,
        help="JSON object overriding MappingConfig fields",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / ".datasets" / "kitti",
    )
    parser.add_argument(
        "--poses-root",
        type=Path,
        default=REPO_ROOT / ".datasets" / "kitti" / "poses",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=REPO_ROOT / "results" / "loop_ledger.jsonl",
    )
    parser.add_argument("--hypothesis", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--openblas-threads", type=int, default=1)
    parser.add_argument("--pipeline-extraction", action="store_true")
    parser.add_argument("--pipeline-buffer", type=int, default=1)
    parser.add_argument(
        "--coalesce-prefetch", action="store_true",
        help="Limit scheduled keyframe lookahead to one pending full extraction per last keyframe",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable per-stage StageProfile instrumentation for unprofiled benchmark runs",
    )
    parser.add_argument(
        "--matching-backend",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Optional descriptor matching backend (performance setting only; does not alter MappingConfig)",
    )
    parser.add_argument(
        "--frontend-backend", choices=["cpu", "cuda"], default="cpu",
        help="Frontend for full extractions, including those requested by lazy tracking",
    )
    parser.add_argument("--mark-baseline", action="store_true")

    # Explicit CLI overrides (take highest precedence over --preset and --config-overrides)
    parser.add_argument("--features", type=int, default=None)
    parser.add_argument("--loop-mode", choices=["off", "live", "offline"], default=None)
    parser.add_argument("--disable-bundle", action="store_true", default=None)
    parser.add_argument(
        "--stereo-depth-policy",
        choices=["supported", "verified_fallback", "verified_all"],
        default=None,
    )
    parser.add_argument(
        "--stereo-map-depth-policy",
        choices=["inherit", "verified"],
        default=None,
    )
    parser.add_argument(
        "--bundle-solver-accuracy",
        choices=["default", "precise"],
        default=None,
    )
    return parser


def resolve_loop_cli_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, MappingConfig, dict[str, Any]]:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be between 0 and 2147483647")
    if args.opencv_threads < 1:
        parser.error("--opencv-threads must be positive")
    if args.openblas_threads < 1:
        parser.error("--openblas-threads must be positive")
    if args.pipeline_buffer < 1:
        parser.error("--pipeline-buffer must be positive")
    if args.coalesce_prefetch and not args.pipeline_extraction:
        parser.error("--coalesce-prefetch requires --pipeline-extraction")
    if args.tier == "custom" and not args.sequences:
        parser.error("--tier custom requires --sequences")

    if args.sequences is not None:
        for seq in args.sequences:
            if len(seq) != 2 or not seq.isdigit():
                parser.error(f"Invalid KITTI sequence ID: {seq!r}")
        if len(set(args.sequences)) != len(args.sequences):
            parser.error("--sequences must not contain duplicates")

    try:
        user_max_frames = parse_max_frames_map(args.max_frames_map)
        json_overrides = parse_json_object_arg(args.config_overrides, "--config-overrides")
        cli_overrides = {
            "features": args.features,
            "loop_mode": args.loop_mode,
            "bundle_enabled": (False if args.disable_bundle else None),
            "stereo_depth_policy": args.stereo_depth_policy,
            "stereo_map_depth_policy": args.stereo_map_depth_policy,
            "bundle_solver_accuracy": args.bundle_solver_accuracy,
        }
        resolved_cfg, effective_overrides = resolve_preset_config(
            args.preset,
            stereo=True,
            config_overrides=json_overrides,
            cli_overrides=cli_overrides,
        )
    except ValueError as exc:
        parser.error(str(exc))

    tier_spec = TIER_DEFAULTS[args.tier]
    if args.sequences is not None:
        sequences = list(args.sequences)
    elif args.tier == "tier1":
        seq04_calib = Path(args.data_root) / "sequences" / "04" / "calib.txt"
        sequences = ["04"] if seq04_calib.is_file() else []
    else:
        sequences = list(tier_spec["sequences"])

    resolved_max_frames: dict[str, int | None] = dict(tier_spec["max_frames_map"])
    resolved_max_frames.update(user_max_frames)
    args.resolved_sequences = sequences
    args.resolved_max_frames_map = {
        seq: resolved_max_frames.get(seq) for seq in sequences
    }
    return args, resolved_cfg, effective_overrides


def execute_loop_harness(argv: list[str] | None = None) -> dict[str, Any]:
    args, resolved_cfg, effective_overrides = resolve_loop_cli_args(argv)

    timestamp_now = datetime.now(timezone.utc)
    git_commit = _current_git_commit(REPO_ROOT)
    default_run_id = f"{timestamp_now.strftime('%Y%m%dT%H%M%SZ')}_{args.tier}_{args.preset}_{git_commit[:8]}"
    output_dir = (
        Path(args.output_dir).resolve(strict=False)
        if args.output_dir is not None
        else (REPO_ROOT / "results" / "loop_runs" / default_run_id).resolve(strict=False)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = output_dir.name or default_run_id

    pytest_info = None
    if TIER_DEFAULTS[args.tier]["run_pytest"]:
        pytest_info = run_tier1_pytest_suite(output_dir, REPO_ROOT)

    # Phase 1: Estimation Subprocesses with sanitized argv/env (parent does not open poses_root yet)
    run_payloads: dict[str, dict[str, Any]] = {}
    for seq in args.resolved_sequences:
        seq_max_frames = args.resolved_max_frames_map.get(seq)
        run_payloads[seq] = run_airgapped_estimator_subprocess(
            sequence=seq,
            data_root=args.data_root,
            poses_root=args.poses_root,
            seq_out_dir=output_dir / seq,
            preset=args.preset,
            effective_overrides=effective_overrides,
            seed=args.seed,
            opencv_threads=args.opencv_threads,
            max_frames=seq_max_frames,
            pipeline_extraction=bool(args.pipeline_extraction),
            pipeline_buffer=int(args.pipeline_buffer),
            openblas_threads=int(args.openblas_threads),
            matching_backend=str(args.matching_backend),
            profile=not bool(args.no_profile),
            frontend_backend=args.frontend_backend,
            coalesce_prefetch=args.coalesce_prefetch,
        )

    # Phase 2: Post-Exit Evaluation (parent loads poses_root only AFTER estimator subprocesses exit)
    sequences_summary: dict[str, dict[str, Any]] = {}
    for seq in args.resolved_sequences:
        seq_max_frames = args.resolved_max_frames_map.get(seq)
        seq_eval = evaluate_sequence_posthoc(
            sequence=seq,
            poses_root=args.poses_root,
            seq_out_dir=output_dir / seq,
            run_payload=run_payloads[seq],
            max_frames=seq_max_frames,
        )
        # Keep exact PROJECT.md per-sequence schema keys cleanly ordered, plus online latency/delta metrics
        seq_entry: dict[str, Any] = {
            "frames": seq_eval["frames"],
            "segments": seq_eval["segments"],
            "ate_rmse_m": seq_eval["ate_rmse_m"],
            "raw_ate_rmse_m": seq_eval["raw_ate_rmse_m"],
            "translation_percent": seq_eval["translation_percent"],
            "rotation_deg_per_m": seq_eval["rotation_deg_per_m"],
            "rotation_deg_per_100m": seq_eval["rotation_deg_per_100m"],
            "lost_frames": seq_eval["lost_frames"],
            "verified_loops": seq_eval["verified_loops"],
            "applied_loops": seq_eval["applied_loops"],
            "elapsed_s": seq_eval["elapsed_s"],
            "online_elapsed_s": seq_eval["online_elapsed_s"],
            "shutdown_drain_s": seq_eval["shutdown_drain_s"],
            "processing_fps": seq_eval["processing_fps"],
            "online_fps": seq_eval["online_fps"],
        }
        for opt_key in (
            "matching_backend",
            "frame_latency_ms",
            "loop_events",
            "online_first_metrics",
            "online_first_vs_final_delta",
            "online_end_vs_final_delta",
        ):
            if opt_key in seq_eval:
                seq_entry[opt_key] = seq_eval[opt_key]
        sequences_summary[seq] = seq_entry
        run_payloads[seq]["peak_rss_mb"] = seq_eval["peak_rss_mb"]

    aggregate_with_rss_rows = {
        seq: {**srow, "peak_rss_mb": run_payloads[seq]["peak_rss_mb"]}
        for seq, srow in sequences_summary.items()
    }
    aggregate = compute_loop_aggregate(aggregate_with_rss_rows)

    baselines = load_reference_baselines(REPO_ROOT)
    baseline_deltas, baseline_details = compute_paired_baseline_comparison(
        sequences_summary, baselines
    )

    verdict, verdict_reason = compute_verdict(
        aggregate=aggregate,
        sequences_summary=sequences_summary,
        mark_baseline=bool(args.mark_baseline),
        ledger_path=args.ledger_path,
        repo_root=REPO_ROOT,
    )
    if verdict not in VALID_VERDICTS:
        raise AssertionError(f"Invalid verdict generated: {verdict!r}")

    summary: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": timestamp_now.isoformat(),
        "git_commit": git_commit,
        "tier": args.tier,
        "preset": args.preset,
        "frontend_backend_requested": args.frontend_backend,
        "matching_backend_requested": str(args.matching_backend),
        "hypothesis": args.hypothesis,
        "ground_truth_used_for_estimation": False,
        "evaluation_isolation": {
            "mode": "subprocess_argv_env_sanitized_posthoc_eval",
            "argv_env_sanitized": True,
            "posthoc_gt_evaluation": True,
            "filesystem_sandbox_enforced": False,
        },
        "seed": int(args.seed),
        "config_overrides": effective_overrides,
        "configuration": asdict(resolved_cfg),
        "aggregate": aggregate,
        "baseline_deltas": baseline_deltas,
        "sequences": sequences_summary,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }
    if pytest_info is not None:
        summary["tier1_pytest"] = pytest_info

    _atomic_write_json(output_dir / "summary.json", summary)
    comparison_md = format_comparison_markdown(summary, baseline_details, run_payloads)
    (output_dir / "comparison.md").write_text(comparison_md, encoding="utf-8")

    if args.ledger_path is not None:
        append_ledger_record(args.ledger_path, summary)

    return summary


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--_estimator-worker" in raw_argv:
        return _estimator_worker_main(raw_argv)
    try:
        summary = execute_loop_harness(raw_argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)
    except Exception as exc:
        print(f"[loop_harness] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
