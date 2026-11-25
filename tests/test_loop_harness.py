"""Unit tests for the airgapped loop-engineering harness and M1 contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import CameraCalib
from eval_kitti import compute_aggregate_metrics
from loop_harness import (
    AirgapViolationError,
    append_ledger_record,
    build_airgapped_estimator_invocation,
    compute_loop_aggregate,
    compute_paired_baseline_comparison,
    format_comparison_markdown,
    load_reference_baselines,
    resolve_preset_config,
)
from shared_slam import MappingConfig


def test_mapping_config_presets_and_validation():
    verified = MappingConfig.from_preset("verified_baseline", stereo=True)
    assert verified.stereo_depth_policy == "verified_fallback"
    assert verified.stereo_map_depth_policy == "verified"
    assert verified.stereo_pose_arbitration is True
    assert verified.bundle_solver_accuracy == "precise"
    assert verified.bundle_landmark_policy == "bounded_fixed"
    assert verified.stereo_bundle_gauge_mode == "veto"

    legacy = MappingConfig.from_preset("legacy_default", stereo=True)
    assert legacy == MappingConfig()

    overridden = MappingConfig.from_preset("verified_baseline", stereo=True, features=800)
    assert overridden.features == 800

    with pytest.raises(ValueError):
        MappingConfig.from_preset("unknown_preset")
    with pytest.raises(ValueError):
        MappingConfig.from_preset("verified_baseline", nonexistent_flag=123)


def test_covisible_bundle_window_override_preserves_ordinary_window():
    control, _ = resolve_preset_config()
    candidate, overrides = resolve_preset_config(config_overrides={"covisible_bundle_max_keyframes": 5})
    assert control.covisible_bundle_max_keyframes == 8
    assert candidate.covisible_bundle_max_keyframes == 5
    assert candidate.bundle_window == control.bundle_window
    assert candidate.use_covisible_ba
    assert overrides == {"covisible_bundle_max_keyframes": 5}
    for invalid in (2, True, 5.0):
        with pytest.raises(ValueError, match="covisible_bundle_max_keyframes"):
            resolve_preset_config(config_overrides={"covisible_bundle_max_keyframes": invalid})


def test_airgap_invocation_rejects_overlap_and_sanitizes_env(tmp_path, monkeypatch):
    seq_dir = tmp_path / "kitti" / "sequences" / "04"
    for cam in ("image_0", "image_1"):
        (seq_dir / cam).mkdir(parents=True)
        for idx in range(2):
            (seq_dir / cam / f"{idx:06d}.png").write_bytes(b"png")
    (seq_dir / "calib.txt").write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    poses_dir = tmp_path / "kitti" / "poses"
    poses_dir.mkdir(parents=True)

    monkeypatch.setenv("KITTI_POSES_ROOT", str(poses_dir))
    monkeypatch.setenv("VISUAL_SLAM_SIFT_FIRST_OCTAVE", "0")

    cmd, env, resolved_seq = build_airgapped_estimator_invocation(
        sequence="04",
        data_root=tmp_path / "kitti",
        poses_root=poses_dir,
        seq_out_dir=tmp_path / "out" / "04",
        preset="verified_baseline",
        effective_overrides={},
        seed=0,
        opencv_threads=1,
        max_frames=2,
        frontend_backend="cuda",
        stereo_verification_backend="cuda",
        pipeline_extraction=True,
        coalesce_prefetch=True,
    )
    assert resolved_seq == seq_dir.resolve()
    assert cmd[cmd.index("--frontend-backend") + 1] == "cuda"
    assert cmd[cmd.index("--stereo-verification-backend") + 1] == "cuda"
    assert "--coalesce-prefetch" in cmd
    assert "--poses-root" not in cmd
    assert str(poses_dir) not in " ".join(cmd)
    assert "KITTI_POSES_ROOT" not in env
    assert env["VISUAL_SLAM_SIFT_FIRST_OCTAVE"] == "0"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"

    with pytest.raises(AirgapViolationError):
        build_airgapped_estimator_invocation(
            sequence="04",
            data_root=tmp_path / "kitti",
            poses_root=seq_dir,
            seq_out_dir=tmp_path / "out" / "04",
            preset="verified_baseline",
            effective_overrides={},
            seed=0,
            opencv_threads=1,
            max_frames=2,
        )


def test_loop_aggregate_and_baseline_comparison():
    sequences_summary = {
        "04": {
            "frames": 271,
            "segments": 43,
            "ate_rmse_m": 0.50,
            "raw_ate_rmse_m": 1.10,
            "translation_percent": 0.60,
            "rotation_deg_per_m": 0.0030,
            "rotation_deg_per_100m": 0.30,
            "lost_frames": 0,
            "verified_loops": 0,
            "applied_loops": 0,
            "elapsed_s": 100.0,
            "processing_fps": 2.71,
            "peak_rss_mb": 300.0,
        },
        "07": {
            "frames": 1101,
            "segments": 157,
            "ate_rmse_m": 2.00,
            "raw_ate_rmse_m": 5.00,
            "translation_percent": 1.20,
            "rotation_deg_per_m": 0.0060,
            "rotation_deg_per_100m": 0.60,
            "lost_frames": 0,
            "verified_loops": 2,
            "applied_loops": 1,
            "elapsed_s": 200.0,
            "processing_fps": 5.505,
            "peak_rss_mb": 420.0,
        },
    }
    agg = compute_loop_aggregate(sequences_summary)
    expected_t = (0.60 * 43 + 1.20 * 157) / 200.0
    assert agg["total_segments"] == 200
    assert agg["segment_weighted_t_rel_pct"] == pytest.approx(expected_t)
    assert agg["segment_weighted_r_rel_deg_per_100m"] == pytest.approx(
        100.0 * (0.0030 * 43 + 0.0060 * 157) / 200.0
    )
    assert agg["peak_rss_mb"] == pytest.approx(420.0)

    baselines = load_reference_baselines(REPO_ROOT)
    deltas, details = compute_paired_baseline_comparison(sequences_summary, baselines)
    for key in (
        "vs_legacy_vo_t_rel_pct",
        "vs_legacy_vo_pgo_t_rel_pct",
        "vs_verified_shared_slam_t_rel_pct",
    ):
        assert deltas[key] is not None and math.isfinite(deltas[key])
    assert details["verified_shared_slam"]["paired_sequences"] == 2


def test_comparison_markdown_handles_dict_stage_timings(tmp_path):
    summary = {
        "run_id": "test-run",
        "timestamp_utc": "2026-11-19T00:00:00+00:00",
        "git_commit": "abc123",
        "tier": "custom",
        "preset": "verified_baseline",
        "hypothesis": "dict stage timings",
        "ground_truth_used_for_estimation": False,
        "verdict": "BASELINE",
        "verdict_reason": "marked",
        "aggregate": {
            "sequence_count": 1,
            "total_frames": 271,
            "total_segments": 43,
            "segment_weighted_t_rel_pct": 0.60,
            "segment_weighted_r_rel_deg_per_100m": 0.30,
            "mean_ate_se3_m": 0.50,
            "total_lost_frames": 0,
            "total_elapsed_s": 100.0,
            "overall_fps": 2.71,
            "peak_rss_mb": 300.0,
        },
        "sequences": {
            "04": {
                "frames": 271,
                "segments": 43,
                "ate_rmse_m": 0.50,
                "raw_ate_rmse_m": 1.10,
                "translation_percent": 0.60,
                "rotation_deg_per_100m": 0.30,
                "lost_frames": 0,
                "verified_loops": 0,
                "applied_loops": 0,
                "elapsed_s": 100.0,
                "processing_fps": 2.71,
            }
        },
    }
    run_payloads = {
        "04": {
            "peak_rss_mb": 300.0,
            "stage_timings": {
                "stages": {
                    "_extract": {"inclusive_seconds": 12.5},
                    "_track": {"total_s": 5.25},
                }
            },
        }
    }
    md = format_comparison_markdown(summary, {}, run_payloads)
    assert "`_extract` | 12.500 |" in md
    assert "`_track` | 5.250 |" in md

    ledger = tmp_path / "ledger.jsonl"
    append_ledger_record(ledger, summary)
    loaded = json.loads(ledger.read_text(encoding="utf-8").strip())
    assert loaded["run_id"] == "test-run"


def test_camera_calib_import_and_eval_kitti_aggregate():
    assert hasattr(CameraCalib, "CameraCalib")
    agg = compute_aggregate_metrics(
        {
            "04": {
                "frames": 271,
                "segment_count": 43,
                "translation_percent": 1.0,
                "rotation_deg_per_m": 0.005,
            },
            "07": {
                "frames": 1101,
                "segment_count": 157,
                "translation_percent": 2.0,
                "rotation_deg_per_m": 0.010,
            },
        }
    )
    assert agg["total_segments"] == 200
    assert agg["segment_weighted_translation_percent"] == pytest.approx(
        (1.0 * 43 + 2.0 * 157) / 200.0
    )
    assert agg["segment_weighted_rotation_deg_per_100m"] == pytest.approx(
        100.0 * (0.005 * 43 + 0.010 * 157) / 200.0
    )


def test_loop_harness_importable_without_unix_resource_module(monkeypatch):
    import importlib.util

    monkeypatch.setitem(sys.modules, "resource", None)
    spec = importlib.util.spec_from_file_location(
        "loop_harness_windows_sim", SRC_DIR / "loop_harness.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    assert module.resource is None
    rss_mb = module._self_peak_rss_mb()
    assert isinstance(rss_mb, float) and math.isfinite(rss_mb) and rss_mb >= 0.0


def test_subprocess_isolation_metadata_matches_enforced_boundary(tmp_path, monkeypatch):
    import loop_harness

    seq_dir = tmp_path / "kitti" / "sequences" / "04"
    for cam in ("image_0", "image_1"):
        (seq_dir / cam).mkdir(parents=True)
        for idx in range(2):
            (seq_dir / cam / f"{idx:06d}.png").write_bytes(b"png")
    (seq_dir / "calib.txt").write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    poses_dir = tmp_path / "kitti" / "poses"
    poses_dir.mkdir(parents=True)
    seq_out_dir = tmp_path / "out" / "04"

    class _DummyProc:
        pid = 12345

        def wait(self):
            return 0

    def _fake_popen(command, env, cwd, stdout, stderr):
        (seq_out_dir / "poses.txt").write_text(
            "1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 0 0 1 0 0 0 0 1 1\n", encoding="utf-8"
        )
        (seq_out_dir / "run.json").write_text(
            json.dumps({"sequence": "04", "frames": 2, "peak_rss_mb": 42.5}),
            encoding="utf-8",
        )
        return _DummyProc()

    monkeypatch.setattr(loop_harness.subprocess, "Popen", _fake_popen)
    monkeypatch.delattr(loop_harness.os, "wait4", raising=False)

    payload = loop_harness.run_airgapped_estimator_subprocess(
        sequence="04",
        data_root=tmp_path / "kitti",
        poses_root=poses_dir,
        seq_out_dir=seq_out_dir,
        preset="improved_stereo",
        effective_overrides={},
        seed=0,
        opencv_threads=1,
        max_frames=2,
    )
    assert "airgap_verified" not in payload
    assert payload["isolation_mode"] == "subprocess_argv_env_sanitized_posthoc_eval"
    assert payload["argv_env_sanitized"] is True
    assert payload["posthoc_gt_evaluation"] is True
    assert payload["filesystem_sandbox_enforced"] is False
    assert payload["ground_truth_used_for_estimation"] is False
    assert payload["peak_rss_mb"] == pytest.approx(42.5)


def test_matching_backend_invocation_preserves_estimator_config(tmp_path, monkeypatch):
    import loop_harness

    seq_dir = tmp_path / "kitti" / "sequences" / "04"
    for cam in ("image_0", "image_1"):
        (seq_dir / cam).mkdir(parents=True)
        for idx in range(2):
            (seq_dir / cam / f"{idx:06d}.png").write_bytes(b"png")
    (seq_dir / "calib.txt").write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    poses_dir = tmp_path / "kitti" / "poses"
    poses_dir.mkdir(parents=True)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    args_cpu, cfg_cpu, ov_cpu = loop_harness.resolve_loop_cli_args(
        ["--tier", "custom", "--sequences", "04", "--preset", "improved_stereo"]
    )
    args_cuda, cfg_cuda, ov_cuda = loop_harness.resolve_loop_cli_args(
        [
            "--tier",
            "custom",
            "--sequences",
            "04",
            "--preset",
            "improved_stereo",
            "--matching-backend",
            "cuda",
        ]
    )
    assert args_cpu.matching_backend == "cpu"
    assert args_cuda.matching_backend == "cuda"
    assert cfg_cpu == cfg_cuda
    assert ov_cpu == ov_cuda == {}

    cmd_cpu, env_cpu, _ = build_airgapped_estimator_invocation(
        sequence="04",
        data_root=tmp_path / "kitti",
        poses_root=poses_dir,
        seq_out_dir=tmp_path / "out_cpu" / "04",
        preset="improved_stereo",
        effective_overrides={},
        seed=0,
        opencv_threads=2,
        max_frames=2,
        matching_backend="cpu",
    )
    assert "--matching-backend" not in cmd_cpu
    assert env_cpu["CUDA_VISIBLE_DEVICES"] == "0"

    cmd_cuda, _, _ = build_airgapped_estimator_invocation(
        sequence="04",
        data_root=tmp_path / "kitti",
        poses_root=poses_dir,
        seq_out_dir=tmp_path / "out_cuda" / "04",
        preset="improved_stereo",
        effective_overrides={},
        seed=0,
        opencv_threads=2,
        max_frames=2,
        matching_backend="cuda",
    )
    idx = cmd_cuda.index("--matching-backend")
    assert cmd_cuda[idx + 1] == "cuda"

    with pytest.raises(ValueError):
        build_airgapped_estimator_invocation(
            sequence="04",
            data_root=tmp_path / "kitti",
            poses_root=poses_dir,
            seq_out_dir=tmp_path / "out_bad" / "04",
            preset="improved_stereo",
            effective_overrides={},
            seed=0,
            opencv_threads=2,
            max_frames=2,
            matching_backend="vulkan",
        )


