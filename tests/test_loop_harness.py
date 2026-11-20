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
    )
    assert resolved_seq == seq_dir.resolve()
    assert "--poses-root" not in cmd
    assert str(poses_dir) not in " ".join(cmd)
    assert "KITTI_POSES_ROOT" not in env
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
