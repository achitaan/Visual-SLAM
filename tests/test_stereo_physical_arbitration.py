"""Synthetic invariants for opt-in physical stereo correspondence pooling."""

import hashlib
import importlib.util
import json
from pathlib import Path
import cv2 as cv
import numpy as np
import pytest

import shared_slam as module
from shared_slam import MappingConfig, SharedSlam, StereoCamera
from stereo_pose_arbitration import SupportedStereoFrame


K = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
BASELINE = 0.54
SIZE = (640, 480)


def _camera(*, physical=False, min_inliers=15):
    q = np.array([[1., 0., 0., -K[0, 2]],
                  [0., 1., 0., -K[1, 2]],
                  [0., 0., 0., K[0, 0]],
                  [0., 0., 1. / BASELINE, 0.]])
    return SharedSlam(
        K, StereoCamera(None, q, BASELINE),
        config=MappingConfig(
            loop_mode="off", bundle_enabled=False, min_inliers=min_inliers,
            stereo_pose_arbitration=True,
            stereo_physical_match_pool=physical,
        ),
    )


def _base_frame_arrays(count=48, *, narrow=False):
    if narrow:
        pixels = np.array([(35. + 14. * (i % 6), 35. + 12. * (i // 6))
                           for i in range(count)], dtype=np.float32)
    else:
        pixels = np.array([(55. + 70. * col, 34. + 70. * row)
                           for row in range(6) for col in range(8)], dtype=np.float32)[:count]
    z = np.full(count, 12., dtype=np.float64)
    points = np.column_stack(((pixels[:, 0] - K[0, 2]) * z / K[0, 0],
                              (pixels[:, 1] - K[1, 2]) * z / K[1, 1], z))
    right = pixels[:, 0].astype(np.float64) - K[0, 0] * BASELINE / z
    ids = np.full(count, -1, dtype=np.int64)
    ids[:] = np.arange(1000, 1000 + count, dtype=np.int64)
    descriptors = np.eye(count, 128, dtype=np.float32)
    return pixels, points, right, ids, descriptors


def _frame_from_arrays(arrays, frame_id, calibration="synthetic-cal-v1"):
    pixels, points, right, ids, descriptors = arrays
    return SupportedStereoFrame(
        np.array(pixels, dtype=np.float32, copy=True),
        np.array(descriptors, dtype=np.float32, copy=True),
        np.array(points, dtype=np.float64, copy=True),
        np.array(right, dtype=np.float64, copy=True),
        np.array(ids, dtype=np.int64, copy=True),
        frame_id, SIZE, calibration,
    )


def _mutated_frame(frame, *, pixels=None, points=None, right=None, ids=None):
    """Replace selected immutable extraction arrays with owned test copies."""
    return _frame_from_arrays((
        frame.pixels.copy() if pixels is None else pixels,
        frame.points.copy() if points is None else points,
        frame.right_u.copy() if right is None else right,
        frame.landmark_ids.copy() if ids is None else ids,
        frame.descriptors.copy(),
    ), frame.frame, frame.calibration_identity)


def _insert_aliases(frame, group_rows, *, alias_id=None):
    """Insert extra descriptor rows beside physical points, copying geometry."""
    group_rows = set(group_rows)
    pixels, points, right, ids, desc = [], [], [], [], []
    first, aliases = {}, {}
    for row in range(len(frame.pixels)):
        first[row] = len(pixels)
        pixels.append(frame.pixels[row].copy())
        points.append(frame.points[row].copy())
        right.append(float(frame.right_u[row]))
        ids.append(int(frame.landmark_ids[row]))
        desc.append(frame.descriptors[row].copy())
        if row in group_rows:
            aliases[row] = len(pixels)
            pixels.append(frame.pixels[row].copy())
            points.append(frame.points[row].copy())
            right.append(float(frame.right_u[row]))
            ids.append(int(frame.landmark_ids[row] if alias_id is None else alias_id))
            alias_descriptor = np.zeros(frame.descriptors.shape[1], dtype=np.float32)
            alias_descriptor[(len(pixels) - 1) % len(alias_descriptor)] = 1.
            desc.append(alias_descriptor)
    expanded = _frame_from_arrays(
        (np.asarray(pixels, np.float32), np.asarray(points, np.float64),
         np.asarray(right, np.float64), np.asarray(ids, np.int64),
         np.asarray(desc, np.float32)), frame.frame, frame.calibration_identity)
    return expanded, first, aliases


def _append_aliases(frame, group_order):
    """Append aliases in an order unrelated to physical extraction order."""
    group_order = tuple(group_order)
    arrays = (
        np.concatenate((frame.pixels, frame.pixels[list(group_order)]), axis=0),
        np.concatenate((frame.points, frame.points[list(group_order)]), axis=0),
        np.concatenate((frame.right_u, frame.right_u[list(group_order)]), axis=0),
        np.concatenate((frame.landmark_ids, frame.landmark_ids[list(group_order)]), axis=0),
        np.concatenate((frame.descriptors, frame.descriptors[list(group_order)]), axis=0),
    )
    expanded = _frame_from_arrays(arrays, frame.frame, frame.calibration_identity)
    first = {row: row for row in range(len(frame.pixels))}
    aliases = {group: len(frame.pixels) + index
               for index, group in enumerate(group_order)}
    return expanded, first, aliases


def _edge_pixel_sets(partition, source, target):
    def collect(edges):
        src, dst = set(), set()
        for edge in edges:
            src_rows = edge["source_alias_rows"]
            dst_rows = edge["target_alias_rows"]
            src.add(tuple(np.asarray(source.pixels[src_rows[0]], np.float32).tolist()))
            dst.add(tuple(np.asarray(target.pixels[dst_rows[0]], np.float32).tolist()))
        return src, dst
    return collect(partition["fit_edges"]), collect(partition["held_edges"])


def _ordered_edge_pixels(edges, source, target):
    return [
        (tuple(source.pixels[edge["source_alias_rows"][0]].tolist()),
         tuple(target.pixels[edge["target_alias_rows"][0]].tolist()))
        for edge in edges
    ]


def _raw_matches(first, second, *, alias_rows=()):
    pairs = [[first[i], second[i]] for i in range(48)]
    pairs.extend([[first[i], second[i]] for i in alias_rows])
    return np.asarray(pairs, dtype=np.int64)


def test_mapping_flag_is_opt_in_requires_arbitration_and_separates_reuse(tmp_path, monkeypatch):
    assert MappingConfig().stereo_physical_match_pool is False
    with pytest.raises(ValueError, match="requires stereo_pose_arbitration"):
        MappingConfig(stereo_physical_match_pool=True)
    active = MappingConfig(stereo_pose_arbitration=True, stereo_physical_match_pool=True)
    assert active.stereo_physical_match_pool is True

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo / "scripts"))
    runner_path = repo / "scripts" / "run_development_tests.py"
    spec = importlib.util.spec_from_file_location("physical_pool_dev_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    off_config = runner.current_mapping_configuration("bundle", "supported")
    on_config = runner.current_mapping_configuration(
        "bundle", "supported", stereo_pose_arbitration=True,
        stereo_physical_match_pool=True)
    assert off_config["stereo_physical_match_pool"] is False
    assert on_config["stereo_physical_match_pool"] is True
    with pytest.raises(ValueError, match="not available for baseline"):
        runner.current_mapping_configuration(
            "baseline", "supported", stereo_physical_match_pool=True)

    # Export reuse must compare this opt-in identity and the resolved config.
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "src" / "fixture.py").write_text("value = 1\n", encoding="utf-8")
    evaluator = b"synthetic evaluator\n"
    (repo / "scripts" / "evaluate_shared_slam.py").write_bytes(evaluator)
    monkeypatch.setattr(runner, "REPO", repo)
    configuration = {"bundle_enabled": True, "loop_mode": "off",
                     "stereo_pose_arbitration": True,
                     "stereo_physical_match_pool": True}
    monkeypatch.setattr(runner, "current_mapping_configuration",
                        lambda *args, **kwargs: configuration)
    out = tmp_path / "export"
    (out / "source").mkdir(parents=True)
    source_hashes = {"fixture.py": hashlib.sha256(
        (repo / "src" / "fixture.py").read_bytes()).hexdigest()}
    (out / "source" / "fixture.py").write_bytes((repo / "src" / "fixture.py").read_bytes())
    (out / "evaluator.py").write_bytes(evaluator)
    identity = {
        "revision": "r", "runtime_identity": {"version": 1},
        "sequence": "01", "frames": 1, "coverage": "partial", "variant": "bundle",
        "cached": False, "input": "input", "reference": "ref",
        "stereo_depth_policy": "supported", "stereo_pose_arbitration": True,
        "stereo_raw_reference_retry": False, "stereo_owned_image_bundle": False,
        "stereo_source_history_bundle": False, "stereo_retained_source_observations": False,
        "stereo_physical_match_pool": True, "bundle_solver_accuracy": "default",
    }
    (out / "poses.txt").write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="ascii")
    (out / "run.json").write_text(json.dumps({
        "configuration": configuration, "tracking": [{"frame": 0, "state": "tracking"}],
        "sparse_points": 1,
    }), encoding="utf-8")
    (out / "preview.json").write_text(json.dumps({
        "trajectory": [[0., 0., 0.]], "sparse": [[0., 0., 1.]],
    }), encoding="utf-8")
    (out / "sparse.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
        "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n0 0 1 10 20 30\n", encoding="ascii")
    report_path = out / "evaluation.json"
    report_path.write_text(json.dumps({
        "development_identity": identity, "status": "completed", "frames": 1,
        "sequence": "01", "coverage": "partial", "stereo": True,
        "ground_truth_used_for_estimation": False, "lost_frames": 0,
        "states": {"tracking": 1}, "landmarks": 1,
        "configuration": configuration, "source_sha256": source_hashes,
        "evaluator_sha256": hashlib.sha256(evaluator).hexdigest(),
    }), encoding="utf-8")
    assert runner.reusable_export(report_path, identity)
    assert not runner.reusable_export(
        report_path, {**identity, "stereo_physical_match_pool": False})
    changed_config = {**configuration, "stereo_physical_match_pool": False}
    (out / "run.json").write_text(json.dumps({
        "configuration": changed_config, "tracking": [{"frame": 0, "state": "tracking"}],
        "sparse_points": 1,
    }), encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["configuration"] = changed_config
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert not runner.reusable_export(report_path, identity)


def test_physical_edges_collapse_alias_matches_and_group_ordinal_split_is_stable():
    slam = _camera(physical=True)
    try:
        raw_source = _frame_from_arrays(_base_frame_arrays(), 0)
        raw_target = _frame_from_arrays(_base_frame_arrays(), 1)
        source, source_first, source_alias = _insert_aliases(raw_source, {1, 5, 9})
        target, target_first, target_alias = _insert_aliases(raw_target, {1, 5, 9})
        raw = _raw_matches(source_first, target_first, alias_rows=(1, 5, 9))
        first = slam._partition_physical_stereo_matches(source, target, raw)
        assert first["report"]["physical_edges_accepted"] == 48
        assert first["report"]["raw_descriptor_pair_count"] == 51
        assert first["report"]["dropped_duplicate_descriptor_pairs"] == 3
        assert first["report"]["partition_rule"] == "source_physical_group_ordinal_modulo_2"
        assert len(first["fit_pairs"]) == len(first["held_pairs"]) == 24

        # Change which detector orientation supplied the match and reverse raw
        # row order. Physical edge and held/fit pixel sets must stay the same.
        alternate = []
        for group in range(48):
            si = source_alias.get(group, source_first[group]) if group in (1, 5, 9) else source_first[group]
            ti = target_alias.get(group, target_first[group]) if group in (1, 5, 9) else target_first[group]
            alternate.append([si, ti])
        second = slam._partition_physical_stereo_matches(
            source, target, np.asarray(alternate[::-1], dtype=np.int64))
        assert second["report"]["physical_edges_accepted"] == 48
        assert _edge_pixel_sets(first, source, target) == _edge_pixel_sets(second, source, target)

        # Appending aliases in opposing orders makes descriptor-row sorting
        # disagree with extraction/group order. Output rows must still align.
        base_source, base_target = (_frame_from_arrays(_base_frame_arrays(), 0),
                                    _frame_from_arrays(_base_frame_arrays(), 1))
        interleaved_source, interleaved_source_first, interleaved_source_alias = (
            _append_aliases(base_source, [9, 5, 1]))
        interleaved_target, interleaved_target_first, interleaved_target_alias = (
            _append_aliases(base_target, [1, 5, 9]))
        base_pairs = np.asarray([[i, i] for i in range(48)], dtype=np.int64)
        alias_pairs = base_pairs.copy()
        for group in (1, 5, 9):
            alias_pairs[group] = (interleaved_source_alias[group],
                                  interleaved_target_alias[group])
        by_base_rows = slam._partition_physical_stereo_matches(
            interleaved_source, interleaved_target, base_pairs)
        by_alias_rows = slam._partition_physical_stereo_matches(
            interleaved_source, interleaved_target, alias_pairs[::-1])
        for subset in ("fit_edges", "held_edges"):
            assert _ordered_edge_pixels(by_base_rows[subset], interleaved_source,
                                        interleaved_target) == _ordered_edge_pixels(
                by_alias_rows[subset], interleaved_source, interleaved_target)

        fit, held = first["fit_edges"], first["held_edges"]
        assert {edge["source_group_id"] for edge in fit}.isdisjoint(
            edge["source_group_id"] for edge in held)
        assert {edge["target_group_id"] for edge in fit}.isdisjoint(
            edge["target_group_id"] for edge in held)
        assert all(edge["source_group_id"] % 2 == 0 for edge in fit)
        assert all(edge["source_group_id"] % 2 == 1 for edge in held)
        alias_edge = next(edge for edge in held if edge["source_group_id"] == 1)
        assert alias_edge["source_alias_rows"] == (source_first[1], source_alias[1])
        assert alias_edge["target_alias_rows"] == (target_first[1], target_alias[1])
    finally:
        slam.close()


def test_two_orientations_for_every_physical_point_keep_both_pools():
    slam = _camera(physical=True)
    try:
        source, source_first, source_alias = _insert_aliases(
            _frame_from_arrays(_base_frame_arrays(), 0), set(range(48)))
        target, target_first, target_alias = _insert_aliases(
            _frame_from_arrays(_base_frame_arrays(), 1), set(range(48)))
        raw = _raw_matches(source_first, target_first, alias_rows=range(48))
        result = slam._partition_physical_stereo_matches(source, target, raw)
        assert result["report"]["physical_edges_accepted"] == 48
        assert result["report"]["raw_descriptor_pair_count"] == 96
        assert len(result["fit_edges"]) == len(result["held_edges"]) == 24
        assert all(len(edge["source_alias_rows"]) == 2
                   and len(edge["target_alias_rows"]) == 2
                   for edge in result["fit_edges"] + result["held_edges"])
        fit_pixels, held_pixels = _edge_pixel_sets(result, source, target)
        assert fit_pixels[0].isdisjoint(held_pixels[0])
        assert fit_pixels[1].isdisjoint(held_pixels[1])
        assert len(source_alias) == len(target_alias) == 48
    finally:
        slam.close()


@pytest.mark.parametrize("bad_side,fault", [
    (side, fault) for side in ("source", "target")
    for fault in ("xyz_conflict", "right_conflict", "negative_depth",
                  "pixel_out_of_bounds", "right_out_of_bounds", "nonfinite_pixel")
])
def test_any_invalid_alias_member_rejects_physical_endpoint_group(bad_side, fault):
    source = _frame_from_arrays(_base_frame_arrays(4), 0)
    target = _frame_from_arrays(_base_frame_arrays(4), 1)
    source, _, source_alias = _insert_aliases(source, {0})
    target, _, target_alias = _insert_aliases(target, {0})
    source = _frame_from_arrays(
        (source.pixels.copy(), source.points.copy(), source.right_u.copy(),
         source.landmark_ids.copy(), source.descriptors.copy()), source.frame,
        source.calibration_identity)
    target = _frame_from_arrays(
        (target.pixels.copy(), target.points.copy(), target.right_u.copy(),
         target.landmark_ids.copy(), target.descriptors.copy()), target.frame,
        target.calibration_identity)
    frame = source if bad_side == "source" else target
    alias = (source_alias if bad_side == "source" else target_alias)[0]
    pixels, points = frame.pixels.copy(), frame.points.copy()
    right = frame.right_u.copy()
    if fault == "xyz_conflict":
        points[alias, 0] += 0.02
    elif fault == "right_conflict":
        right[alias] += 0.02
    elif fault == "negative_depth":
        points[alias, 2] = -1.
    elif fault == "pixel_out_of_bounds":
        pixels[[0, alias], 0] = SIZE[0]
    elif fault == "right_out_of_bounds":
        right[[0, alias]] = SIZE[0]
    elif fault == "nonfinite_pixel":
        pixels[[0, alias], 0] = np.nan
    changed = _mutated_frame(frame, pixels=pixels, points=points, right=right)
    if bad_side == "source":
        source = changed
    else:
        target = changed

    slam = _camera(physical=True)
    try:
        result = slam._partition_physical_stereo_matches(
            source, target, np.array([[0, 0], [source_alias[0], target_alias[0]]], dtype=np.int64))
        assert len(result["fit_pairs"]) + len(result["held_pairs"]) == 0
        assert result["report"]["physical_edges_accepted"] == 0
    finally:
        slam.close()


@pytest.mark.parametrize("bad_side", ["source", "target"])
def test_conflicting_landmark_claims_across_aliases_reject_edge(bad_side):
    source = _frame_from_arrays(_base_frame_arrays(4), 0)
    target = _frame_from_arrays(_base_frame_arrays(4), 1)
    source, _, source_alias = _insert_aliases(
        source, {0}, alias_id=101 if bad_side == "source" else 1000)
    target, _, target_alias = _insert_aliases(
        target, {0}, alias_id=201 if bad_side == "target" else 1000)
    frame = source if bad_side == "source" else target
    alias = source_alias[0] if bad_side == "source" else target_alias[0]
    ids = frame.landmark_ids.copy()
    ids[0] = 100 if bad_side == "source" else 200
    frame = _mutated_frame(frame, ids=ids)
    if bad_side == "source":
        source = frame
    else:
        target = frame
    result = SharedSlam._partition_physical_stereo_matches(
        source, target, np.array([[0, 0], [source_alias[0], target_alias[0]]], dtype=np.int64))
    assert result["report"]["dropped_conflicting_landmark_edges"] > 0
    assert result["report"]["physical_edges_accepted"] == 0
    assert not result["fit_edges"] and not result["held_edges"]


@pytest.mark.parametrize("direction", ["one_source_two_targets", "two_sources_one_target"])
def test_competing_alias_topology_is_rejected_before_invalid_competitor_filtering(direction):
    source = _frame_from_arrays(_base_frame_arrays(3), 0)
    target = _frame_from_arrays(_base_frame_arrays(3), 1)
    if direction == "one_source_two_targets":
        source, _, source_alias = _insert_aliases(source, {0})
        # The same source physical group points to two target groups. The
        # second target is invalid, but it still makes the relation ambiguous.
        points = target.points.copy()
        points[1, 2] = np.nan
        target = _mutated_frame(target, points=points)
        pairs = np.array([[0, 0], [source_alias[0], 1]], dtype=np.int64)
    else:
        # Two source groups claim one target group; the competing source is
        # invalid, but topology is determined before geometry filtering.
        points = source.points.copy()
        points[1, 2] = np.nan
        source = _mutated_frame(source, points=points)
        pairs = np.array([[0, 0], [1, 0]], dtype=np.int64)
    result = SharedSlam._partition_physical_stereo_matches(
        source, target, pairs)
    assert result["report"]["physical_edges_before_validation"] == 2
    assert result["report"]["dropped_competing_relations"] == 2
    assert result["report"]["physical_edges_accepted"] == 0


def test_unmatched_bad_alias_is_not_filled_from_matched_valid_orientation():
    source = _frame_from_arrays(_base_frame_arrays(3), 0)
    target = _frame_from_arrays(_base_frame_arrays(3), 1)
    source, _, source_alias = _insert_aliases(source, {0})
    points = source.points.copy()
    points[source_alias[0], 0] += 0.03
    source = _mutated_frame(source, points=points)
    result = SharedSlam._partition_physical_stereo_matches(
        source, target, np.array([[0, 0]], dtype=np.int64))
    assert result["report"]["dropped_invalid_source_groups"] == 1
    assert result["report"]["physical_edges_accepted"] == 0


def _install_prepare_fixture(slam, *, count=48, alias_group=1, narrow=False):
    arrays = _base_frame_arrays(count, narrow=narrow)
    source = _frame_from_arrays(arrays, 0)
    target = _frame_from_arrays(arrays, 1)
    source, source_first, source_alias = _insert_aliases(source, {alias_group})
    target, target_first, target_alias = _insert_aliases(target, {alias_group})
    # Held-group landmark identity is claimed by every alias with one stable ID.
    source_ids = source.landmark_ids.copy()
    source_ids[source_first[alias_group]] = 4242
    source_ids[source_alias[alias_group]] = 4242
    source = _mutated_frame(source, ids=source_ids)
    target_ids = target.landmark_ids.copy()
    target_ids[:] = -1
    target_ids[target_alias[alias_group]] = -1
    target = _mutated_frame(target, ids=target_ids)
    pairs = [[source_first[i], target_first[i]] for i in range(count)]
    pairs.append([source_alias[alias_group], target_alias[alias_group]])
    slam.previous_supported_stereo = source
    from loop_geometry import StereoLoopFrame
    slam.previous_stereo_geometry = (
        StereoLoopFrame(source.pixels, source.points, source.descriptors, source.image_size), 0)
    slam.map.record(np.eye(4), "tracking")
    slam.previous_tracks = [
        (9001, source.pixels[source_first[alias_group]].copy()),
        (9002, source.pixels[source_alias[alias_group]].copy()),
    ]
    calls = []

    def raw_match(first, second):
        calls.append((first, second))
        return np.asarray(pairs, dtype=np.int64).copy()

    slam._match = raw_match
    return source, target, source_first, source_alias, target_first, target_alias, calls


def test_prepare_uses_one_physical_edge_and_excludes_all_held_aliases_and_flow(monkeypatch):
    slam = _camera(physical=True)
    try:
        source, target, sf, sa, tf, ta, raw_calls = _install_prepare_fixture(slam)
        originals = [(frame.pixels.copy(), frame.points.copy(), frame.right_u.copy(),
                      frame.descriptors.copy(), frame.landmark_ids.copy())
                     for frame in (source, target)]
        fit_calls = []

        def estimate(source_frame, target_frame, matrix, **kwargs):
            rows = kwargs["matcher"](source_frame.descriptors, target_frame.descriptors)
            fit_calls.append(rows.copy())
            return {"measurement": np.eye(4), "reverse_checked": True,
                    "matches": len(rows), "inliers": len(rows),
                    "target_features": rows[:, 1].tolist()}

        monkeypatch.setattr(module, "estimate_stereo_reference", estimate)
        context, report = slam._prepare_stereo_arbitration(1, target, capture_rows=True)
        assert context is not None
        assert len(raw_calls) == 1 and len(fit_calls) == 1
        assert report["physical_match_pool_mode"] == "strict_full_alias_edges_v1"
        assert report["partition_rule"] == "source_physical_group_ordinal_modulo_2"
        assert report["physical_edges_accepted"] == 48
        assert report["fit_count"] == report["holdout_count"] == 24
        assert len(context["held_pairs"]) == 24
        assert 4242 in context["excluded_landmarks"]
        assert {9001, 9002}.issubset(context["excluded_landmarks"])
        assert {tf[1], ta[1]}.issubset(context["excluded_targets"])
        assert context["excluded_target_pixels"]
        assert 1 in set(context["evidence"].source_ids)
        assert not {tuple(source.pixels[i]) for i in context["fit"][:, 0]} & {
            tuple(source.pixels[i]) for i in context["held_pairs"][:, 0]}
        assert not {tuple(target.pixels[i]) for i in context["fit"][:, 1]} & {
            tuple(target.pixels[i]) for i in context["held_pairs"][:, 1]}
        for frame, before in zip((source, target), originals):
            for actual, expected in zip((frame.pixels, frame.points, frame.right_u,
                                         frame.descriptors, frame.landmark_ids), before):
                np.testing.assert_array_equal(actual, expected)
    finally:
        slam.close()


@pytest.mark.parametrize("count,narrow", [(28, False), (48, True)])
def test_physical_pool_keeps_existing_minimum_and_spatial_coverage_gates(
        monkeypatch, count, narrow):
    slam = _camera(physical=True)
    try:
        _source, target, *_rest = _install_prepare_fixture(
            slam, count=count, alias_group=min(1, count - 1), narrow=narrow)
        calls = []
        monkeypatch.setattr(module, "estimate_stereo_reference",
                            lambda *args, **kwargs: calls.append(args) or None)
        context, report = slam._prepare_stereo_arbitration(1, target)
        assert context is None
        assert report["reason"] == "insufficient_reserved_support"
        assert not calls
    finally:
        slam.close()


def test_disabled_mode_keeps_legacy_drop_duplicate_behavior(monkeypatch):
    slam = _camera(physical=False)
    try:
        raw_source = _frame_from_arrays(_base_frame_arrays(48), 0)
        raw_target = _frame_from_arrays(_base_frame_arrays(48), 1)
        source, sf, _ = _insert_aliases(raw_source, {1})
        target, tf, _ = _insert_aliases(raw_target, {1})
        slam.previous_supported_stereo = source
        from loop_geometry import StereoLoopFrame
        slam.previous_stereo_geometry = (
            StereoLoopFrame(source.pixels, source.points, source.descriptors, SIZE), 0)
        slam.map.record(np.eye(4), "tracking")
        raw_pairs = _raw_matches(sf, tf, alias_rows=(1,))
        calls = []
        slam._match = lambda *_args: raw_pairs.copy()
        # The new partitioner must not run on the default path.
        slam._partition_physical_stereo_matches = lambda *_args: (_ for _ in ()).throw(
            AssertionError("opt-in physical pool ran while disabled"))

        def estimate(source_frame, target_frame, matrix, **kwargs):
            rows = kwargs["matcher"](source_frame.descriptors, target_frame.descriptors)
            calls.append(rows.copy())
            return {"measurement": np.eye(4), "reverse_checked": True,
                    "matches": len(rows), "inliers": len(rows),
                    "target_features": rows[:, 1].tolist()}

        monkeypatch.setattr(module, "estimate_stereo_reference", estimate)
        context, report = slam._prepare_stereo_arbitration(1, target)
        assert context is not None
        assert report["dropped_duplicate_matches"] == 2
        assert "physical_match_pool_mode" not in report
        assert len(calls) == 1
    finally:
        slam.close()


def test_physical_pool_keeps_reverse_verification_gate(monkeypatch):
    slam = _camera(physical=True)
    try:
        _source, target, *_rest = _install_prepare_fixture(slam)
        def failed_reverse(source_frame, target_frame, matrix, **kwargs):
            rows = kwargs["matcher"](source_frame.descriptors, target_frame.descriptors)
            return {"measurement": np.eye(4), "reverse_checked": False,
                    "matches": len(rows), "inliers": len(rows)}
        monkeypatch.setattr(module, "estimate_stereo_reference", failed_reverse)
        context, report = slam._prepare_stereo_arbitration(1, target)
        assert context is None
        assert report["reason"] == "independent_training_failed"
    finally:
        slam.close()


def test_physical_pool_recovers_real_nonplanar_bidirectional_stereo_pose():
    """Exercise the real PnP, reverse-PnP, and bidirectional refinement path."""
    slam = _camera(physical=True)
    try:
        source_pixels, _, _, ids, descriptors = _base_frame_arrays()
        z = 8.5 + (np.arange(len(source_pixels)) % 7) * 1.05
        source_points = np.column_stack((
            (source_pixels[:, 0] - K[0, 2]) * z / K[0, 0],
            (source_pixels[:, 1] - K[1, 2]) * z / K[1, 1], z,
        ))
        expected = np.eye(4)
        expected[:3, :3] = cv.Rodrigues(np.array([0.012, -0.018, 0.009]))[0]
        expected[:3, 3] = [0.19, -0.055, 0.11]
        target_points = (source_points - expected[:3, 3]) @ expected[:3, :3]
        target_h = target_points @ K.T
        target_pixels = (target_h[:, :2] / target_h[:, 2, None]).astype(np.float32)
        source_right = source_pixels[:, 0] - K[0, 0] * BASELINE / source_points[:, 2]
        target_right = target_pixels[:, 0] - K[0, 0] * BASELINE / target_points[:, 2]
        assert np.all(target_points[:, 2] > 0)
        assert np.all((target_pixels[:, 0] >= 0) & (target_pixels[:, 0] < SIZE[0]))
        assert np.all((target_pixels[:, 1] >= 0) & (target_pixels[:, 1] < SIZE[1]))
        assert np.all((target_right >= 0) & (target_right < SIZE[0]))

        source = _frame_from_arrays(
            (source_pixels, source_points, source_right, np.full_like(ids, -1), descriptors), 0)
        target = _frame_from_arrays(
            (target_pixels, target_points, target_right, np.full_like(ids, -1), descriptors), 1)
        source, source_first, source_alias = _insert_aliases(source, {1, 5, 9})
        target, target_first, target_alias = _insert_aliases(target, {1, 5, 9})
        raw = _raw_matches(source_first, target_first, alias_rows=(1, 5, 9))
        slam.previous_supported_stereo = source
        slam.previous_stereo_geometry = (
            module.StereoLoopFrame(source.pixels, source.points,
                                  source.descriptors, source.image_size), 0)
        slam.map.record(np.eye(4), "tracking")
        slam._match = lambda *_args: raw.copy()

        context, report = slam._prepare_stereo_arbitration(1, target)
        assert report["reason"] == "reserved_supported_evidence"
        assert context is not None
        result = context["verified"]
        assert result["reverse_checked"] is True
        assert result["inliers"] >= 15
        assert result["bidirectional_refinement"] is not None
        np.testing.assert_allclose(result["measurement"], expected, atol=1e-3, rtol=0.)
        assert len(context["fit"]) == 24
        assert len(context["evidence"].points) == 24
    finally:
        slam.close()

