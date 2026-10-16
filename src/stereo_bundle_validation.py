"""Fail-closed reserved stereo evidence check for local bundle candidates.

This module does not fit poses or alter the bundle objective.  It snapshots the
already selected reserved stereo evidence before local BA, then checks the
candidate at the locked precommit boundary against the same evidence and the
actual prepared image-factor rows.
"""

from __future__ import annotations

import numpy as np

from stereo_pose_arbitration import SupportedStereoFrame, SupportedStereoHoldout, _score


def _pixel_key(value):
    pixel = np.asarray(value, dtype=np.float32)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("invalid_float32_pixel")
    return tuple(float(x) for x in pixel)


def _int_id(value, label):
    if (not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))):
        raise ValueError(f"invalid_{label}")
    value = int(value)
    if value < 0:
        raise ValueError(f"invalid_{label}")
    return value


def _pose(value):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("complex_pose")
    result = np.asarray(raw, dtype=np.float64)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], (0., 0., 0., 1.), rtol=0., atol=1e-10)
            or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3),
                               rtol=0., atol=1e-7)
            or not np.isclose(np.linalg.det(result[:3, :3]), 1., rtol=0., atol=1e-7)):
        raise ValueError("invalid_se3_pose")
    return result.copy()


def _score_dict(value):
    if not isinstance(value, dict):
        return None
    cost = value.get("cost")
    if cost is None:
        return None
    cost = float(cost)
    if not np.isfinite(cost):
        return None
    try:
        result = {
            "cost": cost,
            "inliers": int(value["inliers"]),
            "cells": int(value["cells"]),
            "eligible": value["eligible"] is True,
            "reason": str(value.get("reason", "unknown")),
        }
        if result["inliers"] < 0 or result["cells"] < 0:
            return None
        return result
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _frame_snapshot(state, frame_id):
    frame_id = int(frame_id)
    if (frame_id < 0 or frame_id >= min(len(state.poses), len(state.statuses),
                                        len(state.pose_anchors), len(state.relative_poses))):
        raise ValueError("endpoint_frame_unavailable")
    anchor_id = state.pose_anchors[frame_id]
    anchor_obj = None
    anchor_pose_ref = None
    anchor_pose = None
    if anchor_id is not None:
        anchor_id = int(anchor_id)
        anchor_obj = state.keyframes.get(anchor_id)
        if anchor_obj is None:
            raise ValueError("endpoint_anchor_unavailable")
        anchor_pose_ref = anchor_obj.pose
        anchor_pose = _pose(anchor_pose_ref)
        if int(anchor_obj.id) != anchor_id:
            raise ValueError("endpoint_anchor_identity_invalid")
    return {
        "frame": frame_id,
        "pose": _pose(state.poses[frame_id]),
        "status": state.statuses[frame_id],
        "anchor_id": anchor_id,
        "relative_pose": _pose(state.relative_poses[frame_id]),
        "anchor_object": anchor_obj,
        "anchor_pose_ref": anchor_pose_ref,
        "anchor_pose": anchor_pose,
    }


def _valid_evidence(context, source, target, calibration_identity):
    evidence = context.get("evidence")
    if (not isinstance(evidence, SupportedStereoHoldout)
            or evidence.provenance != "immutable_supported_extraction"
            or evidence.calibration_identity != calibration_identity
            or int(evidence.source_frame) != int(source.frame)
            or int(source.frame) >= int(target.frame)
            or int(target.frame) - int(source.frame) > 3):
        raise ValueError("reserved_evidence_missing_or_unbound")
    if (source.calibration_identity != calibration_identity
            or target.calibration_identity != calibration_identity
            or tuple(source.image_size) != tuple(target.image_size)):
        raise ValueError("endpoint_calibration_mismatch")
    n = len(evidence.points)
    if (n < 1 or evidence.points.shape != (n, 3) or evidence.left.shape != (n, 2)
            or evidence.right_u.shape != (n,) or evidence.source_ids.shape != (n,)
            or evidence.target_ids.shape != (n,) or evidence.landmark_ids.shape != (n,)
            or not np.isfinite(evidence.points).all()
            or not np.isfinite(evidence.left).all()
            or not np.isfinite(evidence.right_u).all()
            or np.any(evidence.points[:, 2] <= 0)
            or not np.issubdtype(evidence.source_ids.dtype, np.integer)
            or not np.issubdtype(evidence.target_ids.dtype, np.integer)
            or not np.issubdtype(evidence.landmark_ids.dtype, np.integer)
            or np.any(evidence.source_ids < 0) or np.any(evidence.target_ids < 0)
            or np.any(evidence.source_ids >= len(source.pixels))
            or np.any(evidence.target_ids >= len(target.pixels))
            or len(np.unique(evidence.source_ids)) != n
            or len(np.unique(evidence.target_ids)) != n):
        raise ValueError("invalid_reserved_evidence")
    source_pixels = np.asarray(source.pixels[evidence.source_ids], dtype=np.float32)
    target_pixels = np.asarray(target.pixels[evidence.target_ids], dtype=np.float32)
    if (not np.array_equal(evidence.points, source.points[evidence.source_ids])
            or not np.array_equal(evidence.right_u, target.right_u[evidence.target_ids])):
        raise ValueError("reserved_raw_measurement_binding_mismatch")
    if (source_pixels.shape != (n, 2) or target_pixels.shape != (n, 2)
            or not np.array_equal(target_pixels, np.asarray(evidence.left, dtype=np.float32))):
        raise ValueError("reserved_pixel_binding_mismatch")
    # Aliased physical rows make ownership ambiguous even when feature IDs differ.
    source_keys = [_pixel_key(pixel) for pixel in source_pixels]
    target_keys = [_pixel_key(pixel) for pixel in target_pixels]
    if len(set(source_keys)) != n or len(set(target_keys)) != n:
        raise ValueError("ambiguous_reserved_pixel_alias")
    source_lids = np.asarray(source.landmark_ids[evidence.source_ids], dtype=np.int64)
    target_lids = np.asarray(target.landmark_ids[evidence.target_ids], dtype=np.int64)
    if not np.array_equal(source_lids, np.asarray(evidence.landmark_ids, dtype=np.int64)):
        raise ValueError("reserved_source_landmark_binding_mismatch")
    for values in (source_lids[source_lids >= 0], target_lids[target_lids >= 0]):
        if len(np.unique(values)) != len(values):
            raise ValueError("ambiguous_reserved_landmark_alias")
    return evidence, source_pixels, target_pixels, source_lids, target_lids


def _validate_fit_partition(context, evidence, source_lids, target_lids):
    fit = np.asarray(context.get("fit"))
    if (fit.ndim != 2 or fit.shape[1] != 2 or not np.issubdtype(fit.dtype, np.integer)
            or not len(fit)):
        raise ValueError("invalid_fit_partition")
    if (np.intersect1d(evidence.source_ids, fit[:, 0]).size
            or np.intersect1d(evidence.target_ids, fit[:, 1]).size):
        raise ValueError("fit_held_feature_overlap")
    map_targets = context.get("map_fit_targets")
    map_landmarks = context.get("map_fit_landmarks")
    if not isinstance(map_targets, (set, frozenset, list, tuple)) or not isinstance(
            map_landmarks, (set, frozenset, list, tuple)):
        raise ValueError("map_fit_partition_unavailable")
    if np.intersect1d(evidence.target_ids, np.asarray(list(map_targets), dtype=np.int64)).size:
        raise ValueError("map_fit_held_feature_overlap")
    held_lids = np.r_[source_lids[source_lids >= 0], target_lids[target_lids >= 0]]
    if np.intersect1d(held_lids, np.asarray(list(map_landmarks), dtype=np.int64)).size:
        raise ValueError("map_fit_held_landmark_overlap")


def _objective_rows(prepared):
    cameras = prepared.get("camera_poses")
    if not isinstance(cameras, (list, tuple)):
        raise ValueError("prepared_camera_pose_rows_missing")
    keyframe_to_frame = {}
    for item in cameras:
        if not isinstance(item, dict):
            raise ValueError("malformed_prepared_camera_pose")
        key = _int_id(item["keyframe_id"], "prepared_keyframe_id")
        frame = _int_id(item["frame_id"], "prepared_frame_id")
        if key in keyframe_to_frame or frame < 0:
            raise ValueError("ambiguous_prepared_camera_identity")
        keyframe_to_frame[key] = frame
    result = []
    for kind in ("selected_observations", "excluded_multiview_observations"):
        rows = prepared.get(kind)
        if not isinstance(rows, (list, tuple)):
            raise ValueError("prepared_objective_rows_missing")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("malformed_prepared_objective_row")
            keyframe_id = _int_id(row["keyframe_id"], "objective_keyframe_id")
            if keyframe_id not in keyframe_to_frame:
                raise ValueError("objective_camera_identity_missing")
            pixel = _pixel_key(row["pixel"])
            landmark_id = row["landmark_id"]
            if (not isinstance(landmark_id, (int, np.integer))
                    or isinstance(landmark_id, (bool, np.bool_))):
                raise ValueError("invalid_objective_landmark_id")
            landmark_id = int(landmark_id)
            if landmark_id < -1:
                raise ValueError("invalid_objective_landmark_id")
            result.append((keyframe_to_frame[keyframe_id], pixel, landmark_id, kind))
    return result


def _validate_objective_disjointness(prepared, source, target, source_pixels,
                                     target_pixels, source_lids, target_lids):
    objective = _objective_rows(prepared)
    occupied_pixels = {(frame, pixel) for frame, pixel, _landmark, _kind in objective}
    occupied_landmarks = {(frame, landmark) for frame, _pixel, landmark, _kind in objective
                          if landmark >= 0}
    held = []
    for pixel, landmark in zip(source_pixels, source_lids):
        held.append((int(source.frame), _pixel_key(pixel), int(landmark)))
    for pixel, landmark in zip(target_pixels, target_lids):
        held.append((int(target.frame), _pixel_key(pixel), int(landmark)))
    for frame, pixel, landmark in held:
        if (frame, pixel) in occupied_pixels:
            raise ValueError("reserved_raw_row_in_bundle_objective")
        if landmark >= 0 and (frame, landmark) in occupied_landmarks:
            raise ValueError("reserved_landmark_row_in_bundle_objective")
    return len(objective)


def _candidate_pose_for_frame(endpoint, candidate_keyframe_poses, frame_updates):
    frame = endpoint["frame"]
    if frame_updates is not None and frame in frame_updates:
        return _pose(frame_updates[frame])
    anchor = endpoint["anchor_id"]
    if anchor is None:
        return endpoint["pose"].copy()
    if anchor not in candidate_keyframe_poses:
        raise ValueError("candidate_anchor_pose_missing")
    candidate_anchor = _pose(candidate_keyframe_poses[anchor])
    return _pose(candidate_anchor @ np.linalg.inv(endpoint["anchor_pose"]) @ endpoint["pose"])


def _same_endpoint(state, endpoint):
    frame = endpoint["frame"]
    if (frame >= min(len(state.poses), len(state.statuses), len(state.pose_anchors),
                     len(state.relative_poses))
            or state.statuses[frame] != endpoint["status"]
            or state.pose_anchors[frame] != endpoint["anchor_id"]
            or not np.array_equal(state.poses[frame], endpoint["pose"])
            or not np.array_equal(state.relative_poses[frame], endpoint["relative_pose"])):
        return False
    anchor = endpoint["anchor_id"]
    if anchor is not None:
        live = state.keyframes.get(anchor)
        if (live is not endpoint["anchor_object"]
                or live.pose is not endpoint["anchor_pose_ref"]
                or not np.array_equal(live.pose, endpoint["anchor_pose"])):
            return False
    return True


def _reject(reason, *, before=None, candidate=None, certificate=None):
    return {
        "accepted": False,
        "reason": str(reason),
        "before": before,
        "candidate": candidate,
        "certificate": certificate or {},
    }


def build_bundle_holdout_guard(slam, context, target_frame, arbitration_report=None):
    """Build a pure validator for one already accepted reserved stereo edge.

    The returned callback is intended to run at local BA's locked precommit
    boundary. It never mutates map state. Missing or ambiguous evidence raises
    during construction so the caller can attach a fail-closed validator.
    """
    if not isinstance(context, dict):
        raise ValueError("reserved_arbitration_context_missing")
    config = slam.config
    state = slam.map
    stereo = slam.stereo
    if (stereo is None or not getattr(config, "stereo_pose_arbitration", False)
            or not getattr(state, "metric", False)):
        raise ValueError("guard_requires_metric_stereo_arbitration")
    if (getattr(config, "stereo_owned_image_bundle", False)
            or getattr(config, "stereo_source_history_bundle", False)
            or getattr(config, "stereo_retained_source_observations", False)
            or getattr(config, "stereo_bundle_gauge_mode", "veto") != "veto"):
        raise ValueError("guard_incompatible_bundle_mode")
    selected = arbitration_report or context.get("selected_arbitration_report")
    pre_bundle = context.get("pre_bundle_arbitration_report")
    if (not isinstance(selected, dict) or selected.get("held_out_arbitration_used") is not True
            or selected.get("choice") not in ("map", "independent")
            or not isinstance(pre_bundle, dict)):
        raise ValueError("selected_reserved_arbitration_certificate_missing")
    source = context.get("previous")
    target = context.get("current")
    if not isinstance(source, SupportedStereoFrame) or not isinstance(target, SupportedStereoFrame):
        raise ValueError("supported_endpoint_snapshot_missing")
    if int(target.frame) != int(target_frame):
        raise ValueError("target_frame_binding_mismatch")
    evidence, source_pixels, target_pixels, source_lids, target_lids = _valid_evidence(
        context, source, target, slam.stereo_calibration_identity)
    _validate_fit_partition(context, evidence, source_lids, target_lids)
    n = len(evidence.points)
    for report in (selected, pre_bundle):
        if (int(report.get("source_frame", -1)) != int(source.frame)
                or int(report.get("target_frame", -1)) != int(target.frame)
                or int(report.get("holdout_count", -1)) != n
                or report.get("calibration_identity") != slam.stereo_calibration_identity):
            raise ValueError("arbitration_report_binding_mismatch")
    before_report = _score_dict(pre_bundle.get("map"))
    if before_report is None or before_report["eligible"] is not True:
        raise ValueError("pre_bundle_map_support_unavailable")
    matrix = np.asarray(slam.K, dtype=np.float64).copy()
    baseline = float(stereo.baseline)
    offset = float(stereo.disparity_offset)
    image_size = tuple(int(x) for x in target.image_size)
    minimum = int(config.min_inliers)
    q_snapshot = np.array(stereo.Q, copy=True)
    identity_snapshot = str(slam.stereo_calibration_identity)
    excluded_pixels = context.get("excluded_target_pixels")
    if not isinstance(excluded_pixels, (set, frozenset)):
        raise ValueError("reserved_target_pixel_exclusion_missing")
    if not set(target_pixels_key for target_pixels_key in map(_pixel_key, target_pixels)).issubset(
            set(excluded_pixels)):
        raise ValueError("reserved_target_pixel_exclusion_mismatch")

    with state.lock:
        revision = int(state.revision)
        geometry_revision = int(state.geometry_revision)
        endpoints = (_frame_snapshot(state, source.frame), _frame_snapshot(state, target.frame))
        if not all(endpoint["status"] in ("tracking", "relocalized", "accepted")
                   for endpoint in endpoints):
            raise ValueError("endpoint_status_not_accepted")
        before_relative = np.linalg.inv(endpoints[0]["pose"]) @ endpoints[1]["pose"]
        before = _score(before_relative, evidence, matrix, baseline, offset, image_size, minimum)
        if before.get("eligible") is not True:
            raise ValueError("accepted_pose_no_longer_supports_reserved_rows")
        if (before.get("inliers") != before_report["inliers"]
                or before.get("cells") != before_report["cells"]
                or before.get("cost") is None or before_report["cost"] is None
                or abs(float(before["cost"]) - float(before_report["cost"]))
                > 64*np.finfo(float).eps*max(1., float(before["cost"]), float(before_report["cost"]))):
            raise ValueError("pre_bundle_score_does_not_match_live_pose")

    source_pixels = np.array(source_pixels, dtype=np.float32, copy=True)
    target_pixels = np.array(target_pixels, dtype=np.float32, copy=True)
    source_lids = np.array(source_lids, dtype=np.int64, copy=True)
    target_lids = np.array(target_lids, dtype=np.int64, copy=True)
    for array in (source_pixels, target_pixels, source_lids, target_lids, matrix, q_snapshot):
        array.setflags(write=False)

    def validate(packet):
        certificate = {"source_frame": int(source.frame), "target_frame": int(target.frame),
                       "holdout_count": n, "objective_row_count": None,
                       "map_revision": revision, "geometry_revision": geometry_revision}
        try:
            if not isinstance(packet, dict) or not isinstance(packet.get("prepared"), dict):
                return _reject("prepared_bundle_snapshot_missing", certificate=certificate)
            prepared = packet["prepared"]
            if (prepared.get("input_valid") is not True
                    or int(prepared.get("revision", -1)) != revision
                    or int(prepared.get("geometry_revision", -1)) != geometry_revision
                    or prepared.get("metric") is not True):
                return _reject("prepared_bundle_epoch_or_metric_mismatch", certificate=certificate)
            calibration = prepared.get("calibration")
            if (not isinstance(calibration, dict)
                    or not np.array_equal(np.asarray(calibration.get("matrix")), matrix)
                    or float(calibration.get("baseline")) != baseline
                    or float(calibration.get("disparity_offset")) != offset):
                return _reject("prepared_bundle_calibration_mismatch", certificate=certificate)
            objective_count = _validate_objective_disjointness(
                prepared, source, target, source_pixels, target_pixels, source_lids, target_lids)
            certificate["objective_row_count"] = int(objective_count)
            camera_poses = packet.get("candidate_keyframe_poses")
            if not isinstance(camera_poses, dict):
                return _reject("candidate_camera_poses_missing", certificate=certificate)
            frame_updates = packet.get("frame_updates")
            if frame_updates is not None:
                return _reject("frame_updates_outside_original_metric_scope",
                               certificate=certificate)
            candidate_poses = {}
            for raw_key, value in camera_poses.items():
                key = _int_id(raw_key, "candidate_keyframe_id")
                if key in candidate_poses:
                    return _reject("candidate_keyframe_id_alias", certificate=certificate)
                candidate_poses[key] = _pose(value)
            with state.lock:
                if (int(state.revision) != revision
                        or int(state.geometry_revision) != geometry_revision
                        or not all(_same_endpoint(state, endpoint) for endpoint in endpoints)):
                    return _reject("stale_map_or_endpoint_snapshot", certificate=certificate)
                if any(key not in state.keyframes for key in candidate_poses):
                    return _reject("candidate_keyframe_not_in_snapshot", certificate=certificate)
                for camera in prepared["camera_poses"]:
                    key = _int_id(camera["keyframe_id"], "prepared_keyframe_id")
                    live = state.keyframes.get(key)
                    if (live is None or int(live.frame) != int(camera["frame_id"])
                            or not np.array_equal(_pose(camera["camera_to_world"]), live.pose)):
                        return _reject("prepared_camera_identity_mismatch", certificate=certificate)
                if (slam.stereo is None
                        or str(slam.stereo_calibration_identity) != identity_snapshot
                        or not np.array_equal(np.asarray(slam.K), matrix)
                        or float(slam.stereo.baseline) != baseline
                        or float(slam.stereo.disparity_offset) != offset
                        or not np.array_equal(np.asarray(slam.stereo.Q), q_snapshot)):
                    return _reject("stale_calibration_snapshot", certificate=certificate)
                candidate_source = _candidate_pose_for_frame(
                    endpoints[0], candidate_poses, frame_updates)
                candidate_target = _candidate_pose_for_frame(
                    endpoints[1], candidate_poses, frame_updates)
                candidate_relative = np.linalg.inv(candidate_source) @ candidate_target
                candidate = _score(candidate_relative, evidence, matrix, baseline, offset,
                                   image_size, minimum)
                safe_candidate = _score_dict(candidate)
                safe_before = _score_dict(before)
                if (safe_candidate is None or safe_before is None
                        or safe_candidate["cost"] is None or safe_before["cost"] is None):
                    return _reject("candidate_score_invalid", before=safe_before,
                                   candidate=safe_candidate, certificate=certificate)
                if not safe_before["eligible"] or not safe_candidate["eligible"]:
                    return _reject("reserved_support_ineligible", before=safe_before,
                                   candidate=safe_candidate, certificate=certificate)
                tolerance = 64*np.finfo(float).eps*max(
                    1., float(safe_before["cost"]), float(safe_candidate["cost"]))
                if (safe_candidate["cost"] > safe_before["cost"] + tolerance
                        or safe_candidate["inliers"] < safe_before["inliers"]
                        or safe_candidate["cells"] < safe_before["cells"]):
                    return _reject("reserved_support_degraded", before=safe_before,
                                   candidate=safe_candidate, certificate=certificate)
                return {
                    "accepted": True,
                    "reason": "reserved_support_preserved",
                    "before": safe_before,
                    "candidate": safe_candidate,
                    "certificate": certificate,
                }
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError,
                IndexError, np.linalg.LinAlgError):
            return _reject("certificate_validation_failed", certificate=certificate)

    return validate
