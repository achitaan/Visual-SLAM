import argparse
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
import time
import uuid
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file

from StereoVisualOdometry import StereoVisualOdometry
from VisualOdometry import VisualOdometry
from kitti import load_poses_txt, save_poses_txt, validate_sequence


from config import (
    ENABLE_KEYFRAMES,
    ENABLE_LOCAL_MAP,
    KEYFRAME_INTERVAL,
    LOCAL_MAP_MAX_POINTS,
    LOCAL_MAP_MAX_RANGE,
    LOCAL_MAP_MIN_DEPTH,
    MIN_KEYFRAME_TRANSLATION,
    TELEMETRY,
    USE_RELATIVE_SCALE_FIX,
    ENABLE_RELOCALIZATION,
)
from slam_backend import SlamBackend
from telemetry import TelemetryServer, TelemetryState, encode_image, make_frame_message, now
from shared_slam import SharedSlam, StereoCamera, MappingConfig, validate_feature_budget
from performance import PerformanceConfig
from reconstruction import export_run

def plot(curr_poses, gt_poses: list[NDArray] | None = None) -> None:
    def get_coords(poses):
        x, y, z = [], [], []
        for pose in poses:
            x.append(pose[0, 3])
            y.append(pose[1, 3])
            z.append(pose[2, 3])
        return x, y, z

    x_a, y_a, z_a = get_coords(curr_poses)
    output_file("trajectory.html", title="Visual Odometry Trajectory")
    p = figure(
        title="Visual Odometry Trajectory",
        x_axis_label="X (meters)",
        y_axis_label="Z (meters)",
        width=800,
        height=600
    )
    p.match_aspect = True
    p.line(x_a, z_a, legend_label="Calculated Camera Trajectory", line_width=2)
    if gt_poses:
        x_expected, _, z_expected = get_coords(gt_poses)
        p.line(x_expected, z_expected, legend_label="Expected Camera Trajectory", line_width=2, color="green")
    p.legend.location = "top_left"
    p.legend.title = "Legend"
    p.grid.grid_line_alpha = 0.3
    show(p)

def plot3D(curr_poses, gt_poses: list[NDArray] | None = None) -> None:
    def get_coords(poses):
        x, y, z = [], [], []
        for pose in poses:
            x.append(pose[0, 3])
            y.append(pose[1, 3])
            z.append(pose[2, 3])
        return x, y, z

    x_a, y_a, z_a = get_coords(curr_poses)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_a, y_a, z_a, label="Calculated Camera Trajectory", color="blue", linewidth=2)
    ax.scatter(x_a, y_a, z_a, color="red", s=10, label="Key Points")
    if gt_poses:
        x_expected, y_expected, z_expected = get_coords(gt_poses)
        ax.plot(x_expected, y_expected, z_expected, label="Expected Camera Trajectory", color="green", linewidth=2)
        ax.scatter(x_expected, y_expected, z_expected, color="orange", s=10, label="Expected Key Points")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.set_title("3D Visual Odometry Trajectory")
    ax.legend()
    ax.grid(True)
    plt.show()


def _mapping_config_from_args(args):
    return MappingConfig(features=args.features,
                         stereo_depth_policy=args.stereo_depth_policy,
                         stereo_pose_arbitration=args.stereo_pose_arbitration,
                         stereo_raw_reference_retry=args.stereo_raw_reference_retry,
                         stereo_owned_image_bundle=args.stereo_owned_image_bundle,
                         stereo_source_history_bundle=args.stereo_source_history_bundle,
                         stereo_retained_source_observations=args.stereo_retained_source_observations,
                         stereo_mapping_observation_retention=getattr(
                             args, 'stereo_mapping_observation_retention', False),
                         stereo_two_view_refinement=getattr(
                             args, 'stereo_two_view_refinement', False),
                          stereo_map_depth_policy=getattr(
                              args, 'stereo_map_depth_policy', 'inherit'),
                         bundle_solver_accuracy=args.bundle_solver_accuracy,
                         stereo_bundle_gauge_mode=getattr(args, 'stereo_bundle_gauge_mode', 'veto'))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, help="KITTI root containing sequences/")
    parser.add_argument("--poses-root", type=Path, help="Ground-truth poses for display only")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--features", type=int, default=1500,
                        help="Requested SharedSlam SIFT feature budget (1–10000; default 1500)")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--opencv-threads", type=int, default=1, help="Bound OpenCV worker memory (default: 1)")
    parser.add_argument("--slam", action="store_true")
    parser.add_argument('--matching-backend', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--stereo-depth-policy', choices=['supported', 'verified_fallback', 'verified_all'], default='supported')
    parser.add_argument('--stereo-map-depth-policy', choices=['inherit', 'verified'], default='inherit',
                        help='Map geometry source only; verified uses full-range right-image correspondences')
    parser.add_argument('--stereo-pose-arbitration', action='store_true',
                        help='Use reserved raw stereo observations to arbitrate map and independent poses')
    parser.add_argument('--stereo-two-view-refinement', action='store_true',
                        help='Opt in to capped full-XYZ two-view stereo training refinement before unchanged holdout arbitration')
    parser.add_argument('--stereo-raw-reference-retry', action='store_true',
                        help='Retry a failed configured stereo reference with guarded raw-supported geometry')
    parser.add_argument('--stereo-owned-image-bundle', action='store_true',
                        help='Opt in to bounded image factors from the selected reserved stereo training rows')
    parser.add_argument('--stereo-source-history-bundle', action='store_true',
                        help='Experiment with accepted source-frame left observations in owned stereo bundle adjustment')
    parser.add_argument('--stereo-retained-source-observations', action='store_true',
                        help='Retain source-image constraints using the existing anchored camera model')
    parser.add_argument('--stereo-mapping-observation-retention', action='store_true',
                        help='Retain individually validated old landmark observations after an independently accepted stereo pose fails only spatial coverage')
    parser.add_argument('--bundle-solver-accuracy', choices=['default', 'precise'], default='default',
                        help='Default keeps current policy; precise applies tight LSMR tolerances to all bundle solves')
    parser.add_argument('--stereo-bundle-gauge-mode', choices=['veto', 'canonical_two_bridge'], default='veto',
                        help='Experimental finite gauge convention for certified two-bridge stereo graphs')
    parser.add_argument('--retrieval', choices=['current', 'indexed', 'exhaustive'], default='current')
    parser.add_argument('--no-cpu-optimizations', action='store_true')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--bundle-diagnostics-dir', type=Path,
                        help='Owned run-output folder for selected bundle adjustment snapshots')
    parser.add_argument('--bundle-diagnostics-frames', type=int, nargs='+',
                        help='Frame IDs to capture when bundle adjustment runs')
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--telemetry-port", type=int, default=TELEMETRY.port)
    parser.add_argument("--frame-delay-ms", type=float, default=TELEMETRY.frame_delay_ms)
    parser.add_argument("--output", type=Path, default=Path("results/poses.txt"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        args.features = validate_feature_budget(args.features)
    except ValueError as error:
        parser.error(str(error))
    if not args.slam and args.features != 1500:
        parser.error('--features other than the default requires --slam')
    if args.stereo_depth_policy != 'supported' and not (args.slam and args.stereo):
        parser.error('--stereo-depth-policy verification requires --slam --stereo')
    if args.stereo_map_depth_policy != 'inherit' and not (args.slam and args.stereo):
        parser.error('--stereo-map-depth-policy verified requires --slam --stereo')
    if args.stereo_pose_arbitration and not (args.slam and args.stereo):
        parser.error('--stereo-pose-arbitration requires --slam --stereo')
    if args.stereo_two_view_refinement and not (
            args.slam and args.stereo and args.stereo_pose_arbitration):
        parser.error('--stereo-two-view-refinement requires --slam --stereo --stereo-pose-arbitration')
    if args.stereo_raw_reference_retry and not (args.slam and args.stereo):
        parser.error('--stereo-raw-reference-retry requires --slam --stereo')
    if args.stereo_owned_image_bundle and not (
            args.slam and args.stereo and args.stereo_pose_arbitration):
        parser.error('--stereo-owned-image-bundle requires --slam --stereo --stereo-pose-arbitration')
    if args.stereo_source_history_bundle and not args.stereo_owned_image_bundle:
        parser.error('--stereo-source-history-bundle requires --stereo-owned-image-bundle')
    if args.stereo_retained_source_observations and not args.stereo_source_history_bundle:
        parser.error('--stereo-retained-source-observations requires --stereo-source-history-bundle')
    if args.stereo_mapping_observation_retention and not (
            args.slam and args.stereo
            and (args.stereo_pose_arbitration or args.stereo_raw_reference_retry)):
        parser.error('--stereo-mapping-observation-retention requires --slam --stereo and an independent stereo reference path')
    if args.bundle_solver_accuracy != 'default' and not args.slam:
        parser.error('--bundle-solver-accuracy precise requires --slam')
    if args.stereo_bundle_gauge_mode != 'veto' and not (args.slam and args.stereo):
        parser.error('--stereo-bundle-gauge-mode canonical_two_bridge requires --slam --stereo')
    if (args.bundle_diagnostics_dir is None) != (args.bundle_diagnostics_frames is None):
        parser.error('--bundle-diagnostics-dir and --bundle-diagnostics-frames must be supplied together')
    if args.bundle_diagnostics_frames is not None:
        if not (args.slam and args.stereo):
            parser.error('--bundle-diagnostics requires --slam --stereo')
        if any(frame < 0 for frame in args.bundle_diagnostics_frames):
            parser.error('--bundle-diagnostics-frames must contain nonnegative frame IDs')
        if len(set(args.bundle_diagnostics_frames)) != len(args.bundle_diagnostics_frames):
            parser.error('--bundle-diagnostics-frames must not contain duplicates')
        map_output = (args.output.parent / (args.output.stem + '-map')).resolve()
        diagnostic_dir = args.bundle_diagnostics_dir.resolve()
        if not diagnostic_dir.is_relative_to(map_output):
            parser.error('--bundle-diagnostics-dir must be inside the map output directory')
    if args.opencv_threads < 1:
        parser.error('--opencv-threads must be positive')
    cv.setNumThreads(args.opencv_threads)
    cv.setRNGSeed(0)
    run_id = str(uuid.uuid4())
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    if args.frame_delay_ms < 0:
        parser.error("--frame-delay-ms cannot be negative")
    if not 0 <= args.telemetry_port <= 65535:
        parser.error("--telemetry-port must be between 0 and 65535")

    gt_poses = None
    stereo_root = None
    gt_path = None
    if args.data_root is not None:
        sequence_dir = validate_sequence(args.data_root, args.sequence, args.stereo, args.max_frames)
        folder_path = str(sequence_dir / "image_0")
        calib_path = str(sequence_dir / "calib.txt")
        times_path = str(sequence_dir / "times.txt")
        camera_id = 0
        stereo_root = str(sequence_dir / "image_")
    else:
        if args.stereo:
            parser.error("--stereo requires --data-root with matching image_0/image_1 frames")
        sample_dir = Path(__file__).resolve().parent.parent / "KITTI_sequence_1"
        folder_path = str(sample_dir / "image_l")
        calib_path = str(sample_dir / "calib.txt")
        times_path = None
        camera_id = 0
    if args.poses_root:
        gt_path = str(args.poses_root / f"{args.sequence}.txt")
        gt_poses = load_poses_txt(gt_path)

    use_stereo = bool(args.stereo and stereo_root and os.path.isdir(stereo_root + "1"))
    if use_stereo:
        vo = StereoVisualOdometry(
            stereo_root,
            calib_path,
            use_brute_force=False,  # Match the SIFT benchmark configuration.
            poses_path=None,
            draw_matches=False,
            max_frames=args.max_frames,
        )
        telemetry_state = TelemetryState(mode="slam" if args.slam else "vo",mode_locked=args.slam)
        telemetry = None
        slam_backend = SlamBackend(camera_matrix=vo.K1)
    else:
        vo = VisualOdometry(
            folder_path,
            calib_path,
            use_brute_force=False,
            camera_id=camera_id,
            draw_matches=False,
            max_frames=args.max_frames,
        )
        telemetry_state = TelemetryState(mode="slam" if args.slam else "vo",mode_locked=args.slam)
        telemetry = None
        slam_backend = SlamBackend(camera_matrix=vo.K)
    last_keyframe_pose = vo.poses[0]
    last_keyframe_index = 0
    if TELEMETRY.enabled and not args.no_telemetry:
        telemetry = TelemetryServer(TELEMETRY.host, args.telemetry_port, telemetry_state)
        telemetry.start()
        if args.slam:
            telemetry_state.mode = "slam"
    last_time = now()
    shared_events_sent = 0
    last_payload = None
    stereo_camera=StereoCamera(vo.stereo,vo.Q,vo.baseline) if use_stereo else None
    performance = PerformanceConfig(retrieval=args.retrieval, matching_backend=args.matching_backend,
                                    cpu_optimizations=not args.no_cpu_optimizations, profile=args.profile is not None)
    bundle_diagnostic_writer = None
    if args.bundle_diagnostics_dir is not None:
        from bundle_diagnostics import BundleDiagnosticsWriter
        bundle_diagnostic_writer = BundleDiagnosticsWriter(
            args.bundle_diagnostics_dir,
            frames=tuple(args.bundle_diagnostics_frames),
        )
    shared = SharedSlam(vo.K1 if use_stereo else vo.K, stereo=stereo_camera,
                        config=_mapping_config_from_args(args),
                        performance=performance,
                        bundle_diagnostic_writer=bundle_diagnostic_writer) if args.slam else None
    if shared is not None:
        shared.process(0, vo.Images_1[0] if use_stereo else vo.Images[0], vo.Images_2[0] if use_stereo else None)
        vo.poses = shared.map.poses

    num_frames = len(vo.Images_1) if use_stereo else len(vo.Images)
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)

    frame_times = None
    if args.realtime and times_path and os.path.isfile(times_path):
        with open(times_path, "r") as f:
            frame_times = [float(line.strip()) for line in f if line.strip()]

    try:
        for i in range(1, num_frames):

            mode_is_slam = telemetry_state.mode == "slam"
            use_keyframes = ENABLE_KEYFRAMES and mode_is_slam
            use_local_map = ENABLE_LOCAL_MAP and mode_is_slam
            use_relocalization = ENABLE_RELOCALIZATION and mode_is_slam

            if shared is not None:
                pose, debug = shared.process(i, vo.Images_1[i] if use_stereo else vo.Images[i], vo.Images_2[i] if use_stereo else None)
                vo.poses = shared.map.poses
                p2 = np.asarray(debug.get('feature_points', []), dtype=np.float32).reshape(-1, 2)
            elif use_stereo:
                T, debug = vo.find_transf_pnp_debug(i)
                kp2 = debug.get("keypoints", [])
                desc2 = debug.get("descriptors")
                p2 = np.asarray(debug.get("feature_points", []), dtype=np.float32).reshape(-1, 2)
            else:
                if use_keyframes or use_local_map:
                    match_debug = vo.flann_match_features(i, return_debug=True)
                    p1, p2, kp1, kp2, desc1, desc2, matches = match_debug
                else:
                    p1, p2 = vo.flann_match_features(i)
                T, debug = vo.find_transf(p1, p2, return_debug=True, use_scale_fix=USE_RELATIVE_SCALE_FIX)
        
            if shared is None:
                vo.poses.append(vo.poses[-1] @ T)

            events = []
            if shared is None and (use_keyframes or use_local_map) and debug.get("tracking_ok", False):
                should_add_keyframe = False
                if (i - last_keyframe_index) >= KEYFRAME_INTERVAL:
                    should_add_keyframe = True
                else:
                    delta = vo.poses[-1][:3, 3] - last_keyframe_pose[:3, 3]
                    if use_stereo and np.linalg.norm(delta) >= MIN_KEYFRAME_TRANSLATION:
                        should_add_keyframe = True

                if should_add_keyframe:
                    map_points = None
                    map_descriptors = None
                    if use_local_map and not use_stereo:
                        best_R = debug.get("best_R")
                        best_t = debug.get("best_t")
                        if best_R is not None and best_t is not None:
                            map_points, point_indices = vo.triangulate_points(p1, p2, best_R, best_t, debug.get("inlier_mask"), return_indices=True)
                            if map_points:
                                match_descs = [desc2[matches[j].trainIdx] for j in point_indices] if desc2 is not None else []

                                prev_pose = vo.poses[-2] if len(vo.poses) > 1 else vo.poses[-1]
                                filtered_points = []
                                filtered_descs = []
                                for point, desc in zip(map_points, match_descs):
                                    if point[2] < LOCAL_MAP_MIN_DEPTH:
                                        continue
                                    if np.linalg.norm(point) > LOCAL_MAP_MAX_RANGE:
                                        continue
                                    world_point = (prev_pose @ np.array([point[0], point[1], point[2], 1.0]))[:3]
                                    filtered_points.append(world_point)
                                    filtered_descs.append(desc)

                                map_points = filtered_points[:LOCAL_MAP_MAX_POINTS]
                                map_descriptors = filtered_descs[:LOCAL_MAP_MAX_POINTS]

                    events = slam_backend.maybe_add_keyframe(
                        index=i,
                        pose_T_wc=vo.poses[-1],
                        image=vo.Images_1[i] if use_stereo else vo.Images[i],
                        descriptors=desc2,
                        timestamp=now(),
                        should_add=True,
                        map_points=map_points,
                        map_descriptors=map_descriptors,
                    )
                    last_keyframe_pose = vo.poses[-1]
                    last_keyframe_index = i

            if shared is None and use_relocalization and (use_keyframes or use_local_map) and not use_stereo:
                lost = slam_backend.update_tracking_state(
                    debug.get("num_inliers"),
                    debug.get("inlier_ratio"),
                )
                if lost:
                    relocalized_pose, relocalize_events = slam_backend.try_relocalize(kp2, desc2, now())
                    events.extend(relocalize_events)
                    if relocalized_pose is not None:
                        vo.poses[-1] = relocalized_pose
                        last_keyframe_pose = relocalized_pose
                        last_keyframe_index = i

            if telemetry:
                current_time = now()
                dt = max(current_time - last_time, 1e-6)
                last_time = current_time
                fps = 1.0 / dt if TELEMETRY.stream_fps else None

                features = None
                if telemetry.state.overlay_enabled and TELEMETRY.stream_features:
                    inlier_mask = debug.get("inlier_mask") or []
                    features = []
                    for idx, (x, y) in enumerate(p2):
                        if idx >= TELEMETRY.max_features:
                            break
                        inlier = inlier_mask[idx] if idx < len(inlier_mask) else True
                        features.append({"x": float(x), "y": float(y), "inlier": inlier})

                image_payload = None
                if telemetry.state.stream_enabled and TELEMETRY.stream_images:
                    image_payload = encode_image(vo.Images_1[i] if use_stereo else vo.Images[i], encoding="jpg")

                tracking = {
                    "num_matches": debug.get("num_matches"),
                    "num_inliers": debug.get("num_inliers"),
                    "inlier_ratio": debug.get("inlier_ratio"),
                    "reprojection_error": debug.get('reprojection_error'),
                    "state": debug.get('state'),
                    "tracking_ok": debug.get("tracking_ok", False),
                }
                map_state = shared.map_state() if shared is not None else slam_backend.map_state() if (use_keyframes or use_local_map) else {"keyframes": 0, "map_points": 0}
                pose_graph_state = None if shared is not None else slam_backend.pose_graph_state(i if (use_keyframes or use_local_map) else None)
                if shared is not None and shared.loop_worker.verified:
                    pose_graph_state={'optimized_pose_T_wc':vo.poses[-1].tolist(),'optimized_poses_count':len(vo.poses),'optimized_poses':[p.tolist() for p in vo.poses] if len(vo.poses)<=5000 else None}
                map_points_payload = None
                if use_local_map and TELEMETRY.stream_map_points:
                    map_points_payload = shared.map_points_sample(TELEMETRY.max_map_points) if shared is not None else slam_backend.map_points_sample(TELEMETRY.max_map_points)

                expected_pose = gt_poses[i] if gt_poses and i < len(gt_poses) else None
                payload = make_frame_message(
                    frame_index=i,
                    timestamp=current_time,
                    pose_T_wc=vo.poses[-1],
                    tracking=tracking,
                    map_state=map_state,
                    map_points=map_points_payload,
                    pose_graph=pose_graph_state,
                    expected_pose_T_wc=expected_pose,
                    image_payload=image_payload,
                    features=features,
                    fps=fps,
                    state=telemetry.state,
                    events=([{'timestamp':current_time,'type':e['type'],'message':str(e),'severity':'warn' if e['type'] in ('loop_failed','loop_discarded') else 'info'} for e in shared.loop_worker.events[shared_events_sent:]] if shared is not None else [e.__dict__ for e in events]),
                    translation_scale="metric" if use_stereo else "arbitrary",
                    sequence=args.sequence if args.data_root else "sample",
                    total_frames=num_frames,
                    run_id=run_id,
                )
                telemetry.publish(payload)
                last_payload = payload
                if shared is not None:shared_events_sent=len(shared.loop_worker.events)
            if args.realtime and frame_times and i < len(frame_times):
                time.sleep(max(0.0, frame_times[i] - frame_times[i - 1]))
            elif telemetry and args.frame_delay_ms > 0:
                time.sleep(args.frame_delay_ms / 1000.0)
    finally:
        if shared is not None:
            shared.close()
            vo.poses = shared.map.poses
            if telemetry and last_payload is not None:
                final_payload={**last_payload,'pose_T_wc':vo.poses[-1].tolist(),'map':shared.map_state(),'map_points':shared.map_points_sample(TELEMETRY.max_map_points)}
                if shared.loop_worker.verified:
                    final_payload['pose_graph']={'optimized_pose_T_wc':vo.poses[-1].tolist(),'optimized_poses_count':len(vo.poses),'optimized_poses':[p.tolist() for p in vo.poses] if len(vo.poses)<=5000 else None}
                final_payload['events']=[{'timestamp':now(),'type':e['type'],'message':str(e),'severity':'info'} for e in shared.loop_worker.events[shared_events_sent:]]
                telemetry.publish(final_payload)
        if telemetry:
            telemetry.stop()
    print("Visual Odometry completed.")
    save_poses_txt(args.output, vo.poses)
    if shared is not None:
        inputs = vo.Images_1.paths if use_stereo else vo.Images.paths
        run_output = args.output.parent / (args.output.stem + '-map')
        if bundle_diagnostic_writer is not None:
            for frame in bundle_diagnostic_writer.frames:
                if frame >= len(shared.map.poses):
                    shared._skip_bundle_diagnostic(frame, 'frame_not_processed')
            diagnostic_manifest = shared.bundle_diagnostics_manifest()
        payload = export_run(shared, run_output, inputs)
        if bundle_diagnostic_writer is not None:
            payload['bundle_diagnostics'] = diagnostic_manifest
            payload['bundle_diagnostic_errors'] = list(shared.bundle_diagnostic_errors)
            (run_output / 'run.json').write_text(
                json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
        if args.profile:
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            args.profile.write_text(json.dumps(shared.profile.detailed_report(), indent=2), encoding='utf-8')
    print(f"Saved {len(vo.poses)} KITTI poses to {args.output}")
    if args.plot:
        plot(vo.poses, gt_poses)
        plot3D(vo.poses, gt_poses)

if __name__ == "__main__":
    main()
