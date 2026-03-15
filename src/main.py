import argparse
import os
import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file

from StereoVisualOdometry import StereoVisualOdometry
from VisualOdometry import VisualOdometry


def load_poses_txt(path: str) -> list[np.ndarray]:
    poses = []
    with open(path, "r") as f:
        for line in f:
            values = np.fromstring(line.strip(), dtype=np.float64, sep=" ")
            if values.size != 12:
                continue
            T = values.reshape(3, 4)
            T = np.vstack((T, [0.0, 0.0, 0.0, 1.0]))
            poses.append(T)
    return poses
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"C:\Users\SSGSS\Documents\Visual-SLAM\data_odometry_gray\dataset")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--slam", action="store_true")
    args = parser.parse_args()

    gt_poses = None
    stereo_root = None
    if os.path.isdir(args.data_root):
        folder_path = os.path.join(args.data_root, "sequences", args.sequence, "image_0")
        calib_path = os.path.join(args.data_root, "sequences", args.sequence, "calib.txt")
        times_path = os.path.join(args.data_root, "sequences", args.sequence, "times.txt")
        camera_id = 0
        stereo_root = os.path.join(args.data_root, "sequences", args.sequence, "image_")
        poses_root = r"C:\Users\SSGSS\Documents\Visual-SLAM\data_odometry_poses\dataset\poses"
        gt_path = os.path.join(poses_root, f"{args.sequence}.txt")
        if os.path.isfile(gt_path):
            gt_poses = load_poses_txt(gt_path)
    else:
        folder_path = r"KITTI_sequence_1\image_l"
        calib_path = r"KITTI_sequence_1\calib.txt"
        times_path = None
        camera_id = 1

    use_stereo = bool(args.stereo and stereo_root and os.path.isdir(stereo_root + "1"))
    if use_stereo:
        vo = StereoVisualOdometry(
            stereo_root,
            calib_path,
            use_brute_force=True,  # ORB is faster than SIFT
            poses_path=gt_path,
            draw_matches=False,
            max_frames=args.max_frames,
        )
        telemetry_state = TelemetryState()
        telemetry = None
        slam_backend = SlamBackend(camera_matrix=vo.K1)
    else:
        vo = VisualOdometry(
            folder_path,
            calib_path,
            use_brute_force=False,
            camera_id=camera_id,
            draw_matches=False,
        )
        telemetry_state = TelemetryState()
        telemetry = None
        slam_backend = SlamBackend(camera_matrix=vo.K)
    last_keyframe_pose = vo.poses[0]
    last_keyframe_index = 0
    if TELEMETRY.enabled:
        telemetry = TelemetryServer(TELEMETRY.host, TELEMETRY.port, telemetry_state)
        telemetry.start()
        if args.slam:
            telemetry_state.mode = "slam"
    last_time = now()

    num_frames = len(vo.Images_1) if use_stereo else len(vo.Images)
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)

    frame_times = None
    if args.realtime and times_path and os.path.isfile(times_path):
        with open(times_path, "r") as f:
            frame_times = [float(line.strip()) for line in f if line.strip()]

    for i in range(1, num_frames):

        mode_is_slam = telemetry_state.mode == "slam" if telemetry else True
        use_keyframes = ENABLE_KEYFRAMES and mode_is_slam
        use_local_map = ENABLE_LOCAL_MAP and mode_is_slam
        use_relocalization = ENABLE_RELOCALIZATION and mode_is_slam

        if use_stereo:
            T, debug = vo.find_transf_pnp_debug(i)
            kp2 = debug.get("keypoints", [])
            desc2 = debug.get("descriptors")
            p2 = np.array([kp.pt for kp in kp2], dtype=np.float32) if kp2 else np.empty((0, 2), dtype=np.float32)
        else:
            if use_keyframes or use_local_map:
                match_debug = vo.flann_match_features(i, return_debug=True)
                p1, p2, kp1, kp2, desc1, desc2, matches = match_debug
            else:
                p1, p2 = vo.flann_match_features(i)
            T, debug = vo.find_transf(p1, p2, return_debug=True, use_scale_fix=USE_RELATIVE_SCALE_FIX)
        
        vo.poses.append(vo.poses[-1] @ T)

        events = []
        if use_keyframes or use_local_map:
            should_add_keyframe = False
            if (i - last_keyframe_index) >= KEYFRAME_INTERVAL:
                should_add_keyframe = True
            else:
                delta = vo.poses[-1][:3, 3] - last_keyframe_pose[:3, 3]
                if np.linalg.norm(delta) >= MIN_KEYFRAME_TRANSLATION:
                    should_add_keyframe = True

            if should_add_keyframe:
                map_points = None
                map_descriptors = None
                if use_local_map and not use_stereo:
                    best_R = debug.get("best_R")
                    best_t = debug.get("best_t")
                    if best_R is not None and best_t is not None:
                        map_points = vo.triangulate_points(p1, p2, best_R, best_t, debug.get("inlier_mask"))
                        if map_points:
                            inlier_mask = debug.get("inlier_mask") or []
                            match_descs = [desc2[m.trainIdx] for m in matches] if desc2 is not None else []
                            if inlier_mask:
                                match_descs = [d for d, keep in zip(match_descs, inlier_mask) if keep]

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

                            if filtered_points:
                                if len(filtered_points) > LOCAL_MAP_MAX_POINTS:
                                    filtered_points = filtered_points[:LOCAL_MAP_MAX_POINTS]
                                    filtered_descs = filtered_descs[:LOCAL_MAP_MAX_POINTS]
                                map_points = filtered_points
                                map_descriptors = filtered_descs

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

        if use_relocalization and (use_keyframes or use_local_map) and not use_stereo:
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
                "reprojection_error": None,
            }
            map_state = slam_backend.map_state() if (use_keyframes or use_local_map) else {"keyframes": 0, "map_points": 0}
            pose_graph_state = slam_backend.pose_graph_state(last_keyframe_index if (use_keyframes or use_local_map) else None)
            map_points_payload = None
            if use_local_map and TELEMETRY.stream_map_points:
                map_points_payload = slam_backend.map_points_sample(TELEMETRY.max_map_points)

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
                events=[e.__dict__ for e in events] if events else [],
            )
            telemetry.publish(payload)
        if args.realtime and frame_times and i < len(frame_times):
            time.sleep(max(0.0, frame_times[i] - frame_times[i - 1]))
        elif TELEMETRY.enabled and hasattr(TELEMETRY, "frame_delay_ms") and TELEMETRY.frame_delay_ms > 0:
            time.sleep(TELEMETRY.frame_delay_ms / 1000.0)
    print("Visual Odometry completed.")
    if hasattr(vo, "save_poses"):
        vo.save_poses("poses.txt")
    if not TELEMETRY.enabled:
        plot(vo.poses, vo.true_poses)
        plot3D(vo.poses, vo.true_poses)

if __name__ == "__main__":
    main()