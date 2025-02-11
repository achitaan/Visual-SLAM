import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from numpy.typing import NDArray
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file

from StereoVisualOdometry import StereoVisualOdometry

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
    folder_path = r"sequences\01\image_"  # Folder containing stereo images (subfolders for left/right)
    vo = StereoVisualOdometry(folder_path, r"sequences\01\calib.txt", use_brute_force=True)
    num_frames = len(vo.Images_1)
    for i in range(1, num_frames):
        T = vo.find_transf_pnp(i)

        current_pose = vo.poses[-1] @ T
        vo.poses.append(current_pose)

    plot(vo.poses, vo.true_poses)
    plot3D(vo.poses, vo.true_poses)

if __name__ == "__main__":
    main()