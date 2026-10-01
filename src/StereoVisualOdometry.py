import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from numpy.typing import NDArray
from kitti import image_paths, save_poses_txt
from image_sequence import ImageSequence
from config import (
    STEREO_MIN_MATCHES,
    STEREO_MIN_VALID_3D,
    STEREO_PNP_CONF,
    STEREO_PNP_ITER,
    STEREO_PNP_REPROJ,
)

WINDOW_SIZE = 5
MINIMUM_DISPARITY = 0
NUMBER_OF_DISPARITY = 16 * 6  # divisible by 16

class StereoVisualOdometry:
    def __init__(
        self,
        folder_path: str,
        calibration_path: str,
        use_brute_force: bool,
        poses_path: str | None = None,
        draw_matches: bool = True,
        max_frames: int | None = None,
    ):
        # Load calibration data (projection matrices and intrinsics)
        self.K1, self.P1, self.K2, self.P2 = self.__calib(filepath=calibration_path)
        print("Left Intrinsics:\n", self.K1)
        print("Right Intrinsics:\n", self.K2)

        if poses_path and os.path.exists(poses_path):
            self.true_poses = self.__load_poses(poses_path)
        else:
            self.true_poses = [np.eye(4)]
        self.poses = [np.eye(4)]

        # Load left and right images
        self.Images_1 = self.__load(folder_path + "0", max_frames=max_frames)  # Left images
        self.Images_2 = self.__load(folder_path + "1", max_frames=max_frames)  # Right images
        if [p.name for p in image_paths(folder_path + "0", max_frames)] != [p.name for p in image_paths(folder_path + "1", max_frames)]:
            raise ValueError("Left/right image filenames must match")

        # Compute the Q matrix for reprojectImageTo3D using the correct stereo pair.
        f = float(self.K1[0, 0])
        cx_left = float(self.P1[0, 2])
        cy_left = float(self.P1[1, 2])
        cx_right = float(self.P2[0, 2])
        tx = float(self.P2[0, 3]) / float(self.P2[0, 0])  # Tx = -B
        print((f, cx_left, cy_left))
        baseline = float(self.P1[0, 3] / self.P1[0, 0]) - tx
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("Stereo calibration requires a positive left-to-right baseline")
        self.baseline = float(baseline)
        self.Q = np.array(
            [
                [1, 0, 0, -cx_left],
                [0, 1, 0, -cy_left],
                [0, 0, 0, f],
                [0, 0, 1.0 / baseline, (cx_right - cx_left) / baseline],
            ],
            dtype=np.float32,
        )


        self.stereo = cv.StereoSGBM_create(
            minDisparity=MINIMUM_DISPARITY,
            numDisparities=NUMBER_OF_DISPARITY,
            blockSize=WINDOW_SIZE,
            P1=8 * 3 * WINDOW_SIZE**2,
            P2=32 * 3 * WINDOW_SIZE**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        self.draw_matches = draw_matches

        # Initialize feature detector/matcher (ORB or SIFT)
        if use_brute_force:
            self.__init_orb()
        else:
            self.__init_sift()

        self.debug_stats: dict[str, float] = {}

    def __init_orb(self):
        self.orb = cv.ORB_create(nfeatures=3000)
        self.brute_force = cv.BFMatcher(cv.NORM_HAMMING)

    def __init_sift(self):
        self.sift = cv.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def __load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def __transform(R: NDArray[np.float32], t: NDArray[np.float32]) -> NDArray[np.float32]:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.squeeze(t)
        return T

    @staticmethod
    def __load(filepath: str, max_frames: int | None = None):
        return ImageSequence(filepath, max_frames)

    def save_poses(self, filepath: str) -> None:
        save_poses_txt(filepath, self.poses)

    def __save(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            for i, pose in enumerate(self.poses):
                f.write(f"Pose {i}:\n")
                np.savetxt(f, pose, fmt="%6f")
                f.write("\n")
    
    def __draw_corresponding_points(self, i, kp1, kp2, good_matches):
        draw_params = dict(
            matchColor=-1, 
            singlePointColor=None,
            matchesMask=None,
            flags=2
        )
        # Draw matches between consecutive left images
        image = cv.drawMatches(self.Images_1[i], kp1, self.Images_1[i - 1], kp2, good_matches, None, **draw_params)
        cv.imshow("Feature Matches", image)
        cv.waitKey(1)

    def calc_camera_matrix(self, filepath: str) -> NDArray:
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P
    
    def __calib(self, filepath: str) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        P = {}
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("P"):
                    key = line.split(':', 1)[0].strip()
                    params = np.fromstring(line.split(':', 1)[1], dtype=np.float64, sep=' ')
                    P[key] = np.reshape(params, (3, 4))
        # KITTI gray sequences use P0/P1 for left/right.
        if "P0" in P and "P1" in P:
            P_l, P_r = P["P0"], P["P1"]
        else:
            P_l, P_r = P["P1"], P["P2"]
        K_l = P_l[0:3, 0:3]
        K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    def bf_match_features(self, i: int):
        # Match features between left image at frame i-1 and frame i using ORB + BFMatcher
        kp1, desc1 = self.orb.detectAndCompute(self.Images_1[i - 1], None)
        kp2, desc2 = self.orb.detectAndCompute(self.Images_1[i], None)
        matches = self.brute_force.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        if self.draw_matches:
            self.__draw_corresponding_points(i, kp1, kp2, matches)
        p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return p1, p2, kp1, kp2, matches

    def flann_match_features(self, i: int, orb: bool = False):
        # Match features between left image at frame i-1 and frame i using SIFT + FLANN
        if hasattr(self, "orb"):
            kp1, desc1 = self.orb.detectAndCompute(self.Images_1[i - 1], None)
            kp2, desc2 = self.orb.detectAndCompute(self.Images_1[i], None)
        else:
            kp1, desc1 = self.sift.detectAndCompute(self.Images_1[i - 1], None)
            kp2, desc2 = self.sift.detectAndCompute(self.Images_1[i], None)

        self._current_descriptors = desc2
        if desc1 is None or desc2 is None or len(desc2) < 2:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), kp1, kp2, []

        matcher = self.brute_force if hasattr(self, "orb") else self.flann
        matches = matcher.knnMatch(desc1, desc2, k=2)
        thresh, good_matches = 0.7, []
        for pair in matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < thresh * n.distance:
                good_matches.append(m)
        if self.draw_matches:
            self.__draw_corresponding_points(i, kp1, kp2, good_matches)
        p1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        return p1, p2, kp1, kp2, good_matches
    
    def _compiute_depth(self, disparity):
        # Compute depth from disparity
        depth = self.K1[0, 0] * self.baseline / disparity
        return depth

    def find_transf_pnp(self, i: int):
        transform, self.debug_stats = self.find_transf_pnp_debug(i)
        return transform

    def find_transf_pnp_debug(self, i: int):
        left, right = self.Images_1[i - 1], self.Images_2[i - 1]
        if left.shape != right.shape:
            raise ValueError("Stereo images must have matching dimensions")
        disparity = self.stereo.compute(left, right).astype(np.float32) / 16.0
        valid_disparity = (disparity > MINIMUM_DISPARITY) & (disparity < NUMBER_OF_DISPARITY)
        dense = cv.reprojectImageTo3D(disparity, self.Q)
        p1, p2, old_kp, new_kp, matches = self.flann_match_features(i)
        debug = {"num_matches": len(matches), "num_inliers": 0, "inlier_ratio": 0.0,
                 "keypoints": new_kp, "descriptors": self._current_descriptors,
                 "valid_disparity_ratio": float(valid_disparity.mean()), "baseline": self.baseline,
                 "tracking_ok": False, "num_valid_3d": 0, "feature_points": [], "inlier_mask": []}
        if len(matches) < STEREO_MIN_MATCHES:
            return np.eye(4), debug
        points_3d, points_2d = [], []
        for match in matches:
            u, v = (int(round(coord)) for coord in old_kp[match.queryIdx].pt)
            if not (0 <= u < dense.shape[1] and 0 <= v < dense.shape[0]):
                continue
            point = dense[v, u]
            if not valid_disparity[v, u] or not np.isfinite(point).all() or not (0.1 < point[2] < 100.0):
                continue
            points_3d.append(point)
            points_2d.append(new_kp[match.trainIdx].pt)
        debug["num_valid_3d"] = len(points_3d)
        debug["feature_points"] = points_2d
        debug["inlier_mask"] = [False] * len(points_3d)
        if len(points_3d) < STEREO_MIN_VALID_3D:
            return np.eye(4), debug
        success, rvec, tvec, inliers = cv.solvePnPRansac(
            np.asarray(points_3d, dtype=np.float32), np.asarray(points_2d, dtype=np.float32),
            self.K1, None, iterationsCount=STEREO_PNP_ITER, reprojectionError=STEREO_PNP_REPROJ,
            confidence=STEREO_PNP_CONF, flags=cv.SOLVEPNP_ITERATIVE)
        if not success or inliers is None or len(inliers) < STEREO_MIN_VALID_3D:
            return np.eye(4), debug
        rotation, _ = cv.Rodrigues(rvec)
        transform = self.__transform(rotation, tvec)
        if not np.isfinite(transform).all():
            return np.eye(4), debug
        for index in inliers.ravel():
            debug["inlier_mask"][int(index)] = True
        debug.update({"num_inliers": len(inliers), "inlier_ratio": len(inliers) / len(points_3d),
                      "tracking_ok": True})
        return np.linalg.inv(transform), debug

    def run_vo(self):
        """
        Main loop: for each frame, compute the transformation from frame i-1 to i using PnP,
        then accumulate the pose.
        """
        num_frames = len(self.Images_1)
        for i in range(1, num_frames):
            T = self.find_transf_pnp(i)
            current_pose = self.poses[-1] @ T
            self.poses.append(current_pose)
        print("Done running PnP-based visual odometry!")

# Test
if __name__ == "__main__":
    folder_path = r"sequences\01\image_"
    vo = StereoVisualOdometry(folder_path, r"sequences\01\calib.txt", use_brute_force=True)
    vo.run_vo()
