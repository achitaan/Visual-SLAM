import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from numpy.typing import NDArray
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
        self.poses = [self.true_poses[0]]

        # Load left and right images
        self.Images_1 = self.__load(folder_path + "0", max_frames=max_frames)  # Left images
        self.Images_2 = self.__load(folder_path + "1", max_frames=max_frames)  # Right images

        # Compute the Q matrix for reprojectImageTo3D using the correct stereo pair.
        f = float(self.K1[0, 0])
        cx_left = float(self.P1[0, 2])
        cy_left = float(self.P1[1, 2])
        cx_right = float(self.P2[0, 2])
        tx = float(self.P2[0, 3]) / float(self.P2[0, 0])  # Tx = -B
        print((f, cx_left, cy_left))
        baseline = -tx
        self.baseline = float(baseline)
        self.Q = np.array(
            [
                [1, 0, 0, -cx_left],
                [0, 1, 0, -cy_left],
                [0, 0, 0, f],
                [0, 0, 1.0 / baseline, (cx_left - cx_right) / baseline],
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
        #self.brute_force = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

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
    def __load(filepath: str, max_frames: int | None = None) -> list[NDArray]:
        images = []
        for filename in sorted(os.listdir(filepath)):
            path = os.path.join(filepath, filename)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                if max_frames is not None and len(images) >= max_frames:
                    break
        return images

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
        if orb:
            kp1, desc1 = self.orb.detectAndCompute(self.Images_1[i - 1], None)
            kp2, desc2 = self.orb.detectAndCompute(self.Images_1[i], None)
        else:
            kp1, desc1 = self.sift.detectAndCompute(self.Images_1[i - 1], None)
            kp2, desc2 = self.sift.detectAndCompute(self.Images_1[i], None)

        if desc1 is None or desc2 is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), kp1, kp2, []

        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)
        matches = self.flann.knnMatch(desc1, desc2, k=2)
        thresh, good_matches = 0.7, []
        for m, n in matches:
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
        # (1) Compute disparity on frame i-1 using StereoSGBM
        left_prev = self.Images_1[i - 1]
        right_prev = self.Images_2[i - 1]
        disparity = self.stereo.compute(left_prev, right_prev).astype(np.float32) / 16.0
        valid_disparity = (disparity > MINIMUM_DISPARITY) & (disparity < NUMBER_OF_DISPARITY)
        valid_ratio = float(valid_disparity.sum()) / float(disparity.size)
        self.debug_stats = {
            "baseline": self.baseline,
            "valid_disparity_ratio": valid_ratio,
        }
        
        # (2) Reproject disparity to a dense 3D point cloud using Q matrix
        points_3d_dense = cv.reprojectImageTo3D(disparity, self.Q)
        
        # (3) Match features between frame i-1 and frame i (left images)
        if hasattr(self, 'orb'):
            p1, p2, old_kp, new_kp, matches_2d = self.flann_match_features(i, True)
        else:
            p1, p2, old_kp, new_kp, matches_2d = self.flann_match_features(i)
        if len(matches_2d) < STEREO_MIN_MATCHES:
            print(f"2D matching insufficient between frames {i-1} and {i}")
            return np.eye(4)
        
        pts_3d_list = []
        pts_2d_list = []
        # (4) For each matched feature in the previous left image, get its 3D coordinate
        for m in matches_2d:
            pt = old_kp[m.queryIdx].pt  # coordinate in frame i-1
            u = int(round(pt[0]))
            v = int(round(pt[1]))
            # Make sure the keypoint is within the image bounds
            if u < 0 or u >= points_3d_dense.shape[1] or v < 0 or v >= points_3d_dense.shape[0]:
                continue
            X = points_3d_dense[v, u]
            # Check for valid depth (avoid points with zero, infinite, or NaN depth)
            if X[2] <= 0 or np.isinf(X[2]) or np.isnan(X[2]):
                continue
            pts_3d_list.append(X)
            pts_2d_list.append(new_kp[m.trainIdx].pt)
        
        pts_3d_list = np.array(pts_3d_list, dtype=np.float32)
        pts_2d_list = np.array(pts_2d_list, dtype=np.float32)
        
        if len(pts_3d_list) < STEREO_MIN_VALID_3D:
            print(f"Not enough valid 3D-2D correspondences at frame {i}")
            return np.eye(4)
        
        # (5) Run PnP with RANSAC using the 3D points from frame i-1 and their 2D correspondences in frame i
        success, rvec, tvec, inliers = cv.solvePnPRansac(
            pts_3d_list,
            pts_2d_list,
            self.K1,
            None,
            iterationsCount=STEREO_PNP_ITER,
            reprojectionError=STEREO_PNP_REPROJ,
            confidence=STEREO_PNP_CONF,
            flags=cv.SOLVEPNP_ITERATIVE
        )
        if not success:
            print(f"PnP failed for frame {i}")
            return np.eye(4)
        
        depth = self.K1[0, 0] * self.P2[0, 3] / disparity
        R, _ = cv.Rodrigues(rvec)

        self.debug_stats.update({
            "num_matches": float(len(matches_2d)),
            "num_valid_3d": float(len(pts_3d_list)),
            "num_inliers": float(len(inliers)) if inliers is not None else 0.0,
        })
        if inliers is not None and len(pts_3d_list) > 0:
            self.debug_stats["inlier_ratio"] = float(len(inliers)) / float(len(pts_3d_list))

        T = self.__transform(R, tvec)
        # Return the transformation from frame i-1 to i (inverse if needed by your convention)
        return np.linalg.inv(T)

    def find_transf_pnp_debug(self, i: int):
        # (1) Compute disparity on frame i-1 using StereoSGBM
        left_prev = self.Images_1[i - 1]
        right_prev = self.Images_2[i - 1]
        disparity = self.stereo.compute(left_prev, right_prev).astype(np.float32) / 16.0
        valid_disparity = (disparity > MINIMUM_DISPARITY) & (disparity < NUMBER_OF_DISPARITY)
        valid_ratio = float(valid_disparity.sum()) / float(disparity.size)

        # (2) Reproject disparity to a dense 3D point cloud using Q matrix
        points_3d_dense = cv.reprojectImageTo3D(disparity, self.Q)

        # (3) Match features between frame i-1 and frame i (left images)
        if hasattr(self, 'orb'):
            p1, p2, old_kp, new_kp, matches_2d = self.flann_match_features(i, True)
        else:
            p1, p2, old_kp, new_kp, matches_2d = self.flann_match_features(i)
        if len(matches_2d) < STEREO_MIN_MATCHES:
            return np.eye(4), {
                "num_matches": len(matches_2d),
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "keypoints": new_kp,
                "descriptors": None,
            }

        pts_3d_list = []
        pts_2d_list = []
        for m in matches_2d:
            pt = old_kp[m.queryIdx].pt
            u = int(round(pt[0]))
            v = int(round(pt[1]))
            if u < 0 or u >= points_3d_dense.shape[1] or v < 0 or v >= points_3d_dense.shape[0]:
                continue
            X = points_3d_dense[v, u]
            if X[2] <= 0 or np.isinf(X[2]) or np.isnan(X[2]):
                continue
            pts_3d_list.append(X)
            pts_2d_list.append(new_kp[m.trainIdx].pt)

        pts_3d_list = np.array(pts_3d_list, dtype=np.float32)
        pts_2d_list = np.array(pts_2d_list, dtype=np.float32)
        if len(pts_3d_list) < STEREO_MIN_VALID_3D:
            return np.eye(4), {
                "num_matches": len(matches_2d),
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "keypoints": new_kp,
                "descriptors": None,
            }

        success, rvec, tvec, inliers = cv.solvePnPRansac(
            pts_3d_list,
            pts_2d_list,
            self.K1,
            None,
            iterationsCount=STEREO_PNP_ITER,
            reprojectionError=STEREO_PNP_REPROJ,
            confidence=STEREO_PNP_CONF,
            flags=cv.SOLVEPNP_ITERATIVE
        )
        if not success:
            return np.eye(4), {
                "num_matches": len(matches_2d),
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "keypoints": new_kp,
                "descriptors": None,
            }

        R, _ = cv.Rodrigues(rvec)
        T = self.__transform(R, tvec)
        num_inliers = int(len(inliers)) if inliers is not None else 0
        num_matches = len(matches_2d)
        if hasattr(self, "orb"):
            _, desc = self.orb.detectAndCompute(self.Images_1[i], None)
        else:
            _, desc = self.sift.detectAndCompute(self.Images_1[i], None)
        debug = {
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "inlier_ratio": (num_inliers / num_matches) if num_matches > 0 else 0.0,
            "keypoints": new_kp,
            "descriptors": desc,
            "valid_disparity_ratio": valid_ratio,
            "baseline": self.baseline,
            "num_valid_3d": len(pts_3d_list),
        }
        return np.linalg.inv(T), debug
    
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
