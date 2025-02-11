import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from numpy.typing import NDArray

class StereoVisualOdometry:
    def __init__(self, folder_path: str, calibration_path: str, use_brute_force: bool):
        # Load calibration data (projection matrices and intrinsics)
        self.K1, self.P1, self.K2, self.P2 = self.__calib(filepath=calibration_path)
        print("Left Intrinsics:\n", self.K1)
        print("Right Intrinsics:\n", self.K2)

        self.true_poses = self.__load_poses(r"poses\01.txt")
        self.poses = [self.true_poses[0]] 

        # Load left and right images
        self.Images_1 = self.__load(folder_path + "0")  # Left images
        self.Images_2 = self.__load(folder_path + "1")  # Right images

        # Compute the Q matrix for reprojectImageTo3D.
        # Assumes P1 = [K1 | 0] and P2 = [K1 | -K1*[B,0,0]^T]
        f = self.K1[0, 0]
        cx = self.K1[0, 2]
        cy = self.K1[1, 2]
        print("f, cx, cy:", (f, cx, cy))
        # Baseline from P2 (assuming P2[0,3] = -f*B). Use absolute value.
        baseline = np.abs(self.P2[0, 3]) / f  
        self.baseline = baseline  # Save for later use in computing depth
        self.f = f              # Save focal length for depth formula
        self.Q = np.array([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, f],
            [0, 0, -1 / baseline, 0]
        ], dtype=np.float32)

        # Initialize feature detector/matcher (ORB or SIFT)
        if use_brute_force:
            self.__init_orb()
        else:
            self.__init_sift()

    def __init_orb(self):
        self.orb = cv.ORB_create(nfeatures=3000)
        self.brute_force = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

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
    def __load(filepath: str) -> list[NDArray]:
        images = []
        for filename in sorted(os.listdir(filepath)):
            path = os.path.join(filepath, filename)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
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
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("P1:"):
                    params = np.fromstring(line.split(':', 1)[1], dtype=np.float64, sep=' ')
                    P_l = np.reshape(params, (3, 4))
                    K_l = P_l[0:3, 0:3]
                if line.startswith("P2:"):
                    params = np.fromstring(line.split(':', 1)[1], dtype=np.float64, sep=' ')
                    P_r = np.reshape(params, (3, 4))
                    K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    def bf_match_features(self, i: int):
        # Match features between left image at frame i-1 and frame i using ORB + BFMatcher
        kp1, desc1 = self.orb.detectAndCompute(self.Images_1[i - 1], None)
        kp2, desc2 = self.orb.detectAndCompute(self.Images_1[i], None)
        matches = self.brute_force.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        self.__draw_corresponding_points(i, kp1, kp2, matches)
        p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return p1, p2, kp1, kp2, matches

    def flann_match_features(self, i: int):
        # Match features between left image at frame i-1 and frame i using SIFT + FLANN
        kp1, desc1 = self.sift.detectAndCompute(self.Images_1[i - 1], None)
        kp2, desc2 = self.sift.detectAndCompute(self.Images_1[i], None)
        matches = self.flann.knnMatch(desc1, desc2, k=2)
        thresh, good_matches = 0.7, []
        for m, n in matches:
            if m.distance < thresh * n.distance:
                good_matches.append(m)
        self.__draw_corresponding_points(i, kp1, kp2, good_matches)
        p1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        return p1, p2, kp1, kp2, good_matches

    def compute_disparity(self, left, right):
        # Set up StereoSGBM parameters
        window_size = 5
        min_disp = 0
        num_disp = 16 * 6  # must be divisible by 16
        stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        # Compute disparity and scale to float32 (in pixels)
        disparity = stereo.compute(left, right).astype(np.float32) / 16.0
        return disparity

    def find_transf_pnp(self, i: int):
        # (1) Compute disparity on frame i-1 using StereoSGBM
        left_prev = self.Images_1[i - 1]
        right_prev = self.Images_2[i - 1]
        disparity = self.compute_disparity(left_prev, right_prev)
        
        # (2) Reproject disparity to a dense 3D point cloud using Q matrix
        points_3d_dense = cv.reprojectImageTo3D(disparity, self.Q)
        
        # (3) Match features between frame i-1 and frame i (left images)
        if hasattr(self, 'orb'):
            p1, p2, old_kp, new_kp, matches_2d = self.bf_match_features(i)
        else:
            p1, p2, old_kp, new_kp, matches_2d = self.flann_match_features(i)
        if len(matches_2d) < 5:
            print(f"2D matching insufficient between frames {i-1} and {i}")
            return np.eye(4)
        
        pts_3d_list = []
        pts_2d_list = []
        depths_reproj = []  # depths from reprojected 3D points
        depths_formula = [] # depths computed from disparity using Z = f*B/disparity

        for m in matches_2d:
            pt = old_kp[m.queryIdx].pt  # coordinate in frame i-1
            u = int(round(pt[0]))
            v = int(round(pt[1]))
            # Check bounds
            if u < 0 or u >= points_3d_dense.shape[1] or v < 0 or v >= points_3d_dense.shape[0]:
                continue
            X = points_3d_dense[v, u]
            # Skip if depth is not valid
            if X[2] <= 0 or np.isinf(X[2]) or np.isnan(X[2]):
                continue

            # Append 3D point and its corresponding 2D location in the new frame
            pts_3d_list.append(X)
            pts_2d_list.append(new_kp[m.trainIdx].pt)
            depths_reproj.append(X[2])
            
            # Use the disparity value at (v,u) to compute depth via the stereo formula
            disp_val = disparity[v, u]
            if disp_val <= 0:
                continue
            depth_formula = self.f * self.baseline / disp_val
            depths_formula.append(depth_formula)
        
        pts_3d_list = np.array(pts_3d_list, dtype=np.float32)
        pts_2d_list = np.array(pts_2d_list, dtype=np.float32)
        
        if len(pts_3d_list) < 4:
            print(f"Not enough valid 3D-2D correspondences at frame {i}")
            return np.eye(4)
        
        # (4) Compute a scale factor from the disparity:
        # Compare the median depth from the reprojected 3D points to the median depth
        # computed using the stereo formula. Their ratio will be used to correct the scale.
        if len(depths_reproj) > 0 and len(depths_formula) > 0:
            median_reproj = np.median(depths_reproj)
            median_formula = np.median(depths_formula)
            scale_factor = median_formula / median_reproj
            print(f"Frame {i}: median_reproj = {median_reproj:.3f}, median_formula = {median_formula:.3f}, scale_factor = {scale_factor:.3f}")
        else:
            scale_factor = 1.0
            print(f"Frame {i}: insufficient depth stats; using scale_factor = {scale_factor}")

        # (5) Run PnP with RANSAC using the 3D points from frame i-1 and their 2D correspondences in frame i
        success, rvec, tvec, inliers = cv.solvePnPRansac(
            pts_3d_list,
            pts_2d_list,
            self.K1,
            None,
            iterationsCount=100,
            reprojectionError=3.0,
            confidence=0.99,
            flags=cv.SOLVEPNP_ITERATIVE
        )
        if not success:
            print(f"PnP failed for frame {i}")
            return np.eye(4)
        
        R, _ = cv.Rodrigues(rvec)
        # Multiply the translation vector by the computed scale factor
        tvec = tvec * scale_factor
        T = self.__transform(R, tvec)
        # Return the transformation from frame i-1 to i (inverse if needed by your convention)
        return np.linalg.inv(T)
    
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
