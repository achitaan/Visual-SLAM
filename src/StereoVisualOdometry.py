import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from numpy.typing import NDArray
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file

class StereoVisualOdometry:
    def __init__(self, folder_path: str, calibration_path: str, use_brute_force: bool):
        self.K1, self.P1, self.K2, self.P2 = self.__calib(filepath=calibration_path)  # Intrinsic/Projection from file

        print("Left Intrinsics:\n", self.K1)
        print("Right Intrinsics:\n", self.K2)

        self.true_poses = self.__load_poses(r"poses\01.txt")
        self.poses = [self.true_poses[0]] 

        # Load left images (Images_1) and right images (Images_2)
        self.Images_1 = self.__load(folder_path + "0")  # Left images
        self.Images_2 = self.__load(folder_path + "1")  # Right images

        # Feature detectors
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
                f.write(f"Pose {i}: \n")
                np.savetxt(f, pose, fmt="%6f")
                f.write("\n")
    
    def __draw_corresponding_points(self, i, kp1, kp2, good_matches):
        draw_params = dict(
            matchColor=-1, 
            singlePointColor=None,
            matchesMask=None,
            flags=2
        )
        image = cv.drawMatches(self.Images[i], kp1, self.Images[i - 1], kp2, good_matches, None, **draw_params)
        cv.imshow("Feature Matches", image)
        cv.waitKey(1)

    def calc_camera_matrix(self, filepath: str) -> NDArray:
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P
    
    def __calib(self, filepath: str) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Reads lines from 'calib.txt' with known labels P1: and P2: 
        Then extracts the left camera (P_l, K_l) and right camera (P_r, K_r).
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith(f"P1:"):
                    params = np.fromstring(line.split(':', 1)[1], dtype=np.float64, sep=' ')
                    P_l = np.reshape(params, (3, 4))
                    K_l = P_l[0:3, 0:3]
                if line.startswith(f"P2:"):
                    params = np.fromstring(line.split(':', 1)[1], dtype=np.float64, sep=' ')
                    P_r = np.reshape(params, (3, 4))
                    K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    def bf_match_features(self, i: int):
        """Matches features between frame (i-1) and frame (i) using ORB + BF Matcher."""
        kp1, desc1 = self.orb.detectAndCompute(self.Images_1[i - 1], None)
        kp2, desc2 = self.orb.detectAndCompute(self.Images_1[i], None)

        matches = self.brute_force.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Optional: draw matches
        self.__draw_corresponding_points(i, kp1, kp2, matches)

        p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return p1, p2, kp1, kp2, matches

    def flann_match_features(self, i: int):
        """Matches features between frame (i-1) and frame (i) using SIFT + FLANN."""
        kp1, desc1 = self.sift.detectAndCompute(self.Images_1[i - 1], None)
        kp2, desc2 = self.sift.detectAndCompute(self.Images_1[i], None)

        matches = self.flann.knnMatch(desc1, desc2, k=2)
        thresh, good_matches = 0.7, []
        for m, n in matches:
            if m.distance < thresh * n.distance:
                good_matches.append(m)

        # Optional: draw matches
        self.__draw_corresponding_points(i, kp1, kp2, good_matches)

        p1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        return p1, p2, kp1, kp2, good_matches

    def get_stereo_matches(self, i: int):
        """
        Matches features between the left and right images at index i for triangulation.
        """
        if i >= len(self.Images_1) or i >= len(self.Images_2):
            print("Index out of range for stereo images.")
            return None, None, None, None, []

        # You can choose ORB or SIFT for left-right matching as well.
        # Here, we do ORB on left vs. right images at time i.
        kp1, desc1 = self.orb.detectAndCompute(self.Images_1[i], None)
        kp2, desc2 = self.orb.detectAndCompute(self.Images_2[i], None)
        matches = self.brute_force.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 5:
            print(f"Not enough stereo matches at index {i}")
            return None, None, None, None, []

        # Points in left image and right image
        p_left  = np.float32([kp1[m.queryIdx].pt for m in matches])
        p_right = np.float32([kp2[m.trainIdx].pt for m in matches])
        return p_left, p_right, kp1, kp2, matches

    def triangulate_from_stereo(self, p_left: NDArray, p_right: NDArray) -> NDArray:
        """
        Given matched keypoints from left/right images (same frame),
        triangulate to obtain 3D points in the LEFT camera coordinate system.
        """
        # Triangulate points => shape: (4, N)
        points_4d_hom = cv.triangulatePoints(self.P1, self.P2, p_left.T, p_right.T)
        # Convert homogeneous to 3D
        points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T  # shape: (N, 3)
        return points_3d

    def find_transf_pnp(self, i: int):
        """
        1) Obtain 3D points from the PREVIOUS frame's stereo images (i-1).
        2) Match those same feature locations from (i-1) left image to the i-th left image => 2D points.
        3) Use PnP to solve for R,t between frame i-1 and i.
        """
        # --- (1) Triangulate 3D from the previous stereo pair (left/right at i-1)
        p_left, p_right, kpL, kpR, stereo_matches = self.get_stereo_matches(i - 1)
        if len(stereo_matches) < 5:
            print(f"Stereo matching insufficient at frame {i-1}")
            return np.eye(4)

        pts_3d_prev = self.triangulate_from_stereo(p_left, p_right)  # 3D in the old (i-1) left-cam frame

        # --- (2) Now match features from the old left image (i-1) to the new left image (i).
        # We want the *same* feature points that we used in stereo, so we track them from (i-1)->(i).
        # For simplicity, let's just do a fresh match (not 1:1 the same indices). In a real pipeline,
        # you would track the same feature IDs if you want a perfect 3D->2D pairing.
        
        if hasattr(self, 'orb'):
            p1, p2, old_kp, new_kp, matches_2d = self.bf_match_features(i)
        else:
            p1, p2, old_kp, new_kp, matches_2d = self.flann_match_features(i)

        if len(matches_2d) < 5:
            print(f"2D matching insufficient between frames {i-1} and {i}")
            return np.eye(4)

        # --- (3) Build correspondences: 
        # in a robust pipeline, you'd ensure p1 lines up with p_left from stereo.
        # We'll do a naive approach: try to find nearest neighbor in old_kp for p_left 
        # so we can align which 3D point corresponds to which 2D point in the new image.
        # This can be improved with e.g. BFS or an ID-based approach, but here is a simple demonstration.

        # We do a nearest search in (p1 = old_kp2D from frame i-1) for each stereo keypoint p_left.
        # Then we pick the corresponding p2 for the new image location.
        # Finally we call solvePnP.
        
        # Convert p1 to a list for searching
        p1_list = np.array(p1)  # shape (N, 2)
        pts_3d_list = []
        pts_2d_list = []

        for idx_st, (lx, ly) in enumerate(p_left):
            # 3D point from stereo
            X3d = pts_3d_prev[idx_st]  # shape (3,)

            # Find the closest point in p1_list => same feature in old image
            dists = np.sqrt((p1_list[:,0] - lx)**2 + (p1_list[:,1] - ly)**2)
            min_idx = np.argmin(dists)
            if dists[min_idx] < 2.0:  # threshold in pixels to accept
                # If close enough, the corresponding new image 2D is p2[min_idx]
                pts_3d_list.append(X3d)
                pts_2d_list.append(p2[min_idx])

        pts_3d_list = np.array(pts_3d_list, dtype=np.float32)
        pts_2d_list = np.array(pts_2d_list, dtype=np.float32)

        if len(pts_3d_list) < 4:
            print(f"Not enough matched 3D->2D correspondences at frame {i}")
            return np.eye(4)

        # --- (4) Run solvePnPRansac
        success, rvec, tvec, inliers = cv.solvePnPRansac(
            pts_3d_list, 
            pts_2d_list, 
            self.K1, 
            distCoeffs=None,
            iterationsCount=100,
            reprojectionError=3.0,
            confidence=0.99,
            flags=cv.SOLVEPNP_ITERATIVE
        )
        if not success:
            print(f"PnP failed for frame {i}")
            return np.eye(4)

        R, _ = cv.Rodrigues(rvec)  # convert rvec to rotation matrix
        T = self.__transform(R, tvec)

        # Return the 4x4 transformation from frame (i-1) to frame (i). 
        # Often we store T^-1 if we want camera i in the world frame, etc. 
        # This depends on your coordinate convention. 
        return np.linalg.inv(T)  # so that post-multiplying by T transforms old->new

    def run_vo(self):
        """
        Example main loop for running PnP-based VO across frames.
        """
        num_frames = len(self.Images_1)
        for i in range(1, num_frames):
            # (1) Find transformation from frame i-1 to i via PnP
            T = self.find_transf_pnp(i)

            # (2) Accumulate pose
            current_pose = self.poses[-1] @ T
            self.poses.append(current_pose)
        
        print("Done running PnP-based visual odometry!")

# Test
if __name__ == "__main__":
    folder_path = r"sequences\01\image_" 
    vo = StereoVisualOdometry(folder_path, r"sequences\01\calib.txt", use_brute_force=False)
    vo.run_vo()
    # This will run your new find_transf_pnp approach on the entire sequence.
