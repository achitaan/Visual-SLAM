import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from numpy.typing import NDArray
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file

class StereoVisualOdometry:
    def __init__(self, folder_path: str, calibration_path: str, use_brute_force: bool):
        self.K1, self.P1, self.K2, self.P2 = self.__calib(filepath=calibration_path)  # Intrinsic camera matrix (example values)

        print("Left Intrinsics:\n", self.K1)
        print("Right Intrinsics:\n", self.K2)

        self.true_poses = self.__load_poses(r"poses\01.txt")
        self.poses = [self.true_poses[0]] 
        self.Images_1 = self.__load(folder_path+"0")  # Left images
        self.Images_2 = self.__load(folder_path+"1")  # Right images


        # Correct Initializations 
        self.__init_orb() if use_brute_force else self.__init_sift()

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
        T[:3, :3] = R; T[:3, 3] = np.squeeze(t)
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
    
    def __calib(self, filepath: str) -> tuple[NDArray, NDArray]:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith(f"P1:"):  # Change this to the appropriate projection matrix
                    params = np.fromstring(line.split(':', 1)[1], dtype=np.float64, sep=' ')
                    P_l = np.reshape(params, (3, 4))
                    K_l = P_l[0:3, 0:3]
                if line.startswith(f"P2:"):  # Change this to the appropriate projection matrix
                    params = np.fromstring(line.split(':', 1)[1], dtype=np.float64, sep=' ')
                    P_r = np.reshape(params, (3, 4))
                    K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    def bf_match_features(self, i: int) -> tuple[NDArray, NDArray]:
        kp1, desc1 = self.orb.detectAndCompute(self.Images[i - 1], None)
        kp2, desc2 = self.orb.detectAndCompute(self.Images[i], None)

        matches = self.brute_force.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        self.__draw_corresponding_points(i, kp1, kp2, matches)
        p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return p1, p2, matches

    def flann_match_features(self, i: int) -> tuple[NDArray, NDArray]:
        kp1, desc1 = self.sift.detectAndCompute(self.Images[i - 1], None)
        kp2, desc2 = self.sift.detectAndCompute(self.Images[i], None)

        matches = self.flann.knnMatch(desc1, desc2, k=2)
        matches = sorted(matches, key=lambda x: x.distance)

        thresh, good_matches = 0.7, []
        for m, n in matches:
            if m.distance < thresh * n.distance:
                good_matches.append(m)

        self.__draw_corresponding_points(i, kp1, kp2, good_matches)
        p1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        return p1, p2, matches
    
    def get_stereo_matches(self, i: int):
        if i >= len(self.Images_1) or i >= len(self.Images_2):
            print("Index out of range for stereo images.")
            return None, None, None

        # Detect keypoints & descriptors
        if hasattr(self, 'orb'):
            p1, p2, matches = self.bf_match_features(i)
        else:
            p1, p2, matches = self.flann_match_features(i)

        if len(matches) < 5:
            print(f"Not enough stereo matches at index {i}")
            return p1, p2, []

        # Optionally, draw the matches for debugging
        self.__draw_corresponding_points(i, p1, p2, matches)
        return p1, p2, matches


    def compute_depth(self, p1, p2, scale_factor: float = 1.0) -> NDArray:
        p1, p2, matches = self.get_stereo_matches(i)
        if not matches:
            return np.array([])

        # Triangulate points (shape: (4, N))
        points_4d_hom = cv.triangulatePoints(self.P1, self.P2, p1.T, p2.T)
        # Convert from homogeneous to 3D
        points_3d = points_4d_hom[:3] / points_4d_hom[3]
        
        # Depth array (Z) from camera perspective
        depth = points_3d[2] * scale_factor
        
        return depth

    def find_transf_fast(self, p1: NDArray, p2: NDArray) -> NDArray:
        E, mask = cv.findEssentialMat(p1, p2, self.K1, method=cv.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv.recoverPose(E, p1, p2, self.K1)

        # Normalize the translation vector
        t = t / np.linalg.norm(t)

        T = self.__transform(R, t)
        return np.linalg.inv(T)

    def find_transf(self, p1: NDArray, p2: NDArray) -> NDArray:
        E, mask = cv.findEssentialMat(p1, p2, self.K1, method=cv.RANSAC, prob=0.999, threshold=1.0)
        R1, R2, t = cv.decomposeEssentialMat(E)
        t = np.squeeze(t)

        pairs = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
        P1 = self.K @ np.eye(3, 4)
        
        max_z_count, best_pose = 0, 0
        relative_scale = 1.0  # Default scale factor

        for R, t in pairs:
            P2 = np.concatenate((self.K1, np.zeros((3, 1))), axis=1) @ self.__transform(R, t)

            points_4d_hom = cv.triangulatePoints(self.P, P2, p1.T, p2.T)
            p1_3d_hom = points_4d_hom[:3] / points_4d_hom[3]
            p2_3d_hom = R @ p1_3d_hom + t.reshape(-1, 1)
            z1, z2 = p1_3d_hom[2], p2_3d_hom[2]

            pos_z_count = np.sum((z1 > 0) & (z2 > 0))
            
            # Calculate relative scale using only points with positive depth
            if pos_z_count > max_z_count:
                max_z_count = pos_z_count
                best_pose = (R, t)
                
                # Filter points with positive depth
                valid_points = (z1 > 0) & (z2 > 0)
                if np.sum(valid_points) > 1:  # Need at least 2 points to compute distances
                    p1_valid = p1_3d_hom[:, valid_points]
                    p2_valid = p2_3d_hom[:, valid_points]

                    # Compute distances between corresponding points
                    dist_p1 = np.linalg.norm(p1_valid, axis=0)  # Distances from origin in frame 1
                    dist_p2 = np.linalg.norm(p2_valid, axis=0)  # Distances from origin in frame 2

                    # Avoid division by zero or invalid distances
                    valid_distances = (dist_p1 > 1e-6) & (dist_p2 > 1e-6)
                    if np.sum(valid_distances) > 0:
                        relative_scale = np.median(dist_p1[valid_distances] / dist_p2[valid_distances])
                    else:
                        relative_scale = 1.0  # Fallback to default scale
                else:
                    relative_scale = 1.0  # Fallback to default scale

        R, t = best_pose
        t = t * relative_scale  # Scale the translation vector
        return np.linalg.inv(self.__transform(R, t))


# Test
if __name__ == "__main__":
    folder_path = r"sequences\01\image_0" 
    #folder_path = r"KITTI_sequence_2\image_l"

    vo = StereoVisualOdometry(folder_path, r"sequences\01\calib.txt", False)
    #vo = VisualOdometry(folder_path, r"KITTI_sequence_2\calib.txt", False)
