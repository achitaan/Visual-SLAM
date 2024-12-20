import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from numpy.typing import NDArray

class VisualOdometry:
    def __init__(self, folder_path, calibration_path):
        self.K, _ = self.calc_camera_matrix(calibration_path)  # Intrinsic camera matrix (example values)
        self.true_poses = self._load_poses(r"KITTI_sequence_2\poses.txt")
        self.poses = [self.true_poses[0]]  # Initial pose (identity matrix)

        self.I = self._load_img(folder_path) # Key frames (images)
        self.orb = cv.ORB_create(nfeatures=3000)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

    def calc_camera_matrix(self, filepath: str) -> NDArray:
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _transform(R: NDArray[np.float32], t: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Computes the transformation matrix T_k from R_k and t_k

        Parameters:
            R (ndarray): 2D numpy array of shape (3, 3)
            t (ndarray): 1D numpy array of shape (1,)

        Returns:
            T (ndarray): 2D numpy array of shape (4, 4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R; T[:3, 3] = np.squeeze(t)
        return T

    @staticmethod
    def _load_img(filepath: str) -> list[NDArray]:
        """
        Load images from the specified folder
        
        Parameters:
            filepath (str): path to folder
            
        """
        images = []
        for filename in sorted(os.listdir(filepath)):
            path = os.path.join(filepath, filename)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        return images

    def matchFeatures(self, i: int) -> tuple[NDArray, NDArray]:
        kp1, desc1 = self.orb.detectAndCompute(self.I[i - 1], None)
        kp2, desc2 = self.orb.detectAndCompute(self.I[i], None)

        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)

        matches = self.flann.knnMatch(desc1, desc2, k=2)


        thresh, good_matches = 0.7, []
        for m, n in matches:
            if m.distance < thresh * n.distance:
                good_matches.append(m)

        p1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        return p1, p2

    def find_transf_fast(self, p1: NDArray, p2: NDArray) -> NDArray:
        E, mask = cv.findEssentialMat(p1, p2, self.K, method=cv.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv.recoverPose(E, p1, p2, self.K)

        T = self._transform(R, t)

        return np.linalg.inv(T) 

    def find_transf(self, p1: NDArray, p2: NDArray) -> NDArray:
        E, mask = cv.findEssentialMat(p1, p2, self.K, method=cv.RANSAC, prob=0.999, threshold=1.0)
        R1, R2, t = cv.decomposeEssentialMat(E)
        t = np.squeeze(t)

        pairs = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
        P1 = self.K @ np.eye(3, 4)
        
        max_z_count, best_pose = 0, 0
        for R, t in pairs:
            P2 = np.concatenate((self.K, np.zeros((3, 1))), axis=1) @ self._transform(R, t)

            points_4d_hom = cv.triangulatePoints(P1, P2, p1.T, p2.T)
            p1_3d_hom = points_4d_hom[:3] / points_4d_hom[3]
            p2_3d_hom = R @ p1_3d_hom + t.reshape(-1, 1)
            z1, z2 = p1_3d_hom[2], p2_3d_hom[2]

            pos_z_count = np.sum((z1 > 0) & (z2 > 0))
 
            if pos_z_count > max_z_count:
                max_z_count = pos_z_count
                best_pose = (R, t)

        R, t = best_pose
        return np.linalg.inv(self._transform(R, t))

    def triangulation(self, p1, p2):
        P1 = self.K @ np.eye(3, 4)  # First camera projection matrix
        P2 = self.K @ self.poses[-1][:3]  # Second camera projection matrix
        points_4d = cv.triangulatePoints(P1, P2, p1.T, p2.T)
        points_3d = points_4d[:3] / points_4d[3]  # Convert to 3D points
        return points_3d.T
    
    def save_to_txt(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            for i, pose in enumerate(self.poses):
                f.write(f"Pose {i}: \n")
                np.savetxt(f, pose, fmt="%6f")
                f.write("\n")

    def plot_trajectory(self):
        x, y, z = [], [], []
        for pose in self.poses:
            x.append(pose[0, 3])
            y.append(pose[1, 3])
            z.append(pose[2, 3])

        plt.figure()
        plt.plot(x, z, label="Camera Trajectory")  # Plot x vs z (top-down view)
        plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
        plt.xlabel("X (meters)")
        plt.ylabel("Z (meters)")
        plt.title("Visual Odometry Trajectory (To Scale)")
        plt.legend()
        plt.grid()
        plt.show()

    def main(self):
        for i in range(1, len(self.I)):
            p1, p2 = self.matchFeatures(i)

            T = self.find_transf_fast(p1, p2)
            self.poses.append(self.poses[-1] @ T)

            print((self.poses[i][0, 3], self.poses[i][1, 3], self.poses[i][2, 3]))


        print("Visual Odometry completed.")
        self.save_to_txt("poses.txt")
        self.plot_trajectory()


# Example usage
if __name__ == "__main__":
    folder_path = r"KITTI_sequence_2\image_l" 

    vo = VisualOdometry(folder_path, r"KITTI_sequence_2\calib.txt")
    vo.main()
