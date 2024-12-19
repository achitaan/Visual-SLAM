import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
class VisualOdometry:
    def __init__(self, calibration_path):
        self.K, _ = self.calc_camera_matrix(calibration_path)  # Intrinsic camera matrix (example values)
        self.true_poses = self._load_poses(r"KITTI_sequence_2\poses.txt")
        self.poses = [self.true_poses[0]]  # Initial pose (identity matrix)

        self.I = []  # Key frames (images)
        self.orb = cv.ORB_create(nfeatures=1500)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

    def calc_camera_matrix(self, file_path: str) -> np.ndarray:
        with open(file_path, 'r') as f:
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


    def load_img(self, folder_path):
        """Load images from the specified folder."""
        for filename in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, filename)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                self.I.append(img)

    def matchFeatures(self, i):
        kp1, desc1 = self.orb.detectAndCompute(self.I[i - 1], None)
        kp2, desc2 = self.orb.detectAndCompute(self.I[i], None)

        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)

        #print(f"desc1 type: {desc1.dtype}, shape: {desc1.shape}")
        #print(f"desc2 type: {desc2.dtype}, shape: {desc2.shape}")

        matches = self.flann.knnMatch(desc1, desc2, k=2)


        thresh, good_matches = 0.7, []
        for m, n in matches:
            if m.distance < thresh * n.distance:
                good_matches.append(m)

        p1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        #print("done")
        return p1, p2

    def find_pose(self, p1, p2):
        E, mask = cv.findEssentialMat(p1, p2, self.K, method=cv.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv.recoverPose(E, p1, p2, self.K)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.squeeze(t)

        self.poses.append(self.poses[-1] @ np.linalg.inv(T))

    def triangulation(self, p1, p2):
        P1 = self.K @ np.eye(3, 4)  # First camera projection matrix
        P2 = self.K @ self.poses[-1][:3]  # Second camera projection matrix
        points_4d = cv.triangulatePoints(P1, P2, p1.T, p2.T)
        points_3d = points_4d[:3] / points_4d[3]  # Convert to 3D points
        return points_3d.T
    
    def save_to_txt(self, file_name):
        with open(file_name, 'w') as f:
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

    def main(self, folder_path):
        self.load_img(folder_path)

        for i in range(1, len(self.I)):
            p1, p2 = self.matchFeatures(i)
            #print(f"self.K shape: {self.K.shape}, dtype: {self.K.dtype}")

            self.find_pose(p1, p2)
            points_3d = self.triangulation(p1, p2)
            #print(points_3d)
            #print(f"self.points shape: {points_3d.shape}, dtype: {points_3d.dtype}")

            print((self.poses[i][0, 3], self.poses[i][1, 3], self.poses[i][2, 3]))
            # Visualization (optional): Display trajectory or 3D points
            #print(f"Frame {i}: Estimated pose:\n{self.poses[-1]}")

        print("Visual Odometry completed.")
        self.save_to_txt("poses.txt")
        self.plot_trajectory()


# Example usage
if __name__ == "__main__":
    vo = VisualOdometry(r"KITTI_sequence_2\calib.txt")
    folder_path = r"KITTI_sequence_2\image_l" 
    vo.main(folder_path)
