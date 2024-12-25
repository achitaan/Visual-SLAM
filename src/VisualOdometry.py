import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from numpy.typing import NDArray
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file

class VisualOdometry:
    def __init__(self, folder_path, calibration_path):
        self.K, _ = self.calc_camera_matrix(calibration_path)  # Intrinsic camera matrix (example values)
        self.true_poses = self.__load_poses(r"KITTI_sequence_2\poses.txt")
        self.poses = [self.true_poses[0]]  # Initial pose (identity matrix)

        self.I = self.__load(folder_path) # Key frames (images)
        self.orb = cv.ORB_create(nfeatures=3000)

        FLANN_INDEX_KDTREE = 0 # float32
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
    def __load(filepath: str) -> list[NDArray]:
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

    def __save(self, filepath: str) -> None:
        """
        Saves poses to a specified file
        
        Parameters:
            filepath (str): path to file

        """
        with open(filepath, 'w') as f:
            for i, pose in enumerate(self.poses):
                f.write(f"Pose {i}: \n")
                np.savetxt(f, pose, fmt="%6f")
                f.write("\n")
    
    def calc_camera_matrix(self, filepath: str) -> NDArray:
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P
    
    def matchFeatures(self, i: int) -> tuple[NDArray, NDArray]:
        """
        Finds and matches the coresponding consistent points between images I_k-1 and I_k

        Parameters:
            i (int): image index

        Returns:
            p1 (ndarray): numpy array of points in the previous image
            p2 (ndarray): numpy array of the coresponding subsequent points
        """
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

        draw_params = dict(matchColor = -1, # draw matches in green color
            singlePointColor = None,
            matchesMask = None, # draw only inliers
            flags = 2)
        
        image = cv.drawMatches(self.I[i], kp1, self.I[i-1],kp2, good_matches ,None,**draw_params)
        cv.imshow("image", image)
        cv.waitKey(10)


        p1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        return p1, p2

    def find_transf_fast(self, p1: NDArray, p2: NDArray) -> NDArray:
        """
        Quickly finds the transformation matrix from points p1 and p2

        Parameters:
            p1 (ndarray): numpy array of points in the previous image
            p2 (ndarray): numpy array of the coresponding subsequent points

        Returns:
            T (ndarray): 2D numpy array of shape (4, 4)
        """
        E, mask = cv.findEssentialMat(p1, p2, self.K, method=cv.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv.recoverPose(E, p1, p2, self.K)

        T = self.__transform(R, t)

        return np.linalg.inv(T) 

    def find_transf(self, p1: NDArray, p2: NDArray) -> NDArray:
        """
        Finds the most accurate transformation matrix from points p1 and p2

        Parameters:
            p1 (ndarray): numpy array of points in the previous image
            p2 (ndarray): numpy array of the coresponding subsequent points

        Returns:
            T (ndarray): 2D numpy array of shape (4, 4)
        """
        E, mask = cv.findEssentialMat(p1, p2, self.K, method=cv.RANSAC, prob=0.999, threshold=1.0)
        R1, R2, t = cv.decomposeEssentialMat(E)
        t = np.squeeze(t)

        pairs = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
        P1 = self.K @ np.eye(3, 4)
        
        max_z_count, best_pose = 0, 0
        for R, t in pairs:
            P2 = np.concatenate((self.K, np.zeros((3, 1))), axis=1) @ self.__transform(R, t)

            points_4d_hom = cv.triangulatePoints(P1, P2, p1.T, p2.T)
            p1_3d_hom = points_4d_hom[:3] / points_4d_hom[3]
            p2_3d_hom = R @ p1_3d_hom + t.reshape(-1, 1)
            z1, z2 = p1_3d_hom[2], p2_3d_hom[2]

            pos_z_count = np.sum((z1 > 0) & (z2 > 0))
 
            if pos_z_count > max_z_count:
                max_z_count = pos_z_count
                best_pose = (R, t)

        R, t = best_pose
        return np.linalg.inv(self.__transform(R, t))

    def plot(self) -> None:
        x, y, z = [], [], []
        for pose in self.poses:
            x.append(pose[0, 3])
            y.append(pose[1, 3])
            z.append(pose[2, 3])

        # Create a Bokeh figure
        output_file("trajectory.html", title="Visual Odometry Trajectory")  # Save the plot to an HTML file
        p = figure(title="Visual Odometry Trajectory (To Scale)", x_axis_label="X (meters)", y_axis_label="Z (meters)", width=800, height=600)
        p.line(x, z, legend_label="Camera Trajectory", line_width=2)
        p.scatter(x, z, size=5, color="red", legend_label="Key Points")

        p.legend.location = "top_left"
        p.legend.title = "Legend"
        p.grid.grid_line_alpha = 0.3

        # Show the plot
        show(p)

    def main(self) -> None:
        for i in range(1, len(self.I)):
            p1, p2 = self.matchFeatures(i)

            T = self.find_transf_fast(p1, p2)
            self.poses.append(self.poses[-1] @ T)

        print("Visual Odometry completed.")
        self.__save("poses.txt")
        self.plot()


# Test
if __name__ == "__main__":
    folder_path = r"KITTI_sequence_2\image_l" 

    vo = VisualOdometry(folder_path, r"KITTI_sequence_2\calib.txt")
    vo.main()
