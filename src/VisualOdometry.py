# changes i gotta do: Add pnp since it is most likely better for monocular

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from numpy.typing import NDArray
from kitti import image_paths, save_poses_txt
from image_sequence import ImageSequence

class VisualOdometry:
    def __init__(
        self,
        folder_path: str,
        calibration_path: str,
        use_brute_force: bool,
        camera_id: int = 1,
        draw_matches: bool = True,
        max_frames: int | None = None,
    ):
        self.K, self.P = self.__calib(camera_id=camera_id, filepath=calibration_path)  # Intrinsic camera matrix (example values)
        self.draw_matches = draw_matches

        # Ground truth is for evaluation only, never estimator initialization.
        self.true_poses = []
        self.poses = [np.eye(4)]
        self.Images = self.__load(folder_path, max_frames)


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
    def __load(filepath: str, max_frames: int | None = None):
        return ImageSequence(filepath, max_frames)

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

    def save_poses(self, filepath: str) -> None:
        save_poses_txt(filepath, self.poses)
    
    def __draw_corresponding_points(self, i, kp1, kp2, good_matches):
        draw_params = dict(
            matchColor=-1,  # Draw matches in green color
            singlePointColor=None,
            matchesMask=None,  # Draw only inliers
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
    
    def __calib(self, camera_id: int, filepath: str) -> tuple[NDArray, NDArray]:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Prefer labeled format (e.g., "P1: ..."), otherwise fall back to line index.
        for line in lines:
            if line.startswith(f"P{camera_id}:"):
                params = np.fromstring(line.split(':', 1)[1], dtype=np.float64, sep=' ')
                P = np.reshape(params, (3, 4))
                K = P[0:3, 0:3]
                return K, P

        if len(lines) > camera_id:
            params = np.fromstring(lines[camera_id], dtype=np.float64, sep=' ')
        else:
            params = np.fromstring(lines[0], dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
        return K, P

    def __relative_scale(): pass

    def bf_match_features(self, i: int) -> tuple[NDArray, NDArray]:
        """
        Finds and matches the coresponding consistent points between images I_k-1 and I_k using a brute force approach

        Parameters:
            i (int): image index

        Returns:
            p1 (ndarray): numpy array of points in the previous image
            p2 (ndarray): numpy array of the coresponding subsequent points
        """
        kp1, desc1 = self.orb.detectAndCompute(self.Images[i - 1], None)
        kp2, desc2 = self.orb.detectAndCompute(self.Images[i], None)

        matches = self.brute_force.match(desc1, desc2) if desc1 is not None and desc2 is not None else []

        if self.draw_matches:
            self.__draw_corresponding_points(i, kp1, kp2, matches)
        p1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        p2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        return p1, p2

    def flann_match_features(self, i: int, return_debug: bool = False):
        """
        Finds and matches the coresponding consistent points between images I_k-1 and I_k

        Parameters:
            i (int): image index

        Returns:
            p1 (ndarray): numpy array of points in the previous image
            p2 (ndarray): numpy array of the coresponding subsequent points
        """
        kp1, desc1 = self.sift.detectAndCompute(self.Images[i - 1], None)
        kp2, desc2 = self.sift.detectAndCompute(self.Images[i], None)

        matches = self.flann.knnMatch(desc1, desc2, k=2) if desc1 is not None and desc2 is not None and len(desc2) >= 2 else []

        thresh, good_matches = 0.7, []
        for pair in matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < thresh * n.distance:
                good_matches.append(m)

        if self.draw_matches:
            self.__draw_corresponding_points(i, kp1, kp2, good_matches)
        p1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        p2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        if not return_debug:
            return p1, p2
        return p1, p2, kp1, kp2, desc1, desc2, good_matches

    def extract_features(self, i: int):
        if hasattr(self, "orb"):
            return self.orb.detectAndCompute(self.Images[i], None)
        return self.sift.detectAndCompute(self.Images[i], None)

    def find_transf_fast(self, p1: NDArray, p2: NDArray) -> NDArray:
        return self.find_transf(p1, p2)

    def find_transf(self, p1: NDArray, p2: NDArray, return_debug: bool = False,
                    use_scale_fix: bool = False):
        """Estimate T_previous_current; monocular translation has unit scale.

        OpenCV recovers X_current = R X_previous + t. Invert that transform
        for camera-to-world pose accumulation. Ground truth never sets scale.
        """
        p1, p2 = np.asarray(p1, dtype=np.float32).reshape(-1, 2), np.asarray(p2, dtype=np.float32).reshape(-1, 2)
        if p1.shape != p2.shape:
            raise ValueError("Matched point arrays must have equal shape")
        debug = {"num_matches": len(p1), "num_inliers": 0, "inlier_ratio": 0.0,
                 "inlier_mask": [False] * len(p1), "relative_scale": 1.0,
                 "scale_observable": False, "tracking_ok": False}
        transform = np.eye(4)
        if len(p1) >= 8 and np.isfinite(p1).all() and np.isfinite(p2).all():
            if np.median(np.linalg.norm(p2 - p1, axis=1)) > 1e-3:
                essential, mask = cv.findEssentialMat(p1, p2, self.K, method=cv.RANSAC,
                                                       prob=0.999, threshold=1.0)
                if essential is not None and essential.shape == (3, 3):
                    count, rotation, translation, mask = cv.recoverPose(essential, p1, p2, self.K, mask=mask)
                    if count >= 8 and np.isfinite(rotation).all() and np.isfinite(translation).all():
                        transform = np.linalg.inv(self.__transform(rotation, translation))
                        debug.update({"num_inliers": int(count), "inlier_ratio": count / len(p1),
                                      "inlier_mask": [bool(v) for v in mask.ravel()],
                                      "best_R": rotation, "best_t": translation.reshape(3), "tracking_ok": True})
        return (transform, debug) if return_debug else transform

    def triangulate_points(self, p1: NDArray, p2: NDArray, R: NDArray, t: NDArray, inlier_mask=None, return_indices=False):
        P1 = self.K @ np.eye(3, 4)
        P2 = np.concatenate((self.K, np.zeros((3, 1))), axis=1) @ self.__transform(R, t)
        indices = np.arange(len(p1))
        if inlier_mask is not None:
            mask = np.array(inlier_mask, dtype=bool)
            p1 = p1[mask]
            p2 = p2[mask]
            indices = indices[mask]
        if len(p1) < 2:
            return ([], []) if return_indices else []
        points_4d_hom = cv.triangulatePoints(P1, P2, p1.T, p2.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T
        # Keep points with positive depth in both cameras
        p2_cam = (R @ points_3d.T + t.reshape(-1, 1)).T
        valid = np.isfinite(points_3d).all(axis=1) & (points_3d[:, 2] > 0) & (p2_cam[:, 2] > 0)
        points = list(points_3d[valid])
        return (points, indices[valid].tolist()) if return_indices else points


# Test
if __name__ == "__main__":
    folder_path = r"sequences\01\image_0" 
    #folder_path = r"KITTI_sequence_2\image_l"

    vo = VisualOdometry(folder_path, r"sequences\01\calib.txt", False)
    #vo = VisualOdometry(folder_path, r"KITTI_sequence_2\calib.txt", False)
    vo.main()
