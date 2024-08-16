import os, cv2 as cv, numpy as np
from matplotlib import pyplot as plt

class VisualOdometry:
    def __init__(self):
        self.K = None
        self.poses = None
        self.I = [] # Key points
        self.orb=cv.ORB_create(nfeatures=1500)

        # cv.BFMatcher() Brute force matcher could be used but slow
        FLANN_INDEX_KDTREE = 0
        idxPrams = dict(algorithm=FLANN_INDEX_KDTREE, table_number=6, key_size=12, multi_probe_level=1)
        searchPrams = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(idxPrams,searchPrams) #Fast Library for Approximate Nearest Neighbors

    def loadImg(self):
        pass

    def matchFeatures(self, i): # Index, 

        # detectAndCompute: image -> {keypoint, descriptors}
        kp1, desc1 = self.orb.detectAndCompute(self.I[i-1], None)
        kp2, desc2 = self.orb.detectAndCompute(self.I[i], None)

        # Compare the keypoints between the subsequent frames using k nearest neighbors
        # Returns the best match 
        matches = self.flann.knnMatch(desc1, desc2)

        thresh, goodMatch=0.7, []
        # Filters the matches (uses Lowe's ratio test)
        for m, n in matches:
            if m.distance<thresh*n.distance:
                goodMatch.append(m)

        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = None, flags = 0)
        
        # Draw the matches
        img = cv.drawMatchesKnn(self.I[i], kp1, self.I[i-1], kp2, goodMatch, None, **draw_params)
        cv.imshow("img", img)

        # Get the image points from the good matches
        return np.float32(kp1[m.queryIdx].pt for m in goodMatch), np.float32(kp2[m.trainIdx].pt for m in goodMatch)

    def calcPose(self, p1, p2):
        E, mask = cv.findEssentialMat(p1, p2, self.K, threshold=1)

        # Using the epipolar constraint solving for E
        # A^\top vecE = 0
        A = np.kron(p1,p2).tolist()
        e=np.asarray(E).reshape(-1)    

        # Decompose the Rotation matrix and translation vector
        R, t = self.decompEmatrix(E, p1, p2)

        T = np.eye(4)

        # Compute T = (R|t)
        T[:2, :2], T[3, :3] = R, t

        
        # Encode R and t into the transfomation matrix T

    def decompEmatrix(self, E, p1, p2):
   
        R, t = None, None
        return R, t


