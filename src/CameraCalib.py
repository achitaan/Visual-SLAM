import numpy as np, cv2 as cv, os
from numpy.typing import NDArray


class CameraCalib:
    def __init__(self, camera_id, board_size, square_size, frame):
        self.dir = dir
        self.w, self.l = board_size
        self.camera_id, self.images = camera_id, []
        self.frame = frame

        # 
        self.obj = np.zeros((self.w*self.l, 3), np.float32)
        self.obj[:,:2] = np.mgrid[0:self.w,0:self.l].T.reshape(-1,2) * square_size

        # 3d and 2d points
        self.p3d = []
        self.p2d = []
    

    def __save(self, filepath: str, matrix: NDArray[np.float32]) -> None:
        """
        Saves poses to a specified file
        
        Parameters:
            filepath (str): path to file

        """
        with open(filepath, 'w') as f:
            np.savetxt(f, matrix, fmt="%6f")

    def add_image(self) -> bool:
        cap = cv.VideoCapture(self.camera_id)
        assert cap.isOpened

        while True:
            ret, frame = cap.read()

            if not ret:
                return False
            
            cv.imshow("fram", frame)

            if cv.waitKey(1) & 0xFF == ord(" "): 
                image = cv.cvtColor(frame, cv.IMREAD_GRAYSCALE)
                self.images.append(image)
                print(len(self.images))

            elif cv.waitKey(1) & 0xFF == 27: 
                break

        cap.release()
        cv.destroyAllWindows()
        return True

        
    def find_matrix(self, display: bool = False):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for image in self.images:
            ret, corners = cv.findChessboardCorners(image, (self.w, self.l), None)

            print("askdaksdm")
            if ret:
                self.p3d.append(self.obj)
                self.p2d.append(corners)

                if display:
                    corners2 = cv.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
                    cv.drawChessboardCorners(image, (self.w, self.l), corners2, ret)
                    cv.imshow("chess", image)
                    cv.waitKey(400)
            else:
                return False

        cv.destroyAllWindows()

        
        ret, cameraMatrix, dist, _, _ = cv.calibrateCamera(
            self.p3d, self.p2d, self.frame, None, None
            )
        
        print(cameraMatrix)

        self.__save()
        return True


def main():
    c = CameraCalib(0, (24,17), 1, (1440,1080))
    c.add_image()
    print(c.find_matrix(True))

main()

