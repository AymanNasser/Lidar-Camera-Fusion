import cv2
import numpy as np 
import os

class Box2D:
    def __init__(self, x, y, w, h, score, classID):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score
        self.classID = classID
        self.enclosing_pcs = []


class VideoData:
    """ Load Image & PointCloud data for a demo video """

    def __init__(self, imgL_dir, lidar_dir):
        self.__P = np.array([[649.65825827, 0, 302.21275333, 0], [0, 656.18334152, 244.27286533, 0], [0, 0, 1, 0]], dtype=np.float32)
        self.__T_velo_to_cam = np.array([[0.08088629, -0.99590131, -0.04047205, -0.15610122], 
                            [0.06293044, 0.04562682, -0.9969744, -0.3785559], 
                            [0.99473472, 0.07809463, 0.06636309, -0.59070911], 
                            [0, 0, 0, 1]], dtype=np.float32)
        self.imgL_dir = imgL_dir
        self.lidar_dir = lidar_dir
        self.__calib = {"P": self.__P, "T_velo_to_cam": self.__T_velo_to_cam}

        self.imgL_filenames = sorted(
            [os.path.join(imgL_dir, filename) for filename in os.listdir(imgL_dir)]
        )
        
        self.lidar_filenames = sorted(
            [os.path.join(lidar_dir, filename) for filename in os.listdir(lidar_dir)]
        )
        # print(self.imgL_filenames)
        assert(len(self.imgL_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.imgL_filenames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.__get_image(index), self.__get_lidar(index)

    def __get_image(self, index):
        assert index < self.num_samples
        imgL_filename = self.imgL_filenames[index]
        return cv2.imread(imgL_filename)
    
    def __get_lidar(self, index):
        assert index < self.num_samples
        lidar_filename = self.lidar_filenames[index]

        scan = np.fromfile(lidar_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def get_calibration(self):
        return self.__calib
