import numpy as np
import cv2
import json
from numba import vectorize, float32, float64, jit, boolean
from math import sqrt

class Camera():
    """
        Projective camera
    """
    def __init__(self, K, rvec, tvec, distCoef, w, h, cid=-1):
        K_new, roi = cv2.getOptimalNewCameraMatrix(np.float32(K), np.float32(distCoef), (w, h), 0)
        mapx, mapy = cv2.initUndistortRectifyMap(np.float32(K), np.float32(distCoef), None, K_new, (w, h), 5)
        self.K = np.float32(K)
        self.mapx = mapx
        self.mapy = mapy
        self.K_new = np.float32(K_new)
        self.rvec = np.float32(rvec)
        self.tvec = np.float32(tvec)
        self.distCoef = np.float32(distCoef)
        self.P = Geometry.get_projection_matrix(K_new, rvec, tvec)
        self.w = w
        self.h = h
        self.cid = cid
    
    def get_C(self):
        """
        :return: (x, y, z) of the camera center in world coordinates
        """
        R = cv2.Rodrigues(self.rvec)[0]
        tvec = self.tvec
        return -np.transpose(R) @ tvec

    def undistort(self, im):
        """ undistorts the image
        :param im: {h x w x c}
        :return:
        """
        return cv2.remap(im, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def undistort_points(self, points2d):
        """
        :param points2d: [ (x,y,w), ...]
        :return:
        """
        points2d_ = points2d[:, 0:2].astype('float32')
        points2d_ = np.expand_dims(points2d_, axis=1)  # (n, 1, 2)

        result = np.squeeze(cv2.undistortPoints(points2d_, self.K, self.distCoef))
        if len(result.shape) == 1:  # only a single point
            result = np.expand_dims(result, 0)

        fx = self.K_new[0, 0]
        fy = self.K_new[1, 1]
        cx = self.K_new[0, 2]
        cy = self.K_new[1, 2]

        points2d_undist = np.empty_like(points2d)
        for i, (px, py) in enumerate(result):
            points2d_undist[i, 0] = px * fx + cx
            points2d_undist[i, 1] = py * fy + cy
            points2d_undist[i, 2] = points2d[i, 2]

        return points2d_undist

    def projectPoints_undist(self, points3d):
        """
            projects 3d points into 2d ones with
            no distortion
        :param points3d: {n x 3}
        :return:
        """
        pts2d, _ = cv2.projectPoints(points3d,
                                     self.rvec,
                                     self.tvec,
                                     self.K_new, 0)
        pts2d = np.squeeze(pts2d)
        if len(pts2d.shape) == 1:
            pts2d = np.expand_dims(pts2d, axis=0)
        return pts2d

    def projectPoints(self, points3d, withmask=False, binary_mask=True):
        """
            projects 3d points into 2d with
            distortion being considered
        :param points3d: {n x 3}
        :param withmask: {boolean} if True return mask that tells if a point is in the view or not
        :return:
        """
        if withmask:
            return Geometry.reproject_points_to_2d(
                points3d, self.rvec, self.tvec, self.K, self.w, self.h,
                distCoef=self.distCoef, binary_mask=binary_mask)
        else:
            pts2d, _ = cv2.projectPoints(points3d,
                                         self.rvec,
                                         self.tvec,
                                         self.K, self.distCoef)
            pts2d = np.squeeze(pts2d)
            if len(pts2d.shape) == 1:
                pts2d = np.expand_dims(pts2d, axis=0)
            return pts2d
    
    def to_json(self):
        return json.dumps({'K': self.K.tolist(), 'P': self.P.tolist() ,'rvec': self.rvec.tolist(), 'tvec': self.tvec.tolist(), 'distCoef': self.distCoef.tolist(), 'w': self.w, 'h': self.h})


class Geometry():
    """
        Auxiliar methods used by the camera classes
    """
    @staticmethod
    @vectorize([float64(float64,float64,float64,float64,float64)])
    def line_to_point_distance(a,b,c,x,y):
        return abs(a*x + b*y + c) / sqrt(a**2 + b**2)

    @staticmethod
    def get_projection_matrix(K, rvec, tvec):
        """
        generate the projection matrix from its sub-elements
        :param K: camera matirx
        :param rvec: rodrigues vector
        :param tvec: loc vector
        :return:
        """
        R = cv2.Rodrigues(np.float32(rvec))[0]
        Rt = np.zeros((3, 4))
        Rt[:, 0:3] = R
        Rt[:, 3] = tvec
        return K @ Rt

    @staticmethod
    def from_homogeneous(x):
        """
        convert from homogeneous coordinate
        :param x:
        :return:
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if len(x.shape) == 1:
            h = x[-1]
            if h != 0:
                x = x/h
                return x[0:-1]
            else:
                return None
        else:
            assert len(x.shape) == 2
            h = np.expand_dims(x[:, -1], axis=1)
            x = x / h
            return x[:, 0:-1]

    @staticmethod
    def reproject_points_to_2d(pts3d, rvec, tvec, K, w, h,
                            distCoef = np.zeros((5, 1)),binary_mask=False):
        """
        :param pts3d:
        :param rvec:
        :param tvec:
        :param K:
        :param w:
        :param h:
        :param distCoef:to match OpenCV API
        :return:
        """
        if len(pts3d) == 0:
            return [], []
        Pts3d = pts3d.astype('float32')
        pts2d, _ = cv2.projectPoints(Pts3d, rvec, tvec, K, distCoef)
        pts2d = np.squeeze(pts2d)
        if len(pts2d.shape) == 1:
            pts2d = np.expand_dims(pts2d, axis=0)

        x = pts2d[:, 0]
        y = pts2d[:, 1]

        mask = (x > 0) * 1
        mask *= (x < w) * 1
        mask *= (y > 0) * 1
        mask *= (y < h) * 1

        if not binary_mask:
            mask = np.nonzero(mask)

        return pts2d, mask