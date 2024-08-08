#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg', 'wxAgg', etc.


# from mpl_toolkits.mplot3d.tool import OrbitingViewTool
def rot_z(yaw):
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return rz


def rot_y(pitch):
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    return ry


def rot_x(roll):
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    return rx


def rpy(roll, pitch, yaw):
    r = np.einsum('ij,jk,km', rot_z(yaw), rot_y(pitch), rot_x(roll))
    return r


def rpy_config2matrix(rpy_config):
    roll, pitch, yaw, x, y, z = rpy_config
    rotation = rpy(roll, pitch, yaw)
    zero = np.array([[0, 0, 0]])
    d = np.array([[x], [y], [z], [1]])
    a = np.vstack([rotation, zero])
    h = np.hstack([a, d])
    return h


def inverse_homo(H):
    R = np.transpose(H[0:3, 0:3])
    d = - np.matmul(R, H[0:3, 3]).reshape((3, 1))
    zero = np.array([[0, 0, 0, 1]])
    h = np.hstack((R, d))
    T = np.vstack([h, zero])
    return T


class Kinematics:
    def __init__(self, aruco_data):
        self.aruco_data = aruco_data
        self.franka_base = np.linalg.multi_dot([rpy_config2matrix([0, np.pi / 2, 0, 0.1, 0, 0.1]),
                                                rpy_config2matrix([0, 0, -np.pi / 2, 0, 0, 0])])

        self.base_frame, self.base_frame_inverse, self.object_frames = self.find_frames()

    def find_frames(self):
        aruco_ids = self.aruco_data['ids']
        base_frame = np.eye(4)
        object_frames = []
        for i, info in enumerate(aruco_ids):
            if info == 49:
                base_frame = self.aruco_data['ht'][i]
            else:
                object_frames.append(np.array(self.aruco_data['ht'][i]).reshape((4, 4)))
        base_frame = np.array(base_frame).reshape((4, 4))
        base_frame_inverse = inverse_homo(base_frame)
        return base_frame, base_frame_inverse, object_frames

    def transformed_matrix(self, ht):
        return np.linalg.multi_dot([self.franka_base, self.base_frame_inverse, ht])

    def transformed_points(self, points):
        T = np.eye(4)
        T[:3, 3] = points
        transformed = np.linalg.multi_dot([self.franka_base, self.base_frame_inverse, T])
        return transformed[:3, 3]
