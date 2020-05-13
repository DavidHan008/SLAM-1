import numpy as np
import cv2
import math

def calculate_transformation_matrix(trackable_3D_points_time_i, trackable_left_imagecoordinates_time_i1,
                                        close_3D_points_index, far_3D_points_index, K_left):
    track2dPoints = np.ascontiguousarray(trackable_left_imagecoordinates_time_i1).reshape(
        (-1, 2))
    track3dPoints = np.ascontiguousarray(trackable_3D_points_time_i).reshape((-1, 3))

    _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(track3dPoints,
                                                                       track2dPoints, K_left,
                                                                       np.zeros(5))

    transformation_matrix = translation_and_rotation_vector_to_matrix(rotation_vector, translation_vector)

    return transformation_matrix



def translation_and_rotation_vector_to_matrix(rotvec, transvec):
    # rotm = eulerAnglesToRotationMatrix(rotvec)
    rotm = np.eye(3)
    cv2.Rodrigues(rotvec, rotm)

    rotm = np.linalg.inv(rotm)
    transvec=-1*transvec
    # print(rotm)
    #transvec[2] = -1 * transvec[2]
    #transvec[1] = -1 * transvec[1]
    #transvec[0] = 1 * transvec[0]
    return form_transf(rotm, np.transpose(transvec))

def form_transf(R, t):
    T = np.eye(4, dtype=np.float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# Calculates Rotation Matrix given euler angles.
# Stolen from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
