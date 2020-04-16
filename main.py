from tracking import *
from visual_odometry_solution_methods import *
import cv2
import math
import time

def get_tiled_keypoints(img, tile_h, tile_w):
    def get_kps(x, y):
        impatch = img[y:y + tile_h, x:x + tile_w]
        keypoints, des = orb_extraction(impatch)
        for pt in keypoints:
            pt.pt = (pt.pt[0] + x, pt.pt[1] + y)
        if len(keypoints) >= 1:
            return keypoints, des
        return [], []
    h, w = img.shape
    kp_list = []
    des = []

    cnt = 0
    for y in range(0, h, tile_h):
        for x in range(0, w, tile_w):
            kp1, des1 = get_kps(x, y)
            cnt += len(kp1)
            kp_list.append(kp1)
            des.append(des1)
    des_res = []
    kp0 = []
    for k in range(cnt):
        print(np.shape(des[k]))
        if len(des[k]) != 0:
            for m in range(len(des[k])):
                des_res.append(des[k][m])
            kp0.append(kp_list[k])
    des_res2 = np.reshape(des_res, (-1, 32))
    kp0 = kp0[0]
    return kp0, des_res2

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


def get_descriptors(kp_id, kp, des):
    des_res = []
    for i in range(len(kp_id)):
        for j in range(len(kp)):
            if kp_id[i][0] == kp[j].pt[0] and kp_id[i][1] == kp[j].pt[1]:
                if des[j] is None:
                    continue
                for k in range(len(des[j])):
                    des_res.append(des[j][k])
                break
    des_res = np.reshape(des_res, (-1, 32))
    return des_res

def from_imagecoords_to_keypoints(q):
    return_array = []
    for u, v in q:
        return_array.append(cv2.KeyPoint(u, v, 31.0))
    return return_array


#TODO: Make this method generic
def show_image(img1, points1, img2, points2):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255,0,255), (255,255,0),(0,255,255), (125,125,0)]
    cnt = 0
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for u, v in points1:
        cv2.circle(img1, (u, v), 2, colors[cnt%len(colors)], -1, cv2.LINE_AA)
        cnt +=1
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    cnt = 0
    for u, v in points2:
        cv2.circle(img2, (u, v), 3, colors[cnt%len(colors)], -1, cv2.LINE_AA)
        cnt +=1
    cv2.imshow("Time 1", img1)
    cv2.imshow("Time 0", img2)
    cv2.waitKey(250)


def find_intersection(array1, array2):
    nrows, ncols = array1.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [array1.dtype]}
    trackable_points_time0, indx1, indx2 = np.intersect1d(array1.view(dtype),
                                                          array2.view(dtype), return_indices=True)
    trackable_points_time0 = trackable_points_time0.view(array1.dtype).reshape(-1, ncols)
    return trackable_points_time0, indx1, indx2


# Returns the indices as array2[array1]
def extract_indices(array1, array2):
    right_idx = []

    for i in range(len(array1)):
        for j in range(len(array2)):
            if array2[j][0] == array1[i][0] and array2[j][1] == array1[i][1]:
                right_idx.append(j)
                break
    return right_idx


def get3DPointsFrom2D(Pts2D, Pts3D, Pts2DFull):
    points = []
    for i in Pts2DFull:
        points.append(i.pt)
    index = extract_indices(Pts2D, points)
    points3D = Pts3D[index]
    return points3D

def main():
    image_path = "../KITTI_sequence_1/"

    # Load the images of the left and right camera
    leftimages = load_images(os.path.join(image_path, "image_l"))
    rightimages = load_images(os.path.join(image_path, "image_r"))

    # Load K and P from the calibration file
    K_l, P_l, K_r, P_r = load_calib("../KITTI_sequence_1/calib.txt")
    poses = load_poses("../KITTI_sequence_1/poses.txt")

    # Find the keypoints and descriptors of the first image in the left camera

    tMat = np.eye(4)
    for i in range(len(leftimages)-1):

        kp1_l, des1_l = get_tiled_keypoints(leftimages[i],100, 200)#orb_extraction(leftimages[i])
        # print(np.shape(kp1_l), np.shape(des1_l))
        # pikkp, pikdes = orb_extraction(leftimages[i])
        # print("main:")
        # print(pikkp)
        # print(pikdes)
        # print(np.shape(pikkp))
        # print(np.shape(pikdes))

        # Find which keypoints are trackable between left and right image at time = 0
        points_left_right_time0, points_right_time0 = track_keypoints(leftimages[i],rightimages[i], kp1_l)

        # Triangulate the common points of all 3 views
        triangulatedPts = triangulate_points(np.transpose(points_left_right_time0), np.transpose(points_right_time0), P_l, P_r)

        show_image(leftimages[i], points_left_right_time0, rightimages[i], points_right_time0)

        kp2, des2 = get_tiled_keypoints(leftimages[i+1],100, 200)


        des_t1 = get_descriptors(points_left_right_time0, kp1_l, des1_l)
        points_left_right_time0 = from_imagecoords_to_keypoints(points_left_right_time0)
        #print(np.shape(des_t1))

        # PROBLEM HERUNDER
        img1, img2 = get_matches(points_left_right_time0, des_t1, kp2, des2)


        newtriangulated_points = get3DPointsFrom2D(img1, triangulatedPts, points_left_right_time0)
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(newtriangulated_points, img2, K_l, np.array([]))

        rotm = eulerAnglesToRotationMatrix(rvecs)
        trannyMatrix = form_transf(rotm, np.transpose(tvecs))
        tMat = np.matmul(tMat, trannyMatrix)
        # print(i)

    print(tMat)
    print(poses[-1])


if __name__ == '__main__':
    main()
# for i in triangluatedPts1:
#     print(i)


# print(np.shape(trackpoints_left))
# print(np.shape(trackpoints_right))


#
# kp1_r, des1_r = orb_extraction(rightimages[0])
# q1_l, q1_r = get_matches(kp1_l, des1_l, kp1_r, des1_r)
# #img = cv2.drawKeypoints(leftimages[0], kp1, np.array([]), (0, 0, 255))
# #show_image(img)
#
# K_l, P_l, K_r, P_r = load_calib("../KITTI_sequence_1/calib.txt")
# q1_l = np.transpose(q1_l)
# q1_r = np.transpose(q1_r)
#
# triangluatedPts1 = triangulate_points(q1_l, q1_r, P_l, P_r)  # All coordinates have both positive and negative values ?
#
# kp2_l, des2_l = orb_extraction(leftimages[1])
# kp2_r, des2_r = orb_extraction(rightimages[1])
# q2_l, q2_r = get_matches(kp2_l, des2_l, kp2_r, des2_r)
#
# #img = cv2.drawKeypoints(leftimages[0], kp1, np.array([]), (0, 0, 255))
# #show_image(img)
#
# q2_l = np.transpose(q2_l)
# q2_r = np.transpose(q2_r)
#
# triangluatedPts2 = triangulate_points(q2_l, q2_r, P_l, P_r)  # All coordinates have both positive and negative values ?
#
# # print(triangluatedPts1)
# # print(triangluatedPts2)
# #
# # print(np.shape(triangluatedPts1))
# # print(np.shape(triangluatedPts2))
#
# ql_vertical, q2_vertical = get_matches(kp1_l,des1_l,kp2_l,des2_l)
#
# # print(np.shape(ql_vertical))
# # print(np.shape(q2_vertical))
#
#


#for i in range(2,len(leftimages)):


# TRACKING
# ORB Extraction
# Initial Pose Estimation from Previous Frame
# OR
# Initial Pose Estimation via Global Relocalization
# Track Local Map
# New KeyFrame Decision


# LOCAL MAPPING
# KeyFrame Insertion
# Recent MapPoints Culling
# New MapPoint Creation
# Local BA
# Local KeyFrame Culling




# LOOP CLOSURE
# Loop Candidates Detection
# Compute Similarity Transform
# Loop Fusion
# Essential Graph Optimization




# KeyFrame
# Includes a pose
# (Includes camera intrinsics)
# Includes ORB Features

# COVISIBILITY GRAPH


# ESSENTIAL GRAPH


# MAPPOINT