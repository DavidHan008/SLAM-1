from tracking import *
from visual_odometry_solution_methods import *


def show_image(img1, points1, img2, points2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for u, v in points1:
        cv2.circle(img1, (u, v), 5, (0, 0, 255), -1, cv2.LINE_AA)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for u, v in points2:
        cv2.circle(img2, (u, v), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.imshow("left", img1)
    cv2.imshow("right", img2)
    cv2.waitKey()

def find_intersection(array1, array2):
    nrows, ncols = array1.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [array1.dtype]}
    trackable_points_time0, indx1, indx2 = np.intersect1d(array1.view(dtype),
                                                          array2.view(dtype), return_indices=True)
    trackable_points_time0 = trackable_points_time0.view(array1.dtype).reshape(-1, ncols)
    return trackable_points_time0, indx1, indx2

def find_index_from_array1_in_array2(array1, array2):
    right_idx = []

    for i in range(len(array1)):
        for j in range(len(array2)):
            if array2[j][0] == array1[i][0] and array2[j][1] == \
                    array1[i][1]:
                right_idx.append(j)
                break
    return right_idx

leftimages = load_images("../KITTI_sequence_1/image_l")
rightimages = load_images("../KITTI_sequence_1/image_r")
K_l, P_l, K_r, P_r = load_calib("../KITTI_sequence_1/calib.txt")

kp1_l, des1_l = orb_extraction(leftimages[0])

points_left_right_time0, points_right_time0 = track_keypoints(leftimages[0],rightimages[0], kp1_l)
points_left_time0, points_left_time1 = track_keypoints(leftimages[0], leftimages[1], kp1_l)

trackable_points_time0, indx1, indx2 = find_intersection(points_left_right_time0, points_left_time0)

trackpoints_left = points_left_right_time0[indx1]

right_idx = find_index_from_array1_in_array2(trackpoints_left, points_left_right_time0)

trackpoints_right = points_right_time0[right_idx]


triangluatedPts1 = triangulate_points(np.transpose(trackpoints_left), np.transpose(trackpoints_right), P_l, P_r)  # All coordinates have both positive and negative values ?

print(triangluatedPts1)

show_image(leftimages[0],trackpoints_left, rightimages[0], trackpoints_right)





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