import numpy as np
import cv2

# Returns a mask of all valid points
def sort_3D_points(triangulated_3D_point, close_def_in_m = 20, far_def_in_m = 20):

    close_3D_point = [(abs(x[0]) < close_def_in_m and abs(x[1]) < close_def_in_m and abs(x[2]) < close_def_in_m) for x in triangulated_3D_point]
    far_3D_points = np.bitwise_not(close_3D_point)
    return close_3D_point, far_3D_points



def triangulate_points_local(qs_l, qs_r, P_l, P_r):
    qs_l = np.transpose(qs_l)
    qs_r = np.transpose(qs_r)
    #print(np.shape(qs_l), np.shape(qs_r))
    hom_Qs = cv2.triangulatePoints(P_l, P_r, qs_l, qs_r)
    return np.transpose(hom_Qs[:3] / hom_Qs[3])


def relative_to_abs3DPoints(points3D, camera_frame):
    # Update the 3D points
    ones = np.ones((np.shape(points3D)[0], 1))

    homogen_points = np.hstack((points3D, ones))

    abs_3D_points = np.matmul(camera_frame,np.transpose(homogen_points))

    return np.transpose(abs_3D_points[:3] / abs_3D_points[3])


def find_2D_and_3D_correspondenses(descriptors_time_i, keypoints_left_time_i,  keypoints_left_time_i1, descriptors_left_time_i1, triangulated_3D_points):
    # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    FLANN_INDEX_LSH = 6
    max_Distance = 200
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    matches = flann.knnMatch(descriptors_time_i, descriptors_left_time_i1, k=2)
    good = []
    try:
        for m, n in matches:
            if m.distance < 0.7 * n.distance: #0.7
                if abs(triangulated_3D_points[m.queryIdx, 0]) < max_Distance and abs(triangulated_3D_points[m.queryIdx, 1]) < max_Distance and \
                        abs(triangulated_3D_points[m.queryIdx, 2]) < max_Distance:
                    good.append(m)
    except ValueError:
        pass
    Q1 = np.asarray([triangulated_3D_points[m.queryIdx] for m in good])
    q1 = np.asarray([keypoints_left_time_i[m.queryIdx] for m in good])
    q2 = np.asarray([keypoints_left_time_i1[m.trainIdx].pt for m in good])

    return q2, Q1, q1