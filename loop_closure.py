from orb import *
from keypoint import *
from Point3D import *
from transformation import *


def close_loop(left_image_0, right_image_0, left_image_i, P_left, P_right, K_left):
    key_points_left_time_0, descriptors_left_time_0 = orb_detector_using_tiles(left_image_0,max_number_of_kp=200)
    key_points_right_time_0, descriptors_right_time_0 = orb_detector_using_tiles(right_image_0, max_number_of_kp=200)

    key_points_left_time_i, descriptors_left_time_i = orb_detector_using_tiles(left_image_i, max_number_of_kp=200)

    trackable_keypoints_left_time_0, trackable_keypoints_right_time_0, \
    trackable_descriptors_left_time_0, trackable_descriptors_right_time_0 = track_keypoints_left_to_right_new(
        key_points_left_time_0,
        descriptors_left_time_0, key_points_right_time_0, descriptors_right_time_0, left_image_0, right_image_0)

    relative_triangulated_3D_points_time_0 = triangulate_points_local(trackable_keypoints_left_time_0,
                                                                      trackable_keypoints_right_time_0, P_left, P_right)

    trackable_left_imagecoordinates_time_i, trackable_3D_points_time_0, imagecoords_left_time_0 \
        = find_2D_and_3D_correspondenses(trackable_descriptors_left_time_0, trackable_keypoints_left_time_0,
                                         key_points_left_time_i, descriptors_left_time_i,
                                         relative_triangulated_3D_points_time_0, max_Distance=500)

    close_3D_points_index, far_3D_points_index = sort_3D_points(trackable_3D_points_time_0, close_def_in_m=70)
    # print(len(trackable_3D_points_time_i))
    transformation_matrix = []
    if len(trackable_3D_points_time_0) > 4:
        transformation_matrix, _ ,_  = calculate_transformation_matrix(trackable_3D_points_time_0,
                                                                trackable_left_imagecoordinates_time_i,
                                                                close_3D_points_index, far_3D_points_index, K_left)
    else:
        print("Not enough 3D points to close the lube")
    print("tranny mat ", transformation_matrix)
    return transformation_matrix


def find_error(correct_frame, wrong_frame):
    return correct_frame - wrong_frame


def get_distribution_error(error_frame, index_0, index_i):
    return error_frame / (index_i - index_0)


# Distributes the error over all relevant frames, only translation
def distribute_error(camera_frames, error_frame, index_0, index_i):
    for i in range(index_0, index_i):
        camera_frames[i].pose[:3, 3] += (i - index_0) * error_frame[:3, 3]

    return camera_frames
