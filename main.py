#%%
from tracking import *
from keyframe import *
from visual_odometry_solution_methods import *
from bag_of_words import *
from keypoint import *
from transformation import *
from Point3D import *
from XXXport_files import *
from orb import *
from BundleAdjustment import *
from scipy.spatial.transform import Rotation as R

#%%
def fit_cam_params(camera_frames, P_left):
    cam_params = np.empty((0, 9))
    for i in range(len(camera_frames)):
        cam_param = np.empty((9))

        trans = camera_frames[i].pose
        rotvec = R.from_matrix(trans[:3, :3])
        r = rotvec.as_rotvec()

        cam_param[0] = r[0]
        cam_param[1] = r[1]
        cam_param[2] = r[2]
        cam_param[3] = trans[0, 3]
        cam_param[4] = trans[1, 3]
        cam_param[5] = trans[2, 3]
        cam_param[6] = P_left[0, 0]
        cam_param[7] = 0
        cam_param[8] = 0
        cam_params = np.append(cam_params, cam_param)

    cam_params = cam_params.reshape((-1, 9))
    return cam_params


def write_pose_to_file(file_name, camera_frame):
    f = open(file_name + ".txt", "a")

    f.write(str(camera_frame.pose[0, 3]) + "," + str(camera_frame.pose[1, 3]) + "," + str(
        camera_frame.pose[2, 3]) + ',' + str(camera_frame.pose[0, 0]) + "," + str(camera_frame.pose[0, 1]) + "," + str(
        camera_frame.pose[0, 2]) + ',' + str(camera_frame.pose[1, 0]) + "," + str(camera_frame.pose[1, 1]) + "," + str(
        camera_frame.pose[1, 2]) + ',' + str(camera_frame.pose[2, 0]) + "," + str(camera_frame.pose[2, 1]) + "," + str(
        camera_frame.pose[2, 2]) + "\n")
    f.close()

def main():
    # image_path = "../KITTI_sequence_2/"
    n_images = 100
    image_path = "../data_odometry_gray/dataset/sequences/06/"
    # Load the images of the left and right camera
    leftimages = load_images(os.path.join(image_path, "image_0"))
    # leftimages = leftimages[:n_images]
    rightimages = load_images(os.path.join(image_path, "image_1"))
    # rightimages = rightimages[:100]
    n_clusters = 50
    n_features = 100
    bow_threshold = 100

    bow = BoW(n_clusters, n_features)
    bow.train(leftimages)

    # Load K and P from the calibration file
    K_left, P_left, K_right, P_right = load_calib(image_path+"calib.txt")
    poses = load_poses(image_path+"poses.txt")

    camera_frame = np.eye(4)
    rvec = np.array([0,0,0])
    tvec = np.array([0,0,0])
    camera_frames = []
    camera_frame_pose = np.eye(4)
    camera_frame = KeyFrame(camera_frame_pose)
    camera_frames.append(camera_frame)

    frame_numbers = []
    Qs = np.empty((0, 3))             # 3D points
    observations = []   # An array that includes frameindex, 3Dpoint index and 2D point in that frame

    points_2d = np.empty((0, 2), dtype=int)
    frame_indices = np.empty((0, 1), dtype=int)
    point_indices = np.empty((0, 1), dtype=int)


    clear_textfile("ourCache/path" +str(image_path[-2]) +".txt")
    clear_textfile("ourCache/3DPoints.txt")

    optimization_matrix = np.empty((0,4))        # frame nr, 3d_index and 2d coordinate

    print("Så kører vi sgu")
    offset = 0

#%%
    key_points_left_time_i, descriptors_left_time_i = orb_detector_using_tiles(leftimages[offset],max_number_of_kp=200)
    for i in range(offset, len(leftimages)-1):
        key_points_right_time_i, descriptors_right_time_i = orb_detector_using_tiles(rightimages[i], max_number_of_kp=200)
        key_points_left_time_i1, descriptors_left_time_i1 = orb_detector_using_tiles(leftimages[i+1], max_number_of_kp=200)

        trackable_keypoints_left_time_i, trackable_keypoints_right_time_i, \
        trackable_descriptors_left_time_i, trackable_descriptors_right_time_i = track_keypoints_left_to_right_new(key_points_left_time_i,
                                          descriptors_left_time_i, key_points_right_time_i, descriptors_right_time_i, leftimages[i], rightimages[i])

        relative_triangulated_3D_points_time_i = triangulate_points_local(trackable_keypoints_left_time_i, trackable_keypoints_right_time_i, P_left, P_right)

        trackable_left_imagecoordinates_time_i1, trackable_3D_points_time_i, imagecoords_left_time_i \
            = find_2D_and_3D_correspondenses(trackable_descriptors_left_time_i, trackable_keypoints_left_time_i,
                          key_points_left_time_i1, descriptors_left_time_i1, relative_triangulated_3D_points_time_i, max_Distance=500)

        close_3D_points_index, far_3D_points_index = sort_3D_points(trackable_3D_points_time_i, close_def_in_m=70)

        if len(trackable_3D_points_time_i) > 4:
            transformation_matrix = calculate_transformation_matrix(trackable_3D_points_time_i,
                                                                    trackable_left_imagecoordinates_time_i1,
                                                                    close_3D_points_index, far_3D_points_index, K_left)


        camera_frame_pose = np.matmul(camera_frame.pose, transformation_matrix)
        camera_frame = KeyFrame(camera_frame_pose)
        camera_frames.append(camera_frame)

        absPoint = relative_to_abs3DPoints(trackable_3D_points_time_i, camera_frame.pose)
        # Qs, opt, frame_index = appendKeyPoints(Qs, absPoint, 0.01, imagecoords_left_time_i, i, trackable_3D_points_time_i)
        Qs, opt = appendKeyPoints(Qs, absPoint, 0.01, imagecoords_left_time_i, i, trackable_3D_points_time_i)
        point_ind = opt[:, 1]
        point_indices = np.append(point_indices, point_ind.astype(int))

        optimization_matrix = np.vstack((optimization_matrix,opt))
        save3DPoints("ourCache/3DPoints.txt", absPoint, i)

        write_pose_to_file(os.path.join(image_path, "ourCache/path"), camera_frame)

        key_points_left_time_i = key_points_left_time_i1
        descriptors_left_time_i = descriptors_left_time_i1

        for p in range(len(imagecoords_left_time_i)):
            points_2d = np.append(points_2d, imagecoords_left_time_i[p])

        for j in range(len(absPoint)):
            frame_indices = np.append(frame_indices, i)

        # Make camera_params fit adjust my bundle
    camera_params = fit_cam_params(camera_frames, P_left)

    points_2d = points_2d.reshape((-1, 2))

    res = adjust_my_bundle(camera_params, Qs, frame_indices, point_indices, points_2d)

#%%
        # Save the stuff

if __name__ == '__main__':
    main()


