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
from loop_closure import *

#TODO: Make this method generic
def show_image(img1, points1, img2, points2):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255,0,255), (255,255,0),(0,255,255), (125,125,0)]
    cnt = 0
    imgfirst = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for u, v in points1:
        cv2.circle(imgfirst, (u, v), 2, colors[cnt%len(colors)], -1, cv2.LINE_AA)
        cnt +=1
    imgsecond = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    cnt = 0
    for u, v in points2:
        cv2.circle(imgsecond, (u, v), 3, colors[cnt%len(colors)], -1, cv2.LINE_AA)
        cnt +=1
    cv2.imshow("Time 1", imgfirst)
    cv2.imshow("Time 0", imgsecond)
    cv2.waitKey(250)



def main():
    # image_path = "../KITTI_sequence_2/"
    image_path = "../data_odometry_gray/dataset/sequences/06/"
    # Load the images of the left and right camera
    leftimages = load_images(os.path.join(image_path, "image_0"))
    rightimages = load_images(os.path.join(image_path, "image_1"))
    n_clusters = 50
    n_features = 100
    bow_threshold = 100

    #
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
    Qs = []             # 3D points
    observations = []   # An array that includes frameindex, 3Dpoint index and 2D point in that frame


    clear_textfile("ourCache/path" +str(image_path[-2]) +".txt")
    clear_textfile("ourCache/3DPoints.txt")

    optimization_matrix = np.empty((0,4))        # frame nr, 3d_index and 2d coordinate
    # for i in range(len(leftimages)):
    #     leftimages[i] = cv2.flip(leftimages[i],1)
    #     rightimages[i] = cv2.flip(rightimages[i],1)

    print("FU")
    offset = 0
    key_points_left_time_i, descriptors_left_time_i = orb_detector_using_tiles(leftimages[offset],max_number_of_kp=200)
    for i in range(offset, len(leftimages)-1):
        # print(i,"/",len(leftimages))
        # ----------------- TRACKING AND LOCAL MAPPING -------------------- #
        key_points_right_time_i, descriptors_right_time_i = orb_detector_using_tiles(rightimages[i],max_number_of_kp=200)
        key_points_left_time_i1, descriptors_left_time_i1 = orb_detector_using_tiles(leftimages[i+1],max_number_of_kp=200)

        trackable_keypoints_left_time_i, trackable_keypoints_right_time_i, \
        trackable_descriptors_left_time_i, trackable_descriptors_right_time_i = track_keypoints_left_to_right_new(key_points_left_time_i,
                                          descriptors_left_time_i, key_points_right_time_i, descriptors_right_time_i, leftimages[i], rightimages[i])

        relative_triangulated_3D_points_time_i = triangulate_points_local(trackable_keypoints_left_time_i, trackable_keypoints_right_time_i, P_left, P_right)

        trackable_left_imagecoordinates_time_i1, trackable_3D_points_time_i, imagecoords_left_time_i \
            = find_2D_and_3D_correspondenses(trackable_descriptors_left_time_i, trackable_keypoints_left_time_i,
                          key_points_left_time_i1, descriptors_left_time_i1, relative_triangulated_3D_points_time_i, max_Distance=500)

        close_3D_points_index, far_3D_points_index = sort_3D_points(trackable_3D_points_time_i, close_def_in_m=70)
        # print(len(trackable_3D_points_time_i))
        if len(trackable_3D_points_time_i) > 4:
            transformation_matrix = calculate_transformation_matrix(trackable_3D_points_time_i,
                                                                    trackable_left_imagecoordinates_time_i1,
                                                                    close_3D_points_index, far_3D_points_index, K_left)

        idx, val = bow.predict_previous(leftimages[i], i, bow_threshold)
        if val < 45 and val > 0:
            print("Frame: ", i, ". Val: ", val, ". idx: ", idx)
            bow_threshold = i + 100
            new_transformation_mat = close_loop(leftimages[idx], rightimages[idx], leftimages[i], P_left, P_right,
                                                K_left)

            frame_pose_idx = camera_frames[idx].pose
            camera_frame_pose = np.matmul(frame_pose_idx, new_transformation_mat)

            wrong_frame = np.matmul(camera_frame.pose, transformation_matrix)
            error_frame = find_error(camera_frame_pose, wrong_frame)
            error_frame = get_distribution_error(error_frame, idx, i)
            camera_frames = distribute_error(camera_frames, error_frame, idx, i)
        else:
            camera_frame_pose = np.matmul(camera_frame.pose, transformation_matrix)
        camera_frame = KeyFrame(camera_frame_pose)
        camera_frames.append(camera_frame)

        absPoint = relative_to_abs3DPoints(trackable_3D_points_time_i, camera_frame.pose)
        Qs, opt = appendKeyPoints(Qs, absPoint, 0.01, imagecoords_left_time_i, i, trackable_3D_points_time_i)
        optimization_matrix = np.vstack((optimization_matrix,opt))
        save3DPoints("ourCache/3DPoints.txt", absPoint, i)
        f = open("ourCache/path" +str(image_path[-2]) +".txt", "a")
        f.write(str(camera_frame.pose[0,3])+","+str(camera_frame.pose[1,3])+"," + str(camera_frame.pose[2,3])+','+str(camera_frame.pose[0,0])+","+str(camera_frame.pose[0,1])+"," + str(camera_frame.pose[0,2])+','+str(camera_frame.pose[1,0])+","+str(camera_frame.pose[1,1])+"," + str(camera_frame.pose[1,2])+','+str(camera_frame.pose[2,0])+","+str(camera_frame.pose[2,1])+"," + str(camera_frame.pose[2,2])+"\n")
        f.close()
        key_points_left_time_i = key_points_left_time_i1
        descriptors_left_time_i = descriptors_left_time_i1

        # ----- Show the image with the found keypoints in red dots -----
        imgfirst = leftimages[i+1]
        imgfirst = cv2.cvtColor(imgfirst, cv2.COLOR_GRAY2BGR)
        for u, v in trackable_left_imagecoordinates_time_i1:
            cv2.circle(imgfirst, (int(u), int(v)), 5, (0,0,255), -1, cv2.LINE_AA)
        cv2.imshow("hej", imgfirst)
        cv2.waitKey(30)


    print("Final frame pose: \n", camera_frames[len(camera_frames) - 1].pose,
          "\n\n")  # Det er denne vi skal have exporteret så det bliver vist rigtigt i MATLAB
    print("Real frame: \n", poses[-1], "\n\n")
    print("Difference: \n", np.abs(camera_frame.pose - poses[-1]), "\n\n")

    cv2.destroyAllWindows()

    export_data(optimization_matrix, camera_frames, Qs, P_left)
    # run_BA()
    # f = open("ourCache/optimizing_matrix.txt", "w")
    # for obj in optimization_matrix:
    #     f.write(str(int(obj[0])) + "," + str(int(obj[1])) + "," + str(obj[2]) + "," + str(obj[3]) + "\n")
    # f.close()
    #
    # f = open("ourCache/Q.txt", "w")
    # for coords in Qs:
    #     f.write(str(coords[0]) + "\n" + str(coords[1]) + "\n" + str(coords[2]) + "\n")
    # f.close()
    #
    # # Write camera r1, r2, r3, t1, t2, t3, f, k1, k2
    # f = open("ourCache/cam_params.txt", "w")
    # # rotmat = camera_frame.pose[:3, :3]
    # # scipy.spatial.transform.Rotation
    # for a in range(len(camera_frames)):
    # # for a in range(1):
    #     rotmat = R.from_matrix(camera_frames[a].pose[:3, :3])
    #     r_VEC = rotmat.as_rotvec()
    #     f.write(str(r_VEC[0]) + "\n" + str(r_VEC[1]) + "\n" + str(r_VEC[2]) + "\n")
    #     f.write(str(camera_frames[a].pose[0, 3]) + "\n" + str(camera_frames[a].pose[1, 3]) + "\n" + str(
    #         camera_frames[a].pose[2, 3]) + "\n")
    #     f.write(str(P_left[0][0]) + "\n0\n0\n")
    # f.close()



    """ rotmat = R.from_matrix(camera_frames[a].pose[:3, :3])
        r_VEC = rotmat.as_rotvec()
        f.write(str(r_VEC[0])+"\n" + str(r_VEC[1])+"\n" +str(r_VEC[2]) +"\n")
        f.write(str(camera_frames[a].pose[0,3])+"\n"+str(camera_frames[a].pose[1,3])+"\n" + str(camera_frames[a].pose[2,3])+"\n")
        f.write(str(P_left[0][0])+"\n0\n0\n")"""



if __name__ == '__main__':
    main()