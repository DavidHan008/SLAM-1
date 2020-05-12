from tracking import *
from keyframe import *
from visual_odometry_solution_methods import *
from bag_of_words import *
from keypoint import *
from transformation import *
from Point3D import *
from XXXport_files import *
from orb import *

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
    image_path = "../dataset/sequences/06/"
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

    combined_tvec = []  # np.empty((1,3))
    combided_rvec = [] # np.empty((1,3))

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
            transformation_matrix, rvec, tvec = calculate_transformation_matrix(trackable_3D_points_time_i,
                                                                    trackable_left_imagecoordinates_time_i1,
                                                                    close_3D_points_index, far_3D_points_index, K_left)
        # else:
        #     transformation_matrix = np.eye(4)
        #     rvec = [0,0,0]
        #     tvec = [0,0,0]

        if len(combined_tvec) == 0:
            combided_rvec = rvec.ravel()
            combined_tvec = tvec.ravel()
        else:
            combided_rvec = np.vstack((combided_rvec, rvec.ravel()))
            combined_tvec = np.vstack((combined_tvec, tvec.ravel()))
        # print(transformation_matrix)
        camera_frame_pose = np.matmul(camera_frame.pose, transformation_matrix)
        camera_frame = KeyFrame(camera_frame_pose)
        camera_frames.append(camera_frame)
        absPoint = relative_to_abs3DPoints(trackable_3D_points_time_i, camera_frame.pose)
        Qs, opt = appendKeyPoints(Qs, absPoint, 0.1, imagecoords_left_time_i, i, trackable_3D_points_time_i)
        optimization_matrix = np.vstack((optimization_matrix,opt))
        save3DPoints("ourCache/3DPoints.txt", absPoint, i)
        f = open("ourCache/path" +str(image_path[-2]) +".txt", "a")
        f.write(str(camera_frame.pose[0,3])+","+str(camera_frame.pose[1,3])+"," + str(camera_frame.pose[2,3])+','+str(camera_frame.pose[0,0])+","+str(camera_frame.pose[0,1])+"," + str(camera_frame.pose[0,2])+','+str(camera_frame.pose[1,0])+","+str(camera_frame.pose[1,1])+"," + str(camera_frame.pose[1,2])+','+str(camera_frame.pose[2,0])+","+str(camera_frame.pose[2,1])+"," + str(camera_frame.pose[2,2])+"\n")
        f.close()
        key_points_left_time_i = key_points_left_time_i1
        descriptors_left_time_i = descriptors_left_time_i1

        # ---------------------------- LOOP CLOSURE -------------------------- #
        idx, val = bow.predict_previous(leftimages[i], i, bow_threshold)
        if val < 45 and val > 0:
            print("Frame: ", i, ". Val: ", val, ". idx: " , idx)
            break
            cv2.waitKey(0)
            bow_threshold = i + 100
        # print(idx, val)

        # ----- Show the image with the found keypoints in red dots -----
        # imgfirst = leftimages[i+1]
        # imgfirst = cv2.cvtColor(imgfirst, cv2.COLOR_GRAY2BGR)
        # for u, v in trackable_left_imagecoordinates_time_i1:
        #     cv2.circle(imgfirst, (int(u), int(v)), 5, (0,0,255), -1, cv2.LINE_AA)
        # cv2.imshow("hej", imgfirst)
        # cv2.waitKey(30)


    f = open("ourCache/optimizing_matrix.txt", "w")
    for pik in optimization_matrix:
        f.write(str(int(pik[0])) + "," + str(int(pik[1])) + "," + str(pik[2]) + "," + str(pik[3]) + "\n")
    f.close()

    f = open("ourCache/Q.txt", "w")
    for coords in Qs:
        f.write(str(coords[0]) + "\n" + str(coords[1]) + "\n" + str(coords[2]) + "\n")
    f.close()

    # Write camera r1, r2, r3, t1, t2, t3, f, k1, k2
    f = open("ourCache/cam_params.txt", "w")
    for a in range(len(combided_rvec)):
        f.write(str(combided_rvec[a][0])+"\n" + str(combided_rvec[a][1])+"\n" +str(combided_rvec[a][2]) +"\n")
        f.write(str(combined_tvec[a][0])+"\n" + str(combined_tvec[a][1])+"\n" +str(combined_tvec[a][2]) +"\n")
        f.write(str(P_left[0][0])+"\n0\n0\n")
    f.close()

    print("Final frame pose: \n", camera_frames[len(camera_frames)-1].pose, "\n\n") # Det er denne vi skal have exporteret så det bliver vist rigtigt i MATLAB
    print("Real frame: \n", poses[-1], "\n\n")
    print("Difference: \n", np.abs(camera_frame.pose - poses[-1]), "\n\n")

if __name__ == '__main__':
    main()