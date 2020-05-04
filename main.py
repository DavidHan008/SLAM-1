from tracking import *
from keyframe import *
from visual_odometry_solution_methods import *
from sklearn.neighbors import KDTree
from bag_of_words import *
from keypoint import *
from transformation import *

import cv2
import math


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
    # C:\Users\Ole\Desktop\Project\dataset\sequences
    image_path = "../KITTI_sequence_2/"
    #image_path = "../data_odometry_gray/dataset/sequences/06/"
    # Load the images of the left and right camera
    leftimages = load_images(os.path.join(image_path, "image_l"))
    rightimages = load_images(os.path.join(image_path, "image_r"))

    # Load K and P from the calibration file
    K_left, P_left, _, P_right = load_calib(image_path+"calib.txt")
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
    combined_tvec = []
    combided_rvec = []
    clear_textfile("path" +str(image_path[-2]) +".txt")
    clear_textfile("3DPoints.txt")
    optimization_matrix = np.empty((0,4))        # frame nr, 3d_index and 2d coordinate
    key_points_left_time_i, descriptors_left_time_i = orb_detector_using_tiles(leftimages[0])
    #for i in range(len(leftimages)-1):
    for i in range(2):

        key_points_right_time_i, descriptors_right_time_i = orb_detector_using_tiles(rightimages[i])
        key_points_left_time_i1, descriptors_left_time_i1 = orb_detector_using_tiles(leftimages[i+1])

        trackable_keypoints_left_time_i, trackable_descriptors_left_time_i, \
        trackable_keypoints_right_time_i = track_keypoints_left_to_right(leftimages[i], rightimages[i],
                                                                                 key_points_left_time_i,
                                                                                 descriptors_left_time_i)

        # print(np.shape(key_points_left_time_i))
        # trackable_keypoints_left_time_i, trackable_keypoints_right_time_i, \
        # trackable_descriptors_left_time_i, trackable_descriptors_right_time_i = track_keypoints_left_to_right_new(key_points_left_time_i,
        #                                   descriptors_left_time_i, key_points_right_time_i, descriptors_right_time_i)

        relative_triangulated_3D_points_time_i = triangulate_points_local(trackable_keypoints_left_time_i, trackable_keypoints_right_time_i, P_left, P_right)

        trackable_left_imagecoordinates_time_i1, trackable_3D_points_time_i, imagecoords_left_time_i \
            = find_2D_and_3D_correspondenses(trackable_descriptors_left_time_i, trackable_keypoints_left_time_i,
                          key_points_left_time_i1, descriptors_left_time_i1, relative_triangulated_3D_points_time_i)

        close_3D_points_index, far_3D_points_index = sort_3D_points(trackable_3D_points_time_i, close_def_in_m=200)

        if len(trackable_3D_points_time_i) > 4:
            transformation_matrix, rvec, tvec = calculate_transformation_matrix(trackable_3D_points_time_i,
                                                                    trackable_left_imagecoordinates_time_i1,
                                                                    close_3D_points_index, far_3D_points_index, K_left)
        else:
            transformation_matrix = np.eye(4)
            rvec = [0,0,0]
            tvec = [0,0,0]

        combined_tvec = np.vstack(combined_tvec, tvec)
        combided_rvec = np.vstack(combided_rvec, rvec)
        camera_frame_pose = np.matmul(camera_frame.pose, transformation_matrix)

        camera_frame = KeyFrame(camera_frame_pose)
        camera_frames.append(camera_frame)

        absPoint = relative_to_abs3DPoints(trackable_3D_points_time_i, camera_frame.pose)

        Qs, opt = appendKeyPoints(Qs, absPoint, 0.2, imagecoords_left_time_i, i)
        optimization_matrix = np.vstack((optimization_matrix,opt))

        save3DPoints("3DPoints.txt", absPoint, i)

        f = open("path" +str(image_path[-2]) +".txt", "a")
        f.write(str(-camera_frame.pose[0,3])+"," + str(-camera_frame.pose[2,3])+"\n")
        f.close()

        key_points_left_time_i = key_points_left_time_i1
        descriptors_left_time_i = descriptors_left_time_i1

        imgfirst = leftimages[i+1]
        imgfirst = cv2.cvtColor(imgfirst, cv2.COLOR_GRAY2BGR)
        for u, v in trackable_left_imagecoordinates_time_i1:
            cv2.circle(imgfirst, (int(u), int(v)), 5, (0,0,255), -1, cv2.LINE_AA)
        cv2.imshow("hej", imgfirst)
        cv2.waitKey(30)


    f = open("optimizing_matrix.txt", "w")
    for pik in optimization_matrix:
        f.write(str(pik[0]) + "," + str(pik[1]) + "," + str(pik[2]) + "," + str(pik[3]) + "\n")
    f.close()


    f = open("Q.txt", "w")
    for coords in Qs:
        f.write(str(coords[0]) + "," + str(coords[1]) + "," + str(coords[2]) + "\n")
    f.close()

    # Write camera r1, r2, r3, t1, t2, t3, f, k1, k2
    f = open("cam_params.txt", "w")
    for a in range(len(combided_rvec)):
        f.write(str(combided_rvec[a][0])+", " + str(combided_rvec[a][1])+", " +str(combided_rvec[a][2]) +", ")
        f.write(str(combined_tvec[a][0])+", " + str(combined_tvec[a][1])+", " +str(combined_tvec[a][2]) +", ")
        f.write()

    f.close()

    print("Final frame pose: \n", camera_frames[len(camera_frames)-1].pose, "\n\n")
    print("Real frame: \n", poses[-1], "\n\n")
    print("Difference: \n", np.abs(camera_frame.pose - poses[-1]), "\n\n")

if __name__ == '__main__':
    main()
