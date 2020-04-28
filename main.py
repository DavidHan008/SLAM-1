from tracking import *
from visual_odometry_solution_methods import *
import cv2
import math


def orb_detector_using_tiles(image, max_number_of_kp = 20, overlap_div = 2, height_div = 5, width_div = 10):
    def get_kps(x, y, h, w):
        impatch = image[y:y + h, x:x + w]
        keypoints, descriptors = orb_extraction_detect(impatch)
        for pt in keypoints:
            pt.pt = (pt.pt[0] + x, pt.pt[1] + y)
        if len(keypoints) == 0:
            return [], []
        return keypoints, descriptors
    h, w = image.shape
    tile_h = int(h/height_div)
    tile_w = int(w/width_div)
    kp_list = []
    des = []
    for y in range(0, h-tile_h, tile_h):
        for x in range(0, w-tile_w, tile_w):
            kp1, des1 = get_kps(x, y, int(tile_h+tile_h/overlap_div), int(tile_w+tile_w/overlap_div))
            kp_list.extend(kp1)
            des.extend(des1)
    des = [kp for sublist in des for kp in sublist]
    des = np.reshape(des, (-1, 32))
    return kp_list, des

def orb_extraction_detect(img):
    """FAST corners at 8 scale levels with a scale factor of 1.2.
    For image resolutions from 512×384 to 752×480 pixels we found suitable to extract 1000 corners,
    for higher resolutions, as the 1241 × 376 in the KITTI dataset [40] we extract 2000 corners"""

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures = 40, scaleFactor=1.2)
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)

    return kp, des

def orb_extraction_compute(img, kp):
    orb = cv2.ORB_create(nfeatures = 50, scaleFactor=1.2)
    kp, des = orb.compute(img, kp)
    return kp, des


def track_keypoints_left_to_right_new(key_points_left, descriptors_left, key_points_right, descriptors_right):
    # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)
    good = []
    try:
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # 0.7
                good.append(m)
    except ValueError:
        pass

    pts_left = np.asarray([key_points_left[m.queryIdx].pt for m in good])
    pts_right = np.asarray([key_points_right[m.trainIdx].pt for m in good])

    des_left = np.asarray([descriptors_left[m.queryIdx] for m in good])
    des_right = np.asarray([descriptors_right[m.trainIdx] for m in good])

    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_LMEDS)
    # brug mask
    distances = []
    for i in range(len(pts_left)):
        dist = cv2.sampsonDistance(pts_left[i], pts_right[i], F)
        if dist < 1000:
            distances.append(True)
        else:
            distances.append(False)

    return pts_left[distances], pts_right[distances], des_left[distances], des_right[distances]


def track_keypoints_left_to_right(image_left, image_right, key_points_left, descriptors_left, max_error = 500):
    lk_params = dict(winSize=(15, 15),
                     flags=cv2.MOTION_AFFINE,
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
    trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(key_points_left), axis=1)
    trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(image_left, image_right, trackpoints1, None, **lk_params)
    trackable = st.astype(bool)
    under_thresh = np.where(err[trackable] < max_error, True, False)

    trackpoints1 = trackpoints1[trackable][under_thresh]
    descriptors_left = np.expand_dims(descriptors_left, axis=1)
    descriptors_left = descriptors_left[trackable][under_thresh]
    trackpoints2 = np.around(trackpoints2[trackable][under_thresh])
    h, w = image_right.shape

    in_bounds = np.where(np.logical_and(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w),
                                        np.logical_and(trackpoints2[:, 1] > 0, trackpoints2[:, 0] > 0)), True, False) # dobbelt tjek at and tager flere input

    return trackpoints1[in_bounds], descriptors_left[in_bounds], trackpoints2[in_bounds]


def find_2D_and_3D_correspondenses(descriptors_time_i, keypoints_left_time_i1, descriptors_left_time_i1, triangulated_3D_points):
    # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    matches = flann.knnMatch(descriptors_time_i, descriptors_left_time_i1, k=2)
    good = []
    try:
        for m, n in matches:
            if m.distance < 0.7 * n.distance: #0.7
                good.append(m)
    except ValueError:
        pass
    Q1 = np.asarray([triangulated_3D_points[m.queryIdx] for m in good])
    q2 = np.asarray([keypoints_left_time_i1[m.trainIdx].pt for m in good])
    return q2, Q1


def remove_dublicate_keypoints(keypoints, descriptors):
    oc_set = set()
    res = []
    get_point = lambda x: x.pt
    points = [get_point(i) for i in keypoints]
    points = [(int(u), int(v)) for u,v in points]
    for idx, val in enumerate(points):
        if val not in oc_set:
            oc_set.add(val)
            res.append(idx)
    kp = [keypoints[i] for i in res]
    des = [descriptors[i] for i in res]

    des = [kp for sublist in des for kp in sublist]
    des = np.reshape(des, (-1,32))
    return kp, des

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


# Returns a mask of all valid points
def sort_3D_points(triangulated_3D_point, close_def_in_m = 20, far_def_in_m = 20):

    close_3D_point = [(abs(x[0]) < close_def_in_m and abs(x[1]) < close_def_in_m and abs(x[2]) < close_def_in_m) for x in triangulated_3D_point]
    far_3D_points = np.bitwise_not(close_3D_point)
    return close_3D_point, far_3D_points


def translation_and_rotation_vector_to_matrix(rotvec, transvec):
    rotm = eulerAnglesToRotationMatrix(rotvec)
    #transvec[2] = -1 * transvec[2]
    #transvec[1] = -1 * transvec[1]
    #transvec[0] = 1 * transvec[0]
    return form_transf(rotm, np.transpose(transvec))

def triangulate_points_local(qs_l, qs_r, P_l, P_r):
    qs_l = np.transpose(qs_l)
    qs_r = np.transpose(qs_r)
    hom_Qs = cv2.triangulatePoints(P_l, P_r, qs_l, qs_r)
    return np.transpose(hom_Qs[:3] / hom_Qs[3])

def calculate_transformation_matrix(trackable_3D_points_time_i, trackable_left_imagecoordinates_time_i1,
                                        close_3D_points_index, far_3D_points_index, K_left, rvec, tvec):

    if sum(close_3D_points_index) > 1000000 and sum(far_3D_points_index) > 10:
        image_point = np.ascontiguousarray(trackable_left_imagecoordinates_time_i1[close_3D_points_index]).reshape((-1,2))
        world_coord = np.ascontiguousarray(trackable_3D_points_time_i[close_3D_points_index]).reshape((-1,3))

        _, _, translation_vector, _ = cv2.solvePnPRansac(world_coord,
                                                         image_point,
                                                         K_left,
                                                         np.zeros(5))

        _, rotation_vector, _, _ = cv2.solvePnPRansac(world_coord,
                                                     image_point,
                                                      K_left,
                                                      np.zeros(5))
    else:
        track2dPoints = np.ascontiguousarray(trackable_left_imagecoordinates_time_i1[close_3D_points_index]).reshape((-1,2))
        track3dPoints = np.ascontiguousarray(trackable_3D_points_time_i[close_3D_points_index]).reshape((-1,3))

        _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(track3dPoints,
                                                                       track2dPoints, K_left,
                                                                       np.zeros(5))


    return translation_and_rotation_vector_to_matrix(rotation_vector, translation_vector), rotation_vector, translation_vector

def main():
    # C:\Users\Ole\Desktop\Project\dataset\sequences
    #image_path = "../KITTI_sequence_2/"
    image_path = "../data_odometry_gray/dataset/sequences/06/"
    # Load the images of the left and right camera
    leftimages = load_images(os.path.join(image_path, "image_0"))
    rightimages = load_images(os.path.join(image_path, "image_1"))

    # Load K and P from the calibration file
    K_left, P_left, _, P_right = load_calib(image_path+"calib.txt")
    poses = load_poses(image_path+"poses.txt")

    camera_frame = np.eye(4)
    rvec = np.array([0,0,0])
    tvec = np.array([0,0,0])
    trans_old = np.eye(4)

    f = open("path.txt", "w")
    f.close()

    # key_points_left_time_i, descriptors_left_time_i = get_descriptors_and_keypoints(leftimages[0])
    key_points_left_time_i, descriptors_left_time_i = orb_detector_using_tiles(leftimages[0])
    for i in range(len(leftimages)-1):

        #key_points_right_time_i, descriptors_right_time_i = orb_detector_using_tiles(rightimages[i])
        key_points_left_time_i1, descriptors_left_time_i1 = orb_detector_using_tiles(leftimages[i+1])
        # key_points_left_time_i1, descriptors_left_time_i1 = get_descriptors_and_keypoints(leftimages[i+1])

        trackable_keypoints_left_time_i, trackable_descriptors_left_time_i, \
        trackable_keypoints_right_time_i = track_keypoints_left_to_right(leftimages[i], rightimages[i],
                                                                                  key_points_left_time_i,
                                                                                  descriptors_left_time_i)


        #trackable_keypoints_left_time_i, trackable_keypoints_right_time_i, \
        #trackable_descriptors_left_time_i, trackable_descriptors_right_time_i = track_keypoints_left_to_right_new(key_points_left_time_i,
        #                                  descriptors_left_time_i, key_points_right_time_i, descriptors_right_time_i)



        triangulated_3D_points_time_i = triangulate_points_local(trackable_keypoints_left_time_i,
                                                                 trackable_keypoints_right_time_i, P_left, P_right)

        # print(trackable_keypoints_left_time_i)
        # print(trackable_keypoints_right_time_i)


        trackable_left_imagecoordinates_time_i1, trackable_3D_points_time_i \
            = find_2D_and_3D_correspondenses(trackable_descriptors_left_time_i,
                          key_points_left_time_i1, descriptors_left_time_i1, triangulated_3D_points_time_i)

        close_3D_points_index, far_3D_points_index = sort_3D_points(trackable_3D_points_time_i, close_def_in_m=200)

        if len(trackable_3D_points_time_i) > 4:
            transformation_matrix, rvec, tvec = calculate_transformation_matrix(trackable_3D_points_time_i,
                                                                    trackable_left_imagecoordinates_time_i1,
                                                                    close_3D_points_index, far_3D_points_index, K_left, rvec, tvec)
        else:
            transformation_matrix = np.eye(4)

        camera_frame = np.matmul(camera_frame, transformation_matrix)
        f = open("path.txt", "a")
        f.write(str(camera_frame[0,3])+"," + str(camera_frame[2,3])+"\n")
        f.close()


        key_points_left_time_i = key_points_left_time_i1
        descriptors_left_time_i = descriptors_left_time_i1

        imgfirst = leftimages[i+1]
        imgfirst = cv2.cvtColor(imgfirst, cv2.COLOR_GRAY2BGR)
        for u, v in trackable_left_imagecoordinates_time_i1:
            cv2.circle(imgfirst, (int(u), int(v)), 5, (0,0,255), -1, cv2.LINE_AA)
        cv2.imshow("hej", imgfirst)
        cv2.waitKey(30)


    print("Final frame: \n", camera_frame, "\n\n")
    print("Real frame: \n", poses[-1], "\n\n")
    print("Difference: \n", np.abs(camera_frame-poses[-1]), "\n\n")

if __name__ == '__main__':
    main()
