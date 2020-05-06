import cv2
import numpy as np
from sklearn.neighbors import KDTree
import math

class KeyPoint:
    def __init__(self, x, y, z, des):
        self.x = x
        self.y = y
        self.z = z
        #self.des = des

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


def track_keypoints_left_to_right_new(key_points_left, descriptors_left, key_points_right, descriptors_right, leftimg, rightimg):
    #print("Key points left size: %d" % np.shape(key_points_left))
    #print("Key points right size: %d" % np.shape(key_points_right))

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

    mask = np.array(mask,dtype=bool).ravel() # unravel the mask.

    pts_left = pts_left[mask] # apply mask to keypoints and descriptors
    pts_right = pts_right[mask]
    des_left = des_left[mask]
    des_right = des_right[mask]

    # ----- Show the corresponding keypoints, after masking via Fundamental matrix (Sampsons Distance) -----
    imgfirst = cv2.cvtColor(leftimg, cv2.COLOR_GRAY2BGR)
    imgsecond = cv2.cvtColor(rightimg, cv2.COLOR_GRAY2BGR)
    for point_idx in range(len(pts_left)):
        col = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        siz = np.random.randint(3,6)
        cv2.circle(imgfirst, (int(pts_left[point_idx][0]),int(pts_left[point_idx][1])), siz, col,-1,cv2.LINE_AA)
        cv2.circle(imgsecond, (int(pts_right[point_idx][0]),int(pts_right[point_idx][1])), siz, col,-1,cv2.LINE_AA)
    cv2.imshow("Left",imgfirst)
    cv2.imshow("Right", imgsecond)
    cv2.waitKey(1)

    return pts_left, pts_right, des_left, des_right


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


# Returns array of [frame_index, 3D_index, u, v]
def appendKeyPoints(Qs, absPoint, threshold, points_2d, frame_index, rel_point):
    full_index_array = np.empty((0, 4))  # frame index, 3Dp index, 2d coordinates
    tmp_index_array = []
    if (len(Qs) == 0):
        for i in range(len(absPoint)):
            tmp_index_array = [frame_index, i, points_2d[i][0], points_2d[i][1]]
            full_index_array = np.vstack((full_index_array, tmp_index_array))
        return absPoint, full_index_array
    known_points_tree = KDTree(Qs)
    dist, ind = known_points_tree.query(absPoint, k = 1)
    for d in range(len(dist)):
        # print(threshold*math.sqrt(np.sum(rel_point[d]**2)))
        distance = threshold*math.sqrt(np.sum(rel_point[d]**2))
        if dist[d] < distance:
            tmp_index_array = [frame_index, int(ind[d]), points_2d[d][0], points_2d[d][1]]
        else:
            Qs = np.vstack((Qs, absPoint[d]))
            tmp_index_array = [frame_index, len(Qs) - 1, points_2d[d][0], points_2d[d][1]]
        full_index_array = np.vstack((full_index_array, tmp_index_array))
    Qs = np.reshape(Qs,(-1,3))
    return Qs, full_index_array