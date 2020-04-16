import cv2
import os
import numpy as np

def load_images(filepath):
    image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
    return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]


# Stolen from Visual Odometry Solution (Lesson 3)
# Why are return values not whole numbers?
def get_matches(kp1, des1, kp2, des2):
    # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    print("Length of kp1: ", len(kp1))
    print("Length of kp2: ", len(kp2))
    print("Length of des1: ", len(des1))
    print("Length of des2: ", len(des2))
    matches = flann.knnMatch(des1, des2, k=2)
    # good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    good = []
    try:
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
    except ValueError:
        pass
    # self.draw_matches(self.images[i - 1], kp1, self.images[i], kp2, good)
    q1 = np.float32([kp1[m.queryIdx].pt for m in good])
    q2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return q1, q2

# This method is stolen from stereo_visual_odometry_solution.py on blackboard
def track_keypoints(img1, img2, kp1, max_error=40):
    lk_params = dict(winSize=(15, 15),
                     flags=cv2.MOTION_AFFINE,
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
    trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)
    trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **lk_params)
    trackable = st.astype(bool)
    under_thresh = np.where(err[trackable] < max_error, True, False)

    trackpoints1 = trackpoints1[trackable][under_thresh]
    trackpoints2 = np.around(trackpoints2[trackable][under_thresh])
    h, w = img1.shape
    in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
    in_bounds_new = np.where(np.logical_and(trackpoints2[:, 1] > 0, trackpoints2[:, 0] > 0), True, False)
    in_bounds = np.logical_and(in_bounds, in_bounds_new)
    return trackpoints1[in_bounds], trackpoints2[in_bounds]

# TRACKING
# ORB Extraction
# Skal ORB extraction bruge img1L, img2L etc?
#TODO: Implement as tiled keypoints, as per the Visual Odometry solution
def orb_extraction(img):
    """FAST corners at 8 scale levels with a scale factor of 1.2.
    For image resolutions from 512×384 to 752×480 pixels we found suitable to extract 1000 corners,
    for higher resolutions, as the 1241 × 376 in the KITTI dataset [40] we extract 2000 corners"""

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures = 1000, scaleFactor=1.2)
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    return kp, des


# Initial Pose Estimation from Previous Frame



# OR

# Initial Pose Estimation via Global Relocalization

# Track Local Map

# New KeyFrame Decision

#kp1, des1 = orb_extraction(img_l)
#kp2, des2 = orb_extraction(img_r)
#q1, q2 = get_matches(kp1, des1, kp2, des2)