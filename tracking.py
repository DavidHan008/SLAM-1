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



# TRACKING
# ORB Extraction
# Skal ORB extraction bruge img1L, img2L etc?
#TODO: Implement as tiled keypoints, as per the Visual Odometry solution
def orb_extraction(img):
    """FAST corners at 8 scale levels with a scale factor of 1.2.
    For image resolutions from 512×384 to 752×480 pixels we found suitable to extract 1000 corners,
    for higher resolutions, as the 1241 × 376 in the KITTI dataset [40] we extract 2000 corners"""

    # Initiate ORB detector
    orb = cv2.ORB_create(scaleFactor=1.2)
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