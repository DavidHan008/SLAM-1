import visual_odometry
from tracking import *
import os
import matplotlib.pyplot as plt
import numpy as np

data_dir = '..//KITTI_sequence_1'
images = load_images(os.path.join(data_dir, "images"))


def compare_matches(img1, img2, kp1, kp2, des1, des2, number_of_matches):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:number_of_matches], None, flags=2)

    plt.imshow(img3), plt.show()


def show_keypoints(image, kp):
    img2 = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
    plt.imshow(img2), plt.show()


def form_transf(R, t):
    T = np.eye(4, dtype=np.float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def get_pose(q1, q2, K):
    E, _ = cv2.findEssentialMat(q1, q2, K, threshold=1)
    R, t = decomp_essential_mat(E, q1, q2, K, P)
    # inliers, R, t, mask = cv2.recoverPose(E, q1, q2, self.K)
    return form_transf(R, np.squeeze(t))


def decomp_essential_mat(E, q1, q2, K, P):
    def sum_z(R, t):
        T = form_transf(R, t)
        P = np.matmul(np.concatenate((K, np.zeros((3, 1))), axis=1), T)
        hom_Q1 = cv2.triangulatePoints(P, P, q1.T, q2.T)
        hom_Q2 = np.matmul(T, hom_Q1)
        return sum(hom_Q1[2, :] / hom_Q1[3, :] > 0) + sum(hom_Q2[2, :] / hom_Q2[3, :] > 0)

    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = np.squeeze(t)
    pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
    sums = [sum_z(R, t) for R, t in pairs]
    return pairs[np.argmax(sums)]


def load_calib(filepath):
    with open(filepath, 'r') as f:
        params = np.fromstring(f.readline(), dtype=float, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
    return K, P

# AUTOMATIC MAP INITIALIZATION

# TRACKING
# Initialize camera intrinsics
K, P = load_calib(os.path.join(data_dir, 'calib.txt'))

# ORB Extraction
kp1, des1 = orb_extraction(images[0])
show_keypoints(images[0], kp1)

# Triangulation


# Initial Pose Estimation from Previous Frame
kp2, des2 = orb_extraction(images[1])
q1, q2 = get_matches(kp1, des1, kp2, des2)
pose = get_pose(q1, q2, K)
print(pose)

compare_matches(images[0], images[1], kp1, kp2, des1, des2, 150)



# Initial Pose Estimation via Global Relocalization
# Track Local Map

# New KeyFrame Decision


# LOCAL MAPPING
# KeyFrame Insertion
# Recent MapPoints Culling
# New MapPoint Creation
# Local BA
# Local KeyFrame Culling



# LOOP CLOSURE
# Loop Candidates Detection
# Compute Similarity Transform
# Loop Fusion
# Essential Graph Optimization




# COVISIBILITY GRAPH

# 3D to 2D
# 1) Do only once:
# 1.1) Capture two frames Ik2, Ik1
# 1.2) Extract and match features between them
# 1.3) Triangulate features from Ik2, Ik1
# 2) Do at each iteration:
# 2.1) Capture new frame Ik
# 2.2) Extract features and match with previous frame Ik1
# 2.3) Compute camera pose (PnP) from 3-D-to-2-D matches
# 2.4) Triangulate all new feature matches between Ik and Ik1
# 2.5) Iterate from 2.1).