import os
import numpy as np
import cv2
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

class VisualOdometry():
    def __init__(self, data_dir):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images_l = self._load_images(os.path.join(data_dir, 'image_l'))
        self.images_r = self._load_images(os.path.join(data_dir, 'image_r'))

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0,numDisparities=32, blockSize=block, P1=P1, P2=P2)
        self.disparities = [np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]
        self.fastFeatures = cv2.FastFeatureDetector_create()

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=float, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=float, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=float, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        R, _ = cv2.Rodrigues(dof[:3])
        transf = self._form_transf(R, dof[3:])

        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        q1_pred = Q2.dot(f_projection.T)
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]
        q2_pred = Q1.dot(b_projection.T)
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        return np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()

    def get_tiled_keypoints(self, img, tile_h, tile_w):
        # Split the image into tiles and detect the 10 best keypoints in each tile
        # Return a 1-D list of all keypoints
        # Hint: use sorted(keypoints, key=lambda x: -x.response)
        pass

    def track_keypoints(self, img1, img2, kp1, max_error=4):
        # Convert the keypoints using cv2.KeyPoint_convert
        # Use cv2.calcOpticalFlowPyrLK to estimate the keypoint locations in the second frame. self.lk_params contains parameters for this function.
        # Remove all points which are not trackable, has error over max_error, or where the points moved out of the frame of the second image
        # Return a list of the original converted keypoints (after removal), and their tracked counterparts
        pass

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        # Get the disparity for each keypoint
        # Remove all keypoints where disparity is out of bounds
        # calculate keypoint location in right image by subtracting disparity from x coordinates
        # return left and right keypoints for both frames
        pass

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        # Triangulate points from both images with self.P_l and self.P_r
        pass

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        # Implement RANSAC to estimate the pose using least squares optimization
        # Sample 6 random point sets from q1, q2, Q1, Q2
        # Minimize the given residual using least squares to find the optimal transform between the sampled points
        # Calculate the reprojection error when using the optimal transform with the whole point set
        # Redo with different sampled points, saving the transform with the lowest error, until max_iter or early termination criteria met
        pass

    def get_pose(self, i):
        img1_l, img2_l = self.images_l[i - 1:i + 1]
        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)
        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        self.disparities.append(np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16))

        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        return self.estimate_pose(tp1_l, tp2_l, Q1, Q2)

def visualize_paths(verts1, verts2):
    codes = [Path.LINETO for _ in range(len(verts1))]
    codes[0] = Path.MOVETO

    path1 = Path(verts1, codes)
    path2 = Path(verts2, codes)
    _, ax = plt.subplots()
    patch1 = patches.PathPatch(path1, facecolor='none', edgecolor='green', lw=2)
    patch2 = patches.PathPatch(path2, facecolor='none', edgecolor='red', lw=2)
    ax.add_patch(patch1)
    ax.add_patch(patch2)
    ax.axis('equal')

    plt.show()

def main():
<<<<<<< HEAD:stereo_visual_odometry_template.py
    data_dir = 'KITTI_sequence_2'
=======
    data_dir = '..//KITTI_sequence_1'
>>>>>>> 648305125c69d7104b5044bb1da9c7f23c38fe31:stereo_visual_odometry_solution.py
    vo = VisualOdometry(data_dir)
    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(vo.gt_poses):
        print(i)
        if i < 1:
            cur_pose = gt_pose
        else:
            transf = vo.get_pose(i)
            cur_pose = np.matmul(cur_pose, transf)
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    visualize_paths(gt_path, estimated_path)

if __name__ == "__main__":
    main()
