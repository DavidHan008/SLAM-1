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
        def get_kps(x, y):
            impatch = img[y:y + tile_h, x:x + tile_w]
            keypoints = self.fastFeatures.detect(impatch)
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10]
            return keypoints

        h, w = img.shape
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]
        return [kp for sublist in kp_list for kp in sublist]

    def track_keypoints(self, img1, img2, kp1, max_error=4):
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)
        trackable = st.astype(bool)
        under_thresh = np.where(err[trackable] < max_error, True, False)

        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)

        return trackpoints1[in_bounds], trackpoints2[in_bounds]

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)

        disp1, idxs1 = get_idxs(q1, disp1)
        disp2, idxs2 = get_idxs(q2, disp2)
        in_bounds = np.logical_and(idxs1, idxs2)
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        Q1 = np.transpose(Q1[:3] / Q1[3])
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        min_error = float('inf')
        early_termination = 0
        for _ in range(max_iter):
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]
            in_guess = np.zeros(6)
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == 5:
                break

        R, _ = cv2.Rodrigues(out_pose[:3])
        return self._form_transf(R, out_pose[3:])

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
    data_dir = '..//KITTI_sequence_1'
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
    print("Hej hej, din haj")
    visualize_paths(gt_path, estimated_path)

if __name__ == "__main__":
    main()
