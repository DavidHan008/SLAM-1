import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'images'))

        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=float, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

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

    @staticmethod
    def draw_matches(img1, kp1, img2, kp2, matches):
        matches = sorted(matches, key = lambda x:x.distance)
        vis_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', vis_img)
        cv2.waitKey(0)

    def get_matches(self, i):
        # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        matches = self.flann.knnMatch(des1, des2, k=2)
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

    def get_pose(self, q1, q2):
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)
        R, t = self.decomp_essential_mat(E, q1, q2)
        # inliers, R, t, mask = cv2.recoverPose(E, q1, q2, self.K)
        return self._form_transf(R, np.squeeze(t))

    def decomp_essential_mat(self, E, q1, q2):
        def sum_z(R, t):
            T = self._form_transf(R, t)
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = np.matmul(T, hom_Q1)
            return sum(hom_Q1[2, :] / hom_Q1[3, :] > 0) + sum(hom_Q2[2, :] / hom_Q2[3, :] > 0)

        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
        sums = [sum_z(R, t) for R, t in pairs]
        return pairs[np.argmax(sums)]

def visualize_paths(verts1, verts2):
    codes = [Path.MOVETO]
    for i in range(len(verts1) - 1):
        codes.append(Path.LINETO)

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
    data_dir = 'KITTI_sequence_2'
    vo = VisualOdometry(data_dir)
    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(vo.gt_poses):
        print(i)
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    visualize_paths(gt_path, estimated_path)

if __name__ == "__main__":
    main()
