import urllib.request
import bz2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import cv2

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
URL = BASE_URL + FILE_NAME


def read_bal_data(file_name):

    # with bz2.open(file_name, "rt") as file:
    file = open(file_name, 'r')
    n_cams, n_Qs, n_qs = map(int, file.readline().split())

    cam_idxs = np.empty(n_qs, dtype=int)
    Q_idxs = np.empty(n_qs, dtype=int)
    qs = np.empty((n_qs, 2))

    for i in range(n_qs):
        cam_idx, Q_idx, x, y = file.readline().split()
        cam_idxs[i] = int(cam_idx)
        Q_idxs[i] = int(Q_idx)
        qs[i] = [float(x), float(y)]

    cam_params = np.empty(n_cams * 9)
    for i in range(n_cams * 9):
        cam_params[i] = float(file.readline())
    cam_params = cam_params.reshape((n_cams, -1))

    print("cam_params ", np.shape(cam_params))
    print(cam_params[-1])

    Qs = np.empty(n_Qs * 3)
    for i in range(n_Qs * 3):
        Qs[i] = float(file.readline())
    Qs = Qs.reshape((n_Qs, -1))

    print("Qs ", np.shape(Qs))
    print(Qs[0])
    print(Qs[-1])

    return cam_params, Qs, cam_idxs, Q_idxs, qs


def reindex(idxs):
    keys = np.sort(np.unique(idxs))
    key_dict = {key: value for key, value in zip(keys, range(keys.shape[0]))}
    return [key_dict[idx] for idx in idxs]


def shrink_problem(n, cam_params, Qs, cam_idxs, Q_idxs, qs):
    cam_idxs = cam_idxs[:n]
    Q_idxs = Q_idxs[:n]
    qs = qs[:n]
    cam_params = cam_params[np.isin(np.arange(cam_params.shape[0]), cam_idxs)]
    Qs = Qs[np.isin(np.arange(Qs.shape[0]), Q_idxs)]
    return cam_params, Qs, reindex(cam_idxs), reindex(Q_idxs), qs


def rotate(Qs, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(Qs * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * Qs + sin_theta * np.cross(v, Qs) + dot * (1 - cos_theta) * v

def project(Qs, cam_params):
    f = float(cam_params[0][6])
    cam_mat = np.array([[f, 0,0],[0, f, 0],[0,0,0]])
    qs_proj,_ = cv2.projectPoints(Qs[0], cam_params[0][:3], cam_params[0][3:6], cam_mat, np.zeros(4))
    qs_proj = np.array(qs_proj.ravel())
    for pik in range(1,len(Qs)):
        # print(cam_params[pik])
        qs_temp,_ = cv2.projectPoints(Qs[pik], cam_params[pik][:3], cam_params[pik][3:6], cam_mat, np.zeros(4))
        qs_temp = qs_temp.ravel()
        # print(qs_temp)
        qs_proj = np.vstack((qs_proj, qs_temp))
        if pik

    print(np.shape(qs_proj))
    return qs_proj
# for i in range(10):
# 	hej = np.array([3,4])
# 	pik = np.vstack((pik,hej))
#
# print(pik)

# def project(Qs, cam_params):
#     """Convert 3-D points to 2-D by projecting onto images."""
#     qs_proj = rotate(Qs, cam_params[:, :3])
#     qs_proj += cam_params[:, 3:6]
#     # print(np.shape(qs_proj))
#     #print(qs_proj)
#     # print(qs_proj[2])
#     qs_proj = -qs_proj[:, :2] / qs_proj[:, 2, np.newaxis]
#     f, k1, k2 = cam_params[:, 6:].T
#     n = np.sum(qs_proj ** 2, axis=1)
#     r = 1 + k1 * n + k2 * n ** 2
#     qs_proj *= (r * f)[:, np.newaxis]
#     return qs_proj


def objective(params, n_cams, n_Qs, cam_idxs, Q_idxs, qs):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    cam_params = params[:n_cams * 9].reshape((n_cams, 9))
    Qs = params[n_cams * 9:].reshape((n_Qs, 3))
    qs_proj = project(Qs[Q_idxs], cam_params[cam_idxs])
    # cnt = 0
    # for proj in range(0, len(qs_proj)):
    #     if abs(qs_proj[proj][0]) > 100 or abs(qs_proj[proj][1]) > 100:# or proj == 204:
            # print("------------------------------")
            # print("image correspondance number: ",proj)
            # print("frame number: ", cam_idxs[proj])
            # print("Size of projections: ", qs_proj[proj])
            # print("cam params: ",cam_params[cam_idxs][proj]) #rotation vector, tranlation vector, intrinsics
            # print("3D point: ",Qs[Q_idxs][proj])
            # print("******************************")
            # cnt += 1
            # qs_proj[proj][0] = qs[proj][0]
            # qs_proj[proj][1] = qs[proj][1]
    # print("number of outliers: ", cnt)
    residual = (qs_proj - qs).ravel() # Vægt
    return residual


def bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs):
    params = np.hstack((cam_params.ravel(), Qs.ravel()))
    residual_init = objective(params, cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)
    res = least_squares(objective, params, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))
    return residual_init, res.fun, res.x


def bundle_adjustment_sparsity(n_cams, n_Qs, cam_idxs, Q_idxs):
    m = cam_idxs.size * 2  # number of residuals
    n = n_cams * 9 + n_Qs * 3  # number of parameters
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(cam_idxs.size)
    for s in range(9):
        A[2 * i, cam_idxs * 9 + s] = 1
        A[2 * i + 1, cam_idxs * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cams * 9 + Q_idxs * 3 + s] = 1
        A[2 * i + 1, n_cams * 9 + Q_idxs * 3 + s] = 1

    return A


def bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs, qs, sparse_mat):
    params = np.hstack((cam_params.ravel(), Qs.ravel()))
    residual_init = objective(params, cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)
    res = least_squares(objective, params, jac_sparsity=sparse_mat, verbose=2, x_scale='jac', ftol=1e-1, method='trf',
                        args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))
    return residual_init, res.fun, res.x


def run_BA():
    # if not os.path.isfile(FILE_NAME):
    #     urllib.request.urlretrieve(URL, FILE_NAME)
    # FILE_NAME = 'ourCache/optimize2.txt'
    # cam_params, Qs, cam_idxs, Q_idxs, qs = read_bal_data(FILE_NAME)
    cam_params, Qs, cam_idxs, Q_idxs, qs = read_bal_data("ourCache/BA_file.txt")
    # cam_params_small, Qs_small, cam_idxs_small, Q_idxs_small, qs_small = shrink_problem(1000, cam_params, Qs, cam_idxs, Q_idxs, qs)
    #
    # n_cams_small = cam_params_small.shape[0]
    # n_Qs_small = Qs_small.shape[0]
    # print("n_cameras: {}".format(n_cams_small))
    # print("n_points: {}".format(n_Qs_small))
    # print("Total number of parameters: {}".format(9 * n_cams_small + 3 * n_Qs_small))
    # print("Total number of residuals: {}".format(2 * qs_small.shape[0]))
    #
    # residual_init, residual_minimized, opt_params = bundle_adjustment(cam_params_small, Qs_small, cam_idxs_small, Q_idxs_small, qs_small)
    # x = np.arange(2 * qs_small.shape[0])
    # plt.subplot(2, 1, 1)
    # plt.plot(x, residual_init)
    # plt.title('Bundle adjustment with reduced parameters')
    # plt.ylabel('Initial residuals')
    # plt.subplot(2, 1, 2)
    # plt.plot(x, residual_minimized)
    # plt.ylabel('Optimized residuals')
    # plt.show()

    n_cams = cam_params.shape[0]
    n_Qs = Qs.shape[0]
    print("n_cameras: {}".format(n_cams))
    print("n_points: {}".format(n_Qs))
    print("Total number of parameters: {}".format(9 * n_cams + 3 * n_Qs))
    print("Total number of residuals: {}".format(2 * qs.shape[0]))

    print("n_Qs: ", n_Qs)
    print("max i Q_idxs: ", np.max(Q_idxs))

    # residual_init, residual_minimized, opt_params = bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs)
    A = bundle_adjustment_sparsity(n_cams, n_Qs, cam_idxs, Q_idxs)

    residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs,
                                                                                    qs, A)
    x = np.arange(2 * qs.shape[0])
    plt.subplot(2, 1, 1)
    plt.title('Bundle adjustment with all parameters')
    plt.ylabel('Initial residuals')
    plt.plot(x, residual_init)

    val = np.max(residual_init)
    idx = np.argmax(residual_init)

    print(np.shape(opt_params))

    plt.subplot(2, 1, 2)
    plt.ylabel('Optimized residuals') #rasmus klump
    plt.plot(x, residual_minimized)
    plt.show()


if __name__ == "__main__":
    run_BA()