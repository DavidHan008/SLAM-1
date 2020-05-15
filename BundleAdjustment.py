import urllib.request
import bz2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from transformation import *
import cv2
from scipy.spatial.transform import Rotation as R

file = open("ourCache/equal_frames.txt", "r")
first_image, second_image = map(int, file.readline().split())
file.close()

def load_data(file_name):
    number_of_frames = 1100
    return_val = np.empty((number_of_frames,6))
    file = open(file_name, 'r')

    for j in range(number_of_frames):
        nul, et, to, tre, fire, fem= map(float, file.readline().split())
        return_val[j,0] = nul
        return_val[j, 1] = et
        return_val[j, 2] = to
        return_val[j, 3] = tre
        return_val[j, 4] = fire
        return_val[j, 5] = fem


    return return_val.ravel()# np.reshape(return_val, ((-1, 1))).ravel()



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

def objective_without_loop_closure(car_params):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.

    cost function should create costs:
    change of x high cost
    change of y high cost
    change of z low cost
    rotation about x medium cost
    rotation about y low cost
    rotation about z high cost

    when run due to loop closure, there should be an additional contraint that ensures that final frame is \
    identical to initial frame
    """
    car_params_local = np.reshape(car_params,(-1,6))
    x_cost=1
    y_cost=1
    z_cost=0.0005
    rotx_cost=0.5
    roty_cost=0.05
    rotz_cost=1
    cost = abs(car_params_local[:,0])*rotx_cost
    cost += abs(car_params_local[:,1])*roty_cost
    cost += abs(car_params_local[:,2])*rotz_cost
    cost += abs(car_params_local[:,3])*x_cost
    cost += abs(car_params_local[:,4])*y_cost
    cost += (abs(car_params_local[:,5])-1)*z_cost #evt (car_params-1)*z_cost   , eller (car_params- (1/|rot_vec|))*z_cost
    return cost

def objective(car_params):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.

    cost function should create costs:
    change of x high cost
    change of y high cost
    change of z low cost
    rotation about x medium cost
    rotation about y low cost
    rotation about z high cost

    when run due to loop closure, there should be an additional contraint that ensures that final frame is \
    identical to initial frame
    """
    car_params_local = np.reshape(car_params,(-1,6))
    x_cost=1
    y_cost=1
    z_cost=0.0005
    rotx_cost=0.5
    roty_cost=0.05
    rotz_cost=1
    cost = abs(car_params_local[:,0])*rotx_cost
    cost += abs(car_params_local[:,1])*roty_cost
    cost += abs(car_params_local[:,2])*rotz_cost
    cost += abs(car_params_local[:,3])*x_cost
    cost += abs(car_params_local[:,4])*y_cost
    cost += (abs(car_params_local[:,5])-1)*z_cost #evt (car_params-1)*z_cost   , eller (car_params- (1/|rot_vec|))*z_cost
    rel_tranny = [translation_and_rotation_vector_to_matrix(car_params_local[i][0:3],car_params_local[i][3:6]) for i in range(np.shape(car_params_local)[0])]
    abs_tranny = np.full((len(rel_tranny)+1,4,4),np.eye(4))
    for r in range(1, len(rel_tranny)+1):
        abs_tranny[r] = np.matmul(abs_tranny[r-1], rel_tranny[r-1])
    cost_trans = np.sum(np.abs(np.subtract(abs_tranny[0][0:3,3], abs_tranny[-1][0:3,3]))) * 1000
    cost_rotm = np.sum(np.abs(np.subtract(100*abs_tranny[0][0:3,0:3], 100*abs_tranny[-1][0:3,0:3]))) * 1000
    cost = np.hstack((cost,cost_trans))
    cost = np.hstack((cost, cost_rotm))
    return cost

def bundle_adjustment_sparsity_without_loop_closure(car_params):#n_cams, n_Qs, cam_idxs, Q_idxs):
    m = int(np.shape(car_params)[0]/6 )  # number of residuals
    n = np.shape(car_params)[0]  # number of parameters
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(int(np.shape(car_params)[0]/6))
    for s in range(6):
        A[i, i * 6 + s] = 1



    return A

def bundle_adjustment_sparsity(car_params):#n_cams, n_Qs, cam_idxs, Q_idxs):
    m = int(np.shape(car_params)[0]/6 + 2)  # number of residuals
    n = np.shape(car_params)[0]  # number of parameters
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(int(np.shape(car_params)[0]/6))
    for s in range(6):
        A[i, i * 6 + s] = 1
    for s in range(n):
        A[m - 1, s] = 1
        A[m - 2, s] = 1

    return A

def bundle_adjustment_with_sparsity_without_loop_closure(car_params, sparse_mat):#, Qs, cam_idxs, Q_idxs, qs, sparse_mat):

    residual_init = objective_without_loop_closure(car_params)
    res = least_squares(objective, car_params, jac_sparsity=sparse_mat, verbose=2, x_scale='jac', ftol=1e-1, method='trf')
    return residual_init, res.fun, res.x

def bundle_adjustment_with_sparsity(car_params, sparse_mat):#, Qs, cam_idxs, Q_idxs, qs, sparse_mat):

    residual_init = objective(car_params)
    res = least_squares(objective, car_params, jac_sparsity=sparse_mat, verbose=2, x_scale='jac', ftol=1e-1, method='trf')
    return residual_init, res.fun, res.x

 # jac_sparsity=sparse_mat,
def run_BA():

    car_params= load_data("ourCache/cam_frames_relative.txt")

    car_params_new = car_params[(first_image*6):(second_image*6)]
    A = bundle_adjustment_sparsity(car_params_new)

    residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(car_params_new, A)


    car_params[(first_image * 6):(second_image * 6)] = opt_params

    A_without_loop = bundle_adjustment_sparsity_without_loop_closure(car_params)

    residual_init_without_loop, residual_minimized_without_loop, opt_params_without_loop = bundle_adjustment_with_sparsity_without_loop_closure(car_params, A_without_loop)

    hep = np.reshape(opt_params_without_loop, (-1,6))

    f = open("ourCache/optimized_paramters.txt", "w")
    for val in hep:
        for p in range(6):
            f.write(str(val[p])+" ")
        f.write("\n")
    f.close()


    # x = np.arange(int(car_params.shape[0]/6+1))
    x = np.arange(len((first_image * 6),(second_image * 6)+1))
    plt.subplot(2, 1, 1)
    plt.title('Bundle adjustment with all parameters')
    plt.ylabel('Initial residuals')
    plt.plot(2, residual_init)

    val = np.max(residual_init)
    idx = np.argmax(residual_init)

    plt.subplot(2, 1, 2)
    plt.ylabel('Optimized residuals') #rasmus klump
    plt.plot(x, residual_minimized)
    plt.show()

if __name__ == "__main__":
    run_BA()

"""
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
    

    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(Qs * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * Qs + sin_theta * np.cross(v, Qs) + dot * (1 - cos_theta) * v

# def project(Qs, cam_params):
#     f = float(cam_params[0][6])
#     cam_mat = np.array([[f, 0,0],[0, f, 0],[0,0,0]])
#     temp,_ = cv2.projectPoints(Qs[0], cam_params[0][:3], cam_params[0][3:6], cam_mat, np.zeros(4))
#     temp = np.array(temp.ravel())
#     qs_proj = np.zeros((np.shape(Qs)[0], 2))
#     qs_proj[0,0] = temp[0]
#     qs_proj[0,1] = temp[1]
#     for pik in range(1,len(Qs)):
#         qs_temp,_ = cv2.projectPoints(Qs[pik], cam_params[pik][:3], cam_params[pik][3:6], cam_mat, np.zeros(4))
#         qs_temp = qs_temp.ravel()
#         qs_proj[pik, 0] = qs_temp[0]
#         qs_proj[pik, 1] = qs_temp[1]
#
#     return qs_proj


def project(Qs, cam_params):
    qs_proj = rotate(Qs, cam_params[:, :3])
    qs_proj += cam_params[:, 3:6]
    # print(np.shape(qs_proj))
    #print(qs_proj)
    # print(qs_proj[2])
    qs_proj = -qs_proj[:, :2] / qs_proj[:, 2, np.newaxis]
    f, k1, k2 = cam_params[:, 6:].T
    n = np.sum(qs_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    qs_proj *= (r * f)[:, np.newaxis]
    return qs_proj


def objective(params, n_cams, n_Qs, cam_idxs, Q_idxs, qs):

    cam_params = params[:n_cams * 9].reshape((n_cams, 9))
    Qs = params[n_cams * 9:].reshape((n_Qs, 3))
    qs_proj = project(Qs[Q_idxs], cam_params[cam_idxs])
    # cnt = 0
    residual = (qs_proj - qs) # Vægt
    # print(np.shape(residual))
    big_residual_index_0 = np.where(abs(residual[:,0]) > 5000)

    big_residual_index_0 = np.unique(big_residual_index_0)
    residual[big_residual_index_0] = np.true_divide(residual[big_residual_index_0],
                                                    np.abs(residual[big_residual_index_0][:,0][:,None]))*613*2
    # print("her: ", residual[big_residual_index_0] / (residual[big_residual_index_0]**2).sum()**0.5)


    big_residual_index_1 = np.where(abs(residual[:,1]) > 5000)
    big_residual_index_1 = np.unique(big_residual_index_1)
    residual[big_residual_index_1] = np.true_divide(residual[big_residual_index_1],
                                                    np.abs(residual[big_residual_index_1][:,1][:,None]))*185*2


    # for proj in range(0, len(qs_proj)):
    #     if abs(qs_proj[proj][0] - qs[proj][0])   qs_proj[proj][0]) > 100 or abs(qs_proj[proj][1]) > 100:# or proj == 204:
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
    # residual = (qs_proj - qs).ravel() # Vægt
    residual = residual.ravel()
    print("max error: ", np.max(residual))
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
    
    """