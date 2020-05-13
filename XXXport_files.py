import numpy as np
from scipy.spatial.transform import Rotation as R

def clear_textfile(file_path):
    f = open(file_path, "w")
    f.close()


def save3DPoints(file_name, points, frame):
    f = open(file_name, 'a')
    for x, y, z in points:
        f.write(str(x) +", "+ str(y) +", " + str(z)+ ","+ str(frame)+"\n")
    f.close()

def make_cam_params(camera_frames ,P_left):
    cam_params = []
    for j in range(len(camera_frames)):
        rotmat = R.from_matrix(camera_frames[j].pose[:3, :3])
        r_VEC = rotmat.as_rotvec()
        cam_params.append(r_VEC[0])
        cam_params.append(r_VEC[1])
        cam_params.append(r_VEC[2])
        cam_params.append(camera_frames[j].pose[0, 3])
        cam_params.append(camera_frames[j].pose[1, 3])
        cam_params.append(camera_frames[j].pose[2, 3])
        cam_params.append(P_left[0][0])
        cam_params.append(0)
        cam_params.append(0)

    cams = np.array(cam_params)
    return np.transpose(cams)

def make_Qs_for_BA(Qs):
    return_Q = []
    for coords in Qs:
        return_Q.append(coords[0])
        return_Q.append(coords[1])
        return_Q.append(coords[2])
    rQ = np.array(return_Q)
    return np.transpose(rQ)


def export_data(optimization_matrix,camera_frames, Qs, P_left):
    f = open("ourCache/BA_file.txt", "w")

    f.write(str(int(np.max(optimization_matrix[:,0])+1)) + " " + str(int(np.max(optimization_matrix[:,1])+1)) + " " +
            str(int(np.shape(optimization_matrix)[0]))+"\n")

    for obj in optimization_matrix:
        f.write(str(int(obj[0])) + " " + str(int(obj[1])) + " " + str(obj[2]-(1226/2)) + " " + str(obj[3]-370/2) + "\n")

    for a in range(len(camera_frames)-1):
    # for a in range(1):
        rotmat = R.from_matrix(camera_frames[a].pose[:3, :3])
        r_VEC = rotmat.as_rotvec()
        f.write(str(r_VEC[0]) + "\n" + str(r_VEC[1]) + "\n" + str(r_VEC[2]) + "\n")
        f.write(str(camera_frames[a].pose[0, 3]) + "\n" + str(camera_frames[a].pose[1, 3]) + "\n" + str(
            camera_frames[a].pose[2, 3]) + "\n")
        f.write(str(P_left[0][0]) + "\n0\n0\n")

    for coords in Qs:
        f.write(str(coords[0]) + "\n" + str(coords[1]) + "\n" + str(coords[2]) + "\n")
    f.close()