def clear_textfile(file_path):
    f = open(file_path, "w")
    f.close()


def save3DPoints(file_name, points, frame):
    f = open("3DPoints.txt", 'a')
    for x, y, z in points:
        f.write(str(x) +", "+ str(y) +", " + str(z)+ ","+ str(frame)+"\n")
    f.close()