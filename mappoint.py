import numpy as np

""" Its 3D position Xw, i in the world coordinate system
The viewing    direction     ni, which is the     mean     unit     vector
of     all     its     viewing     directions(the     rays     that
join     the     point    with the optical center of the
keyframes that observe it)
A representative ORB descriptor Di, which is the associated ORB
descriptor whose hamming distance is minimum with respect to all
other associated descriptors in the keyframes in which the point
is observed"""
class MapPoint:
    def __init__(self):
        self.position = np.array([0, 0, 0])
        self.viewing_direction = np.array([0, 0, 0])
        self.D = []
        self.dmin = 0
        self.dmax = 0