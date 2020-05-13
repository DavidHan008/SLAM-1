import numpy as np

class KeyFrame:
    def __init__(self, pose):
        self.pose = pose
        self.intrinsics = 0
        self.orb_features = []
