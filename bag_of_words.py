import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import cv2
from tqdm import tqdm
from visual_odometry_solution_methods import load_images

class BoW:
    def __init__(self, n_clusters=50, n_features=100):
        self.extractor = cv2.ORB_create(nfeatures=n_features)
        self.n_clusters = n_clusters
        self.kmeans = KMeans(self.n_clusters, verbose=0, n_jobs=-1)

    def train(self, imgs):
        print('Computing local descriptors')
        _, dlist = zip(*[self.extractor.detectAndCompute(img, None) for img in tqdm(imgs)])
        dpool = np.concatenate(dlist)
        self.kmeans = self.kmeans.fit(dpool)
        self.db = [self.hist(d) for d in dlist]

    def hist(self, descriptors):
        labels = self.kmeans.predict(descriptors)
        hist, _ = np.histogram(labels, bins=self.n_clusters, range=(0, self.n_clusters - 1))
        return hist

    def predict(self, img):
        def chi2(x, y):
            return np.sum(2 * (x - y)**2 / (np.maximum(1, x + y)))

        _, d = self.extractor.detectAndCompute(img, None)
        h = self.hist(d)
        dist = [chi2(h, entry) for entry in self.db]
        return np.argmin(dist)


def split_data(dataset, train_size=0.9, test_size=0.1):
    images = dataset
    print('Loading dataset')
    return train_test_split(images, train_size=train_size, test_size=test_size)


def bow_main():
    dataset = load_images("../KITTI_sequence_1/image_l")
    n_features = 100
    n_clusters = 50
    train_img, test_img = split_data(dataset)
    bow = BoW(n_clusters, n_features)
    bow.train(train_img)

    for img in test_img:
        idx = bow.predict(img)
        cv2.imshow("query", img)
        cv2.imshow("match", train_img[idx])
        cv2.waitKey()

bow_main()