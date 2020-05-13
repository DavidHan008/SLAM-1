from bag_of_words import *

# LOOP CLOSURE
# Loop Candidates Detection
# Compute Similarity Transform
# Loop Fusion
# Essential Graph Optimization
"""At first we compute the similarity between the bag of words vector of Ki and all its neighbors in the
covisibility graph (θmin = 30) and retain the lowest score smin.
Then we query the recognition database and discard all those keyframes whose score
is lower than smin."""
"""We extract ORB features in 10K images from the dataset and build a
vocabulary of 6 levels and 10 clusters per level, getting one
million words. Such a big vocabulary is suggested in [6] to
be efficient for recognition in large image databases."""
#CITE LOOP CLOSURE
image_path = "C:/Users/janus/Downloads/data_odometry_gray/dataset/sequences/06"
#image_path = "../KITTI_sequence_1"
n_clusters = 50
n_features = 100

# Load the images of the left and right camera
leftimages = load_images(os.path.join(image_path, "image_0"))
rightimages = load_images(os.path.join(image_path, "image_1"))

bow = BoW(n_clusters, n_features)
bow.train(leftimages[:100])

for i in range(830, 900):#len(leftimages)-1):
    idx, val = bow.predict_previous(leftimages[i], i, 0)
    print(idx, val)
    if val < 50:
        cv2.imshow("query", leftimages[i])
        cv2.imshow("match", leftimages[idx])
        cv2.waitKey()