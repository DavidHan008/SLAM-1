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

if __name__ == '__main__':
    main()
