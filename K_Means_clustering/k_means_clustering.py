import math
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np


# Load and shuffle data
def load_data(file_name):
    with open(file_name, 'r') as reader:
        lines = reader. readlines()

        x_data = [[float(i) for i in line.rstrip().split(',')[:-1]] for line in lines]
        y_data = [line.rstrip().split(',')[-1] for line in lines]

        # Load any two classes for classification
        temp_x = []
        temp_y = []
        for x, y in zip(x_data, y_data):
            temp_x.append(x)
            if y == 'Iris-setosa':
                temp_y.append(0)
            elif y == 'Iris-versicolor':
                temp_y.append(1)
            elif y == 'Iris-virginica':
                temp_y.append(2)

        x_data = np.array(temp_x)
        y_data = np.array(temp_y)

        # # shuffle data, doesn't matter if we do it or not
        # idx = [i for i in range(len(y_data))]
        # np.random.shuffle(idx)
        #
        # x_data = np.array([x_data[curr_idx] for curr_idx in idx])
        # y_data = np.array([y_data[curr_idx] for curr_idx in idx])

    return x_data, y_data


def k_means(x_data, y_data, n_iter=15):

    n_clusters = len(set(y_data))

    # define n_clusters cluster centroids from x_data randomly
    cluster_cent = np.array([x_data[idx] for idx in np.random.random_integers(len(x_data), size=n_clusters)])

    print("Initial Cluster Centroids :\n", cluster_cent)

    for i in range(n_iter):
        # calculate distance of points from cluster centroids
        d1 = np.linalg.norm(x_data - cluster_cent[0], axis=-1, keepdims=True)
        d2 = np.linalg.norm(x_data - cluster_cent[1], axis=-1, keepdims=True)
        d3 = np.linalg.norm(x_data - cluster_cent[2], axis=-1, keepdims=True)

        dist = np.concatenate([d1, d2, d3], axis=-1)

        # assign cluster centroid
        # y_hat is the predicted cluster
        y_hat = np.argmin(dist, axis=-1)

        # Update cluster centroid
        cluster_cent[0] = np.sum([x if y_h == 0 else np.zeros(shape=(1, x_data.shape[-1])) for x, y_h in zip(x_data, y_hat)], axis=0)\
                          / np.count_nonzero(y_hat == 0)
        cluster_cent[1] = np.sum([x if y_h == 1 else np.zeros(shape=(1, x_data.shape[-1])) for x, y_h in zip(x_data, y_hat)], axis=0) \
                          / np.count_nonzero(y_hat == 1)
        cluster_cent[2] = np.sum([x if y_h == 2 else np.zeros(shape=(1, x_data.shape[-1])) for x, y_h in zip(x_data, y_hat)], axis=0) \
                          / np.count_nonzero(y_hat == 2)

    print("\nFinal Cluster Centroids :\n", cluster_cent)


if __name__ == "__main__":
    x, y = load_data('iris.txt')

    # The number of clusters is automatically detected from x
    k_means(x, y, n_iter=15)
