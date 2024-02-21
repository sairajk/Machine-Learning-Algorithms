from sklearn.utils import check_X_y
from sklearn.utils import safe_indexing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder

from time import time
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.preprocessing import scale


loader = load_iris()
data = loader.data

n_samples, n_features = data.shape
unique_labels = len(np.unique(loader.target))
labels = loader.target

print("unique_labels: %d, \t n_samples %d, \t n_features %d"
      % (unique_labels, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\t\tScore_Fn_est\t\tScore_Fn_label')


def check_number_of_labels(n_labels, n_samples):
    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are 2 "
                         "to n_samples - 1 (inclusive)" % n_labels)


def score_function(X, labels):
    """Computes Score Function.

    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like, shape (``n_samples``,)
        Predicted labels for each sample.

    Returns
    -------
    sf: float
        The resulting Score Function.

    References
    ----------
    .. [1] Saitta S., Raphael B., Smith I.F.C. (2007).
        `"A Bounded Index for Cluster Validity"
        <https://dl.acm.org/citation.cfm?id=1420344>`__.
        Perner P. (eds) Machine Learning and Data Mining in Pattern Recognition.
        Lecture Notes in Computer Science, vol 4571.
    """
    # Pre-processing and validation of input data and labels
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    
    check_number_of_labels(n_labels, n_samples)
    
    # Mean of the entire data. Same as z_tot in the original paper.
    data_mean = np.mean(X, axis=0)

    # cluster_size : is a 1D-array of length "n_labels" (i.e. number of unique labels or number of unique clusters)
    # It stores the number of data points in each cluster.
    cluster_size = np.zeros(n_labels)
    # intra_dists: It stores the mean euclidean distance between all points belonging to a cluster and their cluster
    # centroid.
    # Since there are "n_labels" unique clusters, the variable is a 1D-array of length "n_labels".
    intra_dists = np.zeros(n_labels)
    # intra_centroid_dists: It stores the product of euclidean distance between a cluster centroid and the centroid of
    # the entire data to the number of elements of that cluster.
    # Acc. to paper : intra_centroid_dists[i] = ||z_i - z_tot|| * n_i, where "i" is a particular cluster.
    # Since there are "n_labels" unique clusters, the variable is a 1D-array of length "n_labels".
    intra_centroid_dists = np.zeros(n_labels)
    # centroids: stores the centroid of each cluster.
    centroids = np.zeros((n_labels, len(X[0])), dtype=np.float)

    for k in range(n_labels):
        # Retrieve all data points belonging to cluster "k".
        cluster_k = safe_indexing(X, labels == k)
        # Find number of data points in cluster "k" and save them.
        cluster_size[k] = len(cluster_k)
        # Finding the centroid of cluster_k
        centroids[k] = cluster_k.mean(axis=0)

        # Compute intra_centroid_dists[k] acc. to the formula in the paper :
        # intra_centroid_dists[k] = ||z_k - z_tot|| * n_k, where "k" is a particular cluster.
        intra_centroid_dists[k] = pairwise_distances([centroids[k]], [data_mean], metric='euclidean') * cluster_size[k]
        # Compute mean euclidean distance between all points belonging to cluster "k" and the centroid of cluster "k".
        intra_dists[k] = np.mean(pairwise_distances(cluster_k, [centroids[k]], metric='euclidean'))

    # Compute "between class distance"(bcd).
    bcd = np.mean(intra_centroid_dists) / n_samples
    # Compute "within class distance"(wcd).
    wcd = np.sum(intra_dists)

    # return score function
    return 1 - 1 / np.exp(np.exp(bcd - wcd))


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t\t%.7f,\t\t%.7f'
          % (name, (time() - t0),
             score_function(data, estimator.labels_),
             score_function(data, labels)))


bench_k_means(KMeans(init='k-means++', n_clusters=unique_labels, n_init=10, max_iter=500, tol=1e-4),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=unique_labels, n_init=10, max_iter=500, tol=1e-4),
              name="random", data=data)

print(82 * '_')



# OTHER's IMPLEMENTATION OF SCORE FUNCTION


from math import sqrt, log


class internal_indices:

    def __init__(self, data, labels, distance_matrix=None):

        # normalising labels
        le = LabelEncoder()
        le.fit(labels)

        # initialise class memebers
        self.data = np.array(data)
        '''Note: Treats noise as a seperate (K + 1 th) partition

        References :	[1]	https://stats.stackexchange.com/questions/291566/cluster-validation-of-incomplete-clustering-algorithms-esp-density-based-db
        '''
        self.labels = le.transform(labels)

        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]

        # compute mean of data
        self.data_mean = np.mean(self.data, axis=0)

        # self.n_clusters=np.unique([x for x in self.labels if x>=0]).shape[0] (to avoid noise)
        self.n_clusters = np.unique(self.labels).shape[0]
        self.clusters_mean = np.zeros((self.n_clusters, self.n_features))
        self.clusters_size = np.zeros(self.n_clusters)

        self.insignificant = 1 / pow(10, 20)
        for cluster_label in range(self.n_clusters):
            # if cluster_label >=0  (to avoid noise)
            cluster_i_pts = (self.labels == cluster_label)
            self.clusters_size[cluster_label] = np.sum(cluster_i_pts)
            self.clusters_mean[cluster_label] = np.mean(self.data[cluster_i_pts], axis=0)

        if distance_matrix is not None:
            self.distance_matrix = distance_matrix

        # print(self.clusters_mean)
        self.compute_scatter_matrices()

    def compute_scatter_matrices(self):
        """
        References:	[1] Clustering Indices, Bernard Desgraupes (April 2013)
                    [2] http://sebastianraschka.com/Articles/2014_python_lda.html (Linear Discriminatory Analysis)
                    [3] Chapter 4, Clustering -- RUI XU,DONALD C. WUNSCH, II (IEEE Press)
        verified with data from References [2]
        """
        self.T = total_scatter_matrix(self.data)

        # WG_clusters : WG matrix for each cluster | WGSS_clusters : trace(WG matrix) for each cluster
        self.WG_clusters = np.empty((self.n_clusters, self.n_features, self.n_features), dtype=np.float64)
        self.WGSS_clusters = np.zeros(self.n_clusters, dtype=np.float64)

        # self.BG = np.zeros((self.n_features,self.n_features),dtype=np.float64)

        for cluster_label in range(self.n_clusters):
            # compute within cluster matrix
            self.WG_clusters[cluster_label] = total_scatter_matrix(self.data[self.labels == cluster_label])
            self.WGSS_clusters[cluster_label] = np.trace(self.WG_clusters[cluster_label])

        self.WGSS_clusters_non_zeros_indices = [i for i, e in enumerate(self.WGSS_clusters) if e != 0]

        self.WGSS_clusters_non_zeros_indices_size = len(self.WGSS_clusters_non_zeros_indices)

        # print(self.WGSS_clusters_non_zeros_indices_size)
        self.WGSS_clusters_non_zeros_BR_index = np.zeros(self.WGSS_clusters_non_zeros_indices_size, dtype=np.float64)
        self.clusters_size_BR_index = np.zeros(self.WGSS_clusters_non_zeros_indices_size, dtype=np.float64)
        for cluster_index in range(self.WGSS_clusters_non_zeros_indices_size):
            self.WGSS_clusters_non_zeros_BR_index[cluster_index] = self.WGSS_clusters[
                self.WGSS_clusters_non_zeros_indices[cluster_index]]
            self.clusters_size_BR_index[cluster_index] = self.clusters_size[
                self.WGSS_clusters_non_zeros_indices[cluster_index]]

            # compute between-cluster matrix
            mean_vec = self.clusters_mean[cluster_label].reshape((self.n_features, 1))
            overall_mean = self.data_mean.reshape((self.n_features, 1))

        # self.BG += np.array(self.clusters_size[i]*(mean_vec - overall_mean).dot((mean_vec - overall_mean).T),dtype=np.float64)
        # self.BG = self.BG + self.clusters_size[i]*np.dot(cluster_data_mean_diff.T,cluster_data_mean_diff)

        self.WG = np.sum(self.WG_clusters, axis=0)

        # print(self.WG)

        self.WGSS = np.trace(self.WG)

        self.BG = self.T - self.WG
        # print(self.BG)

        self.BGSS = np.trace(self.BG)

        self.det_WG = np.linalg.det(self.WG)
        self.det_T = np.linalg.det(self.T)

    def score_function(self):
        """Score Function - SF : works good for hyper-speriodal data
        References :	[1]	https://pdfs.semanticscholar.org/9701/405b0d601e169636a2541940a070087acd5b.PDF
        range : ]0,1[ | rule : max
        """
        # is centroid of all clusters = data_mean
        bcd = np.sum(np.sqrt(np.sum((self.clusters_mean - self.data_mean) ** 2, axis=1)) * self.clusters_size) / (
                self.n_clusters * self.n_samples)

        # print("bcd",bcd)

        clusters_dispersion = np.zeros(self.n_clusters)
        for data_index in range(self.n_samples):
            clusters_dispersion[self.labels[data_index]] += euclidean_distance(self.data[data_index],
                                                                               self.clusters_mean[
                                                                                   self.labels[data_index]])

        wcd = np.sum(clusters_dispersion / self.clusters_size)
        # print("wcd", wcd)
        return (1 - 1 / np.exp(np.exp(bcd - wcd)))

# internal indices -- end

# helper functions -- start
def total_scatter_matrix(data):
    """
    Total sum of square (TSS) : sum of squared distances of points around the baycentre
    References : Clustering Indices, Bernard Desgraupes (April 2013)
    """
    X = np.array(data.T.copy(), dtype=np.float64)

    for feature_i in range(data.shape[1]):
        X[feature_i] = X[feature_i] - np.mean(X[feature_i])

    T = np.dot(X, X.T)
    return T


def euclidean_distance(vector1, vector2, squared=False):
    """calculates euclidean distance between two vectors

    Keyword arguments:
    vector1 -- first data point (type: numpy array)
    vector2 -- second data point (type: numpy array)
    squared -- return square of euclidean distance (default: False)
    """
    euclidean_distance = np.sum((vector1 - vector2) ** 2)

    if squared is False:
        euclidean_distance = sqrt(euclidean_distance)

    return euclidean_distance


est = KMeans(init='random', n_clusters=unique_labels, n_init=10, max_iter=500, tol=1e-4)
est.fit(data)
ind_est = internal_indices(data, est.labels_)
sf_est = ind_est.score_function()
print("\nScore Function (by estimated labels) by other's method :", sf_est)

ind = internal_indices(data, labels)
sf = ind.score_function()
print("Score Function (by ground-truth labels) by other's method :", sf)
