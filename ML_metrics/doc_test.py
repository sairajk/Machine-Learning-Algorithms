import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils import safe_indexing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder


def check_number_of_labels(n_labels, n_samples):
    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are 2 "
                         "to n_samples - 1 (inclusive)" % n_labels)


def score_function(X, labels):
    """Computes Score Function.

    Score Function is a bounded Cluster Validity Index. It is based on
    standard cluster properties and is proved to be better than or atleast as
    good as existing validation indices for hyperspheroidal clusters. The
    score function is a combination of two terms: the distance between
    clusters (intercluster distance) and the distance inside a cluster (intra-
    cluster distance). Also, the score function can used for two purposes:
    1) to estimate the number of clusters; and
    2) to evaluate the quality of the clustering results.

    DRAWBACKS OF EXISTING VALIDITY INDICES :
    Most current validity indices only cover a subset of important aspects of
    clusters. Moreover, these indices are relevant only for data sets
    containing at least two clusters. As a result, these indices are useful in
    certain situations and are not of general-purpose.

    The score function is tested against four existing validity indices, that
    are Dunn index, Davies-Bouldin index, Silhouette index, Maulik-
    Bandyopadhyay index on several artificial and real-life data sets to
    evaluate the performance of the score function. It is found to be always
    as good or better than these indices in the case of hyperspheroidal
    clusters. It also works well on multidimensional data sets and is able to
    accommodate unique and sub-cluster cases.
    * Score Function has a linear computational complexity.

    # DOCTEST FOR SCORE FUNCTION

    >>> from sklearn.cluster import KMeans
    >>> from sklearn.datasets import load_iris, load_wine
    >>>
    >>> # TEST ON IRIS DATA
    >>>
    >>> loader = load_iris()
    >>> data = loader.data
    >>>
    >>> n_samples, n_features = data.shape
    >>> unique_labels = len(np.unique(loader.target))
    >>> labels = loader.target
    >>> est = KMeans(init='random', n_clusters=unique_labels, n_init=10)
    >>> fit_data = est.fit(data)
    >>> score_function(data, est.labels_)
    0.521
    >>>
    >>> # TEST ON WINE DATA
    >>>
    >>> loader = load_wine()
    >>> data = loader.data
    >>>
    >>> n_samples, n_features = data.shape
    >>> unique_labels = len(np.unique(loader.target))
    >>> labels = loader.target
    >>> est = KMeans(init='random', n_clusters=unique_labels, n_init=10)
    >>> fit_data = est.fit(data)
    >>> score_function(data, est.labels_)
    0.161

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
        Perner P.(eds) Machine Learning and Data Mining in Pattern Recognition.
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

    # cluster_size : is a 1D-array of length "n_labels"
    # (i.e. number of unique labels or number of unique clusters)
    # It stores the number of data points in each cluster.
    cluster_size = np.zeros(n_labels)
    # intra_dists: It stores the mean euclidean distance between
    # all points belonging to a cluster and their cluster centroid.
    # Since there are "n_labels" unique clusters, the variable is
    # a 1D-array of length "n_labels".
    intra_dists = np.zeros(n_labels)
    # intra_centroid_dists: It stores the product of euclidean distance
    # between a cluster centroid and the centroid of
    # the entire data to the number of elements of that cluster.
    # Acc. to paper : intra_centroid_dists[i] = ||z_i - z_tot|| * n_i,
    # where "i" is a particular cluster.
    # Since there are "n_labels" unique clusters, the variable is a
    # 1D-array of length "n_labels".
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
        # intra_centroid_dists[k] = ||z_k - z_tot|| * n_k,
        # where "k" is a particular cluster.
        intra_centroid_dists[k] = \
            pairwise_distances([centroids[k]], [data_mean],
                               metric='euclidean') * cluster_size[k]
        # Compute mean euclidean distance between all points belonging to
        # cluster "k" and the centroid of cluster "k".
        intra_dists[k] = np.mean(pairwise_distances(cluster_k, [centroids[k]],
                                                    metric='euclidean'))

    # Compute "between class distance"(bcd).
    bcd = np.mean(intra_centroid_dists) / n_samples
    # Compute "within class distance"(wcd).
    wcd = np.sum(intra_dists)

    # return score function
    return 1 - 1 / np.exp(np.exp(bcd - wcd))
