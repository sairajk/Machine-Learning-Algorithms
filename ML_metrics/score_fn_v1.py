from sklearn.utils import check_X_y
from sklearn.utils import safe_indexing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
import numpy as np


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
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    z_tot = np.mean(X, axis=0)

    cluster_len = np.zeros(n_labels)
    intra_dists = np.zeros(n_labels)
    intra_centroid_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=np.float)

    for k in range(n_labels):
        cluster_k = safe_indexing(X, labels == k)
        cluster_len[k] = len(cluster_k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid

        intra_centroid_dists[k] = np.square(pairwise_distances([centroid], [z_tot])) * cluster_len[k]
        intra_dists[k] = np.sqrt(np.square(pairwise_distances(cluster_k, [centroid])) / cluster_len[k])

    bcd = np.mean(intra_centroid_dists) / n_samples
    wcd = np.mean(intra_dists)

    sf = 1 - 1 / np.exp(np.exp(bcd - wcd))

    return sf
