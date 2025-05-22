# Out of core metrics for clustering using Dask
# I have tested each one with 10 seeds using 1e7 random integers (0, 5), and they all
# returned the same results as sklearn.
import dask.array as da
import numpy as np
from math import log
from sklearn.metrics.cluster._supervised import expected_mutual_information, entropy, _generalized_average


def contingency_matrix(labels_true, labels_pred, dtype=np.int64):
    labels_true = da.asarray(labels_true)
    labels_pred = da.asarray(labels_pred)

    n_classes = da.unique(labels_true).compute().shape[0]
    n_clusters = da.unique(labels_pred).compute().shape[0]

    contingency = da.histogram2d(
        labels_true, labels_pred,
        bins=[n_classes, n_clusters],
        range=[[0, n_classes], [0, n_clusters]]
    )[0].astype(dtype)
    return contingency


def check_clusterings(labels_true, labels_pred):
    # input checks
    if labels_true.ndim != 1:
        raise ValueError("labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError("labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError(
            "labels_true and labels_pred must have the same number of samples: "
            "%d != %d" % (labels_true.shape[0], labels_pred.shape[0])
        )
    return labels_true, labels_pred


def pair_confusion_matrix(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]

    # Computation using the contingency data
    contingency = contingency_matrix(
        labels_true, labels_pred, dtype=np.int64, # sparse=True,
    )
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency**2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples**2 - C[0, 1] - C[1, 0] - sum_squares
    return C


def adjusted_rand_score(labels_true, labels_pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def rand_score(labels_true, labels_pred):
    contingency = pair_confusion_matrix(labels_true, labels_pred)
    numerator = contingency.diagonal().sum()
    denominator = contingency.sum()

    if numerator == denominator or denominator == 0:
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique
        # cluster. These are perfect matches hence return 1.0.
        return 1.0

    return numerator / denominator


def mutual_info_score(labels_true, labels_pred, *, contingency=None):
    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred)

    contingency = contingency.compute()

    nzx, nzy = np.nonzero(contingency)
    nz_val = contingency[nzx, nzy]

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    # Since MI <= min(H(X), H(Y)), any labelling with zero entropy, i.e. containing a
    # single cluster, implies MI = 0
    if pi.size == 1 or pj.size == 1:
        return 0.0

    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
        np.int64, copy=False
    )
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)

def adjusted_mutual_info_score(
    labels_true, labels_pred, *, average_method="arithmetic"
):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    # It corresponds to both labellings having zero entropy.
    # This is a perfect match hence return 1.0.
    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    # Calculate the expected value for the mutual information
    if isinstance(contingency, da.Array):
        contingency = contingency.compute()
    emi = expected_mutual_information(contingency, n_samples)
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    normalizer = _generalized_average(h_true, h_pred, average_method)
    denominator = normalizer - emi
    # Avoid 0.0 / 0.0 when expectation equals maximum, i.e. a perfect match.
    # normalizer should always be >= emi, but because of floating-point
    # representation, sometimes emi is slightly larger. Correct this
    # by preserving the sign.
    if denominator < 0:
        denominator = min(denominator, -np.finfo("float64").eps)
    else:
        denominator = max(denominator, np.finfo("float64").eps)
    ami = (mi - emi) / denominator
    return ami


def normalized_mutual_info_score(
    labels_true, labels_pred, *, average_method="arithmetic"
):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    # It corresponds to both labellings having zero entropy.
    # This is a perfect match hence return 1.0.
    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)

    # At this point mi = 0 can't be a perfect match (the special case of a single
    # cluster has been dealt with before). Hence, if mi = 0, the nmi must be 0 whatever
    # the normalization.
    if mi == 0:
        return 0.0

    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)

    normalizer = _generalized_average(h_true, h_pred, average_method)
    return mi / normalizer


def homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    if len(labels_true) == 0:
        return 1.0, 1.0, 1.0

    entropy_C = entropy(labels_true)
    entropy_K = entropy(labels_pred)

    contingency = contingency_matrix(labels_true, labels_pred)
    MI = mutual_info_score(None, None, contingency=contingency)

    homogeneity = MI / (entropy_C) if entropy_C else 1.0
    completeness = MI / (entropy_K) if entropy_K else 1.0

    if homogeneity + completeness == 0.0:
        v_measure_score = 0.0
    else:
        v_measure_score = (
            (1 + beta)
            * homogeneity
            * completeness 
            / (beta * homogeneity + completeness)
        )

    return homogeneity, completeness, v_measure_score
