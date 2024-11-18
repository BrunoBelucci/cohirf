# This is adapted from the MATLAB code provided by Yohai Devir at https://mathworks.com/matlabcentral/fileexchange/13443-k-means-projective-clustering
# Here is some comments from the author (for the matlab code)
# This function perform projective clustering as described in "k-means
# projective clustering" by Agarwal & Mustafa.
# (C) Yohai Devir, yohai_devir AT YAH00 D0T C0M
# arguments:
# pointsCoords - (I) - DxN coordination matrix
# k            - (I) - number of desired clusters
# options      - (I) - options struct (square brackets - default value):
#     Threshold     - (double)  - minimal rms difference between two iterations [1e-3]
#     KSMIters      - (integer) - number of k-means iterations in alg 3 [10]
#     IsConstantDim - (logical) - do all clusters have the same dimentionality [true]
#     findDimAlfa   - (double)  - alfa parameter for finding cluster dimentionality. [0.5]
#     constFlatDim  - (integer) - dimentionality of every cluster. [2]
#     gamma         - (double)  - gamma parameter for cluster splitting. [0.3]
# clustIndices - (O) - 1*N vector of the cluster number each point belongs to
# ssdVector    - (O) - rms in each iteration
# flatsStructV - (O) - representation of each Q-dimensional subspace in D-dimensional space:
#     P0      - Dx1 vector of a point in this subspace.
#     Vectors - DxQ matrix of Q orthonormal vectors that spans the subspace.
# Few notes:
# A. Unlike the article clusters are selected to be merged if their merging gives the
#    smallest rms.
# B. rms is refered as ssd (sum of squared distances)
# C. options is an optional parameter.

import numpy as np
from scipy.linalg import svd
from sklearn.base import ClusterMixin, BaseEstimator


def k_means_proj_clustering(points_coords, k, threshold=1e-3, kms_iters=10, fdpkm_iters=10, is_constant_dim=True,
                            find_dim_alfa=0.5, const_flat_dim=2, gamma=0.3):
    clust_indices = np.random.randint(1, k + 1, points_coords.shape[1])
    flats_struct_v = calc_flats(clust_indices, points_coords, is_constant_dim, const_flat_dim, find_dim_alfa)
    ssd_vector = [calc_ssd(clust_indices, points_coords, flats_struct_v)]

    for _ in range(kms_iters):
        clust_indices = split_clusters(clust_indices, points_coords, round(k / 1), gamma, is_constant_dim,
                                       const_flat_dim, find_dim_alfa)
        clust_indices = proj_kmeans(clust_indices, points_coords, fdpkm_iters, is_constant_dim, const_flat_dim,
                                    find_dim_alfa)
        clust_indices = merge_clusters(clust_indices, points_coords, min(round(k / 1), max(clust_indices) - k),
                                       is_constant_dim, const_flat_dim, find_dim_alfa)
        clust_indices = remove_empty_clusters(clust_indices)
        new_ssd = calc_ssd(clust_indices, points_coords,
                           calc_flats(clust_indices, points_coords, is_constant_dim, const_flat_dim, find_dim_alfa))
        ssd_vector.append(new_ssd)

        if new_ssd == 0 or abs(ssd_vector[-1] - ssd_vector[-2]) / ssd_vector[-1] < threshold:
            break

    flats_struct_v = calc_flats(clust_indices, points_coords, is_constant_dim, const_flat_dim, find_dim_alfa)
    return clust_indices, ssd_vector, flats_struct_v


def calc_flats(clust_indices, points_coords, is_constant_dim, const_flat_dim, find_dim_alfa):
    flats_struct_v = []
    for i in range(1, max(clust_indices) + 1):
        cluster_points = points_coords[:, clust_indices == i]
        q_dim = find_dimension(cluster_points, is_constant_dim, const_flat_dim, find_dim_alfa)
        flats_struct_v.append(get_best_qflat(cluster_points, q_dim))
    return flats_struct_v


def get_best_qflat(points_coords, q_dim):
    if points_coords.shape[1] == 0:
        raise ValueError("Empty cluster")
    mean_clust = np.mean(points_coords, axis=1, keepdims=True)
    centered_coords = points_coords - mean_clust
    U, _, _ = svd(centered_coords, full_matrices=False)
    return {
        'P0': mean_clust,
        'Vectors': U[:, :q_dim]
    }


def find_dimension(points_coords, is_constant_dim, const_flat_dim, find_dim_alfa):
    if is_constant_dim:
        return const_flat_dim
    d = points_coords.shape[0]
    alfa = find_dim_alfa
    mju_star_p1 = calc_ssd(np.ones(points_coords.shape[1]), points_coords, get_best_qflat(points_coords, 1))
    mju_star_pii = []

    for q in range(d + 1):
        qflat = get_best_qflat(points_coords, q)
        mju_star_pii.append(calc_ssd(np.ones(points_coords.shape[1]), points_coords, qflat))
        if mju_star_pii[-1] > alfa * mju_star_p1:
            q_tag = q
            break
    else:
        raise ValueError("Was unable to find appropriate qDim")

    two_d_points = np.array([np.arange(1, d + 1), mju_star_pii])
    two_d_p0 = np.array([d, 0])
    two_d_vectors = np.array([d - q_tag, 0 - mju_star_pii[q_tag]])
    two_d_vectors = two_d_vectors / np.linalg.norm(two_d_vectors)
    distances = calc_dists_from_qflat(two_d_points, two_d_vectors, two_d_p0)
    distances[:q_tag - 1] = -np.inf
    qi = np.argmax(distances) + 1  # +1 to match MATLAB 1-based indexing
    return qi


def calc_ssd(clust_indices, points_coords, flats_struct_v):
    ssd = 0
    for i in range(1, max(clust_indices) + 1):
        cluster_points = points_coords[:, clust_indices == i]
        flat = flats_struct_v[i - 1]
        ssd += np.sum(calc_dists_from_qflat(cluster_points, flat['Vectors'], flat['P0']) ** 2)
    return ssd


def calc_dists_from_qflat(points_coords, vectors, p0):
    if points_coords.size == 0:
        return np.array([])
    centered_coords = points_coords - p0
    projection = vectors @ (vectors.T @ centered_coords)
    distances = np.linalg.norm(centered_coords - projection, axis=0)
    return distances


def split_clusters(clust_indices, points_coords, m, gamma, is_constant_dim, const_flat_dim, find_dim_alfa):
    qi_mul_norm_Pi = []
    for i in range(1, max(clust_indices) + 1):
        cluster_points = points_coords[:, clust_indices == i]
        qi_mul_norm_Pi.append(
            np.sum(clust_indices == i) * find_dimension(cluster_points, is_constant_dim, const_flat_dim, find_dim_alfa))

    PtagOrder = np.argsort(qi_mul_norm_Pi)[::-1] + 1  # Descending order and MATLAB 1-based indexing
    empty_clusters = [i for i in range(1, 3 * max(clust_indices) + 1) if np.sum(clust_indices == i) == 0]

    new_clust_indices = clust_indices.copy()
    for i in range(m):
        S = [clust_indices == PtagOrder[i]]
        for j in range(int(np.log(0.5) / np.log(gamma)) + 1):
            flats_struct = get_best_qflat(points_coords[:, S[j]],
                                          find_dimension(points_coords[:, S[j]], is_constant_dim, const_flat_dim,
                                                         find_dim_alfa))
            point_to_flat_dist = calc_dists_from_qflat(points_coords[:, S[j]], flats_struct['Vectors'],
                                                       flats_struct['P0'])
            sorted_indices = np.argsort(point_to_flat_dist)
            threshold = int(np.ceil((1 - gamma) * len(sorted_indices)))
            update_vector = np.zeros(len(sorted_indices), dtype=bool)
            update_vector[sorted_indices[:threshold]] = True
            S.append(S[j].copy())
            S[-1][S[j]] = update_vector

        new_clust_indices[(clust_indices == PtagOrder[i]) & ~S[-1]] = empty_clusters[i]

    return new_clust_indices


def proj_kmeans(clust_indices, points_coords, fdpkm_iters, is_constant_dim, const_flat_dim, find_dim_alfa):
    for _ in range(fdpkm_iters):
        flats_struct_v = calc_flats(clust_indices, points_coords, is_constant_dim, const_flat_dim, find_dim_alfa)
        new_indices = assign_points_to_nearest_flat(points_coords, flats_struct_v)
        if np.array_equal(clust_indices, new_indices):
            break
        clust_indices = new_indices
    return clust_indices


def merge_clusters(clust_indices, points_coords, m, is_constant_dim, const_flat_dim, find_dim_alfa):
    for _ in range(m):
        mju_star = np.full((np.max(clust_indices), np.max(clust_indices)), np.inf)
        for ii in range(1, np.max(clust_indices) + 1):
            if np.sum(clust_indices == ii) == 0:
                continue
            ii_dim = find_dimension(points_coords[:, clust_indices == ii], is_constant_dim, const_flat_dim,
                                    find_dim_alfa)
            for jj in range(ii + 1, np.max(clust_indices) + 1):
                if np.sum(clust_indices == jj) == 0:
                    continue
                union_points = points_coords[:, (clust_indices == ii) | (clust_indices == jj)]
                jj_dim = find_dimension(points_coords[:, clust_indices == jj], is_constant_dim, const_flat_dim,
                                        find_dim_alfa)
                union_dim = min(ii_dim, jj_dim)
                union_flats_struct = get_best_qflat(union_points, union_dim)
                mju_star[ii - 1, jj - 1] = calc_ssd(np.ones(union_points.shape[1]), union_points, union_flats_struct)
                for kk in range(1, np.max(clust_indices) + 1):
                    if kk == ii or kk == jj or np.sum(clust_indices == kk) == 0:
                        continue
                    other_clust_points = points_coords[:, clust_indices == kk]
                    flats_struct = get_best_qflat(other_clust_points, union_dim)
                    mju_star[ii - 1, jj - 1] += calc_ssd(np.ones(other_clust_points.shape[1]), other_clust_points,
                                                         flats_struct)
        ii, jj = np.unravel_index(np.argmin(mju_star), mju_star.shape)
        clust_indices[clust_indices == (jj + 1)] = ii + 1
    return clust_indices


def remove_empty_clusters(clust_indices):
    empty_clusters = []
    max_cluster = np.max(clust_indices)
    for ii in range(1, max_cluster + 1):
        if np.sum(clust_indices == ii) == 0:
            empty_clusters.append(ii)
            continue
        if not empty_clusters:
            continue
        clust_indices[clust_indices == ii] = empty_clusters[0]
        empty_clusters = empty_clusters[1:] + [ii]
    return clust_indices


def assign_points_to_nearest_flat(points_coords, flats_struct_v):
    distances = np.vstack([
        calc_dists_from_qflat(points_coords, flat['Vectors'], flat['P0'])
        for flat in flats_struct_v
    ])
    clusters_indices = np.argmin(distances, axis=0) + 1
    return remove_empty_clusters(clusters_indices)


class KMeansProj(BaseEstimator, ClusterMixin):
    def __init__(
            self,
            n_clusters=2,
            threshold=1e-3,
            kms_iters=10,
            fdpkm_iters=10,
            is_constant_dim=True,
            find_dim_alfa=0.5,
            const_flat_dim=2,
            gamma=0.3
    ):
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.kms_iters = kms_iters
        self.fdpkm_iters = fdpkm_iters
        self.is_constant_dim = is_constant_dim
        self.find_dim_alfa = find_dim_alfa
        self.const_flat_dim = const_flat_dim
        self.gamma = gamma
        self.labels_ = None
        self.ssd_vector_ = None
        self.flats_struct_v_ = None

    def fit(self, X, y=None, sample_weight=None):
        clust_indices, ssd_vector, flats_struct_v = k_means_proj_clustering(X.T, self.n_clusters, self.threshold,
                                                                            self.kms_iters, self.fdpkm_iters,
                                                                            self.is_constant_dim, self.find_dim_alfa,
                                                                            self.const_flat_dim, self.gamma)
        self.labels_ = clust_indices
        self.ssd_vector_ = ssd_vector
        self.flats_struct_v_ = flats_struct_v
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.labels_
