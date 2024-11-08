import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


# based on the original implementation in matlab Cai, Xiaosha, Dong Huang, Chang-Dong Wang, and Chee-Keong Kwoh.
# “Spectral Clustering by Subspace Randomization and Graph Fusion for High-Dimensional Data.”
# Advances in Knowledge Discovery and Data Mining 12084 (April 17, 2020): 330.
# https://doi.org/10.1007/978-3-030-47426-3_26.

# and the code from snfpy

def snf_sparse(W_list, K=20, t=20, alpha=1.0):
    C = len(W_list)
    m, n = W_list[0].shape

    # Normalize and symmetrize matrices
    for i in range(C):
        W_list[i] = W_list[i] / W_list[i].sum(axis=1, keepdims=True)
        W_list[i] = (W_list[i] + W_list[i].T) / 2

    # Find dominate set
    new_W_list = [find_dominate_set_sparse(W, K) for W in W_list]
    Wsum = np.sum(W_list, axis=0)

    for _ in range(t):
        Wall0 = [new_W @ ((Wsum - W) / (C - 1)) @ new_W.T for new_W, W in zip(new_W_list, W_list)]
        W_list = [bo_normalized(W0, alpha) for W0 in Wall0]
        Wsum = np.sum(W_list, axis=0)

    W = Wsum / C
    W = W / W.sum(axis=1, keepdims=True)
    W = (W + W.T + np.eye(n)) / 2
    return W


def bo_normalized(W, alpha=1):
    W = W + alpha * np.eye(W.shape[0])
    return (W + W.T) / 2


def find_dominate_set_sparse(W, K):
    idx = np.argsort(-W, axis=1)[:, :K]
    new_W = np.zeros_like(W)
    np.put_along_axis(new_W, idx, np.take_along_axis(W, idx, axis=1), axis=1)
    new_W = new_W / new_W.sum(axis=1, keepdims=True)
    return new_W


def knn_sparse(data, K):
    nearest_neighbors = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    nearest_neighbors.fit(data)
    graph = nearest_neighbors.kneighbors_graph(data, mode='distance')
    sigma = pairwise_distances(data).mean()
    graph.data = np.exp(-graph.data / (2 * sigma))
    return graph


class SpectralSubspaceRandomization(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            knn=5,
            n_similarities=20,
            t_swaps=20,
            alpha=1.0,
            sampling_ratio=0.5,
            sc_n_clusters=8,
            sc_eigen_solver=None,
            sc_n_components=None,
            sc_n_init=10,
            sc_eigen_tol=0.0,
            sc_assign_labels='kmeans',
            sc_verbose=False,
    ):
        self.knn = knn
        self.n_similarities = n_similarities
        self.t_swaps = t_swaps
        self.alpha = alpha
        self.sampling_ratio = sampling_ratio
        self.sc_n_clusters = sc_n_clusters
        self.sc_eigen_solver = sc_eigen_solver
        self.sc_n_components = sc_n_components
        self.sc_n_init = sc_n_init
        self.sc_eigen_tol = sc_eigen_tol
        self.sc_assign_labels = sc_assign_labels
        self.sc_verbose = sc_verbose
        self.labels_ = None

    def fit(self, X, y=None, sample_weight=None):
        all_matrices = []
        for i in range(self.n_similarities):
            seq = np.random.permutation(X.shape[1])
            X_i = X[:, seq[:int(X.shape[1] * self.sampling_ratio)]]
            similarity_matrix = knn_sparse(X_i, self.knn)
            all_matrices.append(similarity_matrix)

        fused_matrix = snf_sparse(all_matrices, K=self.knn, t=self.t_swaps, alpha=self.alpha)
        spectral_clustering = SpectralClustering(
            n_clusters=self.sc_n_clusters,
            eigen_solver=self.sc_eigen_solver,
            n_components=self.sc_n_components,
            n_init=self.sc_n_init,
            affinity='precomputed',
            eigen_tol=self.sc_eigen_tol,
            assign_labels=self.sc_assign_labels,
            verbose=self.sc_verbose,
        )
        self.labels_ = spectral_clustering.fit_predict(fused_matrix)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X).labels_
