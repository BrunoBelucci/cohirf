import numpy as np
import optuna
from scipy.sparse import csr_matrix, issparse
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state


# based on the original implementation in matlab Cai, Xiaosha, Dong Huang, Chang-Dong Wang, and Chee-Keong Kwoh.
# “Spectral Clustering by Subspace Randomization and Graph Fusion for High-Dimensional Data.”
# Advances in Knowledge Discovery and Data Mining 12084 (April 17, 2020): 330.
# https://doi.org/10.1007/978-3-030-47426-3_26.

# and the code from snfpy

def snf_sparse(W_list, K=20, t=20, alpha=1.0, verbose=False):
    C = len(W_list)
    m, n = W_list[0].shape
    is_sparse = issparse(W_list[0])

    # Normalize and symmetrize matrices
    if verbose:
        print("Normalizing and symmetrizing matrices...")
    for i in range(C):
        if is_sparse:
            # keepdims=True) -> already a scipy sparse matrix that keeps dims
            W_list[i] = W_list[i] / W_list[i].sum(axis=1)
        else:
            W_list[i] = W_list[i] / W_list[i].sum(axis=1, keepdims=True)
        W_list[i] = (W_list[i] + W_list[i].T) / 2

    # Find dominate set
    if verbose:
        print("Finding dominate sets...")
    new_W_list = [find_dominate_set_sparse(W, K) for W in W_list]
    if is_sparse:
        Wsum = sum(W_list)
    else:
        Wsum = np.sum(W_list, axis=0)

    if verbose:
        print("Performing similarity network fusion...")
    for _ in range(t):
        Wall0 = [new_W @ ((Wsum - W) / (C - 1)) @ new_W.T for new_W, W in zip(new_W_list, W_list)]
        W_list = [bo_normalized(W0, alpha, is_sparse) for W0 in Wall0]
        if is_sparse:
            Wsum = sum(W_list)
        else:
            Wsum = np.sum(W_list, axis=0)

    if verbose:
        print("Finalizing fused matrix...")
    W = Wsum / C
    if is_sparse:
        W = W / W.sum(axis=1)
        W = W + W.T 
        W.setdiag(W.diagonal() + 1)
        W = W / 2
    else:
        W = W / W.sum(axis=1, keepdims=True)
        W = (W + W.T + np.eye(n)) / 2
    return W


def bo_normalized(W, alpha=1, is_sparse=False):
    if is_sparse:
        W.setdiag(W.diagonal() + alpha)
    else:
        W = W + alpha * np.eye(W.shape[0])
    return (W + W.T) / 2


def find_dominate_set_sparse(W, K):
    is_sparse = issparse(W)
    if is_sparse:
        W_numpy = W.toarray()
    else:
        W_numpy = W
    idx = np.argpartition(-W_numpy, K, axis=1)[:, :K]
    new_W = np.zeros_like(W_numpy)
    np.put_along_axis(new_W, idx, np.take_along_axis(W_numpy, idx, axis=1), axis=1)
    new_W = new_W / new_W.sum(axis=1, keepdims=True)
    if is_sparse:
        new_W = csr_matrix(new_W)
    return new_W


def knn_sparse(data, K, verbose=False):
    nearest_neighbors = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    if verbose:
        print("Fitting nearest neighbors model...")
    nearest_neighbors.fit(data)
    if verbose:
        print("Computing k-nearest neighbors graph...")
    graph = nearest_neighbors.kneighbors_graph(data, mode='distance')
    if verbose:
        print("Computing sigma...")
    sigma = pairwise_distances(data).mean()
    graph.data = np.exp(-graph.data / (2 * sigma))
    return graph


class SpectralSubspaceRandomization(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            knn=5,
            n_similarities=20,
            alpha=1.0,
            sampling_ratio=0.5,
            sc_n_clusters=8,
            sc_eigen_solver=None,
            sc_n_components=None,
            sc_n_init=10,
            sc_eigen_tol=0.0,
            sc_assign_labels='kmeans',
            verbose=False,
            random_state=None,
    ):
        self.knn = knn
        self.n_similarities = n_similarities
        self.alpha = alpha
        self.sampling_ratio = sampling_ratio
        self.sc_n_clusters = sc_n_clusters
        self.sc_eigen_solver = sc_eigen_solver
        self.sc_n_components = sc_n_components
        self.sc_n_init = sc_n_init
        self.sc_eigen_tol = sc_eigen_tol
        self.sc_assign_labels = sc_assign_labels
        self.verbose = verbose
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X, y=None, sample_weight=None):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        random_state = check_random_state(self.random_state)
        all_matrices = []
        for i in range(self.n_similarities):
            if self.verbose:
                print(f"Computing similarity matrix {i + 1}/{self.n_similarities}")
            seq = random_state.permutation(X.shape[1])
            n_features = int(X.shape[1] * self.sampling_ratio)
            n_features = max(n_features, 1)  # Ensure at least one feature is selected
            X_i = X[:, seq[:n_features]]
            similarity_matrix = knn_sparse(X_i, self.knn, verbose=self.verbose)
            all_matrices.append(similarity_matrix)

        fused_matrix = snf_sparse(all_matrices, K=self.knn, t=self.n_similarities, alpha=self.alpha, verbose=self.verbose)
        
        if issparse(fused_matrix):
            if self.sc_n_components is None:
                # default behavior is to use number of clusters
                n_components = self.sc_n_clusters
            else:
                n_components = self.sc_n_components
            
            if n_components >= fused_matrix.shape[0]:
                # need to convert to dense, otherwise spectral clustering fails
                fused_matrix = fused_matrix.toarray()
        
        spectral_clustering = SpectralClustering(
            n_clusters=self.sc_n_clusters,
            eigen_solver=self.sc_eigen_solver,
            n_components=self.sc_n_components,
            n_init=self.sc_n_init,
            affinity="precomputed",
            eigen_tol=self.sc_eigen_tol,
            assign_labels=self.sc_assign_labels,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        self.labels_ = spectral_clustering.fit_predict(fused_matrix)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X).labels_

    @staticmethod
    def create_search_space():
        search_space = dict(
            n_similarities=optuna.distributions.IntDistribution(10, 30),
            sampling_ratio=optuna.distributions.FloatDistribution(0.2, 0.8),
            sc_n_clusters=optuna.distributions.IntDistribution(2, 30),
        )
        default_values = dict(
            n_similarities=20,
            sampling_ratio=0.5,
            sc_n_clusters=8,
        )
        return search_space, default_values
