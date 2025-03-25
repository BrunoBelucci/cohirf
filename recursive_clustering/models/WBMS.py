import optuna
from sklearn.base import ClusterMixin, BaseEstimator
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import euclidean_distances

# Adapted from the R code provided in:https://github.com/SaptarshiC98/WBMS
# Saptarshi Chakraborty, Debolina Paul and Swagatam Das. 2021. Automated Clustering of High-dimensional Data with a
# Feature Weighted Mean-shift Algorithm. In 35th AAAI Conference on Artificial Intelligence, a virtual conference,
# Feb 2-9, 2021.


def compute_K_matrix(X, w, h):
    # Scale features by sqrt(w) to account for weighted distances
    X_weighted = X * np.sqrt(w)

    # Compute the squared Euclidean distance
    distances = euclidean_distances(X_weighted, squared=True)

    # Apply the RBF kernel
    K_matrix = np.exp(-distances / h)

    return K_matrix


def run_WBMS(X, h, lambda_=1, tmax=50, tol=1e-5, t_warmup=20, verbose=False):
    n, p = X.shape
    # K_matrix = np.zeros((n, n))
    w = np.ones(p) / p
    # D = np.zeros(p)

    # X1 = X.copy()
    X2 = X.copy()

    prev_dmax = np.max(euclidean_distances(X2, squared=False))
    # Main loop for tmax iterations
    for t in range(tmax):
        if verbose:
            print(f'Iteration {t}')
        # Vectorized computation of K_matrix
        K_matrix = compute_K_matrix(X2, w, h)

        # Compute weighted mean for all points
        s = np.sum(K_matrix, axis=1, keepdims=True)  # Normalization factors (n, 1)
        X1 = (K_matrix @ X2) / s  # Weighted mean (n, p)

        D = np.sum((X - X1) ** 2, axis=0)
        w = np.exp(-D / lambda_)
        w /= np.sum(w)

        X2 = X1.copy()

        if t == t_warmup:
            # we reset X1 and X2 but keep the weights and then continue
            if verbose:
                print(f'Warmup finished at iteration {t}')
            X1 = X.copy()
            X2 = X.copy()

        # if t > t_warmup:
        #     current_dmax = np.max(euclidean_distances(X2, squared=False))
        #     diff = np.abs(current_dmax - prev_dmax)
        #     if diff < tol:
        #         if verbose:
        #             print(f'Converged at iteration {t}')
        #         break

    return X2, w


def U2clus(U, epsa=1e-5):
    # Construct adjacency matrix
    Adj_matrix = euclidean_distances(U, squared=False)
    # Apply thresholding
    Adj_matrix[Adj_matrix > epsa] = 0
    Adj_matrix[Adj_matrix <= epsa] = 1

    # Find connected components
    _, labels = connected_components(Adj_matrix, directed=False)

    return labels


class WBMS(ClusterMixin, BaseEstimator):
    def __init__(self, h=1, lambda_=1, tmax=50, epsa=1e-5, tol=1e-9, t_warmup=20, verbose=False):
        self.h = h
        self.lambda_ = lambda_
        self.tmax = tmax
        self.epsa = epsa
        self.tol = tol
        self.t_warmup = t_warmup
        self.verbose = verbose

    def fit(self, X, y=None, sample_weight=None):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        X2, weights = run_WBMS(X, self.h, self.lambda_, self.tmax, self.tol, self.t_warmup, self.verbose)
        self.labels_ = U2clus(X2, self.epsa)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X).labels_

    @staticmethod
    def create_search_space():
        search_space = dict(
            h=optuna.distributions.FloatDistribution(0.1, 1),
            lambda_=optuna.distributions.IntDistribution(1, 20),
        )
        default_values = dict(
            h=0.1,
            lambda_=10,
        )
        return search_space, default_values
