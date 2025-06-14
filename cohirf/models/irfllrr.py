# based on matlab code from https://github.com/wangzhi-swu/IRFLLRR/tree/main

import numpy as np
import optuna
import pandas as pd
from scipy.linalg import orth, svd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import SpectralClustering


def reweighted(X, sigma, c, p, mu, k, eps):
    d, n = X.shape
    c = np.sqrt(d * n) * c
    temp = sigma
    for _ in range(k):
        W = c * (temp + eps) ** (p - 2)
        W = 1.0 / (W / mu + 1)
        sigma = W * sigma
        sigma = np.maximum(sigma - eps, 0)
        temp = sigma
    svp = np.sum(sigma > 0)
    sigma = sigma[:svp]  # maybe +1 for matlab 1-indexing
    return sigma, svp


def solve_irfllrr(X, lambda_=1.0, c=1.0, p=1.0, k=1):
    Q1 = orth(X.T)
    Q2 = orth(X)
    A = X @ Q1
    B = Q2.T @ X

    Z, L, E, iter, EE = solve_irfllrra(X, A, B, lambda_, c, p, k)
    Z = Q1 @ Z
    L = L @ Q2.T
    return Z, L, E, iter, EE


def solve_irfllrra(X, A, B, lambda_, c, p, k):
    rho = 1.12
    tol = 1e-6
    eps = 1e-16
    maxIter = int(1e6)
    d, n = X.shape
    nA = A.shape[1]
    dB = B.shape[0]
    max_mu = 1e10
    mu = 1e-6
    ata = A.T @ A
    bbt = B @ B.T

    Z = np.zeros((nA, n))
    L = np.zeros((d, dB))
    E = np.zeros((d, n))

    Y1 = np.zeros((d, n))
    Y2 = np.zeros((nA, n))
    Y3 = np.zeros((d, dB))

    iter = 0
    # display = True
    # print(f'initial, r(Z)={np.linalg.matrix_rank(Z)}, r(L)={np.linalg.matrix_rank(L)}, |E|_1={np.sum(np.abs(E))}')

    while iter < maxIter:
        iter += 1

        # Update J
        temp1 = Z + Y2 / mu
        U, sigma, Vt = svd(temp1, full_matrices=False)
        sigma, svp = reweighted(X, sigma, c, p, mu, k, eps)
        J = U[:, :svp] @ np.diag(sigma) @ Vt[:svp, :]

        # Update S
        temp2 = L + Y3 / mu
        U, sigma, Vt = svd(temp2, full_matrices=False)
        sigma, svp = reweighted(X, sigma, c, p, mu, k, eps)
        S = U[:, :svp] @ np.diag(sigma) @ Vt[:svp, :]

        # Update Z
        Z = np.linalg.inv(ata + np.eye(nA)) @ (A.T @ (X - L @ B - E) + J + (A.T @ Y1 - Y2) / mu)

        # Update L
        L = ((X - A @ Z - E) @ B.T + S + (Y1 @ B.T - Y3) / mu) @ np.linalg.inv(bbt + np.eye(dB))

        # Update E
        temp3 = X - A @ Z - L @ B + Y1 / mu
        E = np.maximum(0, temp3 - lambda_ / mu) + np.minimum(0, temp3 + lambda_ / mu)

        # Update the multipliers
        leq1 = X - A @ Z - L @ B - E
        leq2 = Z - J
        leq3 = L - S
        stopC = max(np.max(np.abs(leq3)), np.max(np.abs(leq1)), np.max(np.abs(leq2)))
        # if display and (iter == 1 or iter % 50 == 0 or stopC < tol):
        #     print(f'iter {iter}, mu={mu:.1e}, stopALM={stopC:.3e}, |E|_1={np.sum(np.abs(E))}')
        if stopC < tol:
            break
        else:
            Y1 += mu * leq1
            Y2 += mu * leq2
            Y3 += mu * leq3
            mu = min(max_mu, mu * rho)
        EE = stopC

    return Z, L, E, iter, EE

# def solve_l1l2(W, lambda_):
#     n = W.shape[1]
#     E = W.copy()
#     for i in range(n):
#         E[:, i] = solve_l2(W[:, i], lambda_)
#     return E
#
# def solve_l2(w, lambda_):
#     nw = np.linalg.norm(w)
#     if nw > lambda_:
#         x = (nw - lambda_) * w / nw
#     else:
#         x = np.zeros_like(w)
#     return x


class IRFLLRR(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            alpha=4,
            lambda_=1,
            p=0.95,
            c=0.10,
            k=3,
            sc_n_clusters=8,
            sc_eigen_solver=None,
            sc_n_components=None,
            sc_n_init=10,
            sc_eigen_tol=0.0,
            sc_assign_labels='kmeans',
            sc_verbose=False,
            random_state=None,
    ):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.p = p
        self.c = c
        self.k = k
        self.sc_n_clusters = sc_n_clusters
        self.sc_eigen_solver = sc_eigen_solver
        self.sc_n_components = sc_n_components
        self.sc_n_init = sc_n_init
        self.sc_eigen_tol = sc_eigen_tol
        self.sc_assign_labels = sc_assign_labels
        self.sc_verbose = sc_verbose
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X, y=None, sample_weight=None):
        Z, L, E, iter, EE = solve_irfllrr(X.T, lambda_=self.lambda_, c=self.c, p=self.p, k=self.k)
        U, sigma, Vt = svd(Z, full_matrices=False)
        r = np.sum(sigma > 1e-4 * sigma[0])
        U = U[:, :r]
        sigma = sigma[:r]
        U = U @ np.diag(np.sqrt(sigma))
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        L = (U @ U.T) ** (2 * self.alpha)
        spectral_clustering = SpectralClustering(
            n_clusters=self.sc_n_clusters,
            eigen_solver=self.sc_eigen_solver,
            n_components=self.sc_n_components,
            n_init=self.sc_n_init,
            affinity='precomputed',
            eigen_tol=self.sc_eigen_tol,
            assign_labels=self.sc_assign_labels,
            verbose=self.sc_verbose,
            random_state=self.random_state
        )
        self.labels_ = spectral_clustering.fit_predict(L)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.fit(X, y, sample_weight)
        return self.labels_

    @staticmethod
    def create_search_space():
        search_space = dict(
            p=optuna.distributions.FloatDistribution(0.0, 1.0),
            c=optuna.distributions.CategoricalDistribution([1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]),
            lambda_=optuna.distributions.CategoricalDistribution([1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]),
            alpha=optuna.distributions.IntDistribution(1, 4),
            sc_n_clusters=optuna.distributions.IntDistribution(2, 30),
        )
        default_values = dict(
            p=0.95,
            c=0.10,
            lambda_=1,
            alpha=4,
            sc_n_clusters=8,
        )
        return search_space, default_values

