#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Implementation of the PROCLUS algorithm
# for subspace clustering.
#
# Author: Cássio M. M. Pereira <cassiomartini@gmail.com>
#
# Date: Mon Dec 22 14:00:03 BRST 2014
#
# from https://github.com/cmmp/pyproclus
# Updated for python 3 and wrapped in sklearn estimator by Bruno Belucci


import numpy as np
import optuna
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.utils import check_random_state


def greedy(X, S, k):
    # remember that k = B * k here...
    M = [np.random.permutation(S)[0]]  # M = {m_1}, a random point in S
    A = np.setdiff1d(S, M)  # A = S \ M
    dists = np.zeros(len(A))

    for i in range(len(A)):
        dists[i] = np.linalg.norm(X[A[i]] - X[M[0]])  # Euclidean distance

    for i in range(1, k):
        # choose medoid m_i as the farthest from previous medoids

        midx = np.argmax(dists)
        mi = A[midx]
        M.append(mi)

        # update the distances, so they reflect the dist to the closest medoid:
        for j in range(len(A)):
            dists[j] = min(dists[j], np.linalg.norm(X[A[j]] - X[mi]))

        # remove mi entries from A and dists:
        A = np.delete(A, midx)
        dists = np.delete(dists, midx)

    return np.array(M)


def findDimensions(X, k, l, L, Mcurr):
    N, d = X.shape
    Dis = []  # dimensions picked for the clusters
    Zis = []  # Z for the remaining dimensions
    Rem = []  # remaining dimensions
    Mselidx = []  # id of the medoid indexing the dimensions in Zis and Rem

    for i in range(len(Mcurr)):
        mi = Mcurr[i]
        # Xij is the average distance from the points in L_i to m_i
        # Xij here is an array, containing the avg dists in each dimension
        Xij = np.abs(X[L[i]] - X[mi]).sum(axis=0) / len(L[i])
        Yi = Xij.sum() / d  # average distance over all dimensions
        Di = []  # relevant dimensions for m_i
        si = np.sqrt(((Xij - Yi) ** 2).sum() / (d - 1))  # standard deviations
        Zij = (Xij - Yi) / si  # z-scores of distances

        # pick the smallest two:
        o = np.argsort(Zij)
        Di.append(o[0])
        Di.append(o[1])
        Dis.append(Di)

        for j in range(2, d):
            Zis.append(Zij[o[j]])
            Rem.append(o[j])
            Mselidx.append(i)

    if l != 2:
        # we need to pick the remaining dimensions
        o = np.argsort(Zis)
        nremaining = k * l - k * 2

        # we pick the remaining dimensions using a greedy strategy:
        j = 0
        while nremaining > 0:
            midx = Mselidx[o[j]]
            Dis[midx].append(Rem[o[j]])
            j += 1
            nremaining -= 1

    return Dis


def manhattanSegmentalDist(x, y, Ds):
    """ Compute the Manhattan Segmental Distance between x and y considering
        the dimensions on Ds."""
    dist = 0
    for d in Ds:
        dist += np.abs(x[d] - y[d])
    return dist / len(Ds)


def assignPoints(X, Mcurr, Dis):
    assigns = np.ones(X.shape[0]) * -1

    for i in range(X.shape[0]):
        minDist = np.inf
        best = -1
        for j in range(len(Mcurr)):
            dist = manhattanSegmentalDist(X[i], X[Mcurr[j]], Dis[j])
            if dist < minDist:
                minDist = dist
                best = Mcurr[j]

        assigns[i] = best

    return assigns


def evaluateClusters(X, assigns, Dis, Mcurr):
    upperSum = 0.0

    for i in range(len(Mcurr)):
        C = X[np.where(assigns == Mcurr[i])[0]]  # points in cluster M_i
        Cm = C.sum(axis=0) / C.shape[0]  # cluster centroid
        Ysum = 0.0

        for d in Dis[i]:
            # avg dist to centroid along dim d:
            Ysum += np.sum(np.abs(C[:, d] - Cm[d])) / C.shape[0]
        wi = Ysum / len(Dis[i])

        upperSum += C.shape[0] * wi

    return upperSum / X.shape[0]


def computeBadMedoids(X, assigns, Dis, Mcurr, minDeviation):
    N, d = X.shape
    k = len(Mcurr)
    Mbad = []
    counts = [len(np.where(assigns == i)[0]) for i in Mcurr]
    cte = int(np.ceil((N / k) * minDeviation))

    # get the medoid with least points:
    Mbad.append(Mcurr[np.argsort(counts)[0]])

    for i in range(len(counts)):
        if counts[i] < cte and Mcurr[i] not in Mbad:
            Mbad.append(Mcurr[i])

    return Mbad


def proclus(X, k=2, l=3, minDeviation=0.1, A=30, B=3, niters=30, seed=1234):
    """ Run PROCLUS on a database to obtain a set of clusters and
        dimensions associated with each one.

        Parameters:
        ----------
        - X: 	   		the data set
        - k: 	   		the desired number of clusters
        - l:	   		average number of dimensions per cluster
        - minDeviation: for selection of bad medoids
        - A: 	   		constant for initial set of medoids
        - B: 	   		a smaller constant than A for the final set of medoids
        - niters:  		maximum number of iterations for the second phase
        - seed:    		seed for the RNG
    """
    np.random.seed(seed)

    N, d = X.shape

    if B > A:
        raise Exception("B has to be smaller than A.")

    if l < 2:
        raise Exception("l must be >=2.")

    ###############################
    # 1.) Initialization phase
    ###############################

    # first find a superset of the set of k medoids by random sampling
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    S = idxs[0:(A * k)]
    M = greedy(X, S, B * k)

    ###############################
    # 2.) Iterative phase
    ###############################

    BestObjective = np.inf

    # choose a random set of k medoids from M:
    Mcurr = np.random.permutation(M)[0:k]  # M current
    Mbest = None  # Best set of medoids found

    D = squareform(pdist(X))  # precompute the euclidean distance matrix

    it = 0  # iteration counter
    L = []  # locality sets of the medoids, i.e., points within delta_i of m_i.
    Dis = []  # important dimensions for each cluster
    assigns = []  # cluster membership assignments

    while True:
        it += 1
        L = []

        for i in range(len(Mcurr)):
            mi = Mcurr[i]
            # compute delta_i, the distance to the nearest medoid of m_i:
            di = D[mi, np.setdiff1d(Mcurr, mi)].min()
            # compute L_i, points in sphere centered at m_i with radius d_i
            L.append(np.where(D[mi] <= di)[0])

        # find dimensions:
        Dis = findDimensions(X, k, l, L, Mcurr)

        # form the clusters:
        assigns = assignPoints(X, Mcurr, Dis)

        # evaluate the clusters:
        ObjectiveFunction = evaluateClusters(X, assigns, Dis, Mcurr)

        badM = []  # bad medoids

        Mold = Mcurr.copy()

        if ObjectiveFunction < BestObjective:
            BestObjective = ObjectiveFunction
            Mbest = Mcurr.copy()
            # compute the bad medoids in Mbest:
            badM = computeBadMedoids(X, assigns, Dis, Mcurr, minDeviation)

        if len(badM) > 0:
            # replace the bad medoids with random points from M:
            Mavail = np.setdiff1d(M, Mbest)
            newSel = np.random.choice(Mavail, size=len(badM), replace=False)
            Mcurr = np.setdiff1d(Mbest, badM)
            Mcurr = np.union1d(Mcurr, newSel)

        if np.allclose(Mold, Mcurr) or it >= niters:
            break

    ###############################
    # 3.) Refinement phase
    ###############################

    # compute a new L based on assignments:
    L = []
    for i in range(len(Mcurr)):
        mi = Mcurr[i]
        L.append(np.where(assigns == mi)[0])

    Dis = findDimensions(X, k, l, L, Mcurr)
    assigns = assignPoints(X, Mcurr, Dis)

    # handle outliers:

    # smallest Manhattan segmental distance of m_i to all (k-1)
    # other medoids with respect to D_i:
    deltais = np.zeros(k)
    for i in range(k):
        minDist = np.inf
        for j in range(k):
            if j != i:
                dist = manhattanSegmentalDist(X[Mcurr[i]], X[Mcurr[j]], Dis[i])
                if dist < minDist:
                    minDist = dist
        deltais[i] = minDist

    # mark as outliers the points that are not within delta_i of any m_i:
    for i in range(len(assigns)):
        clustered = False
        for j in range(k):
            d = manhattanSegmentalDist(X[Mcurr[j]], X[i], Dis[j])
            if d <= deltais[j]:
                clustered = True
                break
        if not clustered:
            # print "marked an outlier"
            assigns[i] = -1

    return Mcurr, Dis, assigns


def computeBasicAccuracy(pred, expect):
    """ Computes the clustering accuracy by assigning
        a class to each cluster based on majority
        voting and then comparing with the expected
        class. """

    if len(pred) != len(expect):
        raise Exception("pred and expect must have the same length.")

    uclu = np.unique(pred)

    acc = 0.0

    for cl in uclu:
        points = np.where(pred == cl)[0]
        pclasses = expect[points]
        uclass = np.unique(pclasses)
        counts = [len(np.where(pclasses == u)[0]) for u in uclass]
        mcl = uclass[np.argmax(counts)]
        acc += np.sum(np.repeat(mcl, len(points)) == expect[points])

    acc /= len(pred)

    return acc


class Proclus(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            n_clusters=2,
            avg_dims=3,
            min_deviation=0.1,
            cte_A=30,
            cte_B=3,
            max_iter=30,
            random_state=None
    ):
        self.n_clusters = n_clusters
        self.avg_dims = avg_dims
        self.min_deviation = min_deviation
        self.cte_A = cte_A
        self.cte_B = cte_B
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoids_ = None
        self.medoids_dims_ = None
        self.labels_ = None

    def fit(self, X, y=None, sample_weight=None):
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, 1e6)
        medoids, medoids_dims, labels = proclus(X, k=self.n_clusters, l=self.avg_dims,
                                                minDeviation=self.min_deviation, A=self.cte_A,
                                                B=self.cte_B, niters=self.max_iter, seed=seed)
        self.medoids_ = medoids
        self.medoids_dims_ = medoids_dims
        self.labels_ = labels
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.fit(X)
        return self.labels_

    @staticmethod
    def create_search_space():
        search_space = dict(
            n_clusters=optuna.distributions.IntDistribution(2, 30),
            avg_dims=optuna.distributions.IntDistribution(2, 30),
        )
        default_values = dict(
            n_clusters=8,
            avg_dims=3,
        )
        return search_space, default_values
