import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans as KMeansSklearn
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics.pairwise import (cosine_distances, rbf_kernel, laplacian_kernel, euclidean_distances,
                                      manhattan_distances)
import dask.array as da
import dask.dataframe as dd
from dask_ml.cluster import KMeans as KMeansDask
from joblib import Parallel, delayed
import optuna
import pandas as pd


class RecursiveClustering(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            components_size=10,
            repetitions=10,
            kmeans_n_clusters=3,
            # kmeans_init='k-means++',
            kmeans_init='auto',
            kmeans_n_init='auto',
            kmeans_max_iter=300,
            kmeans_tol=1e-4,
            kmeans_verbose=0,
            random_state=None,
            kmeans_algorithm='lloyd',
            representative_method='closest_overall',
            n_jobs=1,
            use_dask='auto',
            dask_chunk_size='auto',
            # if we have a X_j array with a memory less than this threshold (in bytes),
            # we will use numpy instead of dask, default is 1GB
            dask_memory_threshold=1e9,
    ):
        self.components_size = components_size
        self.repetitions = repetitions
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol
        self.kmeans_verbose = kmeans_verbose
        self.random_state = random_state
        self.kmeans_algorithm = kmeans_algorithm
        self.representative_method = representative_method
        self.n_jobs = n_jobs
        self.use_dask = use_dask
        self.dask_chunk_size = dask_chunk_size
        self.dask_memory_threshold = dask_memory_threshold
        self.n_clusters_ = None
        self.labels_ = None
        self.cluster_representatives_ = None
        self.cluster_representatives_labels_ = None
        self.n_iter_ = None
        self.n_clusters_iter_ = []
        self.labels_sequence_ = None

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]
        n_components = X.shape[1]
        random_state = check_random_state(self.random_state)

        # we will work with numpy (arrays) for speed
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(X, dd.DataFrame):
            X = X.to_dask_array(lengths=True)

        if self.use_dask == 'auto':
            if isinstance(X, da.Array):
                use_dask = True
            else:
                use_dask = False
        elif self.use_dask:
            if not isinstance(X, da.Array):
                X = da.from_array(X, chunks=self.dask_chunk_size)
            use_dask = True
        else:
            use_dask = False

        # labels_sequence_ is always a numpy array (no dask)
        self.labels_sequence_ = np.empty((n_samples, 0), dtype=int)

        def run_one_repetition(X_j, r, use_dask):
            repetition_random_seed = random_state.randint(0, 1e6) + r
            n_j_samples = X_j.shape[0]
            kmeans_n_clusters = min(self.kmeans_n_clusters, n_j_samples)
            if use_dask:
                if self.kmeans_init == 'auto':
                    init = 'k-means||'
                else:
                    init = self.kmeans_init
                k_means_estimator = KMeansDask(n_clusters=kmeans_n_clusters, init=init, n_init=self.kmeans_n_init,
                                               max_iter=self.kmeans_max_iter, tol=self.kmeans_tol,
                                               random_state=repetition_random_seed)
            else:
                if self.kmeans_init == 'auto':
                    init = 'k-means++'
                else:
                    init = self.kmeans_init
                k_means_estimator = KMeansSklearn(n_clusters=kmeans_n_clusters, init=init,
                                                  n_init=self.kmeans_n_init,
                                                  max_iter=self.kmeans_max_iter, tol=self.kmeans_tol,
                                                  verbose=self.kmeans_verbose,
                                                  random_state=repetition_random_seed, algorithm=self.kmeans_algorithm)
            # random sample of components
            components = sample_without_replacement(n_components, min(self.components_size, n_components - 1),
                                                    random_state=repetition_random_seed)
            X_p = X_j[:, components]
            if use_dask:
                k_means_estimator.fit(X_p)
                labels_r = k_means_estimator.labels_
            else:
                labels_r = k_means_estimator.fit_predict(X_p)
            return labels_r

        X_j = X
        global_clusters_indexes_i = []
        i = 0
        # initialize with different length of codes and uniques to enter the while loop
        codes = [0, 1]
        unique_labels = [0]
        X_j_indexes_i_last = None
        # iterate until every sequence of labels is unique
        while len(codes) != len(unique_labels):

            # run the repetitions in parallel
            # obs.: For most cases, the overhead of parallelization is not worth it
            if not use_dask:
                labels_i = Parallel(n_jobs=self.n_jobs)(
                    delayed(run_one_repetition)(X_j, r, use_dask) for r in range(self.repetitions))
                labels_i = np.array(labels_i).T
            else:
                labels_i = []
                for r in range(self.repetitions):
                    print('running repetition', r, 'with dask')
                    labels_i.append(run_one_repetition(X_j, r, use_dask))
                labels_i = da.stack(labels_i, axis=1).compute()  # convert to numpy array

            # factorize labels using numpy
            unique_labels, codes = np.unique(labels_i, axis=0, return_inverse=True)

            # store for development/experimentation purposes
            self.n_clusters_iter_.append(len(unique_labels))

            # add to the sequence of labels
            if i == 0:
                # every sample is present in the first iteration
                label_sequence_i = codes
                self.labels_sequence_ = np.concatenate((self.labels_sequence_, label_sequence_i[:, None]), axis=1)
            else:
                # only some samples are present in the following iterations
                # so we need to add the same label as the representative sample to the rest of the samples
                label_sequence_i = np.empty((n_samples, 1), dtype=int)
                for j, cluster_idxs in enumerate(global_clusters_indexes_i):
                    cluster_label = codes[j]
                    label_sequence_i[cluster_idxs] = cluster_label
                self.labels_sequence_ = np.concatenate((self.labels_sequence_, label_sequence_i), axis=1)
                # we could replace it with something like
                # label_sequence_i = np.empty((n_samples), dtype=int)
                # # Concatenate all cluster indexes and corresponding codes into arrays
                # all_cluster_idxs = np.concatenate(global_clusters_indexes_i)
                # all_labels = np.repeat(codes, [len(cluster) for cluster in global_clusters_indexes_i])
                # # Assign labels to label_sequence_i based on these arrays
                # label_sequence_i[all_cluster_idxs] = all_labels
                # label_sequence = np.concatenate((label_sequence, label_sequence_i[:, None]), axis=1)
                # but this is actually slower

            # find the one sample of each cluster that is the closest to every other sample in the cluster

            # X_j_indexes_i[i] will contain the index of the sample that is the closest to every other sample
            # in the i-th cluster, in other words, the representative sample of the i-th cluster
            X_j_indexes_i = np.empty(len(unique_labels), dtype=int)
            # global_clusters_indexes_i[i] will contain the indexes of ALL the samples in the i-th cluster
            global_clusters_indexes_i = []
            # we need the loop because the number of elements of each cluster is not the same
            # so it is difficult to vectorize
            for j, code in enumerate(np.unique(codes)):
                local_cluster_idx = np.where(codes == code)[0]
                # we need to transform the indexes to the original indexes
                if X_j_indexes_i_last is not None:
                    local_cluster_idx = X_j_indexes_i_last[local_cluster_idx]
                local_cluster = X[local_cluster_idx, :]

                if self.representative_method == 'closest_overall':
                    # calculate the distances between all samples in the cluster and pick the one with the smallest sum
                    # this is the most computationally expensive method (O(n^2))
                    local_cluster_similarities = local_cluster @ local_cluster.T
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_idx[np.argmax(local_cluster_similarities_sum)]
                    X_j_indexes_i[j] = most_similar_sample_idx
                elif self.representative_method == 'closest_overall_1000':
                    # calculate the distances between a maximum of 1000 samples in the cluster and pick the one with the
                    # smallest sum
                    # this puts a limit on the computational cost of O(n^2) to O(1000^2)
                    n_resample = min(1000, local_cluster.shape[0])
                    local_cluster_random_sate = check_random_state(random_state.randint(0, 1e6) + i)
                    local_cluster_sampled_idx = sample_without_replacement(local_cluster.shape[0], n_resample,
                                                                           random_state=local_cluster_random_sate)
                    local_cluster_sampled = local_cluster[local_cluster_sampled_idx, :]
                    local_cluster_similarities = local_cluster_sampled @ local_cluster.T
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = (
                        local_cluster_idx)[local_cluster_sampled_idx[np.argmax(local_cluster_similarities_sum)]]
                    X_j_indexes_i[j] = most_similar_sample_idx
                elif self.representative_method == 'closest_to_centroid':
                    # calculate the centroid of the cluster and pick the sample closest to it
                    # this is the second most computationally expensive method (O(n))
                    centroid = local_cluster.mean(axis=0)
                    local_cluster_similarities = local_cluster @ centroid
                    most_similar_sample_idx = local_cluster_idx[np.argmax(local_cluster_similarities)]
                    X_j_indexes_i[j] = most_similar_sample_idx
                elif self.representative_method == 'centroid':
                    if use_dask:
                        raise ValueError('centroid method is not supported with dask')
                    # calculate the centroid of the cluster and use it as the representative sample
                    # this is the least computationally expensive method (O(1))
                    centroid = local_cluster.mean(axis=0)
                    # we arbitrarily pick the first sample as the representative of the cluster and change its
                    # values to the centroid values so we can use the same logic as the other methods
                    closest_sample_idx = local_cluster_idx[0]
                    # we need to change the original value in X
                    X[closest_sample_idx, :] = centroid
                    X_j_indexes_i[j] = closest_sample_idx
                elif self.representative_method == 'rbf':
                    if use_dask:
                        raise ValueError('rbf method is not supported with dask')
                    # replace cosine_distance by rbf_kernel
                    local_cluster_similarities = rbf_kernel(local_cluster)
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_idx[np.argmax(local_cluster_similarities_sum)]
                    X_j_indexes_i[j] = most_similar_sample_idx
                elif self.representative_method == 'rbf_median':
                    if use_dask:
                        raise ValueError('rbf_median method is not supported with dask')
                    # replace cosine_distance by rbf_kernel with gamma = median
                    local_cluster_distances = euclidean_distances(local_cluster)
                    median_distance = np.median(local_cluster_distances)
                    gamma = 1 / (2 * median_distance)
                    local_cluster_similarities = np.exp(-gamma * local_cluster_distances)
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_idx[np.argmax(local_cluster_similarities_sum)]
                    X_j_indexes_i[j] = most_similar_sample_idx
                elif self.representative_method == 'laplacian':
                    if use_dask:
                        raise ValueError('laplacian method is not supported with dask')
                    # replace cosine_distance by laplacian_kernel
                    local_cluster_similarities = laplacian_kernel(local_cluster)
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_idx[np.argmax(local_cluster_similarities_sum)]
                    X_j_indexes_i[j] = most_similar_sample_idx
                elif self.representative_method == 'laplacian_median':
                    if use_dask:
                        raise ValueError('laplacian_median method is not supported with dask')
                    # replace cosine_distance by laplacian_kernel with gamma = median
                    local_cluster_distances = manhattan_distances(local_cluster)
                    median_distance = np.median(local_cluster_distances)
                    gamma = 1 / (2 * median_distance)
                    local_cluster_similarities = np.exp(-gamma * local_cluster_distances)
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_idx[np.argmax(local_cluster_similarities_sum)]
                    X_j_indexes_i[j] = most_similar_sample_idx

                global_cluster_idx = np.where(label_sequence_i == code)[0]
                global_clusters_indexes_i.append(global_cluster_idx)

            # sort the indexes to make the comparison easier between change in the same algorithm
            # maybe we will eliminate this at the end for speed
            sorted_indexes = np.argsort(X_j_indexes_i)
            X_j_indexes_i = X_j_indexes_i[sorted_indexes]
            global_clusters_indexes_i = [global_clusters_indexes_i[i] for i in sorted_indexes]
            X_j_indexes_i_last = X_j_indexes_i.copy()
            X_j = X[X_j_indexes_i, :]
            if isinstance(X, da.Array) and X_j.nbytes < self.dask_memory_threshold:
                X_j = X_j.compute()
                use_dask = False
            i += 1

        self.n_clusters_ = len(unique_labels)
        self.labels_ = self.labels_sequence_[:, -1]
        self.cluster_representatives_ = X_j
        self.cluster_representatives_labels_ = np.unique(codes)
        self.n_iter_ = i
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X).labels_

    def predict(self, X):
        # find from each cluster representative each sample is closest to
        distances = cosine_distances(X, self.cluster_representatives_)
        labels = np.argmin(distances, axis=1)
        labels = self.cluster_representatives_labels_[labels]
        return labels

    @staticmethod
    def create_search_space():
        search_space = dict(
            components_size=optuna.distributions.IntDistribution(2, 30),
            repetitions=optuna.distributions.IntDistribution(3, 10),
            kmeans_n_clusters=optuna.distributions.IntDistribution(2, 10),
        )
        default_values = dict(
            components_size=10,
            repetitions=10,
            kmeans_n_clusters=3,
        )
        return search_space, default_values
