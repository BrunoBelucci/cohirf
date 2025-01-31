import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics.pairwise import (cosine_distances, rbf_kernel, laplacian_kernel, euclidean_distances,
                                      manhattan_distances)
from joblib import Parallel, delayed
import optuna


class CoHiRF(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            components_size=10,
            repetitions=10,
            kmeans_n_clusters=3,
            kmeans_init='k-means++',
            kmeans_n_init=1,
            kmeans_max_iter=300,
            kmeans_tol=1e-4,
            kmeans_verbose=0,
            random_state=None,
            kmeans_algorithm='lloyd',
            representative_method='closest_overall',
            # MiniBatchKMeans parameters
            # kmeans_batch_size=None,
            # kmeans_max_no_improvement=10,
            # kmeans_init_size=None,
            # kmeans_reassignment_ratio=0.01,
            # normalization=False,
            n_jobs=1
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
        # self.normalization = normalization
        # self.kmeans_batch_size = kmeans_batch_size
        # self.kmeans_max_no_improvement = kmeans_max_no_improvement
        # self.kmeans_init_size = kmeans_init_size
        # self.kmeans_reassignment_ratio = kmeans_reassignment_ratio
        self.n_jobs = n_jobs
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

        # we will work with numpy for speed
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        self.labels_sequence_ = np.empty((n_samples, 0), dtype=int)

        def run_one_repetition(X_j, r):
            repetition_random_sate = check_random_state(random_state.randint(0, 1e6) + r)
            n_j_samples = X_j.shape[0]
            kmeans_n_clusters = min(self.kmeans_n_clusters, n_j_samples)
            # if n_samples > 1e3 or self.kmeans_batch_size is not None:
            #     kmeans_batch_size = self.kmeans_batch_size if self.kmeans_batch_size is not None else 1024
            #     kmeans_cls = MiniBatchKMeans
            #     extra_kwargs = dict(batch_size=kmeans_batch_size,
            #                         max_no_improvement=self.kmeans_max_no_improvement,
            #                         init_size=self.kmeans_init_size, reassignment_ratio=self.kmeans_reassignment_ratio)
            # else:
            #     kmeans_cls = KMeans
            #     extra_kwargs = dict(algorithm=self.kmeans_algorithm)
            k_means_estimator = KMeans(n_clusters=kmeans_n_clusters, init=self.kmeans_init,
                                       n_init=self.kmeans_n_init,
                                       max_iter=self.kmeans_max_iter, tol=self.kmeans_tol,
                                       verbose=self.kmeans_verbose,
                                       random_state=repetition_random_sate, algorithm=self.kmeans_algorithm)
            # random sample of components
            components = sample_without_replacement(n_components, min(self.components_size, n_components - 1),
                                                    random_state=repetition_random_sate)
            X_p = X_j[:, components]
            # if self.normalization:
            #     # normalize data
            #     mean = X_p.mean(axis=0)
            #     std = X_p.std(axis=0)
            #     std[std == 0] = 1
            #     X_p = (X_p - mean) / std
            labels_r = k_means_estimator.fit_predict(X_p)
            # data back to original scale
            # if self.normalization:
            #     X_p = X_p * std + mean
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
            labels_i = Parallel(n_jobs=self.n_jobs)(
                delayed(run_one_repetition)(X_j, r) for r in range(self.repetitions))

            # factorize labels using numpy
            labels_i = np.array(labels_i).T
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
                    local_cluster_distances = cosine_distances(local_cluster)
                    local_cluster_distances_sum = local_cluster_distances.sum(axis=0)
                    closest_sample_idx = local_cluster_idx[np.argmin(local_cluster_distances_sum)]
                    X_j_indexes_i[j] = closest_sample_idx
                elif self.representative_method == 'closest_overall_1000':
                    # calculate the distances between a maximum of 1000 samples in the cluster and pick the one with the
                    # smallest sum
                    # this puts a limit on the computational cost of O(n^2) to O(1000n)
                    n_resample = min(1000, local_cluster.shape[0])
                    local_cluster_random_sate = check_random_state(random_state.randint(0, 1e6) + i)
                    local_cluster_sampled_idx = sample_without_replacement(local_cluster.shape[0], n_resample,
                                                                           random_state=local_cluster_random_sate)
                    local_cluster_sampled = local_cluster[local_cluster_sampled_idx, :]
                    local_cluster_distances = cosine_distances(local_cluster_sampled)
                    local_cluster_distances_sum = local_cluster_distances.sum(axis=0)
                    closest_sample_idx = (
                        local_cluster_idx)[local_cluster_sampled_idx[np.argmin(local_cluster_distances_sum)]]
                    X_j_indexes_i[j] = closest_sample_idx
                elif self.representative_method == 'closest_to_centroid':
                    # calculate the centroid of the cluster and pick the sample closest to it
                    # this is the second most computationally expensive method (O(n))
                    centroid = local_cluster.mean(axis=0)
                    local_cluster_distances = cosine_distances(local_cluster, centroid.reshape(1, -1))
                    closest_sample_idx = local_cluster_idx[np.argmin(local_cluster_distances)]
                    X_j_indexes_i[j] = closest_sample_idx
                elif self.representative_method == 'centroid':
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
                    # replace cosine_distance by rbf_kernel
                    local_cluster_similarities = rbf_kernel(local_cluster)
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_idx[np.argmax(local_cluster_similarities_sum)]
                    X_j_indexes_i[j] = most_similar_sample_idx
                elif self.representative_method == 'rbf_median':
                    # replace cosine_distance by rbf_kernel with gamma = median
                    local_cluster_distances = euclidean_distances(local_cluster)
                    median_distance = np.median(local_cluster_distances)
                    gamma = 1 / (2 * median_distance)
                    local_cluster_similarities = np.exp(-gamma * local_cluster_distances)
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_idx[np.argmax(local_cluster_similarities_sum)]
                    X_j_indexes_i[j] = most_similar_sample_idx
                elif self.representative_method == 'laplacian':
                    # replace cosine_distance by laplacian_kernel
                    local_cluster_similarities = laplacian_kernel(local_cluster)
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_idx[np.argmax(local_cluster_similarities_sum)]
                    X_j_indexes_i[j] = most_similar_sample_idx
                elif self.representative_method == 'laplacian_median':
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
