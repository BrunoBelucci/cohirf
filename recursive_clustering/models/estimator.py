import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans as KMeansSklearn, HDBSCAN
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics.pairwise import (cosine_distances, rbf_kernel, laplacian_kernel, euclidean_distances,
                                      manhattan_distances)
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
import dask.array as da
import dask.dataframe as dd
from dask_ml.cluster import KMeans as KMeansDask
from joblib import Parallel, delayed
import optuna
import pandas as pd
from recursive_clustering.models.lazy_minibatchkmeans import LazyMiniBatchKMeans
from recursive_clustering.models.kernel_kmeans import KernelKMeans
from recursive_clustering.models.scsrgf import SpectralSubspaceRandomization
from pathlib import Path


class RecursiveClustering(ClusterMixin, BaseEstimator):
    default_kmeans_max_iter = 300
    minibatch_kmeans_max_iter = 100
    default_kmeans_init = 'k-means++'
    dask_kmeans_init = 'k-means||'

    def __init__(
            self,
            components_size=10,
            repetitions=10,
            verbose=0,
            # base model parameters
            base_model='kmeans',
            kmeans_n_clusters=3,
            kmeans_init='auto',
            kmeans_n_init='auto',
            kmeans_max_iter='auto',
            kmeans_tol=1e-4,
            random_state=None,
            kmeans_algorithm='lloyd',
            representative_method='closest_overall',
            n_samples_representative=None,
            n_jobs=1,
            # kernel kmeans parameters
            kernel_kmeans=False,
            kkmeans_kernel='rbf',
            kkmeans_gamma='median',
            kkmeans_degree=3,
            kkmeans_coef0=1.0,
            kkmeans_params=None,
            # hdbscan parameters
            hdbscan_min_cluster_size=5,
            # sc-srgf parameters
            scsrgf_n_similarities=20,
            scsrgf_sampling_ratio=0.5,
            scsrgf_sc_n_clusters=3,
            # dask and minibatch kmeans parameters
            use_dask='auto',
            batch_size=1024,
            scalable_strategy='minibatch',
            mkmeans_max_no_improvement=10,
            mkmeans_init_size=None,
            mkmeans_reassignment_ratio=0.01,
            mkmeans_shuffle_every_n_epochs=10,
            mkmeans_tmp_dir=Path.cwd(),
            dkmeans_oversampling_factor=2,
            # if we have an X_j array with number of samples smaller than this threshold,
            # we will use numpy instead of dask, and KMeans instead of MiniBatchKMeans
            # this can be useful to ensure that we do not run out of memory as with the default method, if we had only
            # one cluster (worst case scenario), we would calculate a similarity matrix of size n_samples x n_samples
            n_samples_threshold=10000,
            # weighted components
            weighted_components=False,
            exploration_factor=0.5,
            # gaussian random projection
            use_grp=False,
            use_srp=False,
    ):
        self.components_size = components_size
        self.repetitions = repetitions
        self.base_model = base_model
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol
        self.verbose = verbose
        self.random_state = random_state
        self.kmeans_algorithm = kmeans_algorithm
        self.representative_method = representative_method
        self.n_samples_representative = n_samples_representative
        self.n_jobs = n_jobs
        self.kernel_kmeans = kernel_kmeans
        self.kkmeans_kernel = kkmeans_kernel
        self.kkmeans_gamma = kkmeans_gamma
        self.kkmeans_degree = kkmeans_degree
        self.kkmeans_coef0 = kkmeans_coef0
        self.kkmeans_params = kkmeans_params
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.scsrgf_n_similarities = scsrgf_n_similarities
        self.scsrgf_sampling_ratio = scsrgf_sampling_ratio
        self.scsrgf_sc_n_clusters = scsrgf_sc_n_clusters
        self.use_dask = use_dask
        self.batch_size = batch_size
        self.scalable_strategy = scalable_strategy
        self.mkmeans_max_no_improvement = mkmeans_max_no_improvement
        self.mkmeans_init_size = mkmeans_init_size
        self.mkmeans_reassignment_ratio = mkmeans_reassignment_ratio
        self.mkmeans_shuffle_every_n_epochs = mkmeans_shuffle_every_n_epochs
        self.mkmeans_tmp_dir = mkmeans_tmp_dir
        self.dkmeans_oversampling_factor = dkmeans_oversampling_factor
        self.n_samples_threshold = n_samples_threshold
        self.weighted_components = weighted_components
        self.exploration_factor = exploration_factor
        self.use_grp = use_grp
        self.use_srp = use_srp
        self.n_clusters_ = None
        self.labels_ = None
        self.cluster_representatives_ = None
        self.cluster_representatives_labels_ = None
        self.n_iter_ = None
        self.n_clusters_iter_ = []
        self.labels_sequence_ = None
        self.features_weights_ = None
        self.features_sampling_counts_ = None
        self.features_best_inertia_ = None

    def fit(self, X, y=None, sample_weight=None):
        if self.verbose:
            print('Starting fit')
        n_samples = X.shape[0]
        n_components = X.shape[1]
        random_state = check_random_state(self.random_state)
        self.features_weights_ = np.ones(n_components) / n_components
        self.features_sampling_counts_ = np.zeros(n_components)
        self.features_best_inertia_ = np.full(n_components, np.inf)

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
                X = da.from_array(X, chunks=(self.batch_size, -1))
            use_dask = True
        else:
            use_dask = False

        # labels_sequence_ is always a numpy array (no dask)
        self.labels_sequence_ = np.empty((n_samples, 0), dtype=int)

        def run_one_repetition(X_j, r, use_dask):
            repetition_random_seed = random_state.randint(0, 1e6) + r
            n_j_samples = X_j.shape[0]
            kmeans_n_clusters = min(self.kmeans_n_clusters, n_j_samples)
            if use_dask and self.scalable_strategy != 'sampling' and self.base_model == 'kmeans':
                if self.scalable_strategy == 'dask':
                    if self.kmeans_init == 'auto':
                        init = self.dask_kmeans_init
                    else:
                        init = self.kmeans_init
                    if self.kmeans_max_iter == 'auto':
                        max_iter = self.default_kmeans_max_iter
                    else:
                        max_iter = self.kmeans_max_iter
                    base_estimator = KMeansDask(n_clusters=kmeans_n_clusters, init=init, n_init=self.kmeans_n_init,
                                                max_iter=max_iter, tol=self.kmeans_tol,
                                                oversampling_factor=self.dkmeans_oversampling_factor,
                                                random_state=repetition_random_seed)
                elif self.scalable_strategy == 'minibatch':
                    if self.kmeans_init == 'auto':
                        init = self.default_kmeans_init
                    else:
                        init = self.kmeans_init
                    if self.kmeans_max_iter == 'auto':
                        max_iter = self.minibatch_kmeans_max_iter
                    else:
                        max_iter = self.kmeans_max_iter
                    base_estimator = LazyMiniBatchKMeans(n_clusters=kmeans_n_clusters, init=init, max_iter=max_iter,
                                                         batch_size=self.batch_size, verbose=self.verbose,
                                                         compute_labels=True, random_state=repetition_random_seed,
                                                         tol=self.kmeans_tol,
                                                         max_no_improvement=self.mkmeans_max_no_improvement,
                                                         init_size=self.mkmeans_init_size, n_init=self.kmeans_n_init,
                                                         reassignment_ratio=self.mkmeans_reassignment_ratio,
                                                         shuffle_every_n_epochs=self.mkmeans_shuffle_every_n_epochs,
                                                         tmp_dir=self.mkmeans_tmp_dir)
                else:
                    raise ValueError('scalable_strategy must be "dask", "minibatch" or "sampling"')
            elif self.base_model == 'kmeans':
                if self.kmeans_init == 'auto':
                    init = 'k-means++'
                else:
                    init = self.kmeans_init
                if self.kmeans_max_iter == 'auto':
                    max_iter = self.default_kmeans_max_iter
                else:
                    max_iter = self.kmeans_max_iter
                if self.kernel_kmeans:
                    if self.kmeans_n_init == 'auto':
                        n_init = 10
                    else:
                        n_init = self.kmeans_n_init
                    base_estimator = KernelKMeans(n_clusters=self.kmeans_n_clusters, max_iter=max_iter,
                                                  tol=self.kmeans_tol, n_init=n_init,
                                                  kernel=self.kkmeans_kernel, gamma=self.kkmeans_gamma,
                                                  degree=self.kkmeans_degree, coef0=self.kkmeans_coef0,
                                                  kernel_params=self.kkmeans_params,
                                                  random_state=repetition_random_seed, verbose=self.verbose)
                else:
                    base_estimator = KMeansSklearn(n_clusters=kmeans_n_clusters, init=init,
                                                   n_init=self.kmeans_n_init,
                                                   max_iter=max_iter, tol=self.kmeans_tol,
                                                   verbose=self.verbose,
                                                   random_state=repetition_random_seed,
                                                   algorithm=self.kmeans_algorithm)
            elif self.base_model == 'hdbscan':
                base_estimator = HDBSCAN(min_cluster_size=self.hdbscan_min_cluster_size)
            elif self.base_model == 'sc-srgf':
                base_estimator = SpectralSubspaceRandomization(n_similarities=self.scsrgf_n_similarities,
                                                               sampling_ratio=self.scsrgf_sampling_ratio,
                                                               sc_n_clusters=self.scsrgf_sc_n_clusters, )
            else:
                raise ValueError('base_model must be "kmeans", "hdbscan" or "sc-srgf"')

            # random sample of components
            if isinstance(self.components_size, int):
                if self.use_grp:
                    # use Gaussian random projection to reduce the number of components
                    grp = GaussianRandomProjection(n_components=self.components_size, random_state=repetition_random_seed)
                    X_p = grp.fit_transform(X_j)
                elif self.use_srp:
                    # use Sparse random projection to reduce the number of components
                    srp = SparseRandomProjection(n_components=self.components_size, random_state=repetition_random_seed)
                    X_p = srp.fit_transform(X_j)
                else:
                    components = random_state.choice(n_components, size=min(self.components_size, n_components - 1),
                                                     p=self.features_weights_, replace=False)
                    X_p = X_j[:, components]
            elif isinstance(self.components_size, float):
                # sample a percentage of components
                if self.components_size < 0 or self.components_size > 1:
                    raise ValueError('components_size must be between 0 and 1')
                components_size = int(self.components_size * n_components)
                if self.use_grp:
                    # use Gaussian random projection to reduce the number of components
                    grp = GaussianRandomProjection(n_components=components_size, random_state=repetition_random_seed)
                    X_p = grp.fit_transform(X_j)
                elif self.use_srp:
                    # use Sparse random projection to reduce the number of components
                    srp = SparseRandomProjection(n_components=components_size, random_state=repetition_random_seed)
                    X_p = srp.fit_transform(X_j)
                else:
                    components = random_state.choice(n_components, size=min(components_size, n_components - 1),
                                                     p=self.features_weights_, replace=False)
                    X_p = X_j[:, components]
            elif self.components_size == 'full':
                # full kmeans
                components = np.arange(n_components)
                X_p = X_j
            else:
                raise ValueError('components_size must be an int or "full"')

            if self.verbose:
                print('Fitting kmeans')
            if use_dask and self.scalable_strategy == 'dask':
                base_estimator.fit(X_p)
                labels_r = base_estimator.labels_
                inertia = base_estimator.inertia_
            else:
                labels_r = base_estimator.fit_predict(X_p)
                inertia = base_estimator.inertia_

            if self.weighted_components:
                # we update the weights of the features based on the inertia of the components and the number of times
                # they were sampled
                self.features_best_inertia_[components] = np.minimum(self.features_best_inertia_[components], inertia)
                self.features_sampling_counts_[components] += 1
                normalized_inv_inertias = 1 / (self.features_best_inertia_ + 1e-10)
                normalized_inv_inertias /= np.sum(normalized_inv_inertias)
                normalized_inv_counts = 1 / (self.features_sampling_counts_ + 1e-10)
                normalized_inv_counts /= np.sum(normalized_inv_counts)
                self.features_weights_ = ((1-self.exploration_factor)*normalized_inv_inertias
                                          + self.exploration_factor*normalized_inv_counts)
                self.features_weights_ = self.features_weights_/self.features_weights_.sum()

            return labels_r

        X_j = X
        global_clusters_indexes_i = []
        i = 0
        # initialize with different length of codes and uniques to enter the while loop
        codes = [0, 1]
        n_clusters_iter = n_samples
        X_j_indexes_i_last = None
        # iterate until every sequence of labels is unique
        while len(codes) != n_clusters_iter:
            if self.verbose:
                print('Iteration', i)

            if use_dask and self.scalable_strategy == 'sampling':
                n_resample = min(self.batch_size, X_j.shape[0])
                X_j_sampled_indexes = random_state.permutation(X_j.shape[0])[:n_resample]
                X_j_not_sampled_indexes = np.setdiff1d(np.arange(X_j.shape[0]), X_j_sampled_indexes)
                X_j_sampled_not_sampled_indexes = np.concatenate((X_j_sampled_indexes, X_j_not_sampled_indexes))
                X_j_sampled = X_j[X_j_sampled_indexes, :].persist()
                sampled_X_j = True
            else:
                X_j_sampled = X_j
                X_j_sampled_indexes = np.arange(X_j.shape[0])
                X_j_not_sampled_indexes = np.array([])
                X_j_sampled_not_sampled_indexes = np.arange(X_j.shape[0])
                sampled_X_j = False

            # run the repetitions in parallel
            # obs.: For most cases, the overhead of parallelization is not worth it as internally KMeans is already
            # parallelizing with threads, but it may be useful for very large datasets.
            if not use_dask:
                labels_i_sampled = Parallel(n_jobs=self.n_jobs)(
                    delayed(run_one_repetition)(X_j_sampled, r, use_dask) for r in range(self.repetitions))
                labels_i_sampled = np.array(labels_i_sampled).T
            else:
                # if we are using dask probably we cannot run this in parallel as we do not have enough memory
                # unless we are using a distributed cluster which we are not assuming here
                labels_i_sampled = []
                for r in range(self.repetitions):
                    labels_i_sampled.append(run_one_repetition(X_j_sampled, r, use_dask))
                if self.scalable_strategy == 'dask':
                    labels_i_sampled = da.stack(labels_i_sampled, axis=1).compute()  # convert to numpy array
                elif self.scalable_strategy == 'minibatch' or self.scalable_strategy == 'sampling':
                    labels_i_sampled = np.array(labels_i_sampled).T

            # factorize labels using numpy
            unique_labels_sampled, codes_sampled = np.unique(labels_i_sampled, axis=0, return_inverse=True)
            unique_codes_sampled = np.unique(codes_sampled)

            n_clusters_iter = len(unique_labels_sampled)
            if self.verbose:
                print('Number of clusters (before unsampled):', n_clusters_iter)
            if sampled_X_j:
                # each unsampled sample is considered a cluster
                codes_unsampled = np.arange(n_clusters_iter, n_clusters_iter + len(X_j_not_sampled_indexes))
                n_clusters_iter += len(X_j_not_sampled_indexes)
                codes = np.concatenate((codes_sampled, codes_unsampled))
                if self.verbose:
                    print('Number of clusters (after unsampled):', n_clusters_iter)
            else:
                codes = codes_sampled
                codes_unsampled = np.array([])

            # store for development/experimentation purposes
            self.n_clusters_iter_.append(n_clusters_iter)

            # add to the sequence of labels
            if i == 0:
                # we put the codes in the correct indexes
                label_sequence_i = np.empty((n_samples, 1), dtype=int)
                label_sequence_i[X_j_sampled_not_sampled_indexes] = codes[:, None]
                # global_clusters_indexes_i[i] will contain the indexes of ALL the samples in the i-th cluster
                global_clusters_indexes_i = []
                for code in unique_codes_sampled:
                    cluster_idx = np.where(codes_sampled == code)[0]
                    global_cluster_idx = X_j_sampled_indexes[cluster_idx]
                    global_clusters_indexes_i.append(global_cluster_idx)
                global_clusters_indexes_i.extend([[idx] for idx in X_j_not_sampled_indexes])
                self.labels_sequence_ = np.concatenate((self.labels_sequence_, label_sequence_i), axis=1)
            else:
                # only some samples are present in the following iterations
                # so we need to add the same label as the representative sample to the rest of the samples
                label_sequence_i = np.empty((n_samples, 1), dtype=int)
                # first use last global_clusters_indexes_i to get the indexes of the samples in the last i-th cluster
                global_clusters_indexes_i_sampled_not_sampled = [global_clusters_indexes_i[j] for j in
                                                                 X_j_sampled_not_sampled_indexes]
                # next global_clusters_indexes_i will contain the indexes of ALL the samples in the current i-th cluster
                global_clusters_indexes_i = [[] for _ in range(n_clusters_iter)]
                for j, cluster_idxs in enumerate(global_clusters_indexes_i_sampled_not_sampled):
                    cluster_label = codes[j]
                    label_sequence_i[cluster_idxs] = cluster_label
                    global_clusters_indexes_i[cluster_label].extend(cluster_idxs)
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
            X_j_indexes_i = np.empty(n_clusters_iter, dtype=int)
            # we need the loop because the number of elements of each cluster is not the same
            # so it is difficult to vectorize
            for j, code in enumerate(unique_codes_sampled):
                if self.verbose:
                    print('Choosing representative sample for cluster', j)
                local_cluster_idx = np.where(codes_sampled == code)[0]
                local_cluster = X_j_sampled[local_cluster_idx, :]
                # then we get the original cluster indexes
                # we first need to correct the local_cluster_idx to the indexes of the sampled array
                local_cluster_original_idx = X_j_sampled_indexes[local_cluster_idx]
                # then we need to transform the indexes to the original indexes
                if X_j_indexes_i_last is not None:
                    local_cluster_original_idx = X_j_indexes_i_last[local_cluster_original_idx]

                if self.n_samples_representative is not None:
                    n_samples_representative = min(self.n_samples_representative, local_cluster.shape[0])
                    local_cluster_sampled_idx = random_state.permutation(local_cluster.shape[0])[
                                                :n_samples_representative]
                    local_cluster_sampled = local_cluster[local_cluster_sampled_idx, :]
                else:
                    # this will allow us to generalize every representative_method with a sampled version
                    local_cluster_sampled_idx = np.arange(local_cluster.shape[0])  # all samples
                    local_cluster_sampled = local_cluster

                if self.representative_method == 'closest_overall':
                    # calculate the cosine similarities (without normalization) between all samples in the cluster and
                    # pick the one with the largest sum
                    # this is the most computationally expensive method (O(n^2))
                    local_cluster_similarities = local_cluster_sampled @ local_cluster_sampled.T
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_original_idx[
                        local_cluster_sampled_idx[np.argmax(local_cluster_similarities_sum)]]

                elif self.representative_method == 'closest_to_centroid':
                    # calculate the centroid of the cluster and pick the sample most similar to it
                    # this is the second most computationally expensive method (O(n))
                    centroid = local_cluster_sampled.mean(axis=0)
                    local_cluster_similarities = local_cluster_sampled @ centroid
                    most_similar_sample_idx = local_cluster_original_idx[
                        local_cluster_sampled_idx[np.argmax(local_cluster_similarities)]]

                elif self.representative_method == 'centroid':
                    if use_dask and self.scalable_strategy != 'sampling':
                        raise ValueError(
                            'centroid method is not supported with dask and a scalable strategy different from sampling')
                    # calculate the centroid of the cluster and use it as the representative sample
                    # this is the least computationally expensive method (O(1))
                    centroid = local_cluster_sampled.mean(axis=0)
                    # we arbitrarily pick the first sample as the representative of the cluster and change its
                    # values to the centroid values so we can use the same logic as the other methods
                    most_similar_sample_idx = local_cluster_original_idx[local_cluster_sampled_idx[0]]
                    # we need to change the original value in X
                    X[most_similar_sample_idx, :] = centroid

                elif self.representative_method == 'rbf':
                    if use_dask and self.scalable_strategy != 'sampling':
                        raise ValueError(
                            'rbf method is not supported with dask and a scalable strategy different from sampling')
                    # replace cosine_distance by rbf_kernel
                    local_cluster_similarities = rbf_kernel(local_cluster_sampled)
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_original_idx[
                        local_cluster_sampled_idx[np.argmax(local_cluster_similarities_sum)]]

                elif self.representative_method == 'rbf_median':
                    if use_dask and self.scalable_strategy != 'sampling':
                        raise ValueError(
                            'rbf_median method is not supported with dask and a scalable strategy different from sampling')
                    # replace cosine_distance by rbf_kernel with gamma = median
                    local_cluster_distances = euclidean_distances(local_cluster_sampled)
                    median_distance = np.median(local_cluster_distances)
                    gamma = 1 / (2 * median_distance)
                    local_cluster_similarities = np.exp(-gamma * local_cluster_distances)
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_original_idx[
                        local_cluster_sampled_idx[np.argmax(local_cluster_similarities_sum)]]

                elif self.representative_method == 'laplacian':
                    if use_dask and self.scalable_strategy != 'sampling':
                        raise ValueError(
                            'laplacian method is not supported with dask and a scalable strategy different from sampling')
                    # replace cosine_distance by laplacian_kernel
                    local_cluster_similarities = laplacian_kernel(local_cluster_sampled)
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_original_idx[
                        local_cluster_sampled_idx[np.argmax(local_cluster_similarities_sum)]]

                elif self.representative_method == 'laplacian_median':
                    if use_dask and self.scalable_strategy != 'sampling':
                        raise ValueError(
                            'laplacian_median method is not supported with dask and a scalable strategy different from sampling')
                    # replace cosine_distance by laplacian_kernel with gamma = median
                    local_cluster_distances = manhattan_distances(local_cluster_sampled)
                    median_distance = np.median(local_cluster_distances)
                    gamma = 1 / (2 * median_distance)
                    local_cluster_similarities = np.exp(-gamma * local_cluster_distances)
                    local_cluster_similarities_sum = local_cluster_similarities.sum(axis=0)
                    most_similar_sample_idx = local_cluster_original_idx[
                        local_cluster_sampled_idx[np.argmax(local_cluster_similarities_sum)]]
                else:
                    raise ValueError('representative_method must be closest_overall, closest_to_centroid, centroid,'
                                     ' rbf, rbf_median, laplacian or laplacian_median')

                X_j_indexes_i[j] = most_similar_sample_idx
                # global_cluster_idx = np.where(label_sequence_i == code)[0]
                # global_clusters_indexes_i.append(global_cluster_idx)

            if sampled_X_j:
                all_unsampled_clusters_original_idx = X_j_not_sampled_indexes
                if X_j_indexes_i_last is not None:
                    all_unsampled_clusters_original_idx = X_j_indexes_i_last[all_unsampled_clusters_original_idx]
                X_j_indexes_i[len(unique_codes_sampled):] = all_unsampled_clusters_original_idx

            # sort the indexes to make the comparison easier between change in the same algorithm
            # maybe we will eliminate this at the end for speed
            # note: I think that sorting is now essential due to the use of global_clusters_indexes_i starting from i=1
            sorted_indexes = np.argsort(X_j_indexes_i)
            X_j_indexes_i = X_j_indexes_i[sorted_indexes]
            global_clusters_indexes_i = [global_clusters_indexes_i[i] for i in sorted_indexes]
            X_j_indexes_i_last = X_j_indexes_i.copy()
            X_j = X[X_j_indexes_i, :]
            if isinstance(X_j, da.Array) and X_j.shape[0] < self.n_samples_threshold:
                if self.verbose:
                    print('Number of samples threshold reached, converting to numpy')
                X_j = X_j.compute()
                use_dask = False
            i += 1
            del X_j_sampled

        self.n_clusters_ = n_clusters_iter
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
            repetitions=optuna.distributions.IntDistribution(1, 10),
            kmeans_n_clusters=optuna.distributions.IntDistribution(2, 10),
        )
        default_values = dict(
            components_size=10,
            repetitions=10,
            kmeans_n_clusters=3,
        )
        return search_space, default_values


class RecursiveClusteringHDBSCAN(RecursiveClustering):
    @staticmethod
    def create_search_space():
        search_space = dict(
            components_size=optuna.distributions.IntDistribution(2, 30),
            repetitions=optuna.distributions.IntDistribution(3, 10),
            hdbscan_min_cluster_size=optuna.distributions.IntDistribution(2, 10)
        )
        default_values = dict(
            min_cluster_size=5,
            components_size=10,
            hdbscan_min_cluster_size=10,
        )
        return search_space, default_values


class RecursiveClusteringSCSRGF(RecursiveClustering):
    @staticmethod
    def create_search_space():
        search_space = dict(
            components_size=optuna.distributions.IntDistribution(2, 30),
            repetitions=optuna.distributions.IntDistribution(3, 10),
            scsrgf_n_similarities=optuna.distributions.IntDistribution(10, 30),
            scsrgf_sampling_ratio=optuna.distributions.FloatDistribution(0.2, 0.8),
            scsrgf_sc_n_clusters=optuna.distributions.IntDistribution(2, 30),
        )
        default_values = dict(
            components_size=10,
            repetitions=10,
            scsrgf_n_similarities=20,
            scsrgf_sampling_ratio=0.5,
            scsrgf_sc_n_clusters=3,
        )
        return search_space, default_values


class RecursiveClusteringPct(RecursiveClustering):
    @staticmethod
    def create_search_space():
        search_space = dict(
            components_size=optuna.distributions.FloatDistribution(0.1, 0.5),
            repetitions=optuna.distributions.IntDistribution(1, 10),
            kmeans_n_clusters=optuna.distributions.IntDistribution(2, 10),
        )
        default_values = dict(
            components_size=0.3,
            repetitions=10,
            kmeans_n_clusters=3,
        )
        return search_space, default_values
