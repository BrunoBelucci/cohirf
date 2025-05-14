from typing import Optional
import numpy as np
import dask.array as da
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import (cosine_distances, rbf_kernel, laplacian_kernel, euclidean_distances,
                                      manhattan_distances)
from sklearn.kernel_approximation import RBFSampler, Nystroem
import dask.array as da
import dask.dataframe as dd
from dask_ml.metrics.pairwise import (euclidean_distances as dask_euclidean_distances, rbf_kernel as dask_rbf_kernel, 
                                      pairwise_distances as dask_pairwise_distances)
from joblib import Parallel, delayed
import optuna
import pandas as pd
from recursive_clustering.models.kernel_kmeans import KernelKMeans


class BaseCoHiRF:
    def __init__(
            self,
            repetitions: int = 10,
            verbose: int | bool = 0,
            representative_method: str = 'closest_overall',
            n_samples_representative: Optional[int] = None,
            random_state: Optional[int] = None,
            n_jobs: int = 1,
            max_iter: int = 100,
            save_path: bool = False,
            # base model parameters
            base_model: str | type[BaseEstimator] = 'kmeans',
            base_model_kwargs: Optional[dict] = None,
            # sampling parameters
            n_features: int | float | str = 10,  # number of random features that will be sampled
            transform_method: str | type[TransformerMixin] = None,
            transform_kwargs: Optional[dict] = None,
            sample_than_transform: bool = True,
            transform_once_per_iteration: bool = False,
            # batch parameters
            batch_size: Optional[int] = None,
            n_samples_threshold: int | str = 'batch_size',
            use_dask: bool | str = 'auto',
            **kwargs
    ):
        self.n_features = n_features
        self.repetitions = repetitions
        self.verbose = verbose
        self.base_model = base_model
        self.base_model_kwargs = base_model_kwargs if base_model_kwargs is not None else {}
        self.transform_method = transform_method
        self.transform_kwargs = transform_kwargs if transform_kwargs is not None else {}
        self.sample_than_transform = sample_than_transform
        self.transform_once_per_iteration = transform_once_per_iteration
        self.representative_method = representative_method
        self.n_samples_representative = n_samples_representative
        self._random_state = random_state
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.save_path = save_path
        self.batch_size = batch_size
        self._n_samples_threshold = n_samples_threshold
        self.use_dask = use_dask
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def n_samples_threshold(self):
        if isinstance(self._n_samples_threshold, int):
            return self._n_samples_threshold
        elif isinstance(self._n_samples_threshold, str):
            if self._n_samples_threshold == 'batch_size':
                return self.batch_size
            else:
                raise ValueError('n_samples_threshold must be an int or "batch_size"')
        else:
            raise ValueError('n_samples_threshold must be an int or "batch_size"')

    @property
    def random_state(self):
        if self._random_state is None:
            self._random_state = np.random.default_rng()
        elif isinstance(self._random_state, int):
            self._random_state = np.random.default_rng(self._random_state)
        return self._random_state

    def get_base_model(self, child_random_state):
        random_seed = child_random_state.integers(0, 1e6)
        if isinstance(self.base_model, str):
            if self.base_model == 'kmeans':
                return KMeans(**self.base_model_kwargs, random_state=random_seed)
            else:
                raise ValueError(f'base_model {self.base_model} is not valid.')
        elif issubclass(self.base_model, BaseEstimator):
            base_model = self.base_model(**self.base_model_kwargs)
            if hasattr(base_model, 'random_state'):
                base_model.set_params(random_state=random_seed)
            elif hasattr(base_model, 'random_seed'):
                base_model.set_params(random_seed=random_seed)
            return base_model
        else:
            raise ValueError(f'base_model {self.base_model} is not valid.')
        
    def sampling_transform_X(self, X, child_random_state):
        if isinstance(self.transform_method, str):
            if self.transform_method == 'random':
                X_p = X  # we already sampled 
        elif isinstance(self.transform_method, type) and issubclass(self.transform_method, TransformerMixin):
            # perform a transformation
            transformer = self.transform_method(**self.transform_kwargs)
            if hasattr(transformer, 'random_state'):
                transformer.set_params(random_state=child_random_state.integers(0, 1e6))
            elif hasattr(transformer, 'random_seed'):
                transformer.set_params(random_seed=child_random_state.integers(0, 1e6))
            X_p = transformer.fit_transform(X)
        elif self.transform_method is None:
            X_p = X
        else:
            raise ValueError(f'sampling_method {self.transform_method} is not valid.')
        return X_p
        
    def random_sample(self, X, child_random_state):
        n_all_features = X.shape[1]
        # random sample
        if isinstance(self.n_features, int):
            size = min(self.n_features, n_all_features - 1)
            features = child_random_state.choice(n_all_features, size=size, replace=False)
        elif isinstance(self.n_features, float):
            # sample a percentage of features
            if self.n_features < 0 or self.n_features > 1:
                raise ValueError('n_features must be between 0 and 1')
            features_size = int(self.n_features * n_all_features)
            size = min(features_size, n_all_features - 1)
            features = child_random_state.choice(n_all_features, size=size, replace=False)
        elif self.n_features == 'full':
            # full kmeans
            features = np.arange(n_all_features)
        else:
            raise ValueError(f'n_features {self.n_features} not valid.')
        X_p = X[:, features]
        return X_p

    def sample_X_j(self, X_representative, child_random_state):
        if self.transform_once_per_iteration:
            # we only do random sampling because data is already transformed
            X_p = self.random_sample(X_representative, child_random_state)
        else:
            if self.sample_than_transform:
                X_p = self.random_sample(X_representative, child_random_state)
                X_p = self.sampling_transform_X(X_p, child_random_state)
            
            else:
                X_p = self.sampling_transform_X(X_representative, child_random_state)
                X_p = self.random_sample(X_p, child_random_state)
            
        return X_p
    
    def get_labels_from_base_model(self, base_model, X_p):
        labels = base_model.fit_predict(X_p)
        return labels

    def run_one_repetition(self, X_representative, repetition):
        if self.verbose:
            print('Starting repetition', repetition)
        child_random_state = np.random.default_rng([self.random_state.integers(0, 1e6), repetition])
        base_model = self.get_base_model(child_random_state)
        X_p = self.sample_X_j(X_representative, child_random_state)
        labels = self.get_labels_from_base_model(base_model, X_p)
        return labels
    
    def get_representative_cluster_assignments(self, X_representative):
        # run the repetitions in parallel
        # obs.: For most cases, the overhead of parallelization is not worth it as internally KMeans is already
        # parallelizing with threads, but it may be useful for very large datasets.
        if self.verbose:
            print('Starting consensus assignment')
        labels_i = Parallel(n_jobs=self.n_jobs)(delayed(self.run_one_repetition)(X_representative, r) for r in range(self.repetitions))
        labels_i = np.array(labels_i).T

        # factorize labels using numpy (codes are from 0 to n_clusters-1)
        _, codes = np.unique(labels_i, axis=0, return_inverse=True)
        return codes
    
    def compute_similarities(self, X_cluster, use_dask):
        if self.verbose:
            print('Computing similarities with method', self.representative_method)

        if self.representative_method == 'closest_overall':
            # calculate the cosine similarities (without normalization) between all samples in the cluster and
            # pick the one with the largest sum
            # this is the most computationally expensive method (O(n^2))
            cluster_similarities = X_cluster @ X_cluster.T

        elif self.representative_method == 'closest_to_centroid':
            # calculate the centroid of the cluster and pick the sample most similar to it
            # this is the second most computationally expensive method (O(n))
            centroid = X_cluster.mean(axis=0)
            cluster_similarities = X_cluster @ centroid

        # elif self.representative_method == 'centroid':
        #     # calculate the centroid of the cluster and use it as the representative sample
        #     # this is the least computationally expensive method (O(1))
        #     centroid = X_cluster.mean(axis=0)
        #     # we arbitrarily pick the first sample as the representative of the cluster and change its
        #     # values to the centroid values so we can use the same logic as the other methods
        #     most_similar_sample_idx = local_cluster_original_idx[local_cluster_sampled_idx[0]]
        #     # we need to change the original value in X
        #     X[most_similar_sample_idx, :] = centroid

        elif self.representative_method == 'rbf':
            # replace cosine_distance by rbf_kernel
            if use_dask:
                cluster_similarities = dask_rbf_kernel(X_cluster)
            else:
                cluster_similarities = rbf_kernel(X_cluster)

        elif self.representative_method == 'rbf_median':
            # replace cosine_distance by rbf_kernel with gamma = median
            if use_dask:
                cluster_distances = dask_euclidean_distances(X_cluster)
                median_distance = da.median(cluster_distances)
                gamma = 1 / (2 * median_distance)
                cluster_similarities = da.exp(-gamma * cluster_distances)
            else:
                cluster_distances = euclidean_distances(X_cluster)
                median_distance = np.median(cluster_distances)
                gamma = 1 / (2 * median_distance)
                cluster_similarities = np.exp(-gamma * cluster_distances)

        elif self.representative_method == 'laplacian':
            # replace cosine_distance by laplacian_kernel
            if use_dask:
                cluster_similarities = dask_pairwise_distances(X_cluster, metric='manhattan')
                gamma = 1 / (X_cluster.shape[0])  # default sklearn gamma
                cluster_similarities = da.exp(-gamma * cluster_similarities)
            else:
                cluster_similarities = laplacian_kernel(X_cluster)

        elif self.representative_method == 'laplacian_median':
            # replace cosine_distance by laplacian_kernel with gamma = median
            if use_dask:
                cluster_distances = dask_pairwise_distances(X_cluster, metric='manhattan')
                median_distance = da.median(cluster_distances)
                gamma = 1 / (2 * median_distance)
                cluster_similarities = da.exp(-gamma * cluster_distances)
            else:
                cluster_distances = manhattan_distances(X_cluster)
                median_distance = np.median(cluster_distances)
                gamma = 1 / (2 * median_distance)
                cluster_similarities = np.exp(-gamma * cluster_distances)
        else:
            raise ValueError('representative_method must be closest_overall, closest_to_centroid,'
                                ' rbf, rbf_median, laplacian or laplacian_median')
        return cluster_similarities

    
    def choose_new_representatives(self, X_representatives, old_representatives_indexes,
                                   new_representative_cluster_assignments, new_clusters_labels, use_dask):
        new_representatives_indexes = []
        for label in new_clusters_labels:
            if self.verbose:
                print('Choosing new representative sample for cluster', label)
            cluster_mask = new_representative_cluster_assignments == label
            X_cluster = X_representatives[cluster_mask]
            X_cluster_indexes = old_representatives_indexes[cluster_mask]
            # sample a representative sample from the cluster
            if self.n_samples_representative is not None:
                n_samples_representative = min(self.n_samples_representative, X_cluster.shape[0])
                sampled_idx = self.random_state.choice(X_cluster.shape[0], size=n_samples_representative, replace=False)
                X_cluster = X_cluster[sampled_idx]
                X_cluster_indexes = X_cluster_indexes[sampled_idx]
            cluster_similarities = self.compute_similarities(X_cluster, use_dask)
            cluster_similarities_sum = cluster_similarities.sum(axis=0)
            most_similar_sample_idx = X_cluster_indexes[np.argmax(cluster_similarities_sum)]
            new_representatives_indexes.append(most_similar_sample_idx)
        return np.array(new_representatives_indexes)
    
    def get_new_clusters(self, old_clusters, new_representative_cluster_assignments, new_n_clusters):
        new_clusters = [[] for _ in range(new_n_clusters)]
        if self.verbose:
            print('Getting new clusters')
        for i, cluster in enumerate(old_clusters):
            cluster_assignment = new_representative_cluster_assignments[i]
            new_clusters[cluster_assignment].extend(cluster)
        return new_clusters
    
    def get_labels_from_clusters(self, clusters):
        if self.verbose:
            print('Getting labels from clusters')
        cluster_lengths = [len(cluster) for cluster in clusters]
        cluster_indexes = np.concatenate(clusters)
        cluster_labels = np.repeat(np.arange(len(clusters)), cluster_lengths)
        labels = np.empty(cluster_indexes.shape[0], dtype=int)
        labels[cluster_indexes] = cluster_labels
        return labels

    def fit(self, X, y=None, sample_weight=None):
        if self.verbose:
            print('Starting fit')
        n_samples = X.shape[0]

        # we will work with numpy (arrays) for speed
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, dd.DataFrame):
            X = X.to_dask_array(lengths=True)

        if self.use_dask == True:
            if not isinstance(X, da.Array):
                X = da.from_array(X, chunks=(self.batch_size, -1))
            use_dask = True
        elif self.use_dask == False:
            if isinstance(X, da.Array):
                X = X.compute()
            use_dask = False
        elif self.use_dask == 'auto':
            # we use whatever is passed
            if isinstance(X, da.Array):
                use_dask = True
            else:
                use_dask = False

        # representative samples
        X_representatives = X

        # indexes of the representative samples, start with (n_samples) but will be updated when we have less than
        # n_samples as representatives
        representatives_indexes = np.arange(n_samples)
        # each sample starts as its own cluster
        representative_cluster_assignments = representatives_indexes
        clusters_labels = representatives_indexes

        # list of lists of indexes of the samples of each cluster
        # clusters = [[[i] for i in range(n_samples)]]
        clusters = []  # optimization for first iteration, we dont need to iterate through all the samples

        i = 0
        self.n_clusters_iter_ = []
        self.labels_iter_ = []
        # iterate until every sequence of labels is unique
        while (len(representative_cluster_assignments) != len(clusters_labels) and i < self.max_iter) or i == 0:
            if self.verbose:
                print('Iteration', i)

            if self.batch_size is not None:
                # sample a batch of samples
                n_resample = min(self.batch_size, X.shape[0])
                resampled_indexes = self.random_state.choice(X_representatives.shape[0], size=n_resample, replace=False)
                not_resampled_indexes = np.setdiff1d(np.arange(X_representatives.shape[0]), resampled_indexes)
                X_representatives = X_representatives[resampled_indexes, :]
                if use_dask:
                    X_representatives.persist()
                representatives_indexes = representatives_indexes[resampled_indexes]
            else:
                not_resampled_indexes = None

            # consensus assignment (it is here that we repeatdly apply our base model)
            if self.transform_once_per_iteration:
                # we transform once the data here
                X_representatives = self.sampling_transform_X(X_representatives, self.random_state)
            new_representative_cluster_assignments = self.get_representative_cluster_assignments(X_representatives)
            new_clusters_labels = np.unique(new_representative_cluster_assignments)
            new_n_clusters = len(new_clusters_labels)

            # using representative_method
            new_representatives_indexes = self.choose_new_representatives(
                X_representatives, representatives_indexes, new_representative_cluster_assignments, new_clusters_labels,
                use_dask)
            
            if not_resampled_indexes is not None:
                # we consider the not resampled samples as invididual clusters
                # we create the labels of the clusters
                not_sampled_labels = np.arange(new_n_clusters, new_n_clusters + len(not_resampled_indexes))
                # we update representative_cluster_assignments
                new_representative_cluster_assignments = np.concatenate((new_representative_cluster_assignments, 
                                                                        not_sampled_labels))
                new_clusters_labels = np.concatenate((new_clusters_labels, not_sampled_labels))
                # we add the not resampled_indexes to the new_representatives_indexes
                new_representatives_indexes = np.concatenate((new_representatives_indexes, not_resampled_indexes))
                # we update the number of clusters
                new_n_clusters += len(not_resampled_indexes)
            
            self.n_clusters_iter_.append(new_n_clusters)

            if i == 0:
                # small optimization for first iteration: we dont need to iterate through all the clusters (samples)
                # we can just use the new representative cluster assignments
                new_clusters = [[] for _ in range(new_n_clusters)]
                for j in range(new_n_clusters):
                    representative_cluster_assignments = np.where(new_representative_cluster_assignments == j)[0]
                    new_clusters[j] = representative_cluster_assignments
            else:
                # we need to iterate through all the (old) clusters
                new_clusters = self.get_new_clusters(clusters, new_representative_cluster_assignments, new_n_clusters)
            
            if self.save_path:
                labels = self.get_labels_from_clusters(new_clusters)
                self.labels_iter_.append(labels)

            # update variables
            clusters = new_clusters
            representatives_indexes = new_representatives_indexes
            representative_cluster_assignments = new_representative_cluster_assignments
            clusters_labels = new_clusters_labels
            X_representatives = X[representatives_indexes, :]
            if isinstance(X_representatives, da.Array) and X_representatives.shape[0] <= self.n_samples_threshold:
                X_representatives = X_representatives.compute()
                use_dask = False
            i += 1

        self.n_clusters_ = len(clusters)
        self.labels_ = self.get_labels_from_clusters(clusters)
        self.cluster_representatives_ = X_representatives
        self.cluster_representatives_labels_ = representative_cluster_assignments
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
    

class ModularCoHiRF(BaseCoHiRF, ClusterMixin, BaseEstimator):
    def __init__(
            self,
            repetitions: int = 10,
            verbose: int | bool = 0,
            representative_method: str = 'closest_overall',
            n_samples_representative: Optional[int] = None,
            random_state: Optional[int] = None,
            n_jobs: int = 1,
            max_iter: int = 100,
            save_path: bool = False,
            # base model parameters
            base_model: str | type[BaseEstimator] = 'kmeans',
            base_model_kwargs: Optional[dict] = None,
            # sampling parameters
            n_features: int | float | str = 10,  # number of random features that will be sampled
            transform_method: str | type[TransformerMixin] = None,
            transform_kwargs: Optional[dict] = None,
            sample_than_transform: bool = True,
            transform_once_per_iteration: bool = False,
            # batch parameters
            batch_size: Optional[int] = None,
            n_samples_threshold: int | str = 'batch_size',
            use_dask: bool | str = 'auto',
    ):
        super().__init__(repetitions, verbose, representative_method, n_samples_representative, random_state,
                         n_jobs, max_iter, save_path, base_model=base_model, base_model_kwargs=base_model_kwargs,
                         transform_method=transform_method, n_features=n_features, transform_kwargs=transform_kwargs,
                         batch_size=batch_size, n_samples_threshold=n_samples_threshold, use_dask=use_dask,
                         sample_than_transform=sample_than_transform, transform_once_per_iteration=transform_once_per_iteration)


class CoHiRF(BaseCoHiRF, ClusterMixin, BaseEstimator):
    def __init__(
            self,
            repetitions: int = 10,
            verbose: int | bool = 0,
            representative_method: str = 'closest_overall',
            n_samples_representative: Optional[int] = None,
            random_state: Optional[int] = None,
            n_jobs: int = 1,
            max_iter: int = 100,
            save_path: bool = False,
            # base model parameters
            base_model: str | type[BaseEstimator] = 'kmeans',
            base_model_kwargs: Optional[dict] = None,
            # sampling parameters
            n_features: int | float | str = 10,  # number of random features that will be sampled
            transform_method: str | type[TransformerMixin] = None,
            transform_kwargs: Optional[dict] = None,
            sample_than_transform: bool = True,
            transform_once_per_iteration: bool = False,
            # batch parameters
            batch_size: Optional[int] = None,
            n_samples_threshold: int | str = 'batch_size',
            use_dask: bool | str = 'auto',
    ):
        super().__init__(repetitions, verbose, representative_method, n_samples_representative, random_state,
                         n_jobs, max_iter, save_path, base_model=base_model, base_model_kwargs=base_model_kwargs,
                         transform_method=transform_method, n_features=n_features, transform_kwargs=transform_kwargs,
                         batch_size=batch_size, n_samples_threshold=n_samples_threshold, use_dask=use_dask,
                         sample_than_transform=sample_than_transform, transform_once_per_iteration=transform_once_per_iteration)
        
    def get_base_model(self, random_seed):
        if self.base_model == 'kmeans':
            return KMeans(n_clusters=self.kmeans_n_clusters, init=self.kmeans_init,
                                 n_init=self.kmeans_n_init, max_iter=self.kmeans_max_iter, tol=self.kmeans_tol,
                                 verbose=self.verbose, random_state=random_seed)
        else:
            raise ValueError(f'base_model {self.base_model} is not valid.')

    @staticmethod
    def create_search_space():
        search_space = dict(
            n_features=optuna.distributions.FloatDistribution(0.1, 1),
            repetitions=optuna.distributions.IntDistribution(3, 10),
            kmeans_n_clusters=optuna.distributions.IntDistribution(2, 10),
        )
        default_values = dict(
            n_features=0.3,
            repetitions=10,
            kmeans_n_clusters=3,
        )
        return search_space, default_values
    