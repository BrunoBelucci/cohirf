from typing import Optional
import numpy as np
import dask.array as da
from dask.array.core import Array as DaskArray
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import (
    cosine_distances,
    rbf_kernel,
    laplacian_kernel,
    euclidean_distances,
    manhattan_distances,
)
from sklearn.decomposition import PCA
import dask.array as da
import dask.dataframe as dd
from dask_ml.metrics.pairwise import (
    euclidean_distances as dask_euclidean_distances,
    rbf_kernel as dask_rbf_kernel,
    pairwise_distances as dask_pairwise_distances,
)
from joblib import Parallel, delayed
import optuna
import pandas as pd


class BaseCoHiRF:
    def __init__(
        self,
        repetitions: int = 10,
        verbose: int | bool = 0,
        representative_method: str = "closest_overall",
        n_samples_representative: Optional[int] = None,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        max_iter: int = 100,
        save_path: bool = False,
        # base model parameters
        base_model: str | type[BaseEstimator] = "kmeans",
        base_model_kwargs: Optional[dict] = None,
        # sampling parameters
        n_features: int | float | str = 10,  # number of random features that will be sampled
        transform_method: Optional[str | type[TransformerMixin]]= None,
        transform_kwargs: Optional[dict] = None,
        sample_than_transform: bool = True,
        transform_once_per_iteration: bool = False,
        # batch parameters
        batch_size: Optional[int] = None,
        n_samples_threshold: int | str = "batch_size",
        use_dask: bool | str = "auto",
        **kwargs,
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
        self.X_to_store = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def n_samples_threshold(self):
        if isinstance(self._n_samples_threshold, int):
            return self._n_samples_threshold
        elif isinstance(self._n_samples_threshold, str):
            if self._n_samples_threshold == "batch_size":
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
            if self.base_model == "kmeans":
                return KMeans(**self.base_model_kwargs, random_state=random_seed)
            else:
                raise ValueError(f"base_model {self.base_model} is not valid.")
        elif issubclass(self.base_model, BaseEstimator):
            base_model = self.base_model(**self.base_model_kwargs)
            if hasattr(base_model, "random_state"):
                base_model.set_params(random_state=random_seed)
            elif hasattr(base_model, "random_seed"):
                base_model.set_params(random_seed=random_seed)
            return base_model
        else:
            raise ValueError(f"base_model {self.base_model} is not valid.")

    def sampling_transform_X(self, X, child_random_state):
        if isinstance(self.transform_method, str):
            if self.transform_method == "random":
                X_p = X  # we already sampled
            else:
                raise ValueError(f"sampling_method {self.transform_method} is not valid.")
        elif isinstance(self.transform_method, type) and issubclass(self.transform_method, TransformerMixin):
            # perform a transformation
            transform_kwargs = self.transform_kwargs.copy()
            if issubclass(self.transform_method, PCA):
                n_components = self.transform_kwargs.get("n_components", None)
                if n_components is not None:
                    n_components = min(n_components, X.shape[0])  # PCA n_components must be <= n_samples
                    transform_kwargs["n_components"] = n_components
            transformer = self.transform_method(**transform_kwargs)
            if hasattr(transformer, "random_state"):
                try:
                    transformer.set_params(random_state=child_random_state.integers(0, 1e6)) # type: ignore
                except AttributeError:
                    setattr(transformer, "random_state", child_random_state.integers(0, 1e6))
            elif hasattr(transformer, "random_seed"):
                try:
                    transformer.set_params(random_seed=child_random_state.integers(0, 1e6)) # type: ignore
                except AttributeError:
                    setattr(transformer, "random_seed", child_random_state.integers(0, 1e6))
            X_p = transformer.fit_transform(X)
        elif self.transform_method is None:
            X_p = X
        else:
            raise ValueError(f"sampling_method {self.transform_method} is not valid.")
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
                raise ValueError("n_features must be between 0 and 1")
            features_size = int(self.n_features * n_all_features)
            size = min(features_size, n_all_features - 1)
            features = child_random_state.choice(n_all_features, size=size, replace=False)
        elif self.n_features == "full":
            # full kmeans
            features = np.arange(n_all_features)
        else:
            raise ValueError(f"n_features {self.n_features} not valid.")
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
            print("Starting repetition", repetition)
        child_random_state = np.random.default_rng([self.random_state.integers(0, int(1e6)), repetition])
        base_model = self.get_base_model(child_random_state)
        X_p = self.sample_X_j(X_representative, child_random_state)
        labels = self.get_labels_from_base_model(base_model, X_p)
        return labels

    def get_representative_cluster_assignments(self, X_representative):
        # run the repetitions in parallel
        # obs.: For most cases, the overhead of parallelization is not worth it as internally KMeans is already
        # parallelizing with threads, but it may be useful for very large datasets.
        if self.verbose:
            print("Starting consensus assignment")
        labels_i = Parallel(n_jobs=self.n_jobs)(
            delayed(self.run_one_repetition)(X_representative, r) for r in range(self.repetitions)
        )
        labels_i = np.array(labels_i).T

        # factorize labels using numpy (codes are from 0 to n_clusters-1)
        _, codes = np.unique(labels_i, axis=0, return_inverse=True)
        return codes

    def compute_similarities(self, X_cluster: DaskArray | np.ndarray):
        if self.verbose:
            print("Computing similarities with method", self.representative_method)

        if isinstance(X_cluster, DaskArray):
            use_dask = True
        else:
            use_dask = False

        if self.representative_method == "closest_overall":
            # calculate the cosine similarities (without normalization) between all samples in the cluster and
            # pick the one with the largest sum
            # this is the most computationally expensive method (O(n^2))
            cluster_similarities = X_cluster @ X_cluster.T

        elif self.representative_method == "closest_to_centroid":
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

        elif self.representative_method == "rbf":
            # replace cosine_distance by rbf_kernel
            if use_dask:
                cluster_similarities = dask_rbf_kernel(X_cluster) # type: ignore
            else:
                cluster_similarities = rbf_kernel(X_cluster) # type: ignore

        elif self.representative_method == "rbf_median":
            # replace cosine_distance by rbf_kernel with gamma = median
            if use_dask:
                cluster_distances = dask_euclidean_distances(X_cluster) # type: ignore
                median_distance = da.median(cluster_distances) # type: ignore
                gamma = 1 / (2 * median_distance)
                cluster_similarities = da.exp(-gamma * cluster_distances) # type: ignore
            else:
                cluster_distances = euclidean_distances(X_cluster) # type: ignore
                median_distance = np.median(cluster_distances)
                gamma = 1 / (2 * median_distance)
                cluster_similarities = np.exp(-gamma * cluster_distances)

        elif self.representative_method == "laplacian":
            # replace cosine_distance by laplacian_kernel
            if use_dask:
                cluster_similarities = dask_pairwise_distances(X_cluster, X_cluster, metric="manhattan") # type: ignore
                gamma = 1 / (X_cluster.shape[0])  # default sklearn gamma
                cluster_similarities = da.exp(-gamma * cluster_similarities) # type: ignore
            else:
                cluster_similarities = laplacian_kernel(X_cluster) # type: ignore

        elif self.representative_method == "laplacian_median":
            # replace cosine_distance by laplacian_kernel with gamma = median
            if use_dask:
                cluster_distances = dask_pairwise_distances(X_cluster, X_cluster, metric="manhattan") # type: ignore
                median_distance = da.median(cluster_distances) # type: ignore
                gamma = 1 / (2 * median_distance)
                cluster_similarities = da.exp(-gamma * cluster_distances) # type: ignore
            else:
                cluster_distances = manhattan_distances(X_cluster) # type: ignore
                median_distance = np.median(cluster_distances)
                gamma = 1 / (2 * median_distance)
                cluster_similarities = np.exp(-gamma * cluster_distances)
        else:
            raise ValueError(
                "representative_method must be closest_overall, closest_to_centroid,"
                " rbf, rbf_median, laplacian or laplacian_median"
            )
        return cluster_similarities

    def choose_new_representatives(
        self,
        X_representatives,
        new_representative_cluster_assignments,
        new_clusters_labels,
    ):
        new_representatives_local_indexes = []
        for label in new_clusters_labels:
            if self.verbose:
                print("Choosing new representative sample for cluster", label)
            cluster_mask = new_representative_cluster_assignments == label
            X_cluster = X_representatives[cluster_mask]
            X_cluster_indexes = np.where(cluster_mask)[0]

            # sample a representative sample from the cluster
            if self.n_samples_representative is not None:
                n_samples_representative = min(self.n_samples_representative, X_cluster.shape[0])
                sampled_indexes = self.random_state.choice(X_cluster.shape[0], size=n_samples_representative, replace=False)
                X_cluster = X_cluster[sampled_indexes]
                X_cluster_indexes = X_cluster_indexes[sampled_indexes]

            cluster_similarities = self.compute_similarities(X_cluster)
            cluster_similarities_sum = cluster_similarities.sum(axis=0)
            most_similar_sample_local_idx = X_cluster_indexes[cluster_similarities_sum.argmax()]
            new_representatives_local_indexes.append(most_similar_sample_local_idx)
        new_representatives_local_indexes = np.array(new_representatives_local_indexes)
        return new_representatives_local_indexes

    def get_new_clusters(self, old_clusters, new_representative_cluster_assignments, new_n_clusters):
        new_clusters = [[] for _ in range(new_n_clusters)]
        if self.verbose:
            print("Getting new clusters")
        for i, cluster in enumerate(old_clusters):
            cluster_assignment = new_representative_cluster_assignments[i]
            new_clusters[cluster_assignment].extend(cluster)
        return new_clusters

    def update_parents(
        self,
        old_parents,
        old_representatives_absolute_indexes,
        new_clusters_labels,
        new_representative_cluster_assignments,
        new_representatives_absolute_indexes,
    ):
        if self.verbose:
            print("Updating parents")
        new_parents = old_parents.copy()
        for new_representative_index, new_cluster_label in zip(
            new_representatives_absolute_indexes, new_clusters_labels
        ):
            parents_indexes_to_update = old_representatives_absolute_indexes[
                new_representative_cluster_assignments == new_cluster_label
            ]
            new_parents[parents_indexes_to_update] = new_representative_index
        return new_parents

    def get_all_parents_indexes(self, parents, representative_index):
        all_indexes = set()
        indexes_to_append = [representative_index]
        first = True
        while len(indexes_to_append) > 0:  # the representative_index itself will always be in the list
            all_indexes.update(indexes_to_append)
            indexes_to_append = np.where(np.isin(parents, indexes_to_append))[0]
            if first:
                first = False
                indexes_to_append = np.setdiff1d(indexes_to_append, representative_index, assume_unique=True)
        return list(all_indexes)

    def get_labels_from_parents(self, parents, representative_indexes):
        if self.verbose:
            print("Getting labels from parents")
        labels = np.empty(parents.shape[0], dtype=int)
        for i, representative_index in enumerate(representative_indexes):
            all_indexes = self.get_all_parents_indexes(parents, representative_index)
            labels[all_indexes] = i
        return labels

    def get_labels_from_clusters(self, clusters):
        if self.verbose:
            print("Getting labels from clusters")
        cluster_lengths = [len(cluster) for cluster in clusters]
        cluster_indexes = np.concatenate(clusters)
        cluster_labels = np.repeat(np.arange(len(clusters)), cluster_lengths)
        labels = np.empty(cluster_indexes.shape[0], dtype=int)
        labels[cluster_indexes] = cluster_labels
        return labels

    def random_batch(self, X, representatives_absolute_indexes):
        # sample a batch of samples
        n_samples = representatives_absolute_indexes.shape[0]
        if self.batch_size is None:
            n_resample = n_samples
        else:
            n_resample = min(self.batch_size, n_samples)
        resampled_indexes = self.random_state.choice(n_samples, size=n_resample, replace=False)
        absolute_resampled_indexes = representatives_absolute_indexes[resampled_indexes]
        resampled_X = X[absolute_resampled_indexes]
        return resampled_X, absolute_resampled_indexes

    def sample_batch(self, X, representatives_absolute_indexes, iteration):
        if self.batch_size is not None:
            if isinstance(X, DaskArray):
                # iterating by block is MUCH faster than indexing, so we ensure to at least do the first iteration through
                # the whole dataset by block and then (when we already have the representatives_absolute_indexes of each
                # block) we sample normally and we keep the X in memory
                n_batches = np.ceil(X.shape[0] / self.batch_size)
                if iteration < n_batches:
                    # we still have to iterate through the whole dataset
                    start_index = iteration * self.batch_size
                    end_index = min((iteration + 1) * self.batch_size, X.shape[0])
                    absolute_resampled_indexes = np.arange(start_index, end_index, dtype=int)
                    resampled_X = X[start_index:end_index].compute()
                else:
                    # we have already iterated once through the whole dataset
                    if len(representatives_absolute_indexes) <= self.batch_size:
                        if self.X_to_store is None:
                            # first time that representatives_absolute_indexes <= batch_size
                            self.X_to_store = X[representatives_absolute_indexes].compute()
                            self.X_to_store_indexes = representatives_absolute_indexes
                            resampled_X = self.X_to_store
                        else:
                            # we already have X_to_store in memory, we will take the indexes from it
                            indexes_to_take = np.isin(self.X_to_store_indexes, representatives_absolute_indexes)
                            resampled_X = self.X_to_store[indexes_to_take]
                        absolute_resampled_indexes = representatives_absolute_indexes
                    else:
                        # we still have more representatives than batch size, we will sample a batch of them which will be
                        # slower until we can fit the representatives in memory (hopefully we will not have to do this, or
                        # at least not too many times)
                        resampled_X, absolute_resampled_indexes = self.random_batch(X, representatives_absolute_indexes)
                        resampled_X = resampled_X.compute()

            else:
                resampled_X, absolute_resampled_indexes = self.random_batch(X, representatives_absolute_indexes)

        else:
            # we have no batch size, we will take all the representatives
            resampled_X = X[representatives_absolute_indexes]
            absolute_resampled_indexes = representatives_absolute_indexes

        return resampled_X, absolute_resampled_indexes

    def update_with_unsampled_batch(self, representatives_absolute_indexes, new_representatives_local_indexes, 
                                    cluster_assignments, clusters_labels, representatives_absolute_resampled_indexes):
        n_not_sampled = len(representatives_absolute_indexes) - len(representatives_absolute_resampled_indexes)

        new_representatives_absolute_indexes = representatives_absolute_resampled_indexes[new_representatives_local_indexes]
        new_cluster_assignments = cluster_assignments
        new_clusters_labels = clusters_labels
        new_n_clusters = len(new_clusters_labels)

        if n_not_sampled > 0:
            not_sampled_labels = np.arange(new_n_clusters, new_n_clusters + n_not_sampled)
            new_cluster_assignments = np.concatenate((new_cluster_assignments, not_sampled_labels))
            new_clusters_labels = np.concatenate((new_clusters_labels, not_sampled_labels))

            not_sampled_indexes = np.setdiff1d(representatives_absolute_indexes, representatives_absolute_resampled_indexes)

            new_representatives_absolute_indexes = np.concatenate(
                (new_representatives_absolute_indexes, not_sampled_indexes)
            )
        return (
            new_representatives_absolute_indexes,
            new_cluster_assignments,
            new_clusters_labels,
        )

    def fit(self, X, y=None, sample_weight=None):
        if self.verbose:
            print("Starting fit")
        n_samples = X.shape[0]

        # we will work with numpy (arrays) for speed
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, dd.DataFrame):
            X = X.to_dask_array(lengths=True)

        if self.use_dask == True:
            if not isinstance(X, DaskArray):
                X = da.from_array(X, chunks=(self.batch_size, -1)) # type: ignore
            else:
                # we ensure that the chunks are the same size as the batch size
                X = X.rechunk((self.batch_size, -1))  # type: ignore
        elif self.use_dask == False:
            if isinstance(X, DaskArray):
                X = X.compute()
        elif self.use_dask == "auto":
            # we use whatever is passed
            if isinstance(X, DaskArray):
                # we ensure that the chunks are the same size as the batch size
                X = X.rechunk((self.batch_size, -1))  # type: ignore

        # indexes of the representative samples, start with (n_samples) but will be updated when we have less than
        # n_samples as representatives
        representatives_absolute_indexes = np.arange(n_samples)
        representatives_local_indexes = representatives_absolute_indexes
        # each sample starts as its own cluster (and its own parent)
        representative_cluster_assignments = representatives_absolute_indexes
        clusters_labels = representatives_absolute_indexes
        parents = representatives_absolute_indexes
        # # list of lists of indexes of the samples of each cluster
        # clusters = [[i] for i in range(n_samples)]

        i = 0
        self.n_clusters_iter_ = []
        self.labels_iter_ = []
        # iterate until every sequence of labels is unique
        while (len(representative_cluster_assignments) != len(clusters_labels) and i < self.max_iter) or i == 0:
            if self.verbose:
                print("Iteration", i)

            X_batch, representatives_absolute_resampled_indexes = self.sample_batch(X, representatives_absolute_indexes, i)

            if self.transform_once_per_iteration:
                # we transform once the data here
                X_batch = self.sampling_transform_X(X_batch, self.random_state)

            # consensus assignment (it is here that we repeatedly apply our base model)
            batch_cluster_assignments = self.get_representative_cluster_assignments(X_batch)
            batch_clusters_labels = np.unique(batch_cluster_assignments)

            # using representative_method
            batch_new_representatives_local_indexes = self.choose_new_representatives(
                X_batch,
                batch_cluster_assignments,
                batch_clusters_labels,
            )

            parents = self.update_parents(
                parents,
                representatives_absolute_resampled_indexes,  # absolute indexes of the sampled batch
                batch_clusters_labels,
                batch_cluster_assignments,
                representatives_absolute_resampled_indexes[
                    batch_new_representatives_local_indexes
                ],  # absolute indexes of the new representatives that were sampled
            )

            # if there is no batch it is a simple update, otherwise it depends on the batch strategy
            representatives_absolute_indexes, representative_cluster_assignments, clusters_labels = (
                self.update_with_unsampled_batch(
                    representatives_absolute_indexes,
                    batch_new_representatives_local_indexes,
                    batch_cluster_assignments,
                    batch_clusters_labels,
                    representatives_absolute_resampled_indexes,
                )
            )
            i += 1

        self.n_clusters_ = len(clusters_labels)
        self.labels_ = self.get_labels_from_parents(parents, representatives_absolute_indexes)
        self.parents_ = parents
        self.representatives_indexes_ = representatives_absolute_indexes
        self.cluster_representatives_ = X[representatives_absolute_indexes] # type: ignore
        self.n_iter_ = i
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X).labels_

    def predict(self, X):
        # find from each cluster representative each sample is closest to
        cluster_representatives_ = getattr(self, "cluster_representatives_", None)
        if cluster_representatives_ is None:
            raise ValueError("The model has not been fitted yet. Please call fit() before predict().")
        else:
            distances = cosine_distances(X, np.asarray(self.cluster_representatives_))
            labels = np.argmin(distances, axis=1)
            labels = self.labels_[labels]
        return labels


class ModularCoHiRF(BaseCoHiRF, ClusterMixin, BaseEstimator):
    def __init__(
        self,
        repetitions: int = 10,
        verbose: int | bool = 0,
        representative_method: str = "closest_overall",
        n_samples_representative: Optional[int] = None,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        max_iter: int = 100,
        save_path: bool = False,
        # base model parameters
        base_model: str | type[BaseEstimator] = "kmeans",
        base_model_kwargs: Optional[dict] = None,
        # sampling parameters
        n_features: int | float | str = 10,  # number of random features that will be sampled
        transform_method: Optional[str | type[TransformerMixin]] = None,
        transform_kwargs: Optional[dict] = None,
        sample_than_transform: bool = True,
        transform_once_per_iteration: bool = False,
        # batch parameters
        batch_size: Optional[int] = None,
        n_samples_threshold: int | str = "batch_size",
        use_dask: bool | str = "auto",
    ):
        super().__init__(
            repetitions,
            verbose,
            representative_method,
            n_samples_representative,
            random_state,
            n_jobs,
            max_iter,
            save_path,
            base_model=base_model,
            base_model_kwargs=base_model_kwargs,
            transform_method=transform_method,
            n_features=n_features,
            transform_kwargs=transform_kwargs,
            batch_size=batch_size,
            n_samples_threshold=n_samples_threshold,
            use_dask=use_dask,
            sample_than_transform=sample_than_transform,
            transform_once_per_iteration=transform_once_per_iteration,
        )


class CoHiRF(BaseCoHiRF, ClusterMixin, BaseEstimator):
    def __init__(
        self,
        repetitions: int = 10,
        verbose: int | bool = 0,
        representative_method: str = "closest_overall",
        n_samples_representative: Optional[int] = None,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        max_iter: int = 100,
        save_path: bool = False,
        # base model parameters
        base_model: str | type[BaseEstimator] = "kmeans",
        base_model_kwargs: Optional[dict] = None,
        # sampling parameters
        n_features: int | float | str = 10,  # number of random features that will be sampled
        transform_method: Optional[str | type[TransformerMixin]] = None,
        transform_kwargs: Optional[dict] = None,
        sample_than_transform: bool = True,
        transform_once_per_iteration: bool = False,
        # batch parameters
        batch_size: Optional[int] = None,
        n_samples_threshold: int | str = "batch_size",
        use_dask: bool | str = "auto",
        # kmeans parameters
        kmeans_n_clusters: int = 3,
        kmeans_init: str = "k-means++",
        kmeans_n_init: str | int = 'auto',
        kmeans_max_iter: int = 300,
        kmeans_tol: float = 1e-4,
    ):
        super().__init__(
            repetitions,
            verbose,
            representative_method,
            n_samples_representative,
            random_state,
            n_jobs,
            max_iter,
            save_path,
            base_model=base_model,
            base_model_kwargs=base_model_kwargs,
            transform_method=transform_method,
            n_features=n_features,
            transform_kwargs=transform_kwargs,
            batch_size=batch_size,
            n_samples_threshold=n_samples_threshold,
            use_dask=use_dask,
            sample_than_transform=sample_than_transform,
            transform_once_per_iteration=transform_once_per_iteration,
        )
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol

    def get_base_model(self, child_random_state):
        if self.base_model == "kmeans":
            return KMeans(
                n_clusters=self.kmeans_n_clusters,
                init=self.kmeans_init, # type: ignore
                n_init=self.kmeans_n_init, # type: ignore
                max_iter=self.kmeans_max_iter,
                tol=self.kmeans_tol,
                verbose=self.verbose,
                random_state=child_random_state,
            )
        else:
            raise ValueError(f"base_model {self.base_model} is not valid.")

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
