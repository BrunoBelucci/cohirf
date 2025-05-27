from typing import Literal, Optional
import numpy as np
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
from joblib import Parallel, delayed
import optuna
import pandas as pd


def update_labels(old_labels, old_representatives_absolute_indexes, old_n_clusters,
                   new_representative_cluster_assignments, verbose=0):
    if verbose > 0:
        print("Updating labels")
    # if it is the first iteration we can directly use
    # new_representative_cluster_assignments
    if old_labels is None:
        new_labels = new_representative_cluster_assignments
    # else we have to iterate through the old number of clusters and assign them to the new labels
    else:
        old_labels_values = old_labels[old_representatives_absolute_indexes]
        d_array = np.arange(old_n_clusters)
        d_array[old_labels_values] = new_representative_cluster_assignments
        new_labels = d_array[old_labels]
    return new_labels


def get_labels_from_parents(parents, representative_indexes, verbose=0):
    if verbose > 0:
        print("Getting labels from parents")
    # Find the root parent for each sample
    roots = np.arange(len(parents))
    while True:
        new_roots = parents[roots]
        if np.all(new_roots == roots):
            break
        roots = new_roots
    # Map each representative to a cluster label
    rep_to_label = {rep: i for i, rep in enumerate(representative_indexes)}
    # Assign label to each sample based on its root
    labels = np.array([rep_to_label.get(root, -1) for root in roots], dtype=int)
    return labels


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
        hierarchy_strategy: Literal["parents", "labels"] = "parents",
        automatically_get_labels: bool = True,
        # base model parameters
        base_model: str | type[BaseEstimator] = "kmeans",
        base_model_kwargs: Optional[dict] = None,
        # sampling parameters
        n_features: int | float | str = 10,  # number of random features that will be sampled
        transform_method: Optional[str | type[TransformerMixin]] = None,
        transform_kwargs: Optional[dict] = None,
        sample_than_transform: bool = True,
        transform_once_per_iteration: bool = False,
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
        self.hierarchy_strategy = hierarchy_strategy
        self.automatically_get_labels = automatically_get_labels
        for key, value in kwargs.items():
            setattr(self, key, value)

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
        unique, codes = np.unique(labels_i, axis=0, return_inverse=True)
        n_clusters = len(unique)
        return codes, n_clusters

    def compute_similarities(self, X_cluster: np.ndarray):
        if self.verbose:
            print("Computing similarities with method", self.representative_method)

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
            cluster_similarities = rbf_kernel(X_cluster) 

        elif self.representative_method == "rbf_median":
            # replace cosine_distance by rbf_kernel with gamma = median
            cluster_distances = euclidean_distances(X_cluster) 
            median_distance = np.median(cluster_distances)
            gamma = 1 / (2 * median_distance)
            cluster_similarities = np.exp(-gamma * cluster_distances)

        elif self.representative_method == "laplacian":
            # replace cosine_distance by laplacian_kernel
            cluster_similarities = laplacian_kernel(X_cluster)

        elif self.representative_method == "laplacian_median":
            # replace cosine_distance by laplacian_kernel with gamma = median
            cluster_distances = manhattan_distances(X_cluster) 
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
        new_unique_clusters_labels,
    ):
        new_representatives_local_indexes = []
        for label in new_unique_clusters_labels:
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

    def update_parents(
        self,
        old_parents,
        old_representatives_absolute_indexes,
        new_unique_clusters_labels,
        new_n_clusters,
        new_representative_cluster_assignments,
        new_representatives_absolute_indexes,
    ):
        if self.verbose:
            print("Updating parents")
        new_parents = old_parents.copy()
        d_array = np.arange(new_n_clusters)
        d_array[new_unique_clusters_labels] = new_representatives_absolute_indexes
        new_representative_indexes_assignments = d_array[new_representative_cluster_assignments]
        new_parents[old_representatives_absolute_indexes] = new_representative_indexes_assignments
        # for new_representative_index, new_cluster_label in zip(
        #     new_representatives_absolute_indexes, new_unique_clusters_labels
        # ):
        #     parents_indexes_to_update = old_representatives_absolute_indexes[
        #         new_representative_cluster_assignments == new_cluster_label
        #     ]
        #     new_parents[parents_indexes_to_update] = new_representative_index
        return new_parents

    # def get_all_parents_indexes(self, parents, representative_index):
    #     all_indexes = set()
    #     indexes_to_append = [representative_index]
    #     first = True
    #     while len(indexes_to_append) > 0:  # the representative_index itself will always be in the list
    #         all_indexes.update(indexes_to_append)
    #         indexes_to_append = np.where(np.isin(parents, indexes_to_append))[0]
    #         if first:
    #             first = False
    #             indexes_to_append = np.setdiff1d(indexes_to_append, representative_index, assume_unique=True)
    #     return list(all_indexes)

    # def get_labels_from_parents(self, parents, representative_indexes):
    #     if self.verbose:
    #         print("Getting labels from parents")
    #     labels = np.empty(parents.shape[0], dtype=int)
    #     for i, representative_index in enumerate(representative_indexes):
    #         all_indexes = self.get_all_parents_indexes(parents, representative_index)
    #         labels[all_indexes] = i
    #     return labels

    def fit(self, X: pd.DataFrame | np.ndarray, y=None, sample_weight=None):
        if self.verbose:
            print("Starting fit")
        n_samples = X.shape[0]

        # we will work with numpy (arrays) for speed
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # indexes of the representative samples, start with (n_samples) but will be updated when we have less than
        # n_samples as representatives
        representatives_absolute_indexes = np.arange(n_samples)
        # representatives_local_indexes = representatives_absolute_indexes
        # each sample starts as its own cluster (and its own parent)
        representatives_cluster_assignments = representatives_absolute_indexes
        n_clusters = 0  # actually len(representatives_cluster_assignments) but 0 in the beggining for optimization
        self.labels_ = None
        if self.hierarchy_strategy == "parents":
            parents = representatives_absolute_indexes
        elif self.hierarchy_strategy == "labels":
            parents = None
        else:
            raise ValueError("hierarchy_strategy must be 'parents' or 'labels'")

        i = 0
        self.n_clusters_iter_ = []
        self.labels_iter_ = []
        # iterate until every sequence of labels is unique
        while (len(representatives_cluster_assignments) != n_clusters and i < self.max_iter) or i == 0:
            if self.verbose:
                print("Iteration", i)

            X_representatives = X[representatives_absolute_indexes]

            if self.transform_once_per_iteration:
                # we transform once the data here
                X_representatives = self.sampling_transform_X(X_representatives, self.random_state)

            # consensus assignment (it is here that we repeatedly apply our base model)
            representatives_cluster_assignments, new_n_clusters = self.get_representative_cluster_assignments(
                X_representatives
            )
            unique_clusters_labels = np.arange(new_n_clusters)

            # using representative_method
            new_representatives_local_indexes = self.choose_new_representatives(
                X_representatives,
                representatives_cluster_assignments,
                unique_clusters_labels,
            )

            if self.hierarchy_strategy == "parents":
                parents = self.update_parents(
                    parents,
                    representatives_absolute_indexes,  # old_representatives_absolute_indexes
                    unique_clusters_labels,
                    new_n_clusters,
                    representatives_cluster_assignments,
                    representatives_absolute_indexes[
                        new_representatives_local_indexes
                    ],  # new_representatives_absolute_indexes
                )
            elif self.hierarchy_strategy == "labels":
                self.labels_ = update_labels(
                    self.labels_,
                    representatives_absolute_indexes,
                    n_clusters,
                    representatives_cluster_assignments,
                    self.verbose,
                )
            else:
                raise ValueError("hierarchy_strategy must be 'parents' or 'labels'")

            representatives_absolute_indexes = representatives_absolute_indexes[new_representatives_local_indexes]
            n_clusters = new_n_clusters

            i += 1

        self.n_clusters_ = n_clusters
        self.parents_ = parents
        self.representatives_indexes_ = representatives_absolute_indexes
        self.cluster_representatives_ = X[representatives_absolute_indexes] 
        if self.automatically_get_labels:
            self.labels_ = self.get_labels()
        self.n_iter_ = i
        return self

    def get_labels(self):
        if self.hierarchy_strategy == "parents":
            if self.parents_ is None or self.representatives_indexes_ is None:
                raise ValueError(
                    "The model has not been fitted yet. Please call fit() before get_labels()."
                )
            self.labels_ = get_labels_from_parents(self.parents_, self.representatives_indexes_, self.verbose)
        elif self.hierarchy_strategy == "labels":
            if self.labels_ is None:
                raise ValueError(
                    "The model has not been fitted yet. Please call fit() before get_labels()."
                )
            self.labels_ = self.labels_
        else:
            raise ValueError("hierarchy_strategy must be 'parents' or 'labels'")
        return self.labels_

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        if not self.automatically_get_labels:
            self.get_labels()
        if self.labels_ is None:
            raise ValueError('Something went wrong, please check the code.')
        return self.labels_

    def predict(self, X):
        # find from each cluster representative each sample is closest to
        cluster_representatives_ = getattr(self, "cluster_representatives_", None)
        if cluster_representatives_ is None or self.labels_ is None:
            raise ValueError(
                "The model has not been fitted yet. Please call fit() before predict(),"
                " if automatically_get_labels is False, you must call get_labels() after fit()."
            )
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
        hierarchy_strategy: Literal['parents', 'labels'] = "parents",
        automatically_get_labels: bool = True,
        # base model parameters
        base_model: str | type[BaseEstimator] = "kmeans",
        base_model_kwargs: Optional[dict] = None,
        # sampling parameters
        n_features: int | float | str = 10,  # number of random features that will be sampled
        transform_method: Optional[str | type[TransformerMixin]] = None,
        transform_kwargs: Optional[dict] = None,
        sample_than_transform: bool = True,
        transform_once_per_iteration: bool = False,
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
            hierarchy_strategy,
            automatically_get_labels,
            base_model=base_model,
            base_model_kwargs=base_model_kwargs,
            transform_method=transform_method,
            n_features=n_features,
            transform_kwargs=transform_kwargs,
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
        hierarchy_strategy: Literal["parents", "labels"] = "parents",
        automatically_get_labels: bool = True,
        # base model parameters
        base_model: str | type[BaseEstimator] = "kmeans",
        base_model_kwargs: Optional[dict] = None,
        # sampling parameters
        n_features: int | float | str = 10,  # number of random features that will be sampled
        transform_method: Optional[str | type[TransformerMixin]] = None,
        transform_kwargs: Optional[dict] = None,
        sample_than_transform: bool = True,
        transform_once_per_iteration: bool = False,
        # kmeans parameters
        kmeans_n_clusters: int = 3,
        kmeans_init: str = "k-means++",
        kmeans_n_init: str | int = "auto",
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
            hierarchy_strategy,
            automatically_get_labels,
            base_model=base_model,
            base_model_kwargs=base_model_kwargs,
            transform_method=transform_method,
            n_features=n_features,
            transform_kwargs=transform_kwargs,
            sample_than_transform=sample_than_transform,
            transform_once_per_iteration=transform_once_per_iteration,
        )
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol

    def get_base_model(self, child_random_state):
        random_seed = child_random_state.integers(0, 1e6)
        if self.base_model == "kmeans":
            return KMeans(
                n_clusters=self.kmeans_n_clusters,
                init=self.kmeans_init,  # type: ignore
                n_init=self.kmeans_n_init,  # type: ignore
                max_iter=self.kmeans_max_iter,
                tol=self.kmeans_tol,
                verbose=self.verbose,
                random_state=random_seed,
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
