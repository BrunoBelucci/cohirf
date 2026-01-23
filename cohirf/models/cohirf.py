from typing import Literal, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics.pairwise import (
    cosine_distances,
    rbf_kernel,
    laplacian_kernel,
    euclidean_distances,
    manhattan_distances,
    cosine_similarity,
)
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline
from cohirf.models.scsrgf import SpectralSubspaceRandomization
from sklearn.metrics import adjusted_rand_score
from warnings import warn, filterwarnings

# Suppress sklearn warning about unique classes in clustering metrics
# this warning appears when using adjusted_rand_score with a large number of clusters
# which is common in our method, especially at first iterations
filterwarnings(
    "ignore",
    message=".*number of unique classes is greater than 50%.*",
    category=UserWarning,
    module="sklearn.metrics.cluster._supervised",
)


def update_parents(
    old_parents,
    old_representatives_absolute_indexes,
    new_unique_clusters_labels,
    new_representative_cluster_assignments,
    new_representatives_absolute_indexes,
    verbose,
):
    if verbose:
        print("Updating parents")
    label_to_representative_idx = np.full(max(new_unique_clusters_labels) + 1, -1, dtype=int)
    label_to_representative_idx[new_unique_clusters_labels] = new_representatives_absolute_indexes
    new_representative_indexes_assignments = label_to_representative_idx[new_representative_cluster_assignments]
    old_parents[old_representatives_absolute_indexes] = new_representative_indexes_assignments
    return old_parents


def choose_new_representatives(
    X_representatives: np.ndarray,
    new_representative_cluster_assignments: np.ndarray,
    new_unique_clusters_labels: np.ndarray,
    representative_method: str,
    verbose: int,
    random_state: np.random.Generator,
    n_samples_representative: Optional[int],
):
    new_representatives_local_indexes = []
    for label in new_unique_clusters_labels:
        if verbose:
            print("Choosing new representative sample for cluster", label)
        cluster_mask = new_representative_cluster_assignments == label
        X_cluster = X_representatives[cluster_mask]
        X_cluster_indexes = np.where(cluster_mask)[0]

        # sample a representative sample from the cluster
        if n_samples_representative is not None:
            n_samples_representative = min(n_samples_representative, X_cluster.shape[0])
            sampled_indexes = random_state.choice(X_cluster.shape[0], size=n_samples_representative, replace=False)
            X_cluster = X_cluster[sampled_indexes]
            X_cluster_indexes = X_cluster_indexes[sampled_indexes]

        cluster_similarities = compute_similarities(X_cluster, representative_method, verbose)
        cluster_similarities_sum = cluster_similarities.sum(axis=0)
        most_similar_sample_local_idx = X_cluster_indexes[cluster_similarities_sum.argmax()]
        new_representatives_local_indexes.append(most_similar_sample_local_idx)
    new_representatives_local_indexes = np.array(new_representatives_local_indexes)
    return new_representatives_local_indexes


def compute_similarities(X_cluster: np.ndarray, representative_method: str, verbose: int) -> np.ndarray:
    if verbose:
        print("Computing similarities with method", representative_method)

    if representative_method == "closest_overall":
        # calculate the cosine similarities between all samples in the cluster and
        # pick the one with the largest sum
        # this is the most computationally expensive method (O(n^2))
        # cluster_similarities = X_cluster @ X_cluster.T -> this only works if X is normalized, which is generally the case
        cluster_similarities = cosine_similarity(X_cluster)  # but this ensures that it works in all cases

    elif representative_method == "closest_to_centroid":
        # calculate the centroid of the cluster and pick the sample most similar to it
        # this is the second most computationally expensive method (O(n))
        centroid = X_cluster.mean(axis=0)
        # cluster_similarities = X_cluster @ centroid
        cluster_similarities = cosine_similarity(X_cluster, centroid.reshape(1, -1))

    # elif self.representative_method == 'centroid':
    #     # calculate the centroid of the cluster and use it as the representative sample
    #     # this is the least computationally expensive method (O(1))
    #     centroid = X_cluster.mean(axis=0)
    #     # we arbitrarily pick the first sample as the representative of the cluster and change its
    #     # values to the centroid values so we can use the same logic as the other methods
    #     most_similar_sample_idx = local_cluster_original_idx[local_cluster_sampled_idx[0]]
    #     # we need to change the original value in X
    #     X[most_similar_sample_idx, :] = centroid

    elif representative_method == "rbf":
        # replace cosine_distance by rbf_kernel
        cluster_similarities = rbf_kernel(X_cluster)

    elif representative_method == "rbf_median":
        # replace cosine_distance by rbf_kernel with gamma = median
        cluster_distances = euclidean_distances(X_cluster)
        median_distance = np.median(cluster_distances)
        gamma = 1 / (2 * median_distance)
        cluster_similarities = np.exp(-gamma * cluster_distances)

    elif representative_method == "laplacian":
        # replace cosine_distance by laplacian_kernel
        cluster_similarities = laplacian_kernel(X_cluster)

    elif representative_method == "laplacian_median":
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


def compute_ari_without_column(codes: np.ndarray, labels_i: np.ndarray, col_idx: int):
        # Create mask to exclude column col_idx
        mask = np.ones(labels_i.shape[1], dtype=bool)
        mask[col_idx] = False
        labels_i_subset = labels_i[:, mask]
        unique_subset, codes_subset = np.unique(labels_i_subset, axis=0, return_inverse=True)
        ari = adjusted_rand_score(codes, codes_subset)
        return ari

def calculate_pairwise_ari(labels_i: np.ndarray, n_jobs: int, verbose: int):
    aris = Parallel(n_jobs=n_jobs, return_as="list", verbose=verbose, backend="loky")(
        delayed(adjusted_rand_score)(labels_i[:, i], labels_i[:, j])
        for i in range(labels_i.shape[1])
        for j in range(i + 1, labels_i.shape[1])
    )
    aris = np.array(aris)
    # reshape into a upper triangular matrix
    aris_matrix = np.zeros((labels_i.shape[1], labels_i.shape[1]))
    upper_tri_idx = np.triu_indices(labels_i.shape[1], k=1)  # k=1 to exclude diagonal
    aris_matrix[upper_tri_idx] = aris
    return aris_matrix

def find_two_most_similar_repetitions(labels_i: np.ndarray, threshold: float, n_jobs: int, verbose: int):
    # Calculate pairwise ARI between all repetitions
    aris_matrix = calculate_pairwise_ari(labels_i, n_jobs, verbose)

    # get the maximum ARI and its indices
    max_ari_idx = np.unravel_index(np.argmax(aris_matrix, axis=None), aris_matrix.shape)
    best_ari = aris_matrix[max_ari_idx]

    if best_ari < threshold:
        # If best pairwise ARI is below threshold, warn the user and still merge best pair
        warn(
            f"Bottom-up consensus strategy: No pair of repetitions have ARI above threshold {threshold}, "
            f"merging best pair with ARI {best_ari}."
        )
    # Merge the two most similar columns
    new_labels_i = labels_i[:, max_ari_idx]  # I am not sure if the syntax is correct here
    unique, codes = np.unique(new_labels_i, axis=0, return_inverse=True)

    # Remove the merged columns from labels_i
    labels_i = np.delete(labels_i, max_ari_idx, axis=1)
    return unique, codes, labels_i

def get_consensus_labels(labels_i: np.ndarray, consensus_strategy: str, consensus_threshold: float, n_jobs: int, verbose: int):
    if consensus_strategy == "factorize" or labels_i.shape[1] == 1:
        # simply consider each unique row as a separate cluster
        unique, codes = np.unique(labels_i, axis=0, return_inverse=True)

    elif consensus_strategy == "top-down":
        # start with all repetitions (columns) and remove one repetition at a time
        # we try to remove every repetition and see what are the new labels without this repetition
        # we then compare the new labels with the old ones using ARI, if the repetition agrees with the other ones
        # we expect a relatively high ARI, so we remove the repetition with the lowest ARI (if below threshold)
        # we continue until no repetition can be removed (removing any repetition does not lower the ARI below threshold)
        unique, codes = np.unique(labels_i, axis=0, return_inverse=True)
        while labels_i.shape[1] > 1:
            aris = Parallel(n_jobs=n_jobs, return_as="list", verbose=verbose, backend="loky")(
                delayed(compute_ari_without_column)(codes, labels_i, i) for i in range(labels_i.shape[1])
            )
            aris = np.array(aris)
            min_ari_idx = np.argmin(aris)
            min_ari = aris[min_ari_idx]
            if min_ari < consensus_threshold:
                # Remove the column and update codes
                mask = np.ones(labels_i.shape[1], dtype=bool)
                mask[min_ari_idx] = False
                labels_i = labels_i[:, mask]
                unique, codes = np.unique(labels_i, axis=0, return_inverse=True)
            else:
                break

    elif consensus_strategy == "top-down-inv":
        # start with all repetitions (columns) and remove one repetition at a time
        # we try to remove every repetition and see what are the new labels without this repetition
        # we then compare the new labels with the old ones using ARI, if the repetition agrees with the other ones
        # we expect a relatively high ARI
        # but now we remove the repetition with the highest ARI (if above threshold), because we expect that this repetition
        # is redundant with the other ones
        unique, codes = np.unique(labels_i, axis=0, return_inverse=True)
        while labels_i.shape[1] > 1:
            aris = Parallel(n_jobs=n_jobs, return_as="list", verbose=verbose, backend="loky")(
                delayed(compute_ari_without_column)(codes, labels_i, i) for i in range(labels_i.shape[1])
            )
            aris = np.array(aris)
            max_ari_idx = np.argmax(aris)
            max_ari = aris[max_ari_idx]
            if max_ari > consensus_threshold:
                # Remove the column and update codes
                mask = np.ones(labels_i.shape[1], dtype=bool)
                mask[max_ari_idx] = False
                labels_i = labels_i[:, mask]
                unique, codes = np.unique(labels_i, axis=0, return_inverse=True)
            else:
                break

    elif consensus_strategy == "top-down-approx":
        # approximate version of top-down where we only do one pass
        unique, codes = np.unique(labels_i, axis=0, return_inverse=True)
        aris = Parallel(n_jobs=n_jobs, return_as="list", verbose=verbose, backend="loky")(
            delayed(compute_ari_without_column)(codes, labels_i, i) for i in range(labels_i.shape[1])
        )
        aris = np.array(aris)
        # Remove columns with ARI below threshold
        mask = aris >= consensus_threshold
        labels_i = labels_i[:, mask]
        unique, codes = np.unique(labels_i, axis=0, return_inverse=True)

    elif consensus_strategy == "top-down-inv-approx":
        # approximate version of top-down-inv where we only do one pass
        unique, codes = np.unique(labels_i, axis=0, return_inverse=True)
        aris = Parallel(n_jobs=n_jobs, return_as="list", verbose=verbose, backend="loky")(
            delayed(compute_ari_without_column)(codes, labels_i, i) for i in range(labels_i.shape[1])
        )
        aris = np.array(aris)
        # Remove columns with ARI above threshold
        mask = aris < consensus_threshold
        labels_i = labels_i[:, mask]
        unique, codes = np.unique(labels_i, axis=0, return_inverse=True)

    elif consensus_strategy == "bottom-up":
        # Start with the two most similar repetitions and iteratively merge more
        unique, codes, labels_i = find_two_most_similar_repetitions(labels_i, consensus_threshold, n_jobs, verbose)

        # Continue merging while we have columns and max ARI is above threshold
        while labels_i.shape[1] > 0:
            # Calculate ARI between current consensus and each remaining column
            aris = Parallel(n_jobs=n_jobs, return_as="list", verbose=verbose, backend="loky")(
                delayed(adjusted_rand_score)(codes, labels_i[:, col_idx]) for col_idx in range(labels_i.shape[1])
            )
            aris = np.array(aris)

            # Find the column with highest ARI to current consensus
            max_ari_idx = np.argmax(aris)
            max_ari = aris[max_ari_idx]

            if max_ari >= consensus_threshold:
                # Merge this column with current consensus
                new_labels_i = np.column_stack([codes.reshape(-1, 1), labels_i[:, max_ari_idx].reshape(-1, 1)])
                unique, codes = np.unique(new_labels_i, axis=0, return_inverse=True)

                # Remove the merged column
                labels_i = np.delete(labels_i, max_ari_idx, axis=1)
            else:
                # No more columns meet threshold, stop merging
                break

    elif consensus_strategy == "bottom-up-inv":
        # Start with the two most similar repetitions and iteratively merge more
        unique, codes, labels_i = find_two_most_similar_repetitions(labels_i, consensus_threshold, n_jobs, verbose)

        # Continue merging while we have columns and min ARI is below threshold
        while labels_i.shape[1] > 0:
            # Calculate ARI between current consensus and each remaining column
            aris = Parallel(n_jobs=n_jobs, return_as="list", verbose=verbose, backend="loky")(
                delayed(adjusted_rand_score)(codes, labels_i[:, col_idx]) for col_idx in range(labels_i.shape[1])
            )
            aris = np.array(aris)

            # Find the column with lowest ARI to current consensus
            min_ari_idx = np.argmin(aris)
            min_ari = aris[min_ari_idx]

            if min_ari < consensus_threshold:
                # Merge this column with current consensus
                new_labels_i = np.column_stack([codes.reshape(-1, 1), labels_i[:, min_ari_idx].reshape(-1, 1)])
                unique, codes = np.unique(new_labels_i, axis=0, return_inverse=True)

                # Remove the merged column
                labels_i = np.delete(labels_i, min_ari_idx, axis=1)
            else:
                # No more columns meet threshold, stop merging
                break

    elif consensus_strategy == "bottom-up-approx":
        # approximate version of bottom-up where we only do one pass
        unique, codes, labels_i = find_two_most_similar_repetitions(labels_i, consensus_threshold, n_jobs, verbose)
        # Merge remaining columns with current consensus if above threshold
        # Calculate ARI between current consensus and each remaining column
        aris = Parallel(n_jobs=n_jobs, return_as="list", verbose=verbose, backend="loky")(
            delayed(adjusted_rand_score)(codes, labels_i[:, col_idx]) for col_idx in range(labels_i.shape[1])
        )
        aris = np.array(aris)
        mask = aris >= consensus_threshold
        new_labels_i = np.column_stack([codes.reshape(-1, 1), labels_i[:, mask]])
        unique, codes = np.unique(new_labels_i, axis=0, return_inverse=True)

    elif consensus_strategy == "bottom-up-inv-approx":
        # approximate version of bottom-up-inv where we only do one pass
        unique, codes, labels_i = find_two_most_similar_repetitions(labels_i, consensus_threshold, n_jobs, verbose)
        # Merge remaining columns with current consensus if below threshold
        # Calculate ARI between current consensus and each remaining column
        aris = Parallel(n_jobs=n_jobs, return_as="list", verbose=verbose, backend="loky")(
            delayed(adjusted_rand_score)(codes, labels_i[:, col_idx]) for col_idx in range(labels_i.shape[1])
        )
        aris = np.array(aris)
        mask = aris < consensus_threshold
        new_labels_i = np.column_stack([codes.reshape(-1, 1), labels_i[:, mask]])
        unique, codes = np.unique(new_labels_i, axis=0, return_inverse=True)

    else:
        raise ValueError("Unknown consensus strategy")

    return unique, codes


def update_labels(
    old_labels, old_representatives_absolute_indexes, old_n_clusters, new_representative_cluster_assignments, verbose=0
):
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


class BaseCoHiRF(ClusterMixin, BaseEstimator):

    def __init__(
        self,
        repetitions: int = 10,
        verbose: int | bool = 0,
        representative_method: Literal[
            "closest_overall", "closest_to_centroid", "rbf", "rbf_median", "laplacian", "laplacian_median"
        ] = "closest_overall",
        n_samples_representative: Optional[int] = None,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        max_iter: int = 100,
        save_path: bool = False,
        hierarchy_strategy: Literal["parents", "labels"] = "parents",
        automatically_get_labels: bool = True,
        # base model parameters
        base_model: str | type[BaseEstimator] | Pipeline = "kmeans",
        base_model_kwargs: Optional[dict] = None,
        # last model parameters
        last_model: Optional[str | type[BaseEstimator] | Pipeline] = None,
        last_model_kwargs: Optional[dict] = None,
        # consensus parameters
        consensus_strategy: Literal[
            "factorize", "top-down", "top-down-approx", "bottom-up", "bottom-up-approx"
        ] = "factorize",
        consensus_threshold: float = 0.8,
        # sampling parameters
        n_features: int | float = 10,  # number of random features that will be sampled
        transform_method: Optional[str | type[TransformerMixin] | Pipeline] = None,
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
        self.last_model = last_model
        self.last_model_kwargs = last_model_kwargs if last_model_kwargs is not None else {}
        self.consensus_strategy = consensus_strategy
        self.consensus_threshold = consensus_threshold
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def random_state(self):
        if self._random_state is None:
            self._random_state = np.random.default_rng()
        elif isinstance(self._random_state, int):
            self._random_state = np.random.default_rng(self._random_state)
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        if value is None:
            self._random_state = np.random.default_rng()
        elif isinstance(value, int):
            self._random_state = np.random.default_rng(value)
        else:
            raise ValueError("random_state must be an integer or None.")

    def get_base_model(self, X, child_random_state, base_model, base_model_kwargs):
        random_seed = child_random_state.integers(0, 1e6)
        if isinstance(self.base_model, str):
            if base_model == "kmeans":
                model = KMeans(**base_model_kwargs, random_state=random_seed)
            else:
                raise ValueError(f"base_model {base_model} is not valid.")
        elif isinstance(self.base_model, type) and issubclass(base_model, BaseEstimator):
            model = base_model()
            model.set_params(**base_model_kwargs)
            if hasattr(model, "random_state"):
                model.set_params(random_state=random_seed)
            elif hasattr(model, "random_seed"):
                model.set_params(random_seed=random_seed)
            if isinstance(model, HDBSCAN):
                params = model.get_params()
                min_samples = params.get("min_samples", None)
                min_cluster_size = params.get("min_cluster_size")
                if min_samples is not None and min_samples > X.shape[0]:
                    # if min_samples is larger than the number of samples, set it to the number of samples
                    model.set_params(min_samples=X.shape[0])
                elif min_samples is None and min_cluster_size > X.shape[0]:
                    # in this case min_samples is equal to min_cluster_size by default
                    model.set_params(min_samples=X.shape[0])
            if isinstance(model, SpectralSubspaceRandomization):
                params = model.get_params()
                knn = params.get("knn", None)
                sc_n_clusters = params.get("sc_n_clusters", None)
                if knn is not None and knn >= X.shape[0]:
                    # if knn is larger than the number of samples, set it to number of samples
                    model.set_params(knn=X.shape[0])
                if sc_n_clusters is not None and sc_n_clusters > X.shape[0]:
                    # if n_clusters is larger than the number of samples, set it to number of samples
                    model.set_params(sc_n_clusters=X.shape[0])
        elif isinstance(base_model, Pipeline):
            model = base_model
            model.set_params(**base_model_kwargs)
            for step_name, step in model.steps:
                if hasattr(step, "random_state"):
                    step.set_params(random_state=random_seed)
                elif hasattr(step, "random_seed"):
                    step.set_params(random_seed=random_seed)
        else:
            raise ValueError(f"base_model {base_model} is not valid.")

        if hasattr(model, "n_clusters"):
            n_clusters = getattr(model, "n_clusters")
            if n_clusters > X.shape[0]:
                if self.verbose:
                    print(
                        f"Warning: base_model n_clusters ({n_clusters}) is larger than the number of samples"
                        f" ({X.shape[0]}). Setting n_clusters to {X.shape[0]}."
                    )
                model.set_params(n_clusters=X.shape[0])

        return model

    def sampling_transform_X(self, X, child_random_state):
        random_seed = child_random_state.integers(0, 1e6)
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
                    transformer.set_params(random_state=random_seed)  # type: ignore
                except AttributeError:
                    setattr(transformer, "random_state", random_seed)
            elif hasattr(transformer, "random_seed"):
                try:
                    transformer.set_params(random_seed=random_seed)  # type: ignore
                except AttributeError:
                    setattr(transformer, "random_seed", random_seed)
            X_p = transformer.fit_transform(X)
        elif isinstance(self.transform_method, Pipeline):
            transformer = self.transform_method
            transformer.set_params(**self.transform_kwargs)
            for step_name, step in transformer.steps:
                if hasattr(step, "random_state"):
                    step.set_params(random_state=random_seed)
                elif hasattr(step, "random_seed"):
                    step.set_params(random_seed=random_seed)
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
            size = min(self.n_features, n_all_features)
            features = child_random_state.choice(n_all_features, size=size, replace=False)
        elif isinstance(self.n_features, float):
            # sample a percentage of features
            if self.n_features < 0 or self.n_features > 1:
                raise ValueError("n_features must be between 0 and 1")
            features_size = int(self.n_features * n_all_features)
            # we ensure that we get at least one feature
            features_size = max(features_size, 1)
            size = min(features_size, n_all_features)
            features = child_random_state.choice(n_all_features, size=size, replace=False)
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

    def run_one_repetition(self, X_representative, repetition, child_random_state):
        if self.verbose:
            print("Starting repetition", repetition)
        X_p = self.sample_X_j(X_representative, child_random_state)
        base_model = self.get_base_model(X_p, child_random_state, self.base_model, self.base_model_kwargs)
        labels = self.get_labels_from_base_model(base_model, X_p)
        if -1 in labels:
            # we will consider each noise label (-1 for DBSCAN/HDBSCAN) as a separate cluster
            # we will assign a new label for each noise sample
            noise_mask = labels == -1
            noise_labels = np.arange(np.sum(noise_mask)) + np.max(labels) + 1
            labels[noise_mask] = noise_labels
        return labels

    def get_representative_cluster_assignments(self, X, representatives_absolute_indexes):

        X_representatives = X[representatives_absolute_indexes]

        if self.transform_once_per_iteration:
            # we transform once the data here
            X_representatives = self.sampling_transform_X(X_representatives, self.random_state)

        # run the repetitions in parallel using loky, which is finally more stable than threading
        # I don't really understand why, but at least this works (even if may consume more memory)
        if self.verbose:
            print("Starting consensus assignment")

        child_random_states = self.random_state.spawn(self.repetitions)

        labels_i = Parallel(n_jobs=self.n_jobs, return_as="list", verbose=self.verbose, backend="loky")(
            delayed(self.run_one_repetition)(X_representatives, r, child_random_states[r]) for r in range(self.repetitions)
        )
        labels_i = np.array(labels_i).T

        # factorize labels using numpy (codes are from 0 to n_clusters-1)
        unique, codes = get_consensus_labels(labels_i, self.consensus_strategy, self.consensus_threshold, self.n_jobs, self.verbose)
        if unique is None:
            raise ValueError("Something went wrong, check your code!")

        n_clusters = len(unique)
        return codes, n_clusters

    def choose_new_representatives(
        self,
        X_representatives: np.ndarray,
        new_representative_cluster_assignments: np.ndarray,
        new_unique_clusters_labels: np.ndarray,
    ):
        return choose_new_representatives(
            X_representatives,
            new_representative_cluster_assignments,
            new_unique_clusters_labels,
            self.representative_method,
            self.verbose,
            self.random_state,
            self.n_samples_representative,
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y=None, sample_weight=None):
        if self.verbose:
            print("Starting fit")
        n_samples = X.shape[0]

        # we will work with numpy (arrays) for speed
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # indexes of the representative samples, start with (n_samples) but will be updated when we have less than
        # n_samples as representatives
        self.representatives_absolute_indexes = np.arange(n_samples)
        # representatives_local_indexes = representatives_absolute_indexes
        # each sample starts as its own cluster (and its own parent)
        representatives_cluster_assignments = self.representatives_absolute_indexes
        n_clusters = 0  # actually len(representatives_cluster_assignments) but 0 in the beggining to enter the loop
        self.labels_ = None
        if self.hierarchy_strategy == "parents":
            parents = self.representatives_absolute_indexes
        elif self.hierarchy_strategy == "labels":
            parents = None
        else:
            raise ValueError("hierarchy_strategy must be 'parents' or 'labels'")

        i = 0
        self.n_clusters_iter_ = []
        self.labels_iter_ = []
        self.representatives_iter_ = []
        # iterate until every sequence of labels is unique
        while (len(representatives_cluster_assignments) != n_clusters and i < self.max_iter):
            if self.verbose:
                print("Iteration", i)

            # consensus assignment (it is here that we repeatedly apply our base model)
            representatives_cluster_assignments, new_n_clusters = self.get_representative_cluster_assignments(
                X, self.representatives_absolute_indexes
            )
            unique_clusters_labels = np.arange(new_n_clusters)

            # using representative_method
            new_representatives_local_indexes = self.choose_new_representatives(
                X[self.representatives_absolute_indexes],
                representatives_cluster_assignments,
                unique_clusters_labels,
            )

            if self.hierarchy_strategy == "parents":
                parents = update_parents(
                    parents,
                    self.representatives_absolute_indexes,  # old_representatives_absolute_indexes
                    unique_clusters_labels,
                    representatives_cluster_assignments,
                    self.representatives_absolute_indexes[
                        new_representatives_local_indexes
                    ],  # new_representatives_absolute_indexes
                    self.verbose
                )
            elif self.hierarchy_strategy == "labels":
                self.labels_ = update_labels(
                    self.labels_,
                    self.representatives_absolute_indexes,
                    n_clusters,
                    representatives_cluster_assignments,
                    self.verbose,
                )
            else:
                raise ValueError("hierarchy_strategy must be 'parents' or 'labels'")

            self.representatives_absolute_indexes = self.representatives_absolute_indexes[new_representatives_local_indexes]
            n_clusters = new_n_clusters
            if self.save_path:
                self.representatives_iter_.append(self.representatives_absolute_indexes)

            i += 1

        if self.last_model is not None:
            # one last clustering step but with a single run of last_model instead of the consensus with base_model
            if self.verbose:
                print("Final clustering with last_model")
            X_representatives = X[self.representatives_absolute_indexes]
            last_model = self.get_base_model(X_representatives, self.random_state, self.last_model, self.last_model_kwargs)
            representatives_cluster_assignments = last_model.fit_predict(X_representatives)
            new_n_clusters = len(np.unique(representatives_cluster_assignments))
            unique_clusters_labels = np.arange(new_n_clusters)
            new_representatives_local_indexes = self.choose_new_representatives(
                X_representatives,
                representatives_cluster_assignments,
                unique_clusters_labels,
            )
            if self.hierarchy_strategy == "parents":
                parents = self.update_parents(
                    parents,
                    self.representatives_absolute_indexes,  # old_representatives_absolute_indexes
                    unique_clusters_labels,
                    new_n_clusters,
                    representatives_cluster_assignments,
                    self.representatives_absolute_indexes[
                        new_representatives_local_indexes
                    ],  # new_representatives_absolute_indexes
                )
            elif self.hierarchy_strategy == "labels":
                self.labels_ = update_labels(
                    self.labels_,
                    self.representatives_absolute_indexes,
                    n_clusters,
                    representatives_cluster_assignments,
                    self.verbose,
                )
            else:
                raise ValueError("hierarchy_strategy must be 'parents' or 'labels'")

            self.representatives_absolute_indexes = self.representatives_absolute_indexes[new_representatives_local_indexes]
            n_clusters = new_n_clusters
            if self.save_path:
                self.representatives_iter_.append(self.representatives_absolute_indexes)

        self.n_clusters_ = n_clusters
        self.parents_ = parents
        self.representatives_indexes_ = self.representatives_absolute_indexes
        del self.representatives_absolute_indexes
        self.cluster_representatives_ = X[self.representatives_indexes_]
        if self.automatically_get_labels:
            self.labels_ = self.get_labels()
        self.n_iter_ = i
        return self

    def get_labels(self):
        if self.hierarchy_strategy == "parents":
            if self.parents_ is None or self.representatives_indexes_ is None:
                raise ValueError("The model has not been fitted yet. Please call fit() before get_labels().")
            self.labels_ = get_labels_from_parents(self.parents_, self.representatives_indexes_, self.verbose)
        elif self.hierarchy_strategy == "labels":
            if self.labels_ is None:
                raise ValueError("The model has not been fitted yet. Please call fit() before get_labels().")
            self.labels_ = self.labels_
        else:
            raise ValueError("hierarchy_strategy must be 'parents' or 'labels'")
        return self.labels_

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        if not self.automatically_get_labels:
            self.get_labels()
        if self.labels_ is None:
            raise ValueError("Something went wrong, please check the code.")
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


class CoHiRF(BaseCoHiRF):

    def __init__(
        self,
        repetitions: int = 10,
        verbose: int | bool = 0,
        representative_method: Literal[
            "closest_overall", "closest_to_centroid", "rbf", "rbf_median", "laplacian", "laplacian_median"
        ] = "closest_overall",
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
        # last model parameters
        last_model: Optional[str | type[BaseEstimator]] = None,
        last_model_kwargs: Optional[dict] = None,
        # consensus parameters
        consensus_strategy: Literal[
            "factorize",
            "top-down",
            "top-down-approx",
            "bottom-up",
            "bottom-up-approx",
        ] = "factorize",
        consensus_threshold: float = 0.8,
        # sampling parameters
        n_features: int | float = 10,  # number of random features that will be sampled
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
            last_model=last_model,
            last_model_kwargs=last_model_kwargs,
            consensus_strategy=consensus_strategy,
            consensus_threshold=consensus_threshold,
        )
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol

    def get_base_model(self, X, child_random_state, base_model, base_model_kwargs):
        random_seed = child_random_state.integers(0, 1e6)
        if base_model == "kmeans":
            n_clusters = self.kmeans_n_clusters
            if n_clusters > X.shape[0]:
                if self.verbose:
                    print(
                        f"Warning: base_model n_clusters ({n_clusters}) is larger than the number of samples"
                        f" ({X.shape[0]}). Setting n_clusters to {X.shape[0]}."
                    )
                n_clusters=X.shape[0]

            return KMeans(
                n_clusters=n_clusters,
                init=self.kmeans_init,  # type: ignore
                n_init=self.kmeans_n_init,  # type: ignore
                max_iter=self.kmeans_max_iter,
                tol=self.kmeans_tol,
                verbose=self.verbose,
                random_state=random_seed,
            )
        else:
            raise ValueError(f"base_model {base_model} is not valid.")

    @staticmethod
    def create_search_space():
        search_space = dict(
            n_features=optuna.distributions.FloatDistribution(0.1, 0.6),
            repetitions=optuna.distributions.IntDistribution(3, 10),
            kmeans_n_clusters=optuna.distributions.IntDistribution(2, 5),
        )
        default_values = dict(
            n_features=0.3,
            repetitions=5,
            kmeans_n_clusters=3,
        )
        return search_space, default_values
