from typing import Optional, Literal
from cohirf.models.cohirf import BaseCoHiRF, compute_similarities
import pandas as pd
from ml_experiments.utils import update_recursively
import numpy as np


# In MixCoHiRF we basically have two different strategies, but with similar code:
# 1.
# We can imagine that each agent works independently, only sharing the information about cluster assignments,
# choosing their own representatives (medoids) and constructing their own hierarchy (parents), each agent
# can find different labels for their samples, they can continue independently after no more information can be shared,
# and we can also get a unique label after the information is shared.
# 2.
# The agents work colaboratively to find a common set of representatives (medoids) and labels.
# They cannot continue independently after, because they share the parents and one agent may not have information
# on one of the parents to continue independently.
#
# For the first case, we need to keep track of absolute_indexes and parents/labels for each agent, while for
# the second case we only need to keep track of one set of absolute_indexes and parents/labels which are common to all agents and
# represents the union of all samples. The first case need much more code changing and so it is implemented in a separate class
# called IndMixCoHiRF, which basically tranform the variables of the classical CoHiRF in lists to keep track of every agent

class MixCoHiRF(BaseCoHiRF):

    def __init__(
        self,
        cohirf_model: type[BaseCoHiRF] | list[type[BaseCoHiRF]] = BaseCoHiRF,
        cohirf_kwargs: Optional[dict] | list[dict] = None,
        cohirf_kwargs_shared: Optional[dict] = None,
        priority_to_shared_kwargs: bool = True,
        hierarchy_strategy: Literal["parents", "labels"] = "parents",
        use_medoid_rank: bool = True,
        max_iter: int = 100,
        verbose: bool = False,
        n_samples_representative: Optional[int] = None,
        random_state: Optional[int] = None,
        representative_method: Literal[
            "closest_overall", "closest_to_centroid", "rbf", "rbf_median", "laplacian", "laplacian_median"
        ] = "closest_overall",
        automatically_get_labels: bool = True,
        n_jobs: int = 1,
        save_path: bool = False,
        # consensus parameters
        consensus_strategy: Literal[
            "factorize", "top-down", "top-down-approx", "bottom-up", "bottom-up-approx"
        ] = "factorize",
        consensus_threshold: float = 0.8,
    ):
        self.cohirf_model = cohirf_model
        self.cohirf_kwargs = cohirf_kwargs if cohirf_kwargs is not None else {}
        self.cohirf_kwargs_shared = cohirf_kwargs_shared if cohirf_kwargs_shared is not None else {}
        self.priority_to_shared_kwargs = priority_to_shared_kwargs
        self.hierarchy_strategy = hierarchy_strategy
        self.use_medoid_rank = use_medoid_rank
        self.max_iter = max_iter
        self.verbose = verbose
        self.transform_once_per_iteration = False
        self.n_samples_representative = n_samples_representative
        self._random_state = random_state
        self.representative_method = representative_method
        self.automatically_get_labels = automatically_get_labels
        self.n_jobs = n_jobs
        self.save_path = save_path
        self.consensus_strategy = consensus_strategy
        self.consensus_threshold = consensus_threshold
        self.last_model = None
        self.last_model_kwargs = {}

    def run_one_repetition(self, X_representative, i_group, child_random_state): # pyright: ignore[reportIncompatibleMethodOverride]
        if self.verbose:
            print("Starting i_group", i_group)

        features = self.features_groups[i_group]
        samples = self.samples_groups[i_group]
        # we get the samples that are in samples and also in representatives_absolute_indexes, [1] means that we get
        # the indexes that we can use to index X_representative
        samples_in_representatives_absolute_indexes_idx = np.intersect1d(
            self.representatives_absolute_indexes,
            samples,
            assume_unique=True,
            return_indices=True
        )[1]
        # if ever I asked myself why we dont obtain the same result as VeCoHiRF if we have all samples in each group,
        # the reason is that when using samples_in_representatives_absolute_indexes_idx we are reordering the samples
        # in X_group, as for VeCoHiRF, the order is the order obtained by the representatives_absolute_indexes
        X_group = X_representative[np.ix_(samples_in_representatives_absolute_indexes_idx, features)]

        if isinstance(self.cohirf_model, list):
            cohirf_model = self.cohirf_model[i_group]
        else:
            cohirf_model = self.cohirf_model

        if isinstance(self.cohirf_kwargs, list):
            cohirf_kwargs = self.cohirf_kwargs[i_group]
        else:
            cohirf_kwargs = self.cohirf_kwargs

        if self.cohirf_kwargs_shared:
            if self.priority_to_shared_kwargs:
                # shared kwargs may override specific kwargs
                cohirf_kwargs = update_recursively(cohirf_kwargs, self.cohirf_kwargs_shared)
            else:
                # specific kwargs may override shared kwargs
                cohirf_kwargs = update_recursively(self.cohirf_kwargs_shared, cohirf_kwargs)

        if "random_state" not in cohirf_kwargs:
            cohirf_kwargs["random_state"] = child_random_state

        if "n_jobs" not in cohirf_kwargs:
            # divide n_jobs among children if possible
            n_jobs = self.n_jobs // self.repetitions
            n_jobs = max(1, n_jobs)
            cohirf_kwargs["n_jobs"] = n_jobs

        cohirf_instance = cohirf_model(**cohirf_kwargs)
        labels = cohirf_instance.fit_predict(X_group)
        # we need to complete the labels with -1 for samples that were not avalable in X_group
        n_samples = len(X_representative)
        labels_complete = np.ones(n_samples, dtype=int) * -1
        labels_complete[samples_in_representatives_absolute_indexes_idx] = labels
        return labels_complete

    def choose_new_representatives(
        self,
        X_representatives,
        new_representative_cluster_assignments,
        new_unique_clusters_labels,
    ):
        if not self.use_medoid_rank:
            return super().choose_new_representatives(
                X_representatives,
                new_representative_cluster_assignments,
                new_unique_clusters_labels,
            )
        else:
            new_representatives_local_indexes = []
            for label in new_unique_clusters_labels:
                if self.verbose:
                    print("Choosing new representative sample for cluster", label)
                cluster_mask = new_representative_cluster_assignments == label
                X_cluster = X_representatives[cluster_mask]
                X_cluster_indexes = np.where(cluster_mask)[0]

                if self.n_samples_representative is not None:
                    # compute weights based on availability to agents
                    weights = np.zeros(X_cluster.shape[0])
                    for agent_i in range(len(self.features_groups)):
                        samples = self.samples_groups[agent_i]
                        # we get the samples that are in samples and also in representatives_absolute_indexes (masked), [1] means that we get
                        # the indexes that we can use to index X_representative (masked)
                        samples_in_cluster_idx = np.intersect1d(
                            self.representatives_absolute_indexes[cluster_mask],
                            samples,
                            assume_unique=True,
                            return_indices=True,
                        )[1]
                        weights[samples_in_cluster_idx] += 1

                    weights /= weights.sum()

                    n_samples_representative = min(self.n_samples_representative, X_cluster.shape[0])
                    sampled_indexes = self.random_state.choice(
                        X_cluster.shape[0], size=n_samples_representative, replace=False, p=weights
                    )
                    X_cluster = X_cluster[sampled_indexes]
                    X_cluster_indexes = X_cluster_indexes[sampled_indexes]

                new_representatives_local_indexes_ranks = []
                for agent_i in range(len(self.features_groups)):
                    features = self.features_groups[agent_i]
                    samples = self.samples_groups[agent_i]
                    # we get the samples that are in samples and also in representatives_absolute_indexes
                    # (masked and possible masked again), [1] means that we get
                    # the indexes that we can use to index X_representative (masked, masked)
                    if self.n_samples_representative is not None:
                        # double masked
                        samples_in_cluster_idx = np.intersect1d(
                            self.representatives_absolute_indexes[cluster_mask][sampled_indexes],
                            samples,
                            assume_unique=True,
                            return_indices=True,
                        )[1]
                    else:
                        # single masked
                        samples_in_cluster_idx = np.intersect1d(
                            self.representatives_absolute_indexes[cluster_mask],
                            samples,
                            assume_unique=True,
                            return_indices=True,
                        )[1]
                    X_group = X_cluster[np.ix_(samples_in_cluster_idx, features)]
                    cluster_similarities = compute_similarities(X_group, self.representative_method, self.verbose)
                    cluster_similarities_sum = cluster_similarities.sum(axis=0)
                    rank_of_most_similar_samples = np.argsort(cluster_similarities_sum)[::-1]  # reversed order: most similar first
                    # not all samples in X_cluster may be available for agent_i, so we complete missing ranks with len(X_cluster)
                    ranks = np.full(len(X_cluster), fill_value=len(X_cluster))
                    ranks[samples_in_cluster_idx] = rank_of_most_similar_samples
                    new_representatives_local_indexes_ranks.append(ranks)

                # now we have a list of array of ranks, one per group
                new_representatives_local_indexes_ranks = np.vstack(new_representatives_local_indexes_ranks)  # shape (n_groups, n_samples_in_cluster)
                # we sum the ranks to get a final rank and get the sample with the best overall rank
                overall_ranks = new_representatives_local_indexes_ranks.sum(axis=0)
                most_similar_sample_local_idx = X_cluster_indexes[overall_ranks.argmin()]
                new_representatives_local_indexes.append(most_similar_sample_local_idx)

            new_representatives_local_indexes = np.array(new_representatives_local_indexes)
            return new_representatives_local_indexes

    def fit(  # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        samples_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        self.features_groups = features_groups
        self.samples_groups = samples_groups
        if len(features_groups) != len(samples_groups):
            raise ValueError("features_groups and samples_groups must have the same length.")
        self.repetitions = len(features_groups)
        return super().fit(X, y)

    def fit_predict(self, X, features_groups: list[list[int]], samples_groups: list[list[int]], y=None, sample_weight=None):  # type: ignore
        self.fit(X, features_groups, samples_groups, y, sample_weight)
        if not self.automatically_get_labels:
            self.get_labels()
        if self.labels_ is None:
            raise ValueError("Something went wrong, please check the code.")
        return self.labels_
