from typing import Optional, Literal
from cohirf.models.cohirf import BaseCoHiRF
import numpy as np
import pandas as pd
from ml_experiments.utils import update_recursively
from joblib import Parallel, delayed
from functools import reduce


class MixCoHiRF(BaseCoHiRF):

    def __init__(
        self,
        cohirf_model: type[BaseCoHiRF] | list[type[BaseCoHiRF]] = BaseCoHiRF,
        cohirf_kwargs: Optional[dict] | list[dict] = None,
        cohirf_kwargs_shared: Optional[dict] = None,
        priority_to_shared_kwargs: bool = True,
        hierarchy_strategy: Literal["parents", "labels"] = "parents",
        medoid_strategy: Literal["rank", "independent", "shared"] = "independent",
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
    ):
        self.cohirf_model = cohirf_model
        self.cohirf_kwargs = cohirf_kwargs if cohirf_kwargs is not None else {}
        self.cohirf_kwargs_shared = cohirf_kwargs_shared if cohirf_kwargs_shared is not None else {}
        self.priority_to_shared_kwargs = priority_to_shared_kwargs
        self.hierarchy_strategy = hierarchy_strategy
        self.medoid_strategy = medoid_strategy
        self.max_iter = max_iter
        self.verbose = verbose
        self.transform_once_per_iteration = False
        self.n_samples_representative = n_samples_representative
        self._random_state = random_state
        self.representative_method = representative_method
        self.automatically_get_labels = automatically_get_labels
        self.n_jobs = n_jobs
        self.save_path = save_path
        self._representatives_absolute_indexes_i = None

    def run_one_repetition(self, X_representative, i_group, child_random_state): # pyright: ignore[reportIncompatibleMethodOverride]
        if len(X_representative) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        if self.verbose:
            print("Starting i_group", i_group)

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
        labels = cohirf_instance.fit_predict(X_representative)
        representatives_indexes = cohirf_instance.representatives_indexes_
        return labels, representatives_indexes

    def get_representative_cluster_assignments(self, X, representatives_absolute_indexes):
        X_representatives = X[representatives_absolute_indexes]

        if self.transform_once_per_iteration:
            # we transform once the data here
            X_representatives = self.sampling_transform_X(X_representatives, self.random_state)

        if self.verbose:
            print("Starting consensus assignment")

        child_random_states = self.random_state.spawn(self.repetitions)

        # find local indexes for each agent (w.r.t. absolute indexes)
        self._local_indexes_agent = []
        for agent_i in range(self.repetitions):
            if self._representatives_absolute_indexes_i is not None and self.medoid_strategy == "independent":
                last_representatives_absolute_indexes_for_agent_i = self._representatives_absolute_indexes_i[agent_i]
                available_samples_agent_i = np.array(set(last_representatives_absolute_indexes_for_agent_i) & set(self.samples_groups[agent_i]))
            else:
                available_samples_agent_i = self.samples_groups[agent_i]
            _, local_indexes_agent_i, _ = np.intersect1d(representatives_absolute_indexes, available_samples_agent_i, assume_unique=True, return_indices=True)
            self._local_indexes_agent.append(local_indexes_agent_i)

        # run the repetitions in parallel using loky, which is finally more stable than threading
        # I don't really understand why, but at least this works (even if may consume more memory)
        results = Parallel(n_jobs=self.n_jobs, return_as="list", verbose=self.verbose)(
            delayed(self.run_one_repetition)(
                X_representatives[
                    self._local_indexes_agent[agent_i],  # available samples
                    self.features_groups[agent_i],  # available features
                ],
                agent_i,
                child_random_states[agent_i],
            )
            for agent_i in range(self.repetitions)
        )
        labels_i, representatives_indexes_i = zip(*results)

        n_samples = len(representatives_absolute_indexes)
        for i in range(len(labels_i)):
            labels = np.ones(n_samples, dtype=int) * -1
            labels[self._local_indexes_agent[i]] = labels_i[i]
            labels_i[i] = labels

        labels_i = np.array(labels_i).T

        # representatives_indexes_i are local indexes (w.r.t. X_agent_i), we need to convert to absolute indexes
        for i in range(len(representatives_indexes_i)):
            absolute_indexes = representatives_absolute_indexes[self._local_indexes_agent[i]][representatives_indexes_i[i]]
            representatives_indexes_i[i] = absolute_indexes

        # factorize labels using numpy (codes are from 0 to n_clusters-1)
        unique, codes = self.get_consensus_labels(labels_i)
        if unique is None:
            raise ValueError("Something went wrong, check your code!")

        n_clusters = len(unique)
        # we keep the list of the last representatives indexes for each agent in memory
        self._representatives_absolute_indexes_i = representatives_indexes_i
        return codes, n_clusters

    def choose_new_representatives(
        self,
        X_representatives,
        new_representative_cluster_assignments,
        new_unique_clusters_labels,
    ):
        if self.medoid_strategy == "shared":
            return super().choose_new_representatives(
                X_representatives,
                new_representative_cluster_assignments,
                new_unique_clusters_labels,
            )
        elif self.medoid_strategy == "rank":
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
                    sampled_indexes = self.random_state.choice(
                        X_cluster.shape[0], size=n_samples_representative, replace=False
                    )
                    X_cluster = X_cluster[sampled_indexes]
                    X_cluster_indexes = X_cluster_indexes[sampled_indexes]

                new_representatives_local_indexes_ranks = []
                for agent_i in range(len(self.features_groups)):
                    X_group = X_cluster[self._local_indexes_agent[agent_i], self.features_groups[agent_i]]
                    cluster_similarities = self.compute_similarities(X_group)
                    cluster_similarities_sum = cluster_similarities.sum(axis=0)
                    rank_of_most_similar_samples = np.argsort(cluster_similarities_sum)[::-1]  # reversed order: most similar first
                    # not all samples in X_cluster may be available for agent_i, so we complete missing ranks with len(X_cluster)
                    ranks = np.full(len(X_cluster), fill_value=len(X_cluster))
                    ranks[self._local_indexes_agent[agent_i]] = rank_of_most_similar_samples
                    new_representatives_local_indexes_ranks.append(ranks)

                # now we have a list of array of ranks, one per group
                new_representatives_local_indexes_ranks = np.vstack(new_representatives_local_indexes_ranks)  # shape (n_groups, n_samples_in_cluster)
                # we sum the ranks to get a final rank and get the sample with the best overall rank
                overall_ranks = new_representatives_local_indexes_ranks.sum(axis=0)
                most_similar_sample_local_idx = X_cluster_indexes[overall_ranks.argmin()]
                new_representatives_local_indexes.append(most_similar_sample_local_idx)

            new_representatives_local_indexes = np.array(new_representatives_local_indexes)
            return new_representatives_local_indexes
        elif self.medoid_strategy == "independent":
            # simply union of each agent's medoids
            new_representatives_local_indexes = reduce(np.union1d, *self._local_indexes_agent)
            return new_representatives_local_indexes
        else:
            raise ValueError(f"Unknown medoid_strategy: {self.medoid_strategy}")

    def fit( # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        samples_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        self.features_groups = features_groups
        self.samples_groups = samples_groups
        if len(self.features_groups) != len(self.samples_groups):
            raise ValueError("features_groups and samples_groups must have the same length.")
        self.repetitions = len(features_groups)
        return super().fit(X, y)

    def fit(  # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        samples_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        self.fit(X, features_groups, samples_groups, y, sample_weight)
        if not self.automatically_get_labels:
            self.get_labels()
        if self.labels_ is None:
            raise ValueError("Something went wrong, please check the code.")
        return self.labels_
