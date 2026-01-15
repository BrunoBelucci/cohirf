from typing import Optional, Literal
from cohirf.models.cohirf import BaseCoHiRF, get_consensus_labels, choose_new_representatives, compute_similarities, update_parents, get_labels_from_parents
import numpy as np
import pandas as pd
from ml_experiments.utils import update_recursively
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClusterMixin


class IndMixCoHiRF(ClusterMixin, BaseEstimator):

    def __init__(
        self,
        cohirf_model: type[BaseCoHiRF] | list[type[BaseCoHiRF]] = BaseCoHiRF,
        cohirf_kwargs: Optional[dict] | list[dict] = None,
        cohirf_kwargs_shared: Optional[dict] = None,
        priority_to_shared_kwargs: bool = True,
        hierarchy_strategy: Literal["parents", "labels"] = "parents",
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
        self._representatives_absolute_indexes_i = None

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
            n_jobs = self.n_jobs // self.n_agents_
            n_jobs = max(1, n_jobs)
            cohirf_kwargs["n_jobs"] = n_jobs

        cohirf_instance = cohirf_model(**cohirf_kwargs)
        labels = cohirf_instance.fit_predict(X_representative)
        return labels

    def get_representative_cluster_assignments(
        self,
        X: np.ndarray,
        agents_representatives_absolute_indexes: list[np.ndarray],
        samples_groups: list[np.ndarray],
        features_groups: list[np.ndarray],
        union_representatives_absolute_indexes: np.ndarray,
    ):

        if self.verbose:
            print("Starting consensus assignment")

        child_random_states = self.random_state.spawn(len(samples_groups))

        # run the repetitions in parallel using loky, which is finally more stable than threading
        # I don't really understand why, but at least this works (even if may consume more memory)
        labels_i = Parallel(n_jobs=self.n_jobs, return_as="list", verbose=self.verbose)(
            delayed(self.run_one_repetition)(
                X[
                    np.ix_(
                        samples_groups[agent_i][agents_representatives_absolute_indexes[agent_i]],  # available samples
                        features_groups[agent_i],  # available features
                    )
                ],
                agent_i,
                child_random_states[agent_i],
            )
            for agent_i in range(len(samples_groups))
        )

        n_samples = len(union_representatives_absolute_indexes)
        for agent_i in range(len(labels_i)):
            labels = np.ones(n_samples, dtype=int) * -1
            agent_i_representatives_absolute_indexes = samples_groups[agent_i][
                agents_representatives_absolute_indexes[agent_i]
            ]
            # put labels of agent_i where it has representatives
            agent_i_representatives_local_indexes = np.intersect1d(
                union_representatives_absolute_indexes,
                agent_i_representatives_absolute_indexes,
                assume_unique=True,
                return_indices=True,
            )[1]
            labels[agent_i_representatives_local_indexes] = labels_i[agent_i]
            labels_i[agent_i] = labels

        labels_i = np.array(labels_i).T
        unique, codes = get_consensus_labels(labels_i, self.consensus_strategy, self.consensus_threshold, self.n_jobs, self.verbose)
        n_clusters = len(unique)
        return codes, n_clusters

    def choose_new_representatives(
        self,
        X: np.ndarray,
        agents_representatives_absolute_indexes: list[np.ndarray],
        common_representatives_cluster_assignments: np.ndarray,
        samples_groups: list[np.ndarray],
        features_groups: list[np.ndarray],
        union_representatives_absolute_indexes: np.ndarray,
    ) -> list[np.ndarray]:
        agents_new_representatives_local_indexes = []
        for agent_i in range(len(samples_groups)):
            agent_i_representatives_local_indexes = np.intersect1d(
                union_representatives_absolute_indexes,
                samples_groups[agent_i][agents_representatives_absolute_indexes[agent_i]],
                assume_unique=True,
                return_indices=True,
            )[1]
            agent_i_new_representatives_cluster_assignments = common_representatives_cluster_assignments[
                agent_i_representatives_local_indexes
            ]
            agent_i_new_representatives_local_indexes = choose_new_representatives(
                X[
                    np.ix_(
                        samples_groups[agent_i][agents_representatives_absolute_indexes[agent_i]],  # available samples
                        features_groups[agent_i],  # available features
                    )
                ],
                agent_i_new_representatives_cluster_assignments,
                np.unique(agent_i_new_representatives_cluster_assignments),
                self.representative_method,
                self.verbose,
                self.random_state,
                self.n_samples_representative,
            )
            agents_new_representatives_local_indexes.append(agent_i_new_representatives_local_indexes)
        return agents_new_representatives_local_indexes

    def update_parents(
        self,
        agents_parents: list[np.ndarray],
        agents_representatives_absolute_indexes: list[np.ndarray],
        common_representatives_cluster_assignments: np.ndarray,
        agents_new_representatives_local_indexes: list[np.ndarray],
        samples_groups: list[np.ndarray],
        union_representatives_absolute_indexes: np.ndarray,
    ):
        for agent_i in range(len(agents_parents)):
            agent_i_representatives_local_indexes = np.intersect1d(
                union_representatives_absolute_indexes,
                samples_groups[agent_i][agents_representatives_absolute_indexes[agent_i]],
                assume_unique=True,
                return_indices=True,
            )[1]
            agent_i_new_representatives_cluster_assignments = common_representatives_cluster_assignments[
                agent_i_representatives_local_indexes
            ]
            agent_i_new_representatives_absolute_indexes = agents_representatives_absolute_indexes[agent_i][
                agents_new_representatives_local_indexes[agent_i]
            ]
            agents_parents[agent_i] = update_parents(
                agents_parents[agent_i],
                agents_representatives_absolute_indexes[agent_i],
                np.unique(agent_i_new_representatives_cluster_assignments),
                agent_i_new_representatives_cluster_assignments,
                agent_i_new_representatives_absolute_indexes,
                self.verbose,
            )
        return agents_parents

    def get_labels(self):
        if self.hierarchy_strategy == "parents":
            if self.agents_parents_ is None or self.agents_representatives_absolute_indexes_ is None:
                raise ValueError("The model has not been fitted yet. Please call fit() before get_labels().")
            labels = []
            for agent_i in range(self.n_agents_):
                agent_i_labels = get_labels_from_parents(
                    self.agents_parents_[agent_i],
                    self.agents_representatives_absolute_indexes_[agent_i],
                    self.verbose,
                )
                labels.append(agent_i_labels)
            self.labels_ = labels
        # elif self.hierarchy_strategy == "labels":
        #     if self.labels_ is None:
        #         raise ValueError("The model has not been fitted yet. Please call fit() before get_labels().")
        #     self.labels_ = self.labels_
        else:
            raise ValueError("hierarchy_strategy must be 'parents'")
        return self.labels_

    def fit(  # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        samples_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        if self.verbose:
            print("Starting fit")

        # we will work with numpy (arrays) for speed
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        self.n_agents_ = len(samples_groups)

        agents_representatives_absolute_indexes = [np.arange(len(samples)) for samples in samples_groups]
        # representatives_local_indexes = representatives_absolute_indexes
        # each sample starts as its own cluster (and its own parent)
        # agents_representatives_cluster_assignments = agents_representatives_absolute_indexes.copy()
        # n_clusters = [0 for _ in agents_representatives_absolute_indexes]  # actually len(representatives_cluster_assignments) but 0 in the beggining to enter the loop
        common_representatives_cluster_assignments = [1]  # to enter the loop
        common_n_clusters = 0
        self.labels_ = None
        if self.hierarchy_strategy == "parents":
            agents_parents = agents_representatives_absolute_indexes.copy()
        elif self.hierarchy_strategy == "labels":
            agents_parents = None
        else:
            raise ValueError("hierarchy_strategy must be 'parents' or 'labels'")

        i = 0
        # self.n_clusters_iter_ = []
        # self.labels_iter_ = []
        # self.representatives_iter_ = []
        # iterate until every sequence of labels is unique
        while len(common_representatives_cluster_assignments) != common_n_clusters and i < self.max_iter:
            if self.verbose:
                print("Iteration", i)

            # get the union of all representatives absolute indexes
            union_representatives_absolute_indexes = np.array([], dtype=int)
            for agent_i in range(len(agents_representatives_absolute_indexes)):
                agent_i_representatives_absolute_indexes = samples_groups[agent_i][
                    agents_representatives_absolute_indexes[agent_i]
                ]
                union_representatives_absolute_indexes = np.union1d(
                    union_representatives_absolute_indexes, agent_i_representatives_absolute_indexes
                )

            # consensus assignment (it is here that we repeatedly apply our base model)
            common_representatives_cluster_assignments, common_new_n_clusters = self.get_representative_cluster_assignments(
                X, agents_representatives_absolute_indexes, samples_groups, features_groups, union_representatives_absolute_indexes
            )

            # using representative_method
            agents_new_representatives_local_indexes = self.choose_new_representatives(
                X,
                agents_representatives_absolute_indexes,
                common_representatives_cluster_assignments,
                samples_groups,
                features_groups,
                union_representatives_absolute_indexes,
            )

            if self.hierarchy_strategy == "parents":
                agents_parents = self.update_parents(
                    agents_parents,
                    agents_representatives_absolute_indexes,  # old_representatives_absolute_indexes
                    common_representatives_cluster_assignments,
                    agents_new_representatives_local_indexes,
                    samples_groups,
                    union_representatives_absolute_indexes,
                )
            # elif self.hierarchy_strategy == "labels":
            #     self.labels_ = update_labels(
            #         self.labels_,
            #         representatives_absolute_indexes,
            #         n_clusters,
            #         representatives_cluster_assignments,
            #         self.verbose,
            #     )
            else:
                raise ValueError("hierarchy_strategy must be 'parents'")

            for agent_i in range(len(agents_representatives_absolute_indexes)):
                agents_representatives_absolute_indexes[agent_i] = agents_representatives_absolute_indexes[agent_i][
                    agents_new_representatives_local_indexes[agent_i]
                ]
            common_n_clusters = common_new_n_clusters
            # if self.save_path:
            #     self.representatives_iter_.append(representatives_absolute_indexes)

            i += 1

        self.agents_parents_ = agents_parents
        self.agents_representatives_absolute_indexes_ = agents_representatives_absolute_indexes
        if self.automatically_get_labels:
            self.labels_ = self.get_labels()
        self.n_iter_ = i
        return self
