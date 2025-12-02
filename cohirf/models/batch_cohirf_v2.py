import numpy as np
import pandas as pd
from typing import Literal, Optional
from cohirf.models.cohirf import BaseCoHiRF
from joblib import Parallel, delayed
import dask.array as da
import dask.dataframe as dd
from sklearn.model_selection import KFold
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class BatchCoHiRF(BaseCoHiRF):

    def __init__(
        self,
        cohirf_model: type[BaseCoHiRF] = BaseCoHiRF,
        cohirf_kwargs: Optional[dict] = None,
        hierarchy_strategy: Literal["parents", "labels"] = "parents",
        n_batches: int = 10,
        batch_sample_strategy: Literal["random", "sequential", "stratified"] = "sequential",
        batch_size: Optional[int] = None,
        max_iter: int = 100,
        verbose: bool = False,
        n_jobs: int = 1,
        automatically_get_labels: bool = True,
        random_state: Optional[int] = None,
        save_path: bool = False,
        # last model parameters
        last_model: Optional[str | type[BaseEstimator] | Pipeline] = None,
        last_model_kwargs: Optional[dict] = None,
    ):
        self.cohirf_model = cohirf_model
        self.cohirf_kwargs = cohirf_kwargs if cohirf_kwargs is not None else {}
        self.hierarchy_strategy = hierarchy_strategy
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.batch_sample_strategy = batch_sample_strategy
        self.max_iter = max_iter
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.automatically_get_labels = automatically_get_labels
        self._random_state = random_state
        self.save_path = save_path
        self.last_model = last_model
        self.last_model_kwargs = last_model_kwargs if last_model_kwargs is not None else {}
        self.transform_once_per_iteration = False

    def choose_new_representatives(
        self,
        X_representatives,
        new_representative_cluster_assignments,
        new_unique_clusters_labels,
    ):
        # we have already stored which are the representatives for each label in self._map_label_to_representative_index
        # we only need to put it in the order of unique_clusters_labels = np.arange(all_n_clusters)
        new_representatives_local_indexes = []
        for label in new_unique_clusters_labels:
            representative_index = self._map_label_to_representative_index[label]
            new_representatives_local_indexes.append(representative_index)

        return np.array(new_representatives_local_indexes)

    def run_one_repetition(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, X_representatives, batch_number, child_random_state
    ):
        indexes = self._batches_indexes[self._batches_numbers[batch_number]]
        X_batch = X_representatives[indexes]

        cohirf_kwargs = deepcopy(self.cohirf_kwargs)

        if "random_state" not in cohirf_kwargs:
            cohirf_kwargs["random_state"] = child_random_state

        if "n_jobs" not in cohirf_kwargs:
            # divide n_jobs among children if possible
            n_jobs = self.n_jobs // len(self._batches_numbers)
            n_jobs = max(1, n_jobs)
            cohirf_kwargs["n_jobs"] = n_jobs

        cohirf_instance = self.cohirf_model(**cohirf_kwargs)
        labels = cohirf_instance.fit_predict(X_batch)

        representatives_indexes = cohirf_instance.representatives_indexes_
        # But the representatives are relative to the batch
        # so we need to update them to be relative to the whole dataset
        representatives_indexes = indexes[representatives_indexes]
        n_clusters = cohirf_instance.n_clusters_

        return labels, representatives_indexes, n_clusters

    def get_representative_cluster_assignments(self, X_representative):
        n_samples = X_representative.shape[0]
        n_batches = n_samples // self.batch_size
        if n_batches <= 1:
            n_batches = 1
            self._batches_indexes = [np.arange(n_samples)]
            self._batches_numbers = np.array([0])
            batch_leave_out_i = None
        else:
            if self.batch_sample_strategy == "random":
                kfold = KFold(n_splits=n_batches, shuffle=True, random_state=self.random_state.integers(0, int(1e6)))
                self._batches_indexes = [test_index for _, test_index in kfold.split(np.arange(n_samples))]
            elif self.batch_sample_strategy == "sequential":
                kfold = KFold(n_splits=n_batches, shuffle=False, random_state=None)
                self._batches_indexes = [test_index for _, test_index in kfold.split(np.arange(n_samples))]
            # elif self.batch_sample_strategy == "stratified":
            #     kfold = StratifiedKFold(n_splits=n_batches, shuffle=True, random_state=self.random_state.integers(0, int(1e6)))
            #     self._batches_indexes = [test_index for _, test_index in kfold.split(np.arange(n_samples), self.y_representatives_)]
            else:
                raise ValueError(f"Unknown batch_sample_strategy: {self.batch_sample_strategy}")

            self._batches_numbers = np.arange(n_batches)
            batch_leave_out_i = self.random_state.integers(0, n_batches)
            self._batches_numbers = np.delete(self._batches_numbers, batch_leave_out_i)
            n_batches = n_batches - 1  # we will leve one batch out

        if self.verbose:
            print("Starting consensus assignment")

        child_random_states = self.random_state.spawn(n_batches)

        results = Parallel(n_jobs=self.n_jobs, return_as="list", verbose=self.verbose)(
            delayed(self.run_one_repetition)(X_representative, n_b, child_random_states[n_b])
            for n_b in range(n_batches)
        )

        all_labels, all_representatives_indexes, all_n_clusters = zip(*results)
        all_labels = list(all_labels)
        all_representatives_indexes = list(all_representatives_indexes)
        all_n_clusters = list(all_n_clusters)

        if batch_leave_out_i is not None:
            # we need to add the batch that we left to each of the results
            left_representatives_indexes = self._batches_indexes[batch_leave_out_i]
            left_n_clusters = len(left_representatives_indexes)
            left_labels = np.arange(left_n_clusters)
            all_representatives_indexes.insert(batch_leave_out_i, left_representatives_indexes)
            all_n_clusters.insert(batch_leave_out_i, left_n_clusters)
            all_labels.insert(batch_leave_out_i, left_labels)

        cluster_offsets = np.cumsum([0] + list(all_n_clusters[:-1]))
        all_labels = [labels + offset for labels, offset in zip(all_labels, cluster_offsets)]

        # Build the mapping more efficiently
        map_label_to_representative_index = {}
        sum_n_clusters = 0
        for rep_indexes, n_clusters in zip(all_representatives_indexes, all_n_clusters):
            unique_labels = np.arange(sum_n_clusters, sum_n_clusters + n_clusters)
            map_label_to_representative_index.update(dict(zip(unique_labels, rep_indexes)))
            sum_n_clusters += n_clusters

        all_labels = np.concatenate(all_labels)

        # fix indices to return in the original order
        unsorted_indexes = np.concatenate(self._batches_indexes)
        sorted_indexes = np.argsort(unsorted_indexes)
        all_labels = all_labels[sorted_indexes]

        self._map_label_to_representative_index = map_label_to_representative_index  # store fore choose_new_representatives
        return all_labels, sum_n_clusters

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | dd.DataFrame | da.Array,
        y=None,
        sample_weight=None,
    ):
        n_samples = X.shape[0]
        if self.batch_size is None:
            # we will use self.n_batches to determine the batch size
            self.batch_size = n_samples // self.n_batches
            if self.batch_size == 0:
                raise ValueError(
                    "The number of samples is less than the number of batches. Please increase the number of samples "
                    "or decrease the number of batches."
                )
        return super().fit(X, y, sample_weight)

    def fit_predict(
        self,
        X: pd.DataFrame | np.ndarray,
        y=None,
        sample_weight=None,
        representatives_indexes=None,
        parents=None,
        labels=None,
    ):
        self.fit(X, y, sample_weight)
        if not self.automatically_get_labels:
            self.get_labels()
        if self.labels_ is None:
            raise ValueError("Something went wrong, please check the code.")
        return self.labels_
