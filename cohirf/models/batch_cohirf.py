import numpy as np
import pandas as pd
from typing import Literal, Optional
from sklearn.base import BaseEstimator, ClusterMixin
from cohirf.models.cohirf import BaseCoHiRF, CoHiRF, update_labels, get_labels_from_parents
from joblib import Parallel, delayed
import dask.array as da
import dask.dataframe as dd
from sklearn.model_selection import KFold, StratifiedKFold


class BatchCoHiRF(ClusterMixin, BaseEstimator):

    def __init__(
        self,
        cohirf_model: type[BaseCoHiRF] = BaseCoHiRF,
        cohirf_kwargs: Optional[dict] = None,
        hierarchy_strategy: Literal["parents", "labels"] = "parents",
        n_batches: int = 10,
        batch_sample_strategy: Literal["random", "sequential", "stratified"] = "sequential",
        batch_size: Optional[int] = None,
        max_epochs: int = 100,
        verbose: bool = False,
        n_jobs: int = 1,
        automatically_get_labels: bool = True,
        random_state: Optional[int] = None,
        save_path: bool = False,
        stop_at_last_epoch: bool = True,
    ):
        self.cohirf_model = cohirf_model
        self.cohirf_kwargs = cohirf_kwargs if cohirf_kwargs is not None else {}
        self.hierarchy_strategy = hierarchy_strategy
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.batch_sample_strategy = batch_sample_strategy
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.automatically_get_labels = automatically_get_labels
        self._random_state = random_state
        self.save_path = save_path
        self.stop_at_last_epoch = stop_at_last_epoch

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

    def run_one_batch(self, X_representatives, i, child_random_state):
        indexes = self._batches_indexes[i]
        X_batch = X_representatives[indexes]

        if isinstance(X_batch, da.Array):
            # if X_batch is a dask array, we need to compute it
            X_batch = X_batch.compute()

        # fit the cohirf model on the batch, update hierarchy_strategy if needed (priority to this class parameter)
        kwargs = self.cohirf_kwargs.copy()
        kwargs["hierarchy_strategy"] = self.hierarchy_strategy
        # update random_state if needed
        if "random_state" not in kwargs:
            kwargs["random_state"] = child_random_state

        # try to distribute n_jobs if possible
        if "n_jobs" not in kwargs:
            n_jobs = self.n_jobs // len(self._batches_i)
            n_jobs = max(1, n_jobs)
            kwargs["n_jobs"] = n_jobs

        cohirf_model = self.cohirf_model(**kwargs)
        cohirf_model.fit(X_batch)

        if self.hierarchy_strategy == "parents":
            parents = cohirf_model.parents_
            # but the parents are relative to the batch
            # so we need to update them to be relative to the whole dataset
            parents = indexes[parents]
            labels = None
        elif self.hierarchy_strategy == "labels":
            labels = cohirf_model.labels_
            parents = None
        else:
            raise ValueError(f"Unknown hierarchy_strategy: {self.hierarchy_strategy}")

        representatives_indexes = cohirf_model.representatives_indexes_
        # but the representatives are relative to the batch
        # so we need to update them to be relative to the whole dataset
        representatives_indexes = indexes[representatives_indexes]

        n_clusters = cohirf_model.n_clusters_
        return parents, labels, representatives_indexes, n_clusters

    def run_one_epoch(self, X_representatives):
        n_samples = X_representatives.shape[0]
        n_batches = (
            n_samples // self.batch_size
        )  # this might create batches with slightly different sizes (+/- 1 sample)
        if n_batches <= 1:
            # last epoch (every sample fits in one batch, we run one batch with all the final samples and stop)
            n_batches = 1
            self._batches_indexes = [np.arange(n_samples)]
            self._batches_i = np.array([0])
            last_epoch = True
        else:
            # we will leave one batch out for the last epoch
            if self.batch_sample_strategy == "random":
                kfold = KFold(n_splits=n_batches, shuffle=True, random_state=self.random_state.integers(0, int(1e6)))
                self._batches_indexes = [test_index for _, test_index in kfold.split(np.arange(n_samples))]
            elif self.batch_sample_strategy == "sequential":
                kfold = KFold(n_splits=n_batches, shuffle=False, random_state=None)
                self._batches_indexes = [test_index for _, test_index in kfold.split(np.arange(n_samples))]
            elif self.batch_sample_strategy == "stratified":
                kfold = StratifiedKFold(n_splits=n_batches, shuffle=True, random_state=self.random_state.integers(0, int(1e6)))
                self._batches_indexes = [test_index for _, test_index in kfold.split(np.arange(n_samples), self.y_representatives_)]
            else:
                raise ValueError(f"Unknown batch_sample_strategy: {self.batch_sample_strategy}")

            self._batches_i = np.arange(n_batches)
            leave_out_i = self.random_state.integers(0, n_batches)
            self._batches_i = np.delete(self._batches_i, leave_out_i)
            last_epoch = False

        parallel = Parallel(n_jobs=self.n_jobs, return_as="list", verbose=self.verbose, backend="loky")
        child_random_states = self.random_state.spawn(len(self._batches_i) + 1)
        results = parallel(delayed(self.run_one_batch)(X_representatives, i, child_random_states[i]) for i in self._batches_i)
        all_parents, all_labels, all_representatives_indexes, all_n_clusters = zip(*results)
        all_parents = list(all_parents)
        all_labels = list(all_labels)
        all_representatives_indexes = list(all_representatives_indexes)
        all_n_clusters = list(all_n_clusters)

        if not last_epoch:
            # we need to add the batch that we left to representatives_indexes and n_clusters
            left_representatives_indexes = self._batches_indexes[leave_out_i]
            left_n_clusters = len(left_representatives_indexes)
            left_parents = left_representatives_indexes
            left_labels = np.arange(left_n_clusters)
            all_representatives_indexes.insert(leave_out_i, left_representatives_indexes)
            all_n_clusters.insert(leave_out_i, left_n_clusters)
            all_parents.insert(leave_out_i, left_parents)
            all_labels.insert(leave_out_i, left_labels)

        if self.hierarchy_strategy == "parents":
            all_parents = np.concatenate(all_parents)
            all_labels = None
        elif self.hierarchy_strategy == "labels":
            all_parents = None
            all_clusters_cumulative = np.cumsum([0] + list(all_n_clusters))
            all_labels = np.concatenate(
                [labels + offset for labels, offset in zip(all_labels, all_clusters_cumulative)]
            )
        else:
            raise ValueError(f"Unknown hierarchy_strategy: {self.hierarchy_strategy}")

        all_representatives_indexes = np.concatenate(all_representatives_indexes)
        all_n_clusters = sum(all_n_clusters)

        # fix indices to return in the original order
        unsorted_indexes = np.concatenate(self._batches_indexes)
        sorted_indexes = np.argsort(unsorted_indexes)
        if all_parents is not None:
            all_parents = all_parents[sorted_indexes]
        if all_labels is not None:
            all_labels = all_labels[sorted_indexes]

        return all_representatives_indexes, all_parents, all_labels, all_n_clusters, last_epoch

    def update_parents(self, old_parents, old_representatives_absolute_indexes, new_absolute_parents):
        old_parents[old_representatives_absolute_indexes] = new_absolute_parents
        return old_parents

    def get_X_representatives(self, X_representatives, representatives_local_indexes):
        # by indexing with the local indexes we avoid the need to index from the whole array again
        # specially useful for dask arrays where indexing is expensive
        # and when we have already less samples than the batch size (X_representatives will be converted to a
        # numpy array just once)
        X_representatives = X_representatives[representatives_local_indexes]

        if self.batch_sample_strategy == "stratified":
            self.y_representatives_ = self.y_[representatives_local_indexes]

        if isinstance(X_representatives, da.Array):
            n_samples = X_representatives.shape[0]
            if n_samples <= self.batch_size:
                # if we have less samples than the batch size we can compute the whole array
                X_representatives = X_representatives.compute()
            else:
                # we still need dask to avoid memory issues, we will try to persist the array for faster access
                # X_representatives = X_representatives.persist() (this seems to be memory intensive)
                pass

        return X_representatives

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | dd.DataFrame | da.Array,
        y=None,
        sample_weight=None,
        representatives_indexes=None,
        parents=None,
        labels=None,
    ):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, dd.DataFrame):
            X = X.to_dask_array(lengths=(self.batch_size, -1))

        if self.batch_sample_strategy == "stratified":
            if y is None:
                raise ValueError("y must be provided when using stratified batch sampling.")
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)  :
                self.y_ = y.to_numpy().ravel()
            elif isinstance(y, dd.Series) or isinstance(y, dd.DataFrame):
                self.y_ = y.to_dask_array(lengths=(self.batch_size,)).compute().ravel()

        n_samples = X.shape[0]

        if self.batch_size is None:
            # we will use self.n_batches to determine the batch size
            self.batch_size = n_samples // self.n_batches
            if self.batch_size == 0:
                raise ValueError(
                    "The number of samples is less than the number of batches. Please increase the number of samples "
                    "or decrease the number of batches."
                )

        if isinstance(X, da.Array):
            X = da.rechunk(X, (self.batch_size, -1))  # rechunk to have batches of size batch_size

        if representatives_indexes is None:
            # indexes of the representative samples, start with (n_samples) but will be updated when we have less than
            # n_samples as representatives
            representatives_absolute_indexes = np.arange(n_samples)
            if self.hierarchy_strategy == "parents":
                parents = representatives_absolute_indexes
            elif self.hierarchy_strategy == "labels":
                parents = None
                self.labels_ = None
            else:
                raise ValueError(f"Unknown hierarchy_strategy: {self.hierarchy_strategy}")
        else:
            # we consider that we are starting from a previous run
            representatives_absolute_indexes = np.array(representatives_indexes)
            if self.hierarchy_strategy == "parents":
                if parents is None:
                    raise ValueError(
                        "When providing representatives_indexes, parents must also be provided for hierarchy_strategy "
                        "'parents'."
                    )
                parents = np.array(parents)
            elif self.hierarchy_strategy == "labels":
                if labels is None:
                    raise ValueError(
                        "When providing representatives_indexes, labels must also be provided for hierarchy_strategy "
                        "'labels'."
                    )
                self.labels_ = np.array(labels)
            else:
                raise ValueError(f"Unknown hierarchy_strategy: {self.hierarchy_strategy}")

        representatives_local_indexes = representatives_absolute_indexes
        X_representatives = X
        i = 0
        n_clusters = 0
        len_representatives_cluster_assignments = 1
        self.representatives_iter_ = []
        # stop when we have run with n_batches == 1 or when we have not changed the representatives or when we reach max_epochs
        while i < self.max_epochs and len_representatives_cluster_assignments != n_clusters:
            if self.verbose > 0:
                print(f"Starting epoch {i}")

            X_representatives = self.get_X_representatives(X_representatives, representatives_local_indexes)

            (
                new_representatives_local_indexes,
                new_local_parents,
                new_local_labels,
                new_n_clusters,
                last_epoch,
            ) = self.run_one_epoch(X_representatives)

            if self.hierarchy_strategy == "parents":
                len_representatives_cluster_assignments = len(new_local_parents)
                new_absolute_parents = representatives_absolute_indexes[new_local_parents]
                parents = self.update_parents(parents, representatives_absolute_indexes, new_absolute_parents)
            elif self.hierarchy_strategy == "labels":
                len_representatives_cluster_assignments = len(new_local_labels)
                self.labels_ = update_labels(
                    self.labels_,
                    representatives_absolute_indexes,
                    n_clusters,
                    new_local_labels,
                    self.verbose,
                )

            # representatives_absolute_indexes = new_representatives_absolute_indexes
            representatives_absolute_indexes = representatives_absolute_indexes[new_representatives_local_indexes]
            if self.save_path:
                self.representatives_iter_.append(representatives_absolute_indexes)

            representatives_local_indexes = new_representatives_local_indexes
            n_clusters = new_n_clusters

            i += 1
            if last_epoch and self.stop_at_last_epoch:
                if self.verbose > 0:
                    print("Last epoch reached, stopping.")
                break

        self.n_clusters_ = n_clusters
        self.parents_ = parents
        self.representatives_indexes_ = representatives_absolute_indexes
        self.cluster_representatives_ = self.get_X_representatives(X_representatives, representatives_local_indexes)
        self.n_epoch_ = i
        if self.automatically_get_labels:
            self.labels_ = self.get_labels()
        return self

    def get_labels(self):
        if self.hierarchy_strategy == "parents":
            self.labels_ = get_labels_from_parents(self.parents_, self.representatives_indexes_, self.verbose)
        elif self.hierarchy_strategy == "labels":
            self.labels_ = self.labels_
        else:
            raise ValueError(f"Unknown hierarchy_strategy: {self.hierarchy_strategy}")
        return self.labels_

    def fit_predict(
        self,
        X: pd.DataFrame | np.ndarray,
        y=None,
        sample_weight=None,
        representatives_indexes=None,
        parents=None,
        labels=None,
    ):
        self.fit(X, y, sample_weight, representatives_indexes, parents, labels)
        if not self.automatically_get_labels:
            self.get_labels()
        if self.labels_ is None:
            raise ValueError("Something went wrong, please check the code.")
        return self.labels_
