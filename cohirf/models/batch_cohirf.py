import numpy as np
import pandas as pd
from typing import Literal, Optional
from cohirf.models.cohirf import BaseCoHiRF, CoHiRF, update_labels, get_labels_from_parents
from joblib import Parallel, delayed
import dask.array as da
import dask.dataframe as dd


class BatchCoHiRF:
    def __init__(
        self,
        cohirf_model: type[BaseCoHiRF] = CoHiRF,
        cohirf_kwargs: Optional[dict] = None,
        hierarchy_strategy: Literal["parents", "labels"] = "parents",
        batch_size: int = 1000,
        max_epochs: int = 10,
        verbose: bool = False,
        n_jobs: int = 1,
        automatically_get_labels: bool = True,
    ):
        self.cohirf_model = cohirf_model
        self.cohirf_kwargs = cohirf_kwargs if cohirf_kwargs is not None else {}
        self.hierarchy_strategy = hierarchy_strategy
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.automatically_get_labels = automatically_get_labels

    def run_one_batch(self, X_representatives, i):
        n_samples = X_representatives.shape[0]
        start = i * self.batch_size
        end = min((i + 1) * self.batch_size, n_samples)
        indexes = np.arange(start, end)
        X_batch = X_representatives[indexes]

        if isinstance(X_batch, da.Array):
            # if X_batch is a dask array, we need to compute it
            X_batch = X_batch.compute()

        # fit the cohirf model on the batch, update hierarchy_strategy if needed (priority to this class parameter)
        kwargs = self.cohirf_kwargs.copy()
        kwargs["hierarchy_strategy"] = self.hierarchy_strategy
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
        n_batches = np.ceil(n_samples / self.batch_size).astype(int)
        if n_batches == 1:
            stop = True
        else:
            stop = False

        parallel = Parallel(n_jobs=self.n_jobs, return_as='list', verbose=self.verbose)
        results = parallel(
            delayed(self.run_one_batch)(X_representatives, i) for i in range(n_batches)
        )
        all_parents, all_labels, all_representatives_indexes, all_n_clusters = zip(*results)

        if self.hierarchy_strategy == "parents":
            all_parents = np.concatenate(all_parents)
            all_labels = None
        elif self.hierarchy_strategy == "labels":
            all_parents = None
            all_clusters_cumulative = np.cumsum([0] + list(all_n_clusters))
            all_labels = np.concatenate([
                labels + offset for labels, offset in zip(all_labels, all_clusters_cumulative)
            ])
        else:
            raise ValueError(f"Unknown hierarchy_strategy: {self.hierarchy_strategy}")

        all_representatives_indexes = np.concatenate(all_representatives_indexes)
        all_n_clusters = sum(all_n_clusters)
        return all_representatives_indexes, all_parents, all_labels, all_n_clusters, stop

    def update_parents(self, old_parents, old_representatives_absolute_indexes, new_absolute_parents):
        old_parents[old_representatives_absolute_indexes] = new_absolute_parents
        return old_parents

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

    def get_X_representatives(self, X_representatives, representatives_local_indexes):
        # by indexing with the local indexes we avoid the need to index from the whole array again
        # specially useful for dask arrays where indexing is expensive
        # and when we have already less samples than the batch size (X_representatives will be converted to a
        # numpy array just once)
        X_representatives = X_representatives[representatives_local_indexes]

        if isinstance(X_representatives, da.Array):
            n_samples = X_representatives.shape[0]
            if n_samples <= self.batch_size:
                # if we have less samples than the batch size we can compute the whole array
                X_representatives = X_representatives.compute()
            else:
                # we still need dask to avoid memory issues, we will try to persist the array for faster access
                X_representatives = X_representatives.persist()

        return X_representatives

    def fit(self, X: pd.DataFrame | np.ndarray | dd.DataFrame | da.Array, y=None, sample_weight=None,
             representatives_indexes=None, parents=None, labels=None):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, dd.DataFrame):
            X = X.to_dask_array(lengths=(self.batch_size, -1))

        if isinstance(X, da.Array):
            X = da.rechunk(X, (self.batch_size, -1))  # rechunk to have batches of size batch_size

        n_samples = X.shape[0]

        if representatives_indexes is None:
            # indexes of the representative samples, start with (n_samples) but will be updated when we have less than
            # n_samples as representatives
            representatives_absolute_indexes = np.arange(n_samples)
            if self.hierarchy_strategy == 'parents':
                parents = representatives_absolute_indexes
            elif self.hierarchy_strategy == 'labels':
                parents = None
                self.labels_ = None
            else:
                raise ValueError(f"Unknown hierarchy_strategy: {self.hierarchy_strategy}")
        else:
            # we consider that we are starting from a previous run
            representatives_absolute_indexes = np.array(representatives_indexes)
            if self.hierarchy_strategy == 'parents':
                if parents is None:
                    raise ValueError(
                        "When providing representatives_indexes, parents must also be provided for hierarchy_strategy "
                        "'parents'."
                    )
                parents = np.array(parents)
            elif self.hierarchy_strategy == 'labels':
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
        stop = False
        n_clusters = 0
        # stop when we have run with n_batches == 1
        while not stop and i < self.max_epochs:
            if self.verbose > 0:
                print(f"Starting epoch {i}")

            X_representatives = self.get_X_representatives(X_representatives, representatives_local_indexes)

            (
                new_representatives_local_indexes,
                new_local_parents,
                new_local_labels,
                new_n_clusters,
                stop,
            ) = self.run_one_epoch(X_representatives)

            if self.hierarchy_strategy == "parents":
                new_absolute_parents = representatives_absolute_indexes[new_local_parents]
                parents = self.update_parents(parents, representatives_absolute_indexes, new_absolute_parents)
            elif self.hierarchy_strategy == "labels":
                self.labels_ = update_labels(
                    self.labels_,
                    representatives_absolute_indexes,
                    n_clusters,
                    new_local_labels,
                    self.verbose,
                )

            # representatives_absolute_indexes = new_representatives_absolute_indexes
            representatives_absolute_indexes = representatives_absolute_indexes[new_representatives_local_indexes]
            representatives_local_indexes = new_representatives_local_indexes
            n_clusters = new_n_clusters

            i += 1

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
        self, X: pd.DataFrame | np.ndarray, y=None, sample_weight=None, representatives_indexes=None, parents=None,
        labels=None
    ):
        self.fit(X, y, sample_weight, representatives_indexes, parents, labels)
        if not self.automatically_get_labels:
            self.get_labels()
        if self.labels_ is None:
            raise ValueError("Something went wrong, please check the code.")
        return self.labels_
