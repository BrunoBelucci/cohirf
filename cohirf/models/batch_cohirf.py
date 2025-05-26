import numpy as np
import pandas as pd
from typing import Optional
from cohirf.models.cohirf import BaseCoHiRF, CoHiRF
from joblib import Parallel, delayed


class BatchCoHiRF:
    def __init__(
        self,
        cohirf_model: type[BaseCoHiRF] = CoHiRF,
        cohirf_kwargs: Optional[dict] = None,
        batch_size: int = 1000,
        max_epochs: int = 10,
        verbose: bool = False,
        n_jobs: int = 1,
    ):
        self.cohirf_model = cohirf_model
        self.cohirf_kwargs = cohirf_kwargs if cohirf_kwargs is not None else {}
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.n_jobs = n_jobs

    def run_one_batch(self, X_representatives, i):
        n_samples = X_representatives.shape[0]
        start = i * self.batch_size
        end = min((i + 1) * self.batch_size, n_samples)
        indexes = np.arange(start, end)
        X_batch = X_representatives[indexes]
        # fit the cohirf model on the batch
        cohirf_model = self.cohirf_model(**self.cohirf_kwargs)
        cohirf_model.fit(X_batch)

        parents = cohirf_model.parents_
        # but the parents are relative to the batch
        # so we need to update them to be relative to the whole dataset
        parents = indexes[parents]
        # all_parents.append(parents)

        representatives_indexes = cohirf_model.representatives_indexes_
        # but the representatives are relative to the batch
        # so we need to update them to be relative to the whole dataset
        representatives_indexes = indexes[representatives_indexes]
        # all_representatives_indexes.append(representatives_indexes)

        n_clusters = cohirf_model.n_clusters_
        # all_n_clusters = all_n_clusters + n_clusters
        return parents, representatives_indexes, n_clusters

    def run_one_epoch(self, X_representatives):
        n_samples = X_representatives.shape[0]
        n_batches = np.ceil(n_samples / self.batch_size).astype(int)
        if n_batches == 1:
            stop = True
        else:
            stop = False
        all_parents = []
        all_representatives_indexes = []
        all_n_clusters = 0

        parallel = Parallel(n_jobs=self.n_jobs, return_as='list')
        results = parallel(
            delayed(self.run_one_batch)(X_representatives, i) for i in range(n_batches)
        )
        all_parents, all_representatives_indexes, all_n_clusters = zip(*results)

        all_parents = np.concatenate(all_parents)
        all_representatives_indexes = np.concatenate(all_representatives_indexes)
        all_n_clusters = sum(all_n_clusters)
        return all_representatives_indexes, all_parents, all_n_clusters, stop

    def update_parents(self, old_parents, old_representatives_absolute_indexes, new_absolute_parents):
        new_parents = old_parents.copy()
        new_parents[old_representatives_absolute_indexes] = new_absolute_parents
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

    def fit(self, X: pd.DataFrame | np.ndarray, y=None, sample_weight=None, representatives_indexes=None, parents=None):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        n_samples = X.shape[0]

        if representatives_indexes is None and parents is None:
            # indexes of the representative samples, start with (n_samples) but will be updated when we have less than
            # n_samples as representatives
            representatives_absolute_indexes = np.arange(n_samples)
            # representatives_local_indexes = representatives_absolute_indexes
            # each sample starts as its own cluster (and its own parent)
            parents = representatives_absolute_indexes
        else:
            # we consider that we are starting from a previous run
            # we make sure that both representatives_indexes and parents are not None
            if representatives_indexes is None or parents is None:
                raise ValueError("If you provide representatives_indexes, you must also provide parents.")
            representatives_absolute_indexes = np.array(representatives_indexes)

        i = 0
        stop = False
        # stop when we have run with n_batches == 1
        while not stop and i < self.max_epochs:
            if self.verbose:
                print(f"Starting epoch {i}")

            X_representatives = X[representatives_absolute_indexes]

            (
                new_representatives_local_indexes,
                new_local_parents,
                n_clusters,
                stop,
            ) = self.run_one_epoch(X_representatives)

            new_absolute_parents = representatives_absolute_indexes[new_local_parents]

            parents = self.update_parents(parents, representatives_absolute_indexes, new_absolute_parents)

            representatives_absolute_indexes = representatives_absolute_indexes[new_representatives_local_indexes]

            i += 1

        self.n_clusters_ = n_clusters
        self.labels_ = self.get_labels_from_parents(parents, representatives_absolute_indexes)
        self.parents_ = parents
        self.representatives_indexes_ = representatives_absolute_indexes
        self.cluster_representatives_ = X[representatives_absolute_indexes]
        self.n_epoch_ = i
        return self

    def fit_predict(
        self, X: pd.DataFrame | np.ndarray, y=None, sample_weight=None, representatives_indexes=None, parents=None
    ):
        return self.fit(X, y, sample_weight, representatives_indexes, parents).labels_
