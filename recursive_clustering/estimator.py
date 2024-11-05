import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics.pairwise import cosine_distances


class RecursiveClustering(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            components_size=10,
            repetitions=10,
            kmeans_n_clusters=3,
            kmeans_init='k-means++',
            kmeans_n_init=1,
            kmeans_max_iter=300,
            kmeans_tol=1e-4,
            kmeans_verbose=0,
            random_state=None,
            kmeans_algorithm='lloyd',
    ):
        self.components_size = components_size
        self.repetitions = repetitions
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol
        self.kmeans_verbose = kmeans_verbose
        self.random_state = random_state
        self.kmeans_algorithm = kmeans_algorithm
        self.n_clusters_ = None
        self.labels_ = None
        self.cluster_representatives_ = None
        self.cluster_representatives_labels_ = None

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]
        n_components = X.shape[1]
        random_state = check_random_state(self.random_state)

        # we will work with pandas DataFrames for the moment and maybe change to numpy arrays later
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df_label_sequence = X.iloc[:, []]

        # find distance between each pair of samples once
        distances = cosine_distances(X)

        X_j = X.copy()
        global_clusters_indexes_i = []
        i = 0
        # initialize with different length of codes and uniques to enter the while loop
        codes = [0, 1]
        uniques = [0]
        # iterate until every sequence of labels is unique
        while len(codes) != len(uniques):

            labels_i = []
            for repetition in range(self.repetitions):
                # random sample of components
                components = sample_without_replacement(n_components, self.components_size, random_state=random_state)
                X_p = X_j.iloc[:, components]
                k_means_estimator = KMeans(n_clusters=self.kmeans_n_clusters, init=self.kmeans_init,
                                           n_init=self.kmeans_n_init,
                                           max_iter=self.kmeans_max_iter, tol=self.kmeans_tol,
                                           verbose=self.kmeans_verbose,
                                           random_state=random_state, algorithm=self.kmeans_algorithm)
                labels_r = k_means_estimator.fit_predict(X_p)
                labels_i.append(labels_r)

            # transform sequence of labels to pandas DataFrame
            labels_i = pd.DataFrame(labels_i).T
            labels_i['labels'] = labels_i.apply(lambda x: tuple(x), axis=1)
            labels_i = labels_i['labels']
            # transform tuples of labels to unique integers for each unique tuple
            codes, uniques = labels_i.factorize()
            labels_i = pd.DataFrame({'labels': codes})
            labels_i.index = X_j.index
            # add to the sequence of labels
            if i == 0:
                # every sample is present in the first iteration
                df_label_sequence_i = labels_i
                df_label_sequence = df_label_sequence.join(df_label_sequence_i)
            else:
                # only some samples are present in the following iterations
                # so we need to add the same label as the representative sample to the rest of the samples
                df_label_sequence_i = []
                for j, cluster in enumerate(global_clusters_indexes_i):
                    df_cat = pd.DataFrame({'labels': codes[j]}, index=cluster)
                    df_label_sequence_i.append(df_cat)
                df_label_sequence_i = pd.concat(df_label_sequence_i)
                df_label_sequence = df_label_sequence.join(df_label_sequence_i, how='left', rsuffix=f'_{i}')

            # find the one sample of each cluster that is the closest to every other sample in the cluster

            # X_j_indexes_i[i] will contain the index of the sample that is the closest to every other sample
            # in the i-th cluster, in other words, the representative sample of the i-th cluster
            X_j_indexes_i = []
            # global_clusters_indexes_i[i] will contain the indexes of ALL the samples in the i-th cluster
            global_clusters_indexes_i = []
            for code in np.unique(codes):
                global_cluster = df_label_sequence_i[df_label_sequence_i['labels'] == code].index
                global_clusters_indexes_i.append(global_cluster)
                # local cluster contains only the samples from the current iteration i
                local_cluster = labels_i[labels_i['labels'] == code].index
                local_cluster_distances = distances[local_cluster][:, local_cluster]
                local_cluster_distances_sum = local_cluster_distances.sum(axis=0)
                closest_sample = local_cluster[np.argmin(local_cluster_distances_sum)]
                X_j_indexes_i.append(closest_sample)

            X_j = X.iloc[X_j_indexes_i, :]
            i += 1

        self.n_clusters_ = len(uniques)
        self.labels_ = df_label_sequence.iloc[:, -1].values
        self.cluster_representatives_ = X_j.to_numpy()
        self.cluster_representatives_labels_ = np.unique(codes)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X).labels_

    def predict(self, X):
        # find from each cluster representative each sample is closest to
        distances = cosine_distances(X, self.cluster_representatives_)
        labels = np.argmin(distances, axis=1)
        labels = self.cluster_representatives_labels_[labels]
        return labels
