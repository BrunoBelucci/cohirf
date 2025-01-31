# from https://github.com/georgekatona/Clique/tree/master

import numpy as np
import optuna
import scipy.sparse.csgraph
from sklearn.base import ClusterMixin, BaseEstimator


class Cluster:
    def __init__(self, dense_units, dimensions, data_point_ids):
        self.id = None
        self.dense_units = dense_units
        self.dimensions = dimensions
        self.data_point_ids = data_point_ids

    def __str__(self):
        return "Dense units: " + str(self.dense_units.tolist()) + "\nDimensions: " \
            + str(self.dimensions) + "\nCluster size: " + str(len(self.data_point_ids)) \
            + "\nData points:\n" + str(self.data_point_ids) + "\n"


# Inserts joined item into candidates list only if its dimensionality fits
def insert_if_join_condition(candidates, item, item2, current_dim):
    joined = item.copy()
    joined.update(item2)
    if (len(joined.keys()) == current_dim) & (not candidates.__contains__(joined)):
        candidates.append(joined)


# Prune all candidates, which have a (k-1) dimensional projection not in (k-1) dim dense units
def prune(candidates, prev_dim_dense_units):
    for c in candidates:
        if not subdims_included(c, prev_dim_dense_units):
            candidates.remove(c)


def subdims_included(candidate, prev_dim_dense_units):
    for feature in candidate:
        projection = candidate.copy()
        projection.pop(feature)
        if not prev_dim_dense_units.__contains__(projection):
            return False
    return True


def self_join(prev_dim_dense_units, dim):
    candidates = []
    for i in range(len(prev_dim_dense_units)):
        for j in range(i + 1, len(prev_dim_dense_units)):
            insert_if_join_condition(
                candidates, prev_dim_dense_units[i], prev_dim_dense_units[j], dim)
    return candidates


def is_data_in_projection(tuple, candidate, xsi):
    for feature_index, range_index in candidate.items():
        feature_value = tuple[feature_index]
        if int(feature_value * xsi % xsi) != range_index:
            return False
    return True


def get_dense_units_for_dim(data, prev_dim_dense_units, dim, xsi, tau):
    candidates = self_join(prev_dim_dense_units, dim)
    prune(candidates, prev_dim_dense_units)

    # Count number of elements in candidates
    projection = np.zeros(len(candidates))
    number_of_data_points = np.shape(data)[0]
    for dataIndex in range(number_of_data_points):
        for i in range(len(candidates)):
            if is_data_in_projection(data[dataIndex], candidates[i], xsi):
                projection[i] += 1
    # print("projection: ", projection)

    # Return elements above density threshold
    is_dense = projection > tau * number_of_data_points
    # print("is_dense: ", is_dense)
    return np.array(candidates)[is_dense]


def build_graph_from_dense_units(dense_units):
    graph = np.identity(len(dense_units))
    for i in range(len(dense_units)):
        for j in range(len(dense_units)):
            graph[i, j] = get_edge(dense_units[i], dense_units[j])
    return graph


def get_edge(node1, node2):
    dim = len(node1)
    distance = 0

    if node1.keys() != node2.keys():
        return 0

    for feature in node1.keys():
        distance += abs(node1[feature] - node2[feature])
        if distance > 1:
            return 0

    return 1


def get_cluster_data_point_ids(data, cluster_dense_units, xsi):
    point_ids = set()

    # Loop through all dense unit
    for u in cluster_dense_units:
        tmp_ids = set(range(np.shape(data)[0]))
        # Loop through all dimensions of dense unit
        for feature_index, range_index in u.items():
            tmp_ids = tmp_ids & set(
                np.where(np.floor(data[:, feature_index] * xsi % xsi) == range_index)[0])
        point_ids = point_ids | tmp_ids

    return point_ids


def get_clusters(dense_units, data, xsi):
    graph = build_graph_from_dense_units(dense_units)
    number_of_components, component_list = scipy.sparse.csgraph.connected_components(
        graph, directed=False)

    dense_units = np.array(dense_units)
    clusters = []
    # For every cluster
    for i in range(number_of_components):
        # Get dense units of the cluster
        cluster_dense_units = dense_units[np.where(component_list == i)]
        # print("cluster_dense_units: ", cluster_dense_units.tolist())

        # Get dimensions of the cluster
        dimensions = set()
        for u in cluster_dense_units:
            dimensions.update(u.keys())

        # Get points of the cluster
        cluster_data_point_ids = get_cluster_data_point_ids(
            data, cluster_dense_units, xsi)
        # Add cluster to list
        clusters.append(Cluster(cluster_dense_units,
                                dimensions, cluster_data_point_ids))

    return clusters


def get_one_dim_dense_units(data, tau, xsi):
    number_of_data_points = np.shape(data)[0]
    number_of_features = np.shape(data)[1]
    projection = np.zeros((xsi, number_of_features))
    for f in range(number_of_features):
        for element in data[:, f]:
            projection[int(element * xsi % xsi), f] += 1
    # print("1D projection:\n", projection, "\n")
    is_dense = projection > tau * number_of_data_points
    # print("is_dense:\n", is_dense)
    one_dim_dense_units = []
    for f in range(number_of_features):
        for unit in range(xsi):
            if is_dense[unit, f]:
                dense_unit = dict({f: unit})
                one_dim_dense_units.append(dense_unit)
    return one_dim_dense_units


# Normalize data in all features (1e-5 padding is added because clustering works on [0,1) interval)
def normalize_features(data):
    normalized_data = data
    number_of_features = np.shape(normalized_data)[1]
    for f in range(number_of_features):
        normalized_data[:, f] -= min(normalized_data[:, f]) - 1e-5
        normalized_data[:, f] *= 1 / (max(normalized_data[:, f]) + 1e-5)
    return normalized_data


def run_clique(data, xsi, tau):
    # Finding 1 dimensional dense units
    dense_units = get_one_dim_dense_units(data, tau, xsi)

    # Getting 1 dimensional clusters
    clusters = get_clusters(dense_units, data, xsi)

    # Finding dense units and clusters for dimension > 2
    current_dim = 2
    number_of_features = np.shape(data)[1]
    while (current_dim <= number_of_features) & (len(dense_units) > 0):
        # print("\n", str(current_dim), " dimensional clusters:")
        dense_units = get_dense_units_for_dim(
            data, dense_units, current_dim, xsi, tau)
        for cluster in get_clusters(dense_units, data, xsi):
            clusters.append(cluster)
        current_dim += 1

    return clusters


class Clique(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            n_partitions=100,
            density_threshold=0.5
    ):
        self.n_partitions = n_partitions
        self.density_threshold = density_threshold
        self.clusters_ = None
        self.labels_ = None

    def fit(self, X, y=None, sample_weight=None):
        clusters = run_clique(X, self.n_partitions, self.density_threshold)
        labels = np.zeros(X.shape[0], dtype=int)
        for i, cluster in enumerate(clusters):
            cluster.id = i
            labels[list(cluster.data_point_ids)] = i
        self.clusters_ = clusters
        self.labels_ = labels
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.labels_

    @staticmethod
    def create_search_space():
        search_space = dict(
            n_partitions=optuna.distributions.IntDistribution(5, 200),
            density_threshold=optuna.distributions.FloatDistribution(0.1, 0.8),
        )
        default_values = dict(
            n_partitions=100,
            density_threshold=0.5,
        )
        return search_space, default_values
