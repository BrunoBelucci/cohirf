from sklearn.cluster import (KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering,
                             DBSCAN, HDBSCAN, OPTICS)
import optuna


def create_search_space_kmeans():
    search_space = dict(n_clusters=optuna.distributions.IntDistribution(2, 30))
    default_values = dict(n_clusters=8)
    return search_space, default_values


def create_search_space_affinity_propagation():
    search_space = dict(damping=optuna.distributions.FloatDistribution(0.5, 1.0))
    default_values = dict(damping=0.5)
    return search_space, default_values


def create_search_space_mean_shift():
    search_space = dict(min_bin_freq=optuna.distributions.IntDistribution(1, 100))
    default_values = dict(min_bin_freq=1)
    return search_space, default_values


def create_search_space_spectral_clustering():
    search_space = dict(n_clusters=optuna.distributions.IntDistribution(2, 30))
    default_values = dict(n_clusters=15)
    return search_space, default_values


def create_search_space_agglomerative_clustering():
    search_space = dict(n_clusters=optuna.distributions.IntDistribution(2, 30))
    default_values = dict(n_clusters=15)
    return search_space, default_values


def create_search_space_dbscan():
    search_space = dict(eps=optuna.distributions.FloatDistribution(1e-1, 1e2, log=True),
                        min_samples=optuna.distributions.IntDistribution(2, 10))
    default_values = dict(eps=0.5, min_samples=5)
    return search_space, default_values


def create_search_space_hdbscan():
    search_space = dict(min_cluster_size=optuna.distributions.IntDistribution(2, 10))
    default_values = dict(min_cluster_size=5)
    return search_space, default_values


def create_search_space_optics():
    search_space = dict(min_samples=optuna.distributions.IntDistribution(2, 10))
    default_values = dict(min_samples=5)
    return search_space, default_values


KMeans.create_search_space = staticmethod(create_search_space_kmeans)
AffinityPropagation.create_search_space = staticmethod(create_search_space_affinity_propagation)
MeanShift.create_search_space = staticmethod(create_search_space_mean_shift)
SpectralClustering.create_search_space = staticmethod(create_search_space_spectral_clustering)
AgglomerativeClustering.create_search_space = staticmethod(create_search_space_agglomerative_clustering)
DBSCAN.create_search_space = staticmethod(create_search_space_dbscan)
HDBSCAN.create_search_space = staticmethod(create_search_space_hdbscan)
OPTICS.create_search_space = staticmethod(create_search_space_optics)
