from recursive_clustering.models.estimator import RecursiveClustering
from recursive_clustering.models.clique import Clique
from recursive_clustering.models.irfllrr import IRFLLRR
from recursive_clustering.models.kmeansproj import KMeansProj
from recursive_clustering.models.proclus import Proclus
from recursive_clustering.models.scsrgf import SpectralSubspaceRandomization
from recursive_clustering.models.sklearn import (KMeans, OPTICS, DBSCAN, AgglomerativeClustering, SpectralClustering,
                                                 MeanShift, AffinityPropagation, HDBSCAN)
from recursive_clustering.models.WBMS import WBMS


models_dict = {
    RecursiveClustering.__name__: (RecursiveClustering, dict()),
    Clique.__name__: (Clique, dict()),
    IRFLLRR.__name__: (IRFLLRR, dict()),
    KMeansProj.__name__: (KMeansProj, dict()),
    Proclus.__name__: (Proclus, dict()),
    SpectralSubspaceRandomization.__name__: (SpectralSubspaceRandomization, dict()),
    KMeans.__name__: (KMeans, dict()),
    OPTICS.__name__: (OPTICS, dict()),
    DBSCAN.__name__: (DBSCAN, dict()),
    SpectralClustering.__name__: (SpectralClustering, dict()),
    MeanShift.__name__: (MeanShift, dict()),
    AffinityPropagation.__name__: (AffinityPropagation, dict()),
    HDBSCAN.__name__: (HDBSCAN, dict()),
    'WardAgglomerativeClustering': (AgglomerativeClustering, dict()),
    'CompleteAgglomerativeClustering': (AgglomerativeClustering, dict(linkage='complete')),
    'AverageAgglomerativeClustering': (AgglomerativeClustering, dict(linkage='average')),
    'SingleAgglomerativeClustering': (AgglomerativeClustering, dict(linkage='single')),
    WBMS.__name__: (WBMS, dict()),
}
