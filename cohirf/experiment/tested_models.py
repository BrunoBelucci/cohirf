from cohirf.models.clique import Clique
from cohirf.models.irfllrr import IRFLLRR
from cohirf.models.kmeansproj import KMeansProj
from cohirf.models.proclus import Proclus
from cohirf.models.scsrgf import SpectralSubspaceRandomization
from cohirf.models.sklearn import (KMeans, OPTICS, DBSCAN, AgglomerativeClustering, SpectralClustering,
                                   MeanShift, AffinityPropagation, HDBSCAN)
from cohirf.models.cohirf import CoHiRF

models_dict = {
    CoHiRF.__name__: (CoHiRF, dict()),
    'CoHiRF_rbf': (CoHiRF, dict(representative_method='rbf')),
    'CoHiRF_1000': (CoHiRF, dict(representative_method='closest_overall_1000')),
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
}
