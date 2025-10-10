from cohirf.models.cohirf import CoHiRF, BaseCoHiRF
from cohirf.models.clique import Clique
from cohirf.models.irfllrr import IRFLLRR
from cohirf.models.kmeansproj import KMeansProj
from cohirf.models.proclus import Proclus
from cohirf.models.scsrgf import SpectralSubspaceRandomization
from cohirf.models.sklearn import (KMeans, OPTICS, DBSCAN, AgglomerativeClustering, SpectralClustering,
                                                 MeanShift, AffinityPropagation, HDBSCAN)
from cohirf.models.batch_cohirf import BatchCoHiRF
from cohirf.models.pseudo_kernel import PseudoKernelClustering
from cohirf.models.WBMS import WBMS
from sklearn.kernel_approximation import Nystroem, RBFSampler
import optuna


models_dict = {
    CoHiRF.__name__: (
        CoHiRF,
        dict(),
        dict(
            n_features=optuna.distributions.FloatDistribution(0.1, 0.6),
            repetitions=optuna.distributions.IntDistribution(2, 10),
            kmeans_n_clusters=optuna.distributions.IntDistribution(2, 5),
        ),
        [
            dict(
                n_features=0.3,
                repetitions=5,
                kmeans_n_clusters=3,
            )
        ],
    ),
    "CoHiRF-KernelRBF": (
        BaseCoHiRF,
        dict(
            base_model=KMeans,
            transform_method=RBFSampler,
            transform_kwargs=dict(n_components=500),
            representative_method="rbf",
        ),
        dict(
            n_features=optuna.distributions.FloatDistribution(0.1, 0.6),
            repetitions=optuna.distributions.IntDistribution(2, 10),
            base_model_kwargs=dict(
                n_clusters=optuna.distributions.IntDistribution(2, 5),
            ),
            transform_kwargs=dict(
                gamma=optuna.distributions.FloatDistribution(0.1, 30),
            ),
        ),
        [
            dict(
                n_features=0.3,
                repetitions=5,
                base_model_kwargs=dict(
                    n_clusters=3,
                ),
                transform_kwargs=dict(
                    gamma=1.0,
                ),
            )
        ],
    ),
    "CoHiRF-DBSCAN": (
        BaseCoHiRF,
        dict(base_model=DBSCAN),
        dict(
            n_features=optuna.distributions.FloatDistribution(0.1, 0.6),
            repetitions=optuna.distributions.IntDistribution(1, 10),
            base_model_kwargs=dict(
                eps=optuna.distributions.FloatDistribution(1e-1, 10),
                min_samples=optuna.distributions.IntDistribution(2, 50),
            ),
        ),
        [
            dict(
                n_features=0.3,
                repetitions=5,
                base_model_kwargs=dict(
                    eps=0.5,
                    min_samples=5,
                ),
            )
        ],
    ),
    "CoHiRF-SC-SRGF": (
        BaseCoHiRF,
        dict(base_model=SpectralSubspaceRandomization, n_features=1.0),
        dict(
            repetitions=optuna.distributions.IntDistribution(2, 10),
            base_model_kwargs=dict(
                n_similarities=optuna.distributions.IntDistribution(10, 30),
                sampling_ratio=optuna.distributions.FloatDistribution(0.2, 0.8),
                sc_n_clusters=optuna.distributions.IntDistribution(2, 5),
            ),
        ),
        [
            dict(
                repetitions=5,
                base_model_kwargs=dict(
                    n_similarities=20,
                    sampling_ratio=0.5,
                    sc_n_clusters=3,
                ),
            )
        ],
    ),
    "BatchCoHiRF": (
        BatchCoHiRF,
        dict(),
        dict(
            cohirf_kwargs=dict(
                n_features=optuna.distributions.FloatDistribution(0.1, 0.6),
                repetitions=optuna.distributions.IntDistribution(2, 10),
                kmeans_n_clusters=optuna.distributions.IntDistribution(2, 5),
            )
        ),
        [
            dict(
                cohirf_kwargs=dict(
                    n_features=0.3,
                    repetitions=5,
                    kmeans_n_clusters=3,
                )
            )
        ],
    ),
    "BatchCoHiRF-1iter": (
        BatchCoHiRF,
        dict(cohirf_kwargs=dict(max_iter=1)),
        dict(
            cohirf_kwargs=dict(
                n_features=optuna.distributions.FloatDistribution(0.1, 0.6),
                repetitions=optuna.distributions.IntDistribution(2, 10),
                kmeans_n_clusters=optuna.distributions.IntDistribution(2, 5),
            )
        ),
        [
            dict(
                cohirf_kwargs=dict(
                    n_features=0.3,
                    repetitions=5,
                    kmeans_n_clusters=3,
                )
            )
        ],
    ),
    "BatchCoHiRF-DBSCAN": (
        BatchCoHiRF,
        dict(
            cohirf_model=BaseCoHiRF,
            cohirf_kwargs=dict(base_model=DBSCAN),
        ),
        dict(
            cohirf_kwargs=dict(
                n_features=optuna.distributions.FloatDistribution(0.1, 0.6),
                repetitions=optuna.distributions.IntDistribution(1, 10),
                base_model_kwargs=dict(
                    eps=optuna.distributions.FloatDistribution(1e-1, 10),
                    min_samples=optuna.distributions.IntDistribution(2, 50),
                ),
            )
        ),
        [
            dict(
                cohirf_kwargs=dict(
                    n_features=0.3,
                    repetitions=5,
                    base_model_kwargs=dict(
                        eps=0.5,
                        min_samples=5,
                    ),
                )
            )
        ],
    ),
    "BatchCoHiRF-DBSCAN-1iter": (
        BatchCoHiRF,
        dict(
            cohirf_model=BaseCoHiRF,
            cohirf_kwargs=dict(base_model=DBSCAN, max_iter=1),
        ),
        dict(
            cohirf_kwargs=dict(
                n_features=optuna.distributions.FloatDistribution(0.1, 0.6),
                repetitions=optuna.distributions.IntDistribution(1, 10),
                base_model_kwargs=dict(
                    eps=optuna.distributions.FloatDistribution(1e-1, 10),
                    min_samples=optuna.distributions.IntDistribution(2, 50),
                ),
            )
        ),
        [
            dict(
                cohirf_kwargs=dict(
                    n_features=0.3,
                    repetitions=5,
                    base_model_kwargs=dict(
                        eps=0.5,
                        min_samples=5,
                    ),
                )
            )
        ],
    ),
    "BatchCoHiRF-KernelRBF-1iter": (
        BatchCoHiRF,
        dict(
            cohirf_model=BaseCoHiRF,
            cohirf_kwargs=dict(
                base_model=KMeans,
                transform_method=RBFSampler,
                transform_kwargs=dict(n_components=500),
                representative_method="rbf",
                max_iter=1,
            ),
        ),
        dict(
            cohirf_kwargs=dict(
                n_features=optuna.distributions.FloatDistribution(0.1, 0.6),
                repetitions=optuna.distributions.IntDistribution(1, 10),
                base_model_kwargs=dict(
                    n_clusters=optuna.distributions.IntDistribution(2, 5),
                ),
                transform_kwargs=dict(
                    gamma=optuna.distributions.FloatDistribution(0.1, 30),
                ),
            )
        ),
        [
            dict(
                cohirf_kwargs=dict(
                    n_features=0.3,
                    repetitions=5,
                    base_model_kwargs=dict(
                        n_clusters=3,
                    ),
                    transform_kwargs=dict(
                        gamma=1.0,
                    ),
                ),
            )
        ],
    ),
    "BatchCoHiRF-SC-SRGF": (
        BatchCoHiRF,
        dict(
            cohirf_model=BaseCoHiRF,
            cohirf_kwargs=dict(base_model=SpectralSubspaceRandomization, max_iter=1, n_features=1.0),
        ),
        dict(
            cohirf_kwargs=dict(
                repetitions=optuna.distributions.IntDistribution(2, 10),
                base_model_kwargs=dict(
                    n_similarities=optuna.distributions.IntDistribution(10, 30),
                    sampling_ratio=optuna.distributions.FloatDistribution(0.2, 0.8),
                    sc_n_clusters=optuna.distributions.IntDistribution(2, 5),
                ),
            )
        ),
        [
            dict(
                cohirf_kwargs=dict(
                    repetitions=5,
                    base_model_kwargs=dict(
                        n_similarities=20,
                        sampling_ratio=0.5,
                        sc_n_clusters=3,
                    ),
                )
            )
        ],
    ),
    "BatchCoHiRF-SC-SRGF-2": (
        BatchCoHiRF,
        dict(
            cohirf_model=BaseCoHiRF,
            cohirf_kwargs=dict(base_model=SpectralSubspaceRandomization, max_iter=1, n_features=1.0),
        ),
        dict(
            cohirf_kwargs=dict(
                repetitions=optuna.distributions.IntDistribution(2, 10),
                base_model_kwargs=dict(
                    n_similarities=optuna.distributions.IntDistribution(10, 30),
                    sampling_ratio=optuna.distributions.FloatDistribution(0.2, 0.8),
                    sc_n_clusters=optuna.distributions.IntDistribution(2, 30),
                ),
            )
        ),
        [
            dict(
                cohirf_kwargs=dict(
                    repetitions=5,
                    base_model_kwargs=dict(
                        n_similarities=20,
                        sampling_ratio=0.5,
                        sc_n_clusters=8,
                    ),
                )
            )
        ],
    ),
    Clique.__name__: (
        Clique,
        dict(),
        dict(
            n_partitions=optuna.distributions.IntDistribution(5, 200),
            density_threshold=optuna.distributions.FloatDistribution(0.1, 0.8),
        ),
        [
            dict(
                n_partitions=100,
                density_threshold=0.5,
            )
        ],
    ),
    IRFLLRR.__name__: (
        IRFLLRR,
        dict(),
        dict(
            p=optuna.distributions.FloatDistribution(0.0, 1.0),
            c=optuna.distributions.CategoricalDistribution([1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]),
            lambda_=optuna.distributions.CategoricalDistribution([1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]),
            alpha=optuna.distributions.IntDistribution(1, 4),
            sc_n_clusters=optuna.distributions.IntDistribution(2, 30),
        ),
        [
            dict(
                p=0.95,
                c=0.10,
                lambda_=1,
                alpha=4,
                sc_n_clusters=8,
            )
        ],
    ),
    KMeansProj.__name__: (
        KMeansProj,
        dict(),
        dict(
            n_clusters=optuna.distributions.IntDistribution(2, 30),
        ),
        [
            dict(
                n_clusters=8,
            )
        ],
    ),
    Proclus.__name__: (
        Proclus,
        dict(),
        dict(
            n_clusters=optuna.distributions.IntDistribution(2, 30),
            avg_dims=optuna.distributions.FloatDistribution(0.1, 1.0),
        ),
        [
            dict(
                n_clusters=8,
                avg_dims=0.5,
            )
        ],
    ),
    SpectralSubspaceRandomization.__name__: (
        SpectralSubspaceRandomization,
        dict(),
        dict(
            n_similarities=optuna.distributions.IntDistribution(10, 30),
            sampling_ratio=optuna.distributions.FloatDistribution(0.2, 0.8),
            sc_n_clusters=optuna.distributions.IntDistribution(2, 30),
        ),
        [
            dict(
                n_similarities=20,
                sampling_ratio=0.5,
                sc_n_clusters=8,
            )
        ],
    ),
    KMeans.__name__: (
        KMeans,
        dict(),
        dict(n_clusters=optuna.distributions.IntDistribution(2, 30)),
        [dict(n_clusters=8)],
    ),
    "KernelRBFKMeans": (
        PseudoKernelClustering,
        dict(
            base_model=KMeans,
            transform_method=RBFSampler,
            transform_kwargs=dict(n_components=500),
        ),
        dict(
            base_model_kwargs=dict(
                n_clusters=optuna.distributions.IntDistribution(2, 30),
            ),
            transform_kwargs=dict(
                gamma=optuna.distributions.FloatDistribution(0.1, 30),
            ),
        ),
        [
            dict(
                base_model_kwargs=dict(
                    n_clusters=8,
                ),
                transform_kwargs=dict(
                    gamma=1.0,
                ),
            )
        ],
    ),
    OPTICS.__name__: (
        OPTICS,
        dict(),
        dict(min_samples=optuna.distributions.IntDistribution(2, 50)),
        [dict(min_samples=5)],
    ),
    DBSCAN.__name__: (
        DBSCAN,
        dict(),
        dict(
            eps=optuna.distributions.FloatDistribution(1e-1, 10),
            min_samples=optuna.distributions.IntDistribution(2, 50),
        ),
        [dict(eps=0.5, min_samples=5)],
    ),
    SpectralClustering.__name__: (
        SpectralClustering,
        dict(),
        dict(n_clusters=optuna.distributions.IntDistribution(2, 30)),
        [dict(n_clusters=15)],
    ),
    MeanShift.__name__: (
        MeanShift,
        dict(),
        dict(min_bin_freq=optuna.distributions.IntDistribution(1, 100)),
        [dict(min_bin_freq=1)],
    ),
    AffinityPropagation.__name__: (
        AffinityPropagation,
        dict(),
        dict(damping=optuna.distributions.FloatDistribution(0.5, 1.0)),
        [dict(damping=0.5)],
    ),
    HDBSCAN.__name__: (
        HDBSCAN,
        dict(),
        dict(min_cluster_size=optuna.distributions.IntDistribution(2, 50)),
        [dict(min_cluster_size=5)],
    ),
    "WardAgglomerativeClustering": (
        AgglomerativeClustering,
        dict(),
        dict(n_clusters=optuna.distributions.IntDistribution(2, 30)),
        [dict(n_clusters=15)],
    ),
    "CompleteAgglomerativeClustering": (
        AgglomerativeClustering,
        dict(linkage="complete"),
        dict(n_clusters=optuna.distributions.IntDistribution(2, 30)),
        [dict(n_clusters=15)],
    ),
    "AverageAgglomerativeClustering": (
        AgglomerativeClustering,
        dict(linkage="average"),
        dict(n_clusters=optuna.distributions.IntDistribution(2, 30)),
        [dict(n_clusters=15)],
    ),
    "SingleAgglomerativeClustering": (
        AgglomerativeClustering,
        dict(linkage="single"),
        dict(n_clusters=optuna.distributions.IntDistribution(2, 30)),
        [dict(n_clusters=15)],
    ),
}
