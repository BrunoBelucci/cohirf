from __future__ import annotations
from typing import Optional
import os
import numpy as np
from cohirf.experiment.open_ml_clustering_experiment import ClusteringExperiment


def make_multivariate_normal(n_samples, n_informative_features, n_random_features, n_centers, distance, std, seed, standardize=False):
    rng = np.random.default_rng(seed)
    # Generate equally spaced centers using a regular simplex
    # Start with a random orthonormal basis in P dimensions
    centers = rng.standard_normal(size=(n_centers, n_informative_features))
    centers, _ = np.linalg.qr(centers.T)  # Orthonormalize columns
    centers = centers.T
    # Scale the simplex to achieve the desired pairwise distance
    centers *= distance / np.sqrt(2)
    # Covariance matrix (same standard deviation for all features) and same for all clusters
    cov = np.eye(n_informative_features) * std ** 2
    # Generate clusters
    X = []
    y = []
    for i, mean in enumerate(centers):
        cluster_samples = rng.multivariate_normal(mean, cov, n_samples, method='cholesky')
        if n_random_features > 0:
            random_features = rng.normal(size=(n_samples, n_random_features))
            cluster_samples = np.hstack([cluster_samples, random_features])
        X.append(cluster_samples)
        y.extend([i] * n_samples)
    X = np.vstack(X)
    y = np.array(y)
    # shuffle data
    indexes = np.arange(X.shape[0])
    rng.shuffle(indexes)
    X = X[indexes]
    y = y[indexes]
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


class GaussianClusteringExperiment(ClusteringExperiment):
    """
    Experiment class for clustering synthetic Gaussian datasets.
    
    This experiment generates synthetic datasets using multivariate normal distributions
    with configurable parameters for cluster separation, noise, and dimensionality.
    The clusters are positioned using a regular simplex structure to ensure controlled
    geometric relationships. The experiment allows fine-grained control over informative
    vs. random features, making it ideal for studying clustering performance under
    different signal-to-noise ratios and dimensionality settings.
    """

    def __init__(
        self,
        *args,
        n_samples: int | list[int] = 100,
        n_features_dataset: int | list[int] = 10,
        n_random_features: Optional[int | list[int] | list[None]] = None,
        n_informative_features: Optional[int | list[int]] = None,
        pct_random_features: Optional[float | list[float]] = None,
        n_centers: int | list[int] = 3,
        distance: float | list[float] = 1.0,
        std: float | list[float] = 1.0,
        seed_dataset: int | list[int] = 0,
        seed_unified: Optional[int | list[int]] = None,
        standardize: bool = False,
        **kwargs,
    ):
        """
        Initialize the GaussianClusteringExperiment.

        Args:
            *args: Variable length argument list passed to parent class.
            n_samples (int | list[int], optional): Number of samples per cluster.
                If list, creates multiple experiments. Defaults to 100.
            n_features (int | list[int], optional): Total number of features in the dataset.
                If list, creates multiple experiments. Defaults to 10.
            n_random_features (Optional[int | list[int] | list[None]], optional): Number of random
                (noise) features without clustering information. If None, computed from other parameters.
                If list, creates multiple experiments. Defaults to None.
            n_informative_features (Optional[int | list[int]], optional): Number of informative
                features that contain clustering signal. If None, computed from other parameters.
                If list, creates multiple experiments. Defaults to None.
            pct_random_features (Optional[float | list[float]], optional): Percentage of features
                that should be random noise (0.0 to 1.0). Alternative to specifying n_random_features
                directly. If list, creates multiple experiments. Defaults to None.
            n_centers (int | list[int], optional): Number of cluster centers to generate.
                If list, creates multiple experiments. Defaults to 3.
            distances (float | list[float], optional): Distance parameter controlling cluster
                separation. Larger values create more separated clusters. If list, creates multiple
                experiments. Defaults to 1.0.
            stds (float | list[float], optional): Standard deviation of the clusters. Larger values create more
                dispersed clusters. If list, creates multiple experiments. Defaults to 1.0.
            seeds_dataset (int | list[int], optional): Random seed(s) for dataset generation.
                If list, creates multiple experiments with different seeds. Defaults to 0.
            seeds_unified (Optional[int | list[int]], optional): Unified random seeds for
                controlling all randomness in the experiment. If specified, overrides other seed
                parameters. If list, creates multiple experiments. Defaults to None.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.n_features_dataset = n_features_dataset
        self.n_random_features = n_random_features
        self.n_informative_features = n_informative_features
        self.pct_random_features = pct_random_features
        self.n_centers = n_centers
        self.distance = distance
        self.seed_dataset = seed_dataset
        self.seed_unified = seed_unified
        self.std = std
        self.standardize = standardize

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        if self.parser is None:
            raise ValueError("Parser must be set before adding arguments")
        self.parser.add_argument('--n_samples', type=int, default=self.n_samples, nargs='*')
        self.parser.add_argument("--n_features_dataset", type=int, default=self.n_features_dataset, nargs="*")
        self.parser.add_argument('--n_random_features', type=int, default=self.n_random_features, nargs='*')
        self.parser.add_argument('--n_informative_features', type=int, default=self.n_informative_features, nargs='*')
        self.parser.add_argument('--pct_random_features', type=float, default=self.pct_random_features, nargs='*')
        self.parser.add_argument('--n_centers', type=int, default=self.n_centers, nargs='*')
        self.parser.add_argument('--distance', type=float, default=self.distance, nargs='*')
        self.parser.add_argument('--seed_dataset', type=int, default=self.seed_dataset, nargs='*')
        self.parser.add_argument('--seed_unified', type=int, default=self.seed_unified, nargs='*')
        self.parser.add_argument('--std', type=float, default=self.std, nargs='*')
        self.parser.add_argument('--standardize', action='store_true', default=self.standardize)

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.n_samples = args.n_samples
        self.n_features_dataset = args.n_features_dataset
        self.n_random_features = args.n_random_features
        self.n_informative_features = args.n_informative_features
        self.pct_random_features = args.pct_random_features
        self.n_centers = args.n_centers
        self.distance = args.distance
        self.seed_dataset = args.seed_dataset
        self.seed_unified = args.seed_unified
        self.std = args.std
        self.standardize = args.standardize
        return args

    def _get_combinations_names(self) -> list[str]:
        combination_names = super()._get_combinations_names()
        combination_names.extend(
            [
                "seed_dataset",
                "seed_unified",
                "n_samples",
                "n_features_dataset",
                "n_centers",
                "distance",
                "n_random_features",
                "pct_random_features",
                "n_informative_features",
                "std",
                "standardize",
            ]
        )
        return combination_names

    def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        n_samples = combination['n_samples']
        n_features = combination["n_features_dataset"]
        n_centers = combination['n_centers']
        distance = combination['distance']
        std = combination['std']
        n_random_features = combination['n_random_features']
        n_informative_features = combination['n_informative_features']
        pct_random_features = combination['pct_random_features']
        seed_dataset = combination['seed_dataset']
        seed_unified = combination['seed_unified']
        standardize = combination['standardize']
        if seed_unified is not None:
            seed_dataset = seed_unified

        if pct_random_features is not None:
            if n_random_features is not None:
                raise ValueError('You cannot specify both n_random_features and pct_random_features')
            if n_informative_features is not None:
                raise ValueError('You cannot specify both n_informative_features and pct_random_features')
            n_random_features = int(n_features * pct_random_features)
            n_informative_features = n_features - n_random_features
        elif n_random_features is not None:
            if n_informative_features is None:
                raise ValueError('You must specify n_informative_features if you specify n_random_features')
        else:
            n_informative_features = n_features
            n_random_features = 0

        dataset_name = f'gaussian_{n_samples}_{n_informative_features}_{n_random_features}_{n_centers}_{distance}_{std}_{seed_dataset}_{standardize}'

        # check if dataset is already saved and load it if it is
        dataset_dir = self.work_root_dir / dataset_name
        X_file = dataset_dir / 'X.npy'
        y_file = dataset_dir / 'y.npy'
        if os.path.exists(X_file) and os.path.exists(y_file):
            X = np.load(X_file)
            y = np.load(y_file)
        else:
            X, y = make_multivariate_normal(
                n_samples=n_samples,
                n_informative_features=n_informative_features,
                n_random_features=n_random_features,
                n_centers=n_centers,
                distance=distance,
                std=std,  # standard deviation for the clusters
                seed=seed_dataset,
                standardize=standardize,
            )
            # save on work_dir for later use
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(X_file, X)
            np.save(y_file, y)
        return {'X': X, 'y': y, 'dataset_name': dataset_name}


if __name__ == '__main__':
    experiment = GaussianClusteringExperiment()
    experiment.run_from_cli()
