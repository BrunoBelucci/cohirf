from __future__ import annotations
import argparse
from itertools import product
from typing import Optional
import os
import numpy as np

from recursive_clustering.experiment.open_ml_clustering_experiment import ClusteringExperiment


class GaussianClusteringExperiment(ClusteringExperiment):
    def __init__(
            self,
            *args,
            n_samples: Optional[int | list[int]] = 100,
            n_features: Optional[int | list[int]] = 10,
            n_random_features: Optional[int | list[int]] = None,
            pct_random_features: Optional[float | list[float]] = None,
            n_centers: Optional[int] = 3,
            distances: Optional[float] = 1.0,
            seeds_dataset: Optional[int | list[int]] = 0,
            seeds_unified: Optional[int | list[int]] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if isinstance(n_samples, int):
            n_samples = [n_samples]
        self.n_samples = n_samples
        if isinstance(n_features, int):
            n_features = [n_features]
        self.n_features = n_features
        if isinstance(n_random_features, int) or n_random_features is None:
            n_random_features = [n_random_features]
        self.n_random_features = n_random_features
        if isinstance(pct_random_features, float) or isinstance(pct_random_features, int) or pct_random_features is None:
            pct_random_features = [pct_random_features]
        self.pct_random_features = pct_random_features
        if isinstance(n_centers, int):
            n_centers = [n_centers]
        self.n_centers = n_centers
        if isinstance(distances, float):
            distances = [distances]
        self.distances = distances
        if isinstance(seeds_dataset, int):
            seeds_dataset = [seeds_dataset]
        self.seeds_dataset = seeds_dataset
        if isinstance(seeds_unified, int) or seeds_unified is None:
            seeds_unified = [seeds_unified]
        self.seeds_unified = seeds_unified

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--n_samples', type=int, default=self.n_samples, nargs='*')
        self.parser.add_argument('--n_features', type=int, default=self.n_features, nargs='*')
        self.parser.add_argument('--n_random_features', type=int, default=self.n_random_features, nargs='*')
        self.parser.add_argument('--pct_random_features', type=float, default=self.pct_random_features, nargs='*')
        self.parser.add_argument('--n_centers', type=int, default=self.n_centers, nargs='*')
        self.parser.add_argument('--distances', type=float, default=self.distances, nargs='*')
        self.parser.add_argument('--seeds_dataset', type=int, default=self.seeds_dataset, nargs='*')
        self.parser.add_argument('--seeds_unified', type=int, default=self.seeds_unified, nargs='*')

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.n_samples = args.n_samples
        self.n_features = args.n_features
        self.n_random_features = args.n_random_features
        self.pct_random_features = args.pct_random_features
        self.n_centers = args.n_centers
        self.distances = args.distances
        self.seeds_dataset = args.seeds_dataset
        self.seeds_unified = args.seeds_unified
        return args

    def _get_combinations(self):
        combinations = list(product(self.models_nickname, self.seeds_models, self.seeds_dataset, self.seeds_unified,
                                    self.n_samples, self.n_features, self.n_centers, self.distances,
                                    self.n_random_features, self.pct_random_features))
        combination_names = ['model_nickname', 'seed_model', 'seed_dataset', 'seed_unified', 'n_samples', 'n_features',
                             'n_centers', 'distance', 'n_random_features', 'pct_random_features']
        combinations = [list(combination) + [self.models_params[combination[0]]] + [self.fits_params[combination[0]]]
                        for combination in combinations]
        combination_names += ['model_params', 'fit_params']
        unique_params = dict()
        extra_params = dict(n_jobs=self.n_jobs, return_results=False, timeout_combination=self.timeout_combination,
                            timeout_fit=self.timeout_fit)
        return combinations, combination_names, unique_params, extra_params

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
        n_samples = combination['n_samples']
        n_features = combination['n_features']
        n_centers = combination['n_centers']
        distance = combination['distance']
        n_random_features = combination['n_random_features']
        pct_random_features = combination['pct_random_features']
        seed_dataset = combination['seed_dataset']
        seed_unified = combination['seed_unified']
        if seed_unified is not None:
            seed_dataset = seed_unified
        dataset_name = f'gaussian_{n_samples}_{n_features}_{n_centers}_{distance}_{seed_dataset}_{n_random_features}_{pct_random_features}'

        if pct_random_features is not None:
            if n_random_features is not None:
                raise ValueError('You cannot specify both n_random_features and pct_random_features')
            n_random_features = int(n_features * pct_random_features)
            n_informative = n_features - n_random_features
        elif n_random_features is not None:
            n_informative = n_features - n_random_features
        else:
            n_informative = n_features
            n_random_features = 0

        # check if dataset is already saved and load it if it is
        dataset_dir = self.work_root_dir / dataset_name
        X_file = dataset_dir / 'X.npy'
        y_file = dataset_dir / 'y.npy'
        if os.path.exists(X_file) and os.path.exists(y_file):
            X = np.load(X_file)
            y = np.load(y_file)
        else:
            rng = np.random.default_rng(seed_dataset)
            # Generate equally spaced centers using a regular simplex
            # Start with a random orthonormal basis in P dimensions
            centers = np.random.randn(n_centers, n_informative)
            centers, _ = np.linalg.qr(centers.T)  # Orthonormalize columns
            centers = centers.T
            # Scale the simplex to achieve the desired pairwise distance
            centers *= distance / np.sqrt(2)
            # Covariance matrix (identity matrix for simplicity)
            cov = np.eye(n_informative)
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
            idx = np.random.permutation(n_samples * n_centers)
            X = X[idx]
            y = y[idx]
            # save on work_dir for later use
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(X_file, X)
            np.save(y_file, y)
        return {'X': X, 'y': y, 'dataset_name': dataset_name}

    def run_gaussian_experiment_combination(self, model_nickname: str, seed_model: int = 0, seed_dataset: int = 0,
                                            seed_unified: Optional[int] = None,
                                            model_params: Optional[dict] = None, fit_params: Optional[dict] = None,
                                            n_samples: int = 100, n_features: int = 2, n_centers: int = 3,
                                            distance: float = 1.0, n_random_features: Optional[int] = None,
                                            pct_random_features: Optional[float] = None,
                                            n_jobs: int = 1, return_results: bool = True, log_to_mlflow: bool = False,
                                            timeout_combination: Optional[int] = None,
                                            timeout_fit: Optional[int] = None,
                                            ):
        combination = {
            'model_nickname': model_nickname,
            'seed_model': seed_model,
            'seed_dataset': seed_dataset,
            'seed_unified': seed_unified,
            'model_params': model_params,
            'fit_params': fit_params,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_centers': n_centers,
            'distance': distance,
            'n_random_features': n_random_features,
            'pct_random_features': pct_random_features,
        }
        unique_params = {

        }
        extra_params = {
            'n_jobs': n_jobs,
            'return_results': return_results,
            'timeout_combination': timeout_combination,
            'timeout_fit': timeout_fit,
        }
        if log_to_mlflow:
            return self._run_mlflow_and_train_model(combination=combination, unique_params=unique_params,
                                                    extra_params=extra_params, return_results=return_results)
        else:
            return self._train_model(combination=combination, unique_params=unique_params, extra_params=extra_params,
                                     return_results=return_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = GaussianClusteringExperiment(parser=parser)
    experiment.run()
