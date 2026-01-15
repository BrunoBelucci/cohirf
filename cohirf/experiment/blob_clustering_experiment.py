from __future__ import annotations

import argparse
from itertools import product
from typing import Optional
import os

from sklearn.datasets import make_blobs
import numpy as np

from cohirf.experiment.open_ml_clustering_experiment import ClusteringExperiment


class BlobClusteringExperiment(ClusteringExperiment):
    """
    Experiment class for clustering synthetic blob datasets.
    
    This experiment generates synthetic datasets using scikit-learn's make_blobs function
    and evaluates clustering algorithms on them. The blob datasets consist of isotropic
    Gaussian blobs that can be configured with various parameters such as number of samples,
    features, cluster standard deviation, and random seeds.
    """

    def __init__(
        self,
        *args,
        n_samples: int | list[int] = 100,
        n_features_dataset: int | list[int] = 2,
        centers: int = 3,
        cluster_std: float = 1.0,
        center_box: tuple[float, float] = (-10.0, 10.0),
        shuffle: bool = True,
        seed_dataset: int | list[int] = 0,
        **kwargs,
    ):
        """
        Initialize the BlobClusteringExperiment.

        Args:
            *args: Variable length argument list passed to parent class.
            n_samples (int | list[int], optional): Number of samples to generate for each blob dataset.
                If int, creates a single dataset size. If list, creates multiple experiments with
                different sample sizes. Defaults to 100.
            n_features_dataset (int | list[int], optional): Number of features (dimensions) for each
                blob dataset. If int, creates datasets with single dimensionality. If list, creates
                multiple experiments with different dimensionalities. Defaults to 2.
            centers (int, optional): Number of centers (clusters) to generate in the blob dataset.
                Defaults to 3.
            cluster_std (float, optional): Standard deviation of the clusters. Controls how spread
                out the points are within each cluster. Defaults to 1.0.
            center_box (tuple[float, float], optional): Bounding box for cluster centers, specified
                as (min_value, max_value). Centers will be randomly placed within this range.
                Defaults to (-10.0, 10.0).
            shuffle (bool, optional): Whether to shuffle the samples after generation. Defaults to True.
            seed_dataset (int | list[int], optional): Random seed(s) for dataset generation.
                If int, uses single seed. If list, creates multiple experiments with different seeds
                for reproducibility. Defaults to 0.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.n_features_dataset = n_features_dataset
        self.centers = centers
        self.cluster_std = cluster_std
        self.center_box = center_box
        self.shuffle = shuffle
        self.seed_dataset = seed_dataset

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--n_samples', type=int, default=self.n_samples, nargs='*')
        self.parser.add_argument('--n_features_dataset', type=int, default=self.n_features_dataset, nargs='*')
        self.parser.add_argument('--centers', type=int, default=self.centers)
        self.parser.add_argument('--cluster_std', type=float, default=self.cluster_std)
        self.parser.add_argument('--center_box', type=tuple, default=self.center_box)
        self.parser.add_argument('--shuffle', type=bool, default=self.shuffle)
        self.parser.add_argument('--seed_dataset', type=int, default=self.seed_dataset, nargs='*')

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.n_samples = args.n_samples
        self.n_features_dataset = args.n_features_dataset
        self.centers = args.centers
        self.cluster_std = args.cluster_std
        self.center_box = args.center_box
        self.shuffle = args.shuffle
        self.seed_dataset = args.seed_dataset
        return args

    def _get_combinations_names(self):
        combination_names = super()._get_combinations_names()
        combination_names.extend(['n_samples', 'n_features_dataset', 'seed_dataset', "centers", "cluster_std", "center_box", "shuffle"])
        return combination_names

    def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        n_samples = combination['n_samples']
        n_features = combination['n_features_dataset']
        centers = combination["centers"]
        cluster_std = combination['cluster_std']
        center_box = combination['center_box']
        shuffle = combination['shuffle']
        seed_dataset = combination['seed_dataset']
        dataset_name = f'blob_{n_samples}_{n_features}_{centers}_{cluster_std}_{center_box}_{shuffle}_{seed_dataset}'
        dataset_dir = self.work_root_dir / dataset_name
        X_file = dataset_dir / 'X.npy'
        y_file = dataset_dir / 'y.npy'
        # check if dataset is already saved and load it if it is
        if os.path.exists(X_file) and os.path.exists(y_file):
            X = np.load(X_file)
            y = np.load(y_file)
        else:
            X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, # type: ignore
                              center_box=center_box, shuffle=shuffle, random_state=seed_dataset)
            # save on work_dir for later use
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(X_file, X)
            np.save(y_file, y)
        return {'X': X, 'y': y, 'dataset_name': dataset_name}


if __name__ == '__main__':
    experiment = BlobClusteringExperiment()
    experiment.run_from_cli()
