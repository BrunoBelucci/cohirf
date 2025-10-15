from __future__ import annotations

import argparse
import os
from itertools import product
from typing import Optional

from sklearn.datasets import make_classification
import numpy as np

from cohirf.experiment.clustering_experiment import ClusteringExperiment


class ClassificationClusteringExperiment(ClusteringExperiment):
    """
    Experiment class for clustering synthetic classification datasets.
    
    This experiment generates synthetic datasets using scikit-learn's make_classification function
    and evaluates clustering algorithms on them. The classification datasets can be configured with
    various parameters such as number of informative/redundant/random features, class separation,
    and noise levels to create diverse clustering challenges.
    """
    
    def __init__(
            self,
            *args,
            n_samples: int | list[int] = 100,
            n_random: int | list[int] = 0,
            n_informative: int | list[int] = 10,
            n_redundant: int = 0,
            n_repeated: int = 0,
            n_classes: int | list[int] = 2,
            n_clusters_per_class: int = 1,
            weights: Optional[list] = None,
            flip_y: float = 0.0,
            class_sep: float | list[float] = 1.0,
            hypercube: bool = True,
            shift: float = 0.0,
            scale: float = 1.0,
            shuffle: bool = True,
            seed_dataset: int | list[int] = 0,
            n_features_dataset: Optional[int | list[int]] = None,
            pct_random: Optional[float | list[float]] = None,
            add_outlier: bool = False,
            std_random: float = 1.0,
            **kwargs
    ):
        """
        Initialize the ClassificationClusteringExperiment.

        Args:
            *args: Variable length argument list passed to parent class.
            n_samples (int | list[int], optional): Number of samples to generate. 
                If list, creates multiple experiments. Defaults to 100.
            n_random (int | list[int], optional): Number of random features without useful information.
                If list, creates multiple experiments. Defaults to 0.
            n_informative (int | list[int], optional): Number of informative features.
                If list, creates multiple experiments. Defaults to 10.
            n_redundant (int, optional): Number of redundant features (linear combinations of informative).
                Defaults to 0.
            n_repeated (int, optional): Number of features that are duplicates of existing features.
                Defaults to 0.
            n_classes (int | list[int], optional): Number of classes (clusters) for the classification.
                If list, creates multiple experiments. Defaults to 2.
            n_clusters_per_class (int, optional): Number of clusters per class. Defaults to 1.
            weights (Optional[list], optional): Proportions of samples assigned to each class.
                If None, classes are balanced. Defaults to None.
            flip_y (float, optional): Fraction of samples whose class is flipped (noise). Defaults to 0.0.
            class_sep (float | list[float], optional): Factor multiplying the hypercube size for class separation.
                If list, creates multiple experiments. Defaults to 1.0.
            hypercube (bool, optional): If True, clusters are placed on vertices of a hypercube.
                If False, clusters are placed on vertices of a random polytope. Defaults to True.
            shift (float, optional): Shift features by the specified value. Defaults to 0.0.
            scale (float, optional): Multiply features by the specified value. Defaults to 1.0.
            shuffle (bool, optional): Whether to shuffle the samples and features. Defaults to True.
            seed_dataset (int | list[int], optional): Random seed(s) for dataset generation.
                If list, creates multiple experiments with different seeds. Defaults to 0.
            n_features_dataset (Optional[int | list[int]], optional): Total number of features.
                If None, computed from other feature parameters. If list, creates multiple experiments.
                Defaults to None.
            pct_random (Optional[float | list[float]], optional): Percentage of random features.
                Alternative to specifying n_random directly. If list, creates multiple experiments.
                Defaults to None.
            add_outlier (bool, optional): Whether to add outlier samples to the dataset. Defaults to False.
            std_random (float, optional): Standard deviation for random features. Defaults to 1.0.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.n_random = n_random
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_repeated = n_repeated
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.weights = weights
        self.flip_y = flip_y
        self.class_sep = class_sep
        self.hypercube = hypercube
        self.shift = shift
        self.scale = scale
        self.shuffle = shuffle
        self.seed_dataset = seed_dataset
        self.n_features_dataset = n_features_dataset
        self.pct_random = pct_random
        self.add_outlier = add_outlier
        self.std_random = std_random

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        if self.parser is None:
            raise ValueError('Parser must be set before adding arguments')
        self.parser.add_argument('--n_samples', type=int, default=self.n_samples, nargs='*')
        self.parser.add_argument('--n_random', type=int, default=self.n_random, nargs='*')
        self.parser.add_argument('--n_informative', type=int, default=self.n_informative, nargs='*')
        self.parser.add_argument('--n_redundant', type=int, default=self.n_redundant)
        self.parser.add_argument('--n_repeated', type=int, default=self.n_repeated)
        self.parser.add_argument('--n_classes', type=int, default=self.n_classes, nargs='*')
        self.parser.add_argument('--n_clusters_per_class', type=int, default=self.n_clusters_per_class)
        self.parser.add_argument('--weights', type=list, default=self.weights)
        self.parser.add_argument('--flip_y', type=float, default=self.flip_y)
        self.parser.add_argument('--class_sep', type=float, default=self.class_sep, nargs='*')
        self.parser.add_argument('--hypercube', type=bool, default=self.hypercube)
        self.parser.add_argument('--shift', type=float, default=self.shift)
        self.parser.add_argument('--scale', type=float, default=self.scale)
        self.parser.add_argument('--shuffle', type=bool, default=self.shuffle)
        self.parser.add_argument('--seed_dataset', type=int, default=self.seed_dataset, nargs='*')
        self.parser.add_argument("--n_features_dataset", type=int, default=self.n_features_dataset, nargs="*")
        self.parser.add_argument('--pct_random', type=float, default=self.pct_random, nargs='*')
        self.parser.add_argument('--add_outlier', action='store_true')
        self.parser.add_argument('--std_random', type=float, default=self.std_random)

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.n_samples = args.n_samples
        self.n_random = args.n_random
        self.n_informative = args.n_informative
        self.n_redundant = args.n_redundant
        self.n_repeated = args.n_repeated
        self.n_classes = args.n_classes
        self.n_clusters_per_class = args.n_clusters_per_class
        self.weights = args.weights
        self.flip_y = args.flip_y
        self.class_sep = args.class_sep
        self.hypercube = args.hypercube
        self.shift = args.shift
        self.scale = args.scale
        self.shuffle = args.shuffle
        self.seed_dataset = args.seed_dataset
        self.n_features_dataset = args.n_features_dataset
        self.pct_random = args.pct_random
        self.add_outlier = args.add_outlier
        self.std_random = args.std_random
        return args

    def _get_combinations_names(self) -> list[str]:
        combination_names = super()._get_combinations_names()
        combination_names.extend(
            [
                "n_samples",
                "n_random",
                "n_informative",
                "n_redundant",
                "n_repeated",
                "n_classes",
                "n_clusters_per_class",
                "weights",
                "flip_y",
                "class_sep",
                "hypercube",
                "shift",
                "scale",
                "shuffle",
                "seed_dataset",
                "n_features_dataset",
                "pct_random",
                "add_outlier",
                "std_random",
            ]
        )
        return combination_names

    def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        n_samples = combination['n_samples']
        n_random = combination['n_random']
        n_classes = combination['n_classes']
        n_informative = combination['n_informative']
        n_redundant = combination["n_redundant"]
        n_repeated = combination["n_repeated"]
        n_clusters_per_class = combination["n_clusters_per_class"]
        weights = combination["weights"]
        flip_y = combination["flip_y"]
        class_sep = combination['class_sep']
        hypercube = combination["hypercube"]
        shift = combination["shift"]
        scale = combination["scale"]
        shuffle = combination["shuffle"]
        seed_dataset = combination['seed_dataset']
        n_features_dataset = combination["n_features_dataset"]
        pct_random = combination['pct_random']
        add_outlier = combination["add_outlier"]
        std_random = combination["std_random"]

        if n_features_dataset is not None:
            if pct_random is not None:
                n_random = int(n_features_dataset * pct_random)
                n_informative = n_features_dataset - n_random
            else:
                raise ValueError('n_features and pct_random must be both None or both not None')

        dataset_name = (f'classif_{n_samples}_{n_random}_{n_informative}_{n_redundant}_{n_repeated}_{n_classes}_'
                        f'{n_clusters_per_class}_{weights}_{flip_y}_{class_sep}_{hypercube}_{shift}_{scale}_{shuffle}_'
                        f'{seed_dataset}_{add_outlier}_{std_random}')
        dataset_dir = self.work_root_dir / dataset_name
        X_file = dataset_dir / 'X.npy'
        y_file = dataset_dir / 'y.npy'
        if X_file.exists() and y_file.exists():
            X = np.load(X_file)
            y = np.load(y_file)
        else:
            if add_outlier:
                if weights:
                    raise ValueError('Weights must be None if add_outlier is True')
                outlier_weight = 1 / (n_samples + 1)  # one outlier class with only one sample
                others_weights = (n_samples/n_classes) / (n_samples + 1)  # n_samples/n_classes samples per class
                weights = [others_weights] * n_classes + [outlier_weight]
                n_samples += 1
                n_classes += 1
            X, y = make_classification(n_samples=n_samples,
                                       n_features=n_informative + n_redundant + n_repeated + n_random,
                                       n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated,
                                       n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, weights=weights,
                                       flip_y=flip_y, class_sep=class_sep, hypercube=hypercube, shift=shift,
                                       scale=scale, shuffle=shuffle, random_state=seed_dataset)
            # re-scale the random features to have a given standard deviation
            if n_random > 0:
                if std_random != 1.0:
                    random_features = X[:, -n_random:]
                    random_features = (random_features - np.mean(random_features, axis=0)) / np.std(random_features, axis=0) * std_random
                    X[:, -n_random:] = random_features
            # save dataset
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(X_file, X)
            np.save(y_file, y)
        return {'X': X, 'y': y, 'dataset_name': dataset_name}

    def _evaluate_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        results = super()._evaluate_model(combination=combination, unique_params=unique_params,
                                          extra_params=extra_params, **kwargs)
        add_outlier = combination["add_outlier"]
        if add_outlier:
            y_true = kwargs['load_data_return']['y']
            y_pred = kwargs['fit_model_return']['y_pred']
            # get outlier
            clusters_true, clusters_counts_true = np.unique(y_true, return_counts=True)
            outlier_true_value = clusters_true[clusters_counts_true == 1]
            outlier_idx = np.where(y_true == outlier_true_value)[0]
            clusters_pred, clusters_counts_pred = np.unique(y_pred, return_counts=True)
            outlier_pred_value = y_pred[outlier_idx][0]
            cluster_pred_outlier = np.where(clusters_pred == outlier_pred_value)[0][0]
            clusters_count_pred_outlier = clusters_counts_pred[cluster_pred_outlier]
            outlier_is_alone = clusters_count_pred_outlier == 1
            # check if other samples were considered outliers, that is they are alone in a cluster with count 1
            clusters_counts_1 = np.where(clusters_counts_pred == 1)[0]
            clusters_counts_1_but_not_outlier_cluster = np.setdiff1d(clusters_counts_1, cluster_pred_outlier)
            number_of_false_outliers = len(clusters_counts_1_but_not_outlier_cluster)
            has_false_outliers = number_of_false_outliers > 0
            results['outlier_is_alone'] = outlier_is_alone
            results['outlier_cluster_count'] = clusters_count_pred_outlier
            results['has_false_outliers'] = has_false_outliers
            results['number_of_false_outliers'] = number_of_false_outliers
        return results


if __name__ == '__main__':
    experiment = ClassificationClusteringExperiment()
    experiment.run_from_cli()
