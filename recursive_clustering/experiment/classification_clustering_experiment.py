from __future__ import annotations

import argparse
import os
from itertools import product
from typing import Optional

from sklearn.datasets import make_classification
import numpy as np

from recursive_clustering.experiment.clustering_experiment import ClusteringExperiment


class ClassificationClusteringExperiment(ClusteringExperiment):
    def __init__(
            self,
            *args,
            n_samples: Optional[int | list[int]] = 100,
            n_random: Optional[int | list[int]] = None,
            n_informative: Optional[int | list[int]] = None,
            n_redundant: Optional[int] = 0,
            n_repeated: Optional[int] = 0,
            n_classes: Optional[int | list[int]] = 2,
            n_clusters_per_class: Optional[int] = 1,
            weights: Optional[list] = None,
            flip_y: Optional[float] = 0.0,
            class_sep: Optional[int | list[int]] = 1.0,
            hypercube: Optional[bool] = True,
            shift: Optional[float] = 0.0,
            scale: Optional[float] = 1.0,
            shuffle: Optional[bool] = True,
            seeds_dataset: Optional[int | list[int]] = 0,
            seeds_unified: Optional[int | list[int]] = None,
            n_features: Optional[int | list[int]] = None,
            pct_random: Optional[float | list[float]] = None,
            add_outlier: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if isinstance(n_samples, int):
            n_samples = [n_samples]
        self.n_samples = n_samples
        if isinstance(n_random, int) or n_random is None:
            n_random = [n_random]
        self.n_random = n_random
        if isinstance(n_informative, int) or n_informative is None:
            n_informative = [n_informative]
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_repeated = n_repeated
        if isinstance(n_classes, int) or n_classes is None:
            n_classes = [n_classes]
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.weights = weights
        self.flip_y = flip_y
        if isinstance(class_sep, float):
            class_sep = [class_sep]
        self.class_sep = class_sep
        self.hypercube = hypercube
        self.shift = shift
        self.scale = scale
        self.shuffle = shuffle
        if isinstance(seeds_dataset, int):
            seeds_dataset = [seeds_dataset]
        self.seeds_dataset = seeds_dataset
        if isinstance(seeds_unified, int) or seeds_unified is None:
            seeds_unified = [seeds_unified]
        self.seeds_unified = seeds_unified
        if isinstance(n_features, int) or n_features is None:
            n_features = [n_features]
        self.n_features = n_features
        if isinstance(pct_random, float) or pct_random is None:
            pct_random = [pct_random]
        self.pct_random = pct_random
        self.add_outlier = add_outlier

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--n_samples', type=int, default=self.n_samples, nargs='*')
        self.parser.add_argument('--n_random', type=int, default=self.n_random, nargs='*')
        self.parser.add_argument('--n_informative', type=int, default=self.n_informative)
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
        self.parser.add_argument('--seeds_dataset', type=int, default=self.seeds_dataset, nargs='*')
        self.parser.add_argument('--seeds_unified', type=int, default=self.seeds_unified, nargs='*')
        self.parser.add_argument('--n_features', type=int, default=self.n_features, nargs='*')
        self.parser.add_argument('--pct_random', type=float, default=self.pct_random, nargs='*')
        self.parser.add_argument('--add_outlier', action='store_true')

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
        self.seeds_dataset = args.seeds_dataset
        self.seeds_unified = args.seeds_unified
        self.n_features = args.n_features
        self.pct_random = args.pct_random
        self.add_outlier = args.add_outlier
        return args

    def _get_combinations(self):
        combinations = list(product(self.models_nickname, self.seeds_models, self.seeds_dataset, self.n_samples,
                                    self.n_random, self.n_informative, self.n_features, self.pct_random,
                                    self.class_sep,
                                    self.seeds_unified, self.n_classes))
        combination_names = ['model_nickname', 'seed_model', 'seed_dataset', 'n_samples', 'n_random', 'n_informative',
                             'n_features', 'pct_random', 'class_sep', 'seed_unified', 'n_classes']
        combinations = [list(combination) + [self.models_params[combination[0]]] + [self.fits_params[combination[0]]]
                        for combination in combinations]
        combination_names += ['model_params', 'fit_params']
        unique_params = dict(n_redundant=self.n_redundant, n_repeated=self.n_repeated,
                             n_clusters_per_class=self.n_clusters_per_class, weights=self.weights, flip_y=self.flip_y,
                             hypercube=self.hypercube, shift=self.shift, scale=self.scale,
                             shuffle=self.shuffle, add_outlier=self.add_outlier)
        extra_params = dict(n_jobs=self.n_jobs, return_results=False, timeout_combination=self.timeout_combination,
                            timeout_fit=self.timeout_fit)
        return combinations, combination_names, unique_params, extra_params

    def _before_load_model(self, combination: dict, unique_params: Optional[dict] = None,
                           extra_params: Optional[dict] = None, **kwargs):
        seed_unified = combination['seed_unified']
        seed_model = combination['seed_model']
        if seed_unified is not None:
            combination['seed_model'] = seed_unified
        return dict(seed_model=seed_model)

    def _after_load_model(self, combination: dict, unique_params: Optional[dict] = None,
                          extra_params: Optional[dict] = None, **kwargs):
        seed_model = kwargs['before_load_model_return']['seed_model']
        combination['seed_model'] = seed_model
        return {}

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
        n_samples = combination['n_samples']
        n_random = combination['n_random']
        n_classes = combination['n_classes']
        n_informative = combination['n_informative']
        n_redundant = unique_params['n_redundant']
        n_repeated = unique_params['n_repeated']
        n_clusters_per_class = unique_params['n_clusters_per_class']
        weights = unique_params['weights']
        flip_y = unique_params['flip_y']
        class_sep = combination['class_sep']
        hypercube = unique_params['hypercube']
        shift = unique_params['shift']
        scale = unique_params['scale']
        shuffle = unique_params['shuffle']
        seed_dataset = combination['seed_dataset']
        n_features = combination['n_features']
        pct_random = combination['pct_random']
        seed_unified = combination['seed_unified']
        add_outlier = unique_params['add_outlier']

        if n_features is not None:
            if pct_random is not None:
                n_random = int(n_features * pct_random)
                n_informative = n_features - n_random
            else:
                raise ValueError('n_features and pct_random must be both None or both not None')

        if seed_unified is not None:
            seed_dataset = seed_unified

        dataset_name = (f'classif_{n_samples}_{n_random}_{n_informative}_{n_redundant}_{n_repeated}_{n_classes}_'
                        f'{n_clusters_per_class}_{weights}_{flip_y}_{class_sep}_{hypercube}_{shift}_{scale}_{shuffle}_'
                        f'{seed_dataset}_{add_outlier}')
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
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(X_file, X)
            np.save(y_file, y)
        return {'X': X, 'y': y, 'dataset_name': dataset_name}

    def run_classification_experiment_combination(
            self, model_nickname: str, seed_model: int = 0, model_params: Optional[dict] = None,
            fit_params: Optional[dict] = None, seed_dataset: int = 0,
            n_samples: int = 100, n_random: int = 16, n_informative: int = 2, n_redundant: int = 2,
            n_repeated: int = 0, n_classes: int = 2, n_clusters_per_class: int = 1, weights: Optional[list] = None,
            flip_y: float = 0.0, class_sep: float = 1.0, hypercube: bool = True, shift: float = 0.0,
            scale: float = 1.0, shuffle: bool = True,
            n_features: Optional[int] = None, pct_random: Optional[float] = None, seed_unified: Optional[int] = None,
            add_outlier: bool = False,
            n_jobs: int = 1, return_results: bool = True,
            timeout_combination: Optional[int] = None, timeout_fit: Optional[int] = None,
            log_to_mlflow: bool = False
    ):

        combination = {
            'model_nickname': model_nickname,
            'seed_model': seed_model,
            'seed_dataset': seed_dataset,
            'model_params': model_params,
            'fit_params': fit_params,
            'n_samples': n_samples,
            'n_random': n_random,
            'n_informative': n_informative,
            'class_sep': class_sep,
            'n_features': n_features,
            'pct_random': pct_random,
            'seed_unified': seed_unified,
            'n_classes': n_classes,
        }
        unique_params = {
            'n_redundant': n_redundant,
            'n_repeated': n_repeated,
            'n_clusters_per_class': n_clusters_per_class,
            'weights': weights,
            'flip_y': flip_y,
            'hypercube': hypercube,
            'shift': shift,
            'scale': scale,
            'shuffle': shuffle,
            'add_outlier': add_outlier,
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

    def _evaluate_model(self, combination: dict, unique_params: Optional[dict] = None,
                        extra_params: Optional[dict] = None, **kwargs):
        results = super()._evaluate_model(combination=combination, unique_params=unique_params,
                                          extra_params=extra_params, **kwargs)
        add_outlier = unique_params['add_outlier']
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
    parser = argparse.ArgumentParser()
    experiment = ClassificationClusteringExperiment(parser=parser)
    experiment.run()
