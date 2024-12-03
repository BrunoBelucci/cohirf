import argparse
import os
from shutil import rmtree
from itertools import product
from typing import Optional

from sklearn.datasets import make_classification
import numpy as np

from recursive_clustering.experiment.clustering_experiment import ClusteringExperiment


class ClassificationClusteringExperiment(ClusteringExperiment):
    def __init__(
            self,
            *args,
            n_samples: Optional[int] = 100,
            n_random: Optional[int] = 16,
            n_informative: Optional[int] = 2,
            n_redundant: Optional[int] = 2,
            n_repeated: Optional[int] = 0,
            n_classes: Optional[int] = 2,
            n_clusters_per_class: Optional[int] = 1,
            weights: Optional[list] = None,
            flip_y: Optional[float] = 0.0,
            class_sep: Optional[float] = 1.0,
            hypercube: Optional[bool] = True,
            shift: Optional[float] = 0.0,
            scale: Optional[float] = 1.0,
            shuffle: Optional[bool] = True,
            seeds_dataset: Optional[int] = 0,
            **kwargs
    ):
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
        self.seeds_dataset = seeds_dataset

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--n_samples', type=int, default=self.n_samples)
        self.parser.add_argument('--n_random', type=int, default=self.n_random)
        self.parser.add_argument('--n_informative', type=int, default=self.n_informative)
        self.parser.add_argument('--n_redundant', type=int, default=self.n_redundant)
        self.parser.add_argument('--n_repeated', type=int, default=self.n_repeated)
        self.parser.add_argument('--n_classes', type=int, default=self.n_classes)
        self.parser.add_argument('--n_clusters_per_class', type=int, default=self.n_clusters_per_class)
        self.parser.add_argument('--weights', type=list, default=self.weights)
        self.parser.add_argument('--flip_y', type=float, default=self.flip_y)
        self.parser.add_argument('--class_sep', type=float, default=self.class_sep)
        self.parser.add_argument('--hypercube', type=bool, default=self.hypercube)
        self.parser.add_argument('--shift', type=float, default=self.shift)
        self.parser.add_argument('--scale', type=float, default=self.scale)
        self.parser.add_argument('--shuffle', type=bool, default=self.shuffle)
        self.parser.add_argument('--seeds_dataset', type=int, default=self.seeds_dataset, nargs='*')

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
        return args

    def _get_combinations(self):
        combinations = list(product(self.models_nickname, self.seeds_models, self.seeds_dataset))
        combination_names = ['model_nickname', 'seed_model', 'seed_dataset']
        combinations = [list(combination) + [self.models_params[combination[0]]] + [self.fits_params[combination[0]]]
                        for combination in combinations]
        combination_names += ['model_params', 'fit_params']
        unique_params = dict(n_samples=self.n_samples, n_random=self.n_random, n_informative=self.n_informative,
                             n_redundant=self.n_redundant, n_repeated=self.n_repeated, n_classes=self.n_classes,
                             n_clusters_per_class=self.n_clusters_per_class, weights=self.weights, flip_y=self.flip_y,
                             class_sep=self.class_sep, hypercube=self.hypercube, shift=self.shift, scale=self.scale,
                             shuffle=self.shuffle)
        extra_params = dict(n_jobs=self.n_jobs, return_results=False, timeout_combination=self.timeout_combination,
                            timeout_fit=self.timeout_fit)
        return combinations, combination_names, unique_params, extra_params

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
        n_samples = unique_params['n_samples']
        n_random = unique_params['n_random']
        n_informative = unique_params['n_informative']
        n_redundant = unique_params['n_redundant']
        n_repeated = unique_params['n_repeated']
        n_classes = unique_params['n_classes']
        n_clusters_per_class = unique_params['n_clusters_per_class']
        weights = unique_params['weights']
        flip_y = unique_params['flip_y']
        class_sep = unique_params['class_sep']
        hypercube = unique_params['hypercube']
        shift = unique_params['shift']
        scale = unique_params['scale']
        shuffle = unique_params['shuffle']
        seed_dataset = combination['seed_dataset']
        dataset_name = (f'classif_{n_samples}_{n_random}_{n_informative}_{n_redundant}_{n_repeated}_{n_classes}_'
                        f'{n_clusters_per_class}_{weights}_{flip_y}_{class_sep}_{hypercube}_{shift}_{scale}_{shuffle}_'
                        f'{seed_dataset}')
        dataset_dir = self.work_root_dir / dataset_name
        X_file = dataset_dir / 'X.npy'
        y_file = dataset_dir / 'y.npy'
        if X_file.exists() and y_file.exists():
            X = np.load(X_file)
            y = np.load(y_file)
        else:
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
            n_jobs: int = 1, return_results: bool = True,
            log_to_mlflow: bool = False
    ):

        combination = {
            'model_nickname': model_nickname,
            'seed_model': seed_model,
            'seed_dataset': seed_dataset,
            'model_params': model_params,
            'fit_params': fit_params,
        }
        unique_params = {
            'n_samples': n_samples,
            'n_random': n_random,
            'n_informative': n_informative,
            'n_redundant': n_redundant,
            'n_repeated': n_repeated,
            'n_classes': n_classes,
            'n_clusters_per_class': n_clusters_per_class,
            'weights': weights,
            'flip_y': flip_y,
            'class_sep': class_sep,
            'hypercube': hypercube,
            'shift': shift,
            'scale': scale,
            'shuffle': shuffle,
        }
        extra_params = {
            'n_jobs': n_jobs,
            'return_results': return_results,
        }
        if log_to_mlflow:
            return self._run_mlflow_and_train_model(combination=combination, unique_params=unique_params,
                                                    extra_params=extra_params, return_results=return_results)
        else:
            return self._train_model(combination=combination, unique_params=unique_params, extra_params=extra_params,
                                     return_results=return_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = ClassificationClusteringExperiment(parser=parser)
    experiment.run()
