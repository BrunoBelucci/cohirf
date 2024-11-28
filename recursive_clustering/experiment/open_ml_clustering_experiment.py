import argparse
from itertools import product
from typing import Optional

import openml

from recursive_clustering.experiment.clustering_experiment import ClusteringExperiment


class OpenmlClusteringExperiment(ClusteringExperiment):
    def __init__(
            self,
            datasets_ids: Optional[list[int]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.datasets_ids = datasets_ids

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--datasets_ids', type=int, nargs='*')

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.datasets_ids = args.datasets_ids
        return args

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
        dataset_id = combination['dataset_id']
        dataset = openml.datasets.get_dataset(dataset_id)
        target = dataset.default_target_attribute
        X, y, cat_ind, att_names = dataset.get_data(target=target)
        cat_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is True]
        cat_dims = [len(X[cat_feature].cat.categories) for cat_feature in cat_features_names]
        n_classes = len(y.unique())
        dataset_name = dataset.name
        return {
            'X': X,
            'y': y,
            'cat_ind': cat_ind,
            'att_names': att_names,
            'cat_features_names': cat_features_names,
            'cat_dims': cat_dims,
            'n_classes': n_classes,
            'dataset_name': dataset_name
        }

    def _get_combinations(self):
        combinations = list(product(self.models_nickname, self.seeds_models, self.datasets_ids))
        combination_names = ['model_nickname', 'seed_model', 'dataset_id']
        combinations = [list(combination) + [self.models_params[combination[0]]] + [self.fits_params[combination[0]]]
                        for combination in combinations]
        combination_names += ['model_params', 'fit_params']
        unique_params = dict()
        extra_params = dict(n_jobs=self.n_jobs, return_results=False)
        return combinations, combination_names, unique_params, extra_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = OpenmlClusteringExperiment(parser=parser)
    experiment.run()
