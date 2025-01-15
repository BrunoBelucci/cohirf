import argparse
from itertools import product
from typing import Optional
import numpy as np
import pandas as pd
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
        cont_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is False]
        cat_dims = [len(X[cat_feature].cat.categories) for cat_feature in cat_features_names]
        n_classes = len(y.unique())
        dataset_name = dataset.name
        # we will preprocess the data always in the same way
        # categorical features
        if cat_features_names:
            # we will convert categorical features to codes
            for cat_feature in cat_features_names:
                X[cat_feature] = X[cat_feature].cat.codes
                X[cat_feature] = X[cat_feature].replace(-1, np.nan).astype('category')
            # we will fill missing values with the most frequent value
            X[cat_features_names] = X[cat_features_names].fillna(X[cat_features_names].mode().iloc[0])
            # we will one hot encode the categorical features and convert them to float
            X = pd.get_dummies(X, columns=cat_features_names, dtype=float)
        # continuous features
        if cont_features_names:
            # we will fill missing values with the median
            X[cont_features_names] = X[cont_features_names].fillna(X[cont_features_names].median())
            # we will standardize the continuous features
            X[cont_features_names] = (X[cont_features_names] - X[cont_features_names].mean()) / X[cont_features_names].std()
            # we will cast them to float
            X[cont_features_names] = X[cont_features_names].astype(float)
        # we will drop 0 variance features
        X = X.dropna(axis=1, how='all')
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
        extra_params = dict(n_jobs=self.n_jobs, return_results=False, timeout_combination=self.timeout_combination,
                            timeout_fit=self.timeout_fit)
        return combinations, combination_names, unique_params, extra_params

    def run_openml_experiment_combination(
            self, model_nickname: str, dataset_id: int, seed_model: int = 0, model_params: Optional[dict] = None,
            fit_params: Optional[dict] = None,
            n_jobs: int = 1, return_results: bool = True,
            log_to_mlflow: bool = False,
            timeout_combination: Optional[int] = None, timeout_fit: Optional[int] = None,
    ):

        combination = {
            'model_nickname': model_nickname,
            'seed_model': seed_model,
            'dataset_id': dataset_id,
            'model_params': model_params,
            'fit_params': fit_params,
        }
        unique_params = {}
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
    experiment = OpenmlClusteringExperiment(parser=parser)
    experiment.run()
