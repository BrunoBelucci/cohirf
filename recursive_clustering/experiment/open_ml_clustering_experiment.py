import argparse
from itertools import product
from typing import Optional
import mlflow
import numpy as np
import pandas as pd
import openml
from recursive_clustering.experiment.clustering_experiment import ClusteringExperiment


class OpenmlClusteringExperiment(ClusteringExperiment):
    def __init__(
            self,
            datasets_ids: Optional[list[int]] = None,
            standardize: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.datasets_ids = datasets_ids
        self.standardize = standardize

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--datasets_ids', type=int, nargs='*')
        self.parser.add_argument('--standardize', action='store_true')

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.datasets_ids = args.datasets_ids
        self.standardize = args.standardize
        return args

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
        dataset_id = combination['dataset_id']
        standardize = unique_params['standardize']
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
            # but only if they have less than 10 categories, else we drop them
            cat_dims = [len(X[cat_feature].cat.categories) for cat_feature in cat_features_names]
            cat_features_names_more_10 = [cat_feature for cat_feature, cat_dim in zip(cat_features_names, cat_dims) if
                                          cat_dim < 10]
            X = pd.get_dummies(X, columns=cat_features_names_more_10, drop_first=True)
            cat_features_drop = [cat_feature for cat_feature in cat_features_names if
                                 cat_feature not in cat_features_names_more_10]
            X = X.drop(columns=cat_features_drop)
        # continuous features
        if cont_features_names:
            # we will fill missing values with the median
            X[cont_features_names] = X[cont_features_names].fillna(X[cont_features_names].median())
            # we will standardize the continuous features
            if standardize:
                X[cont_features_names] = (X[cont_features_names] - X[cont_features_names].mean()) / X[cont_features_names].std()
            # we will cast them to float
            X[cont_features_names] = X[cont_features_names].astype(float)
        # we will drop 0 variance features
        X = X.dropna(axis=1, how='all')

        # log to mlflow to facilitate analysis
        mlflow_run_id = extra_params.get('mlflow_run_id', None)
        if mlflow_run_id is not None:
            mlflow.log_params({
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': n_classes,
            }, run_id=mlflow_run_id)
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
        unique_params = dict(standardize=self.standardize)
        extra_params = dict(n_jobs=self.n_jobs, return_results=False, timeout_combination=self.timeout_combination,
                            timeout_fit=self.timeout_fit)
        return combinations, combination_names, unique_params, extra_params

    def run_openml_experiment_combination(
            self, model_nickname: str, dataset_id: int, seed_model: int = 0, model_params: Optional[dict] = None,
            fit_params: Optional[dict] = None, standardize: bool = False,
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
        unique_params = {
            'standardize': standardize,
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
    experiment = OpenmlClusteringExperiment(parser=parser)
    experiment.run()
