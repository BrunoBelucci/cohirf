import random
from copy import deepcopy
from itertools import product
from typing import Optional

import numpy as np
import openml
from sklearn.metrics import (rand_score, adjusted_rand_score, mutual_info_score, adjusted_mutual_info_score,
                             normalized_mutual_info_score, homogeneity_completeness_v_measure, silhouette_score)


from ml_experiments.base_experiment import BaseExperiment


class ClusteringExperiment(BaseExperiment):
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

    @property
    def models_dict(self):
        pass

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

    def _load_model(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                    **kwargs):
        model_nickname = combination['model_nickname']
        seed_model = combination['seed_model']
        model_params = combination['model_params']
        n_jobs = extra_params.get('n_jobs', self.n_jobs)
        random.seed(seed_model)
        np.random.seed(seed_model)
        model_class, model_default_params = deepcopy(self.models_dict[model_nickname])
        model_default_params.update(model_params)
        model = model_class(**model_default_params)
        if hasattr(model, 'n_jobs'):
            n_jobs = model_params.get('n_jobs', n_jobs)
            model.set_params(n_jobs=n_jobs)
        return {
            'model': model,
        }

    def _get_metrics(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                     **kwargs):
        scores = {
            'rand_score': rand_score,
            'adjusted_rand': adjusted_rand_score,
            'mutual_info': mutual_info_score,
            'adjusted_mutual_info': adjusted_mutual_info_score,
            'normalized_mutual_info': normalized_mutual_info_score,
            'homogeneity_completeness_v_measure': homogeneity_completeness_v_measure,
            'silhouette': silhouette_score,
        }
        return scores

    def _fit_model(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
        model = kwargs['load_model_return']['model']
        X = kwargs['load_data_return']['X']
        y_pred = model.fit_predict(X)
        return { 'y_pred': y_pred }

    def _evaluate_model(self, combination: dict, unique_params: Optional[dict] = None,
                        extra_params: Optional[dict] = None, **kwargs):
        scores = kwargs['get_metrics_return']['scores']
        X = kwargs['load_data_return']['X']
        y_true = kwargs['load_data_return']['y']
        y_pred = kwargs['fit_model_return']['y_pred']
        results = {}
        for score_name, score_fn in scores.items():
            if score_name == 'homogeneity_completeness_v_measure':
                homogeneity, completeness, v_measure = score_fn(y_true, y_pred)
                results['homogeneity'] = homogeneity
                results['completeness'] = completeness
                results['v_measure'] = v_measure
            elif score_name == 'silhouette':
                results['silhouette'] = score_fn(X, y_pred)
            else:
                results[score_name] = score_fn(y_true, y_pred)
        return results

    def _get_combinations(self):
        combinations = list(product(self.models_nickname, self.seeds_models, self.datasets_ids))
        combination_names = ['model_nickname', 'seed_model', 'dataset_id']
        combinations = [list(combination) + [self.models_params[combination[0]]] + [self.fits_params[combination[0]]]
                        for combination in combinations]
        combination_names += ['model_params', 'fit_params']
        unique_params = dict()
        extra_params = dict(n_jobs=self.n_jobs, return_results=False)
        return combinations, combination_names, unique_params, extra_params
