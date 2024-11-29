import random
from copy import deepcopy
from abc import ABC
from typing import Optional
from shutil import rmtree

import mlflow
import numpy as np
from sklearn.metrics import (rand_score, adjusted_rand_score, mutual_info_score, adjusted_mutual_info_score,
                             normalized_mutual_info_score, homogeneity_completeness_v_measure, silhouette_score)


from ml_experiments.base_experiment import BaseExperiment
from recursive_clustering.experiment.tested_models import models_dict


class ClusteringExperiment(BaseExperiment, ABC):
    def __init__(
            self,
            *args,
            clean_data_dir: Optional[bool] = True,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.clean_data_dir = clean_data_dir

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--do_not_clean_data_dir', action='store_true')

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.clean_data_dir = not args.do_not_clean_data_dir
        return args

    @property
    def models_dict(self):
        return models_dict.copy()

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
        return {'y_pred': y_pred}

    def _evaluate_model(self, combination: dict, unique_params: Optional[dict] = None,
                        extra_params: Optional[dict] = None, **kwargs):
        scores = kwargs['get_metrics_return']
        X = kwargs['load_data_return']['X']
        y_true = kwargs['load_data_return']['y']
        y_pred = kwargs['fit_model_return']['y_pred']
        results = {'n_clusters_': len(np.unique(y_pred))}
        for score_name, score_fn in scores.items():
            if callable(score_fn):
                if score_name == 'homogeneity_completeness_v_measure':
                    homogeneity, completeness, v_measure = score_fn(y_true, y_pred)
                    results['homogeneity'] = homogeneity
                    results['completeness'] = completeness
                    results['v_measure'] = v_measure
                elif score_name == 'silhouette':
                    try:
                        results['silhouette'] = score_fn(X, y_pred)
                    except ValueError:
                        results['silhouette'] = -1
                else:
                    results[score_name] = score_fn(y_true, y_pred)
        return results

    def _log_run_results(self, combination: dict, unique_params: Optional[dict] = None,
                         extra_params: Optional[dict] = None, mlflow_run_id=None, **kwargs):
        if mlflow_run_id is None:
            return

        self._log_base_experiment_run_results(combination=combination, unique_params=unique_params,
                                              extra_params=extra_params, mlflow_run_id=mlflow_run_id, **kwargs)

        log_params = {}
        log_metrics = {}

        load_data_return = kwargs['load_data_return']
        if 'dataset_name' in load_data_return:
            log_params['dataset_name'] = load_data_return['dataset_name']

        evaluate_model_return = kwargs['evaluate_model_return']
        log_metrics.update(evaluate_model_return)

        mlflow.log_params(log_params, run_id=mlflow_run_id)
        mlflow.log_metrics(log_metrics, run_id=mlflow_run_id)

    def _on_exception_or_train_end(self, combination: dict, unique_params: Optional[dict] = None,
                                   extra_params: Optional[dict] = None, **kwargs):
        result = super()._on_exception_or_train_end(combination=combination, unique_params=unique_params,
                                                    extra_params=extra_params, **kwargs)
        dataset_name = kwargs['load_data_return']['dataset_name']
        dataset_dir = self.work_root_dir / dataset_name
        if self.clean_data_dir:
            if dataset_dir.exists():
                rmtree(dataset_dir)
        return result
