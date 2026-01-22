from copy import deepcopy
from abc import ABC
from typing import Optional
from shutil import rmtree
from functools import partial
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (rand_score, adjusted_rand_score, mutual_info_score, adjusted_mutual_info_score,
                             normalized_mutual_info_score, homogeneity_completeness_v_measure, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.base import BaseEstimator
from ml_experiments.base_experiment import BaseExperiment
from cohirf.experiment.tested_models import models_dict
from cohirf.metrics import davies_bouldin_score as chunked_davies_bouldin_score
from ml_experiments.utils import update_recursively, profile_memory, profile_time
import json
import os
from warnings import warn
from cohirf.models.batch_cohirf import BatchCoHiRF


def calculate_scores(calculate_metrics_even_if_too_many_clusters, n_clusters, X, y_true, y_pred, scores):
    results = {}
    if not calculate_metrics_even_if_too_many_clusters:
            if n_clusters > 0.5 * X.shape[0]:
                warn(f"Too many clusters ({n_clusters}) for dataset with {X.shape[0]} samples. Skipping metric calculation. If you want to calculate metrics anyway, set `calculate_metrics_even_if_too_many_clusters` to True.")
                return results  # Avoid calculating scores if too many clusters (they are probably not meaningful)
    for score_name, score_fn in scores.items():
        if callable(score_fn):
            if score_name == 'homogeneity_completeness_v_measure':
                homogeneity, completeness, v_measure = score_fn(y_true, y_pred) # type: ignore
                results['homogeneity'] = homogeneity
                results['completeness'] = completeness
                results['v_measure'] = v_measure
            elif score_name == 'silhouette' or score_name == 'silhouette_1000':
                try:
                    results['silhouette'] = score_fn(X, y_pred) # type: ignore
                except ValueError:
                    results['silhouette'] = -1
            elif score_name == 'calinski_harabasz_score':
                try:
                    results['calinski_harabasz_score'] = score_fn(X, y_pred) # type: ignore
                except ValueError:
                    results['calinski_harabasz_score'] = -1
            elif score_name == 'davies_bouldin_score':
                try:
                    results['davies_bouldin_score'] = score_fn(X, y_pred) # type: ignore
                except ValueError:
                    results['davies_bouldin_score'] = 1e3 # type: ignore
            elif score_name == 'inertia_score':
                try:
                    results['inertia_score'] = score_fn(X, y_pred) # type: ignore
                except ValueError:
                    results['inertia_score'] = -1
            else:
                results[score_name] = score_fn(y_true, y_pred) # type: ignore
    return results


def inertia_score(X, y):
    """
    Calculate the inertia score for a clustering result.

    Parameters:
    X : np.ndarray
        The data points, shape (n_samples, n_features).
    y : np.ndarray
        The assigned cluster labels, shape (n_samples,).

    Returns:
    float
        The inertia score.
    """
    # obs not optimized, but should work for testing purposes
    inertia = 0.0
    unique_labels = np.unique(y)

    # Ensure X is a NumPy array
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()

    for label in unique_labels:
        # Extract points belonging to the current cluster
        cluster_points = X[y == label]

        # Calculate the centroid of the cluster
        centroid = cluster_points.mean(axis=0)

        # Calculate the squared distances of points to the centroid
        distances = euclidean_distances(cluster_points, centroid.reshape(1, -1))**2

        # Sum the squared distances to the inertia
        inertia += distances.sum()

    return inertia


class ClusteringExperiment(BaseExperiment, ABC):
    """
    Abstract base class for clustering experiments.
    
    This class provides a framework for running clustering experiments with various algorithms
    and evaluation metrics. It handles model initialization, fitting, prediction, and comprehensive
    evaluation using multiple clustering metrics (both supervised and unsupervised). Subclasses
    should implement data loading logic specific to their dataset type.
    """

    def __init__(
        self,
        *args,
        model: Optional[str | BaseEstimator | type[BaseEstimator] | list[str]] = None,
        model_params: Optional[dict] = None,
        seed_model: int | list[int] = 0,
        n_jobs: int = 1,
        clean_data_dir: Optional[bool] = True,
        calculate_davies_bouldin: bool = False,
        chunk_size_davies_bouldin: Optional[int] = None,
        calculate_full_silhouette: bool = False,
        calculate_metrics_even_if_too_many_clusters: bool = False,
        # if we set this, we will automatically set the number of threads to accomodate the number of jobs
        max_threads: Optional[int] = None,
        save_labels: bool = False,
        **kwargs,
    ):
        """
        Initialize the ClusteringExperiment.

        Args:
            *args: Variable length argument list passed to parent class.
            model (Optional[str | BaseEstimator | type[BaseEstimator] | list[str]], optional):
                Clustering model(s) to use. Can be:
                - String: Model name from the models dictionary
                - BaseEstimator: Instantiated sklearn-compatible model
                - Type: Class of sklearn-compatible model
                - List of strings: Multiple model names for batch processing
                Defaults to None.
            model_params (Optional[dict], optional): Parameters to pass to the clustering model(s).
                Used for hyperparameter configuration. Defaults to None (empty dict).
            seed_model (int | list[int], optional): Random seed(s) for model initialization.
                If list, creates multiple experiments with different seeds for reproducibility.
                Defaults to 0.
            n_jobs (int, optional): Number of parallel jobs to use in clustering algorithms
                that support parallelization. -1 uses all available cores. Defaults to 1.
            clean_data_dir (Optional[bool], optional): Whether to clean the data directory
                after the experiments. Defaults to True.
            calculate_davies_bouldin (bool, optional): Whether to calculate the Davies-Bouldin
                clustering validity index. Can be computationally expensive for large datasets.
                Defaults to False.
            chunk_size_davies_bouldin (Optional[int], optional): Chunk size for Davies-Bouldin calculation.
                If not specified, defaults to None (no chunking) and use sklearn implementation. If specified,
                it will use our custom implementation with the given chunk size.
            calculate_full_silhouette (bool, optional): Whether to calculate the full silhouette
                score using all samples. Can be computationally expensive for large datasets.
                Defaults to False.
            max_threads (Optional[int], optional): Maximum number of threads to use across all jobs.
                Automatically adjusts threading to accommodate the number of parallel jobs.
                Defaults to None (no limit).
            calculate_metrics_even_if_too_many_clusters (bool, optional): Whether to calculate
                metrics even if the number of clusters is too high. Defaults to False.
            save_labels (bool, optional): Whether to save the predicted labels after clustering.
				Defaults to False.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.model_params = model_params if model_params is not None else {}
        self.seed_model = seed_model
        self.n_jobs = n_jobs
        self.clean_data_dir = clean_data_dir
        self.calculate_davies_bouldin = calculate_davies_bouldin
        self.chunk_size_davies_bouldin = chunk_size_davies_bouldin
        self.calculate_full_silhouette = calculate_full_silhouette
        self.max_threads = max_threads
        self.calculate_metrics_even_if_too_many_clusters = calculate_metrics_even_if_too_many_clusters
        self.save_labels = save_labels

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        if self.parser is None:
            raise ValueError("Parser is not initialized.")
        self.parser.add_argument('--model', type=str, nargs='+', help='Model to use for clustering.')
        self.parser.add_argument('--model_params', type=json.loads, default=self.model_params, help='Parameters for the model.')
        self.parser.add_argument('--seed_model', type=int, nargs='*', default=self.seed_model, help='Random seed for model initialization.')
        self.parser.add_argument('--n_jobs', type=int, default=self.n_jobs, help='n_jobs for models.')
        self.parser.add_argument('--do_not_clean_data_dir', action='store_true')
        self.parser.add_argument('--calculate_davies_bouldin', action='store_true', default=self.calculate_davies_bouldin,
                                help='Calculate Davies-Bouldin score.')
        self.parser.add_argument('--chunk_size_davies_bouldin', type=int, default=self.chunk_size_davies_bouldin,
                                 help='Chunk size for Davies-Bouldin calculation.')
        self.parser.add_argument('--calculate_full_silhouette', action='store_true', default=self.calculate_full_silhouette,
                                help='Calculate full silhouette score (not sampled).')
        self.parser.add_argument('--max_threads', type=int, default=self.max_threads, help='Maximum number of threads to use across all jobs.')
        self.parser.add_argument('--calculate_metrics_even_if_too_many_clusters', action='store_true', default=self.calculate_metrics_even_if_too_many_clusters,
                                help='Calculate metrics even if the number of clusters is too high.')
        self.parser.add_argument('--save_labels', action='store_true', default=self.save_labels,
								help='Save predicted labels after clustering.')

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.clean_data_dir = not args.do_not_clean_data_dir
        self.model = args.model
        self.n_jobs = args.n_jobs
        self.model_params = args.model_params
        self.seed_model = args.seed_model
        self.calculate_davies_bouldin = args.calculate_davies_bouldin
        self.chunk_size_davies_bouldin = args.chunk_size_davies_bouldin
        self.calculate_full_silhouette = args.calculate_full_silhouette
        self.max_threads = args.max_threads
        self.calculate_metrics_even_if_too_many_clusters = args.calculate_metrics_even_if_too_many_clusters
        self.save_labels = args.save_labels
        return args

    def _get_combinations_names(self) -> list[str]:
        combination_names = super()._get_combinations_names()
        combination_names.extend(['model', 'seed_model'])
        return combination_names

    def _get_unique_params(self):
        unique_params = super()._get_unique_params()
        unique_params['n_jobs'] = self.n_jobs
        unique_params['model_params'] = self.model_params
        unique_params["max_threads"] = self.max_threads
        unique_params["calculate_davies_bouldin"] = self.calculate_davies_bouldin
        unique_params["chunk_size_davies_bouldin"] = self.chunk_size_davies_bouldin
        unique_params["calculate_full_silhouette"] = self.calculate_full_silhouette
        unique_params["calculate_metrics_even_if_too_many_clusters"] = self.calculate_metrics_even_if_too_many_clusters
        return unique_params

    def _get_extra_params(self):
        extra_params = super()._get_extra_params()
        extra_params["save_labels"] = self.save_labels
        return extra_params

    @property
    def models_dict(self):
        return models_dict.copy()

    def _load_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        model = combination['model']
        seed_model = combination['seed_model']
        n_jobs = unique_params['n_jobs']
        model_params = unique_params['model_params']
        max_threads = unique_params["max_threads"]
        if isinstance(model, str):
            model_class, model_default_params, _, _ = deepcopy(self.models_dict[model])
            model_default_params = update_recursively(model_default_params, model_params)
            model = model_class(**model_default_params)
        elif isinstance(model, type):
            model = model(**model_params)
        else:
            model = deepcopy(model)
            model.set_params(**model_params)
        if hasattr(model, 'n_jobs'):
            # note that I have thested and at least in linux (in my machine) setting these env variables actually works
            # and limits the number of threads used by each process, I am not sure on how effective this is in Windows
            # or MacOS or other systems, maybe we should set them externally before launching the script to be sure.
            # Also I have tested them when doing hpo, so maybe they work because we are spawning other processes when
            # using optuna or similar libraries? Not sure if they work when only running a simple experiment.
            # We can check with the following script: python hpo_classification_clustering_experiment.py --model BatchCoHiRF-SC-SRGF --experiment_name sfni-BatchCoHiRF-SC-SRGF-no-limit --n_jobs 10 --hpo_framework optuna --n_trials 20 --sampler tpe --pruner none --direction maximize --hpo_metric adjusted_rand --n_samples 1000 --n_classes 5 --n_informative 3 --class_sep 5.196152422706632 --n_random 1000 --seed_dataset 0 --hpo_seed 0
            # and comparing it with the same command but adding OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 before "python" and/or --max_threads 10 at the end
            if max_threads is not None:
                threads_per_process = max(1, max_threads // n_jobs)
                # Set environment variables
                for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
                    os.environ[var] = str(threads_per_process)
            model.set_params(n_jobs=n_jobs)
        if hasattr(model, 'random_state'):
            if "random_state" in model_params:  # override with model_params if present
                seed_model = model_params["random_state"]
            model.set_params(random_state=seed_model)
        return {
            'model': model,
        }

    def _get_metrics(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        calculate_davies_bouldin = unique_params["calculate_davies_bouldin"]
        calculate_full_silhouette = unique_params["calculate_full_silhouette"]
        chunk_size_davies_bouldin = unique_params["chunk_size_davies_bouldin"]
        scores = {
            'rand_score': rand_score,
            'adjusted_rand': adjusted_rand_score,
            'mutual_info': mutual_info_score,
            'adjusted_mutual_info': adjusted_mutual_info_score,
            'normalized_mutual_info': normalized_mutual_info_score,
            'homogeneity_completeness_v_measure': homogeneity_completeness_v_measure,
            'silhouette_1000': partial(silhouette_score, sample_size=1000),
            'calinski_harabasz_score': calinski_harabasz_score,
        }
        if calculate_davies_bouldin:
            if chunk_size_davies_bouldin is not None:
                scores['davies_bouldin_score'] = partial(chunked_davies_bouldin_score, chunk_size=chunk_size_davies_bouldin)
            else:
                scores['davies_bouldin_score'] = davies_bouldin_score
        if calculate_full_silhouette:
            scores['silhouette'] = silhouette_score
        return scores

    @profile_time(enable_based_on_attribute="profile_time")
    @profile_memory(enable_based_on_attribute="profile_memory")
    def _fit_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        model = kwargs['load_model_return']['model']
        save_labels = extra_params["save_labels"]
        X = kwargs['load_data_return']['X']
        if not isinstance(model, BatchCoHiRF):
            y_pred = model.fit_predict(X)
        else:
            y = kwargs['load_data_return'].get('y', None)
            y_pred = model.fit_predict(X, y)
        if save_labels:
            save_dir = kwargs.get('save_dir', None)
            if save_dir is None:
                raise ValueError("save_dir is None, please specify a valid directory to save labels.")
            # save labels in a csv file
            df = pd.DataFrame({'labels': y_pred})
            df.to_csv(save_dir / f"predicted_labels.csv", index=False)
        return {'y_pred': y_pred}

    @profile_time(enable_based_on_attribute="profile_time")
    @profile_memory(enable_based_on_attribute="profile_memory")
    def _evaluate_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        scores = kwargs['get_metrics_return']
        X = kwargs['load_data_return']['X']
        y_true = kwargs['load_data_return']['y']
        y_pred = kwargs['fit_model_return']['y_pred']
        calculate_metrics_even_if_too_many_clusters = unique_params["calculate_metrics_even_if_too_many_clusters"]
        n_clusters = len(np.unique(y_pred))
        results = calculate_scores(calculate_metrics_even_if_too_many_clusters, n_clusters, X, y_true, y_pred, scores)
        results["n_clusters_"] = n_clusters
        return results

    def _log_run_results(self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id=None, 
                         **kwargs):
        if mlflow_run_id is None:
            return

        self._log_base_experiment_run_results(combination=combination, unique_params=unique_params,
                                              extra_params=extra_params, mlflow_run_id=mlflow_run_id, **kwargs)

        log_params = {}

        load_data_return = kwargs.get('load_data_return', {})
        if 'dataset_name' in load_data_return:
            log_params['dataset_name'] = load_data_return['dataset_name']

        mlflow.log_params(log_params, run_id=mlflow_run_id)

        log_metrics = {}
        load_model_return = kwargs.get('load_model_return', {})
        if 'model' in load_model_return:
            model = load_model_return["model"]
            if hasattr(model, 'n_iter_'):
                log_metrics["n_iter_"] = model.n_iter_
            if hasattr(model, 'n_epoch_'):
                log_metrics["n_epoch_"] = model.n_epoch_

        mlflow.log_metrics(log_metrics, run_id=mlflow_run_id)

    def _on_exception_or_train_end(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        result = super()._on_exception_or_train_end(combination=combination, unique_params=unique_params, mlflow_run_id=mlflow_run_id,
                                                    extra_params=extra_params, **kwargs)
        dataset_name = kwargs.get('load_data_return', {}).get('dataset_name', None)
        if dataset_name is not None:
            dataset_dir = self.work_root_dir / dataset_name
            if self.clean_data_dir:
                if dataset_dir.exists():
                    rmtree(dataset_dir)
        return result
