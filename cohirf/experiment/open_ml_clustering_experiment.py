import argparse
from email.policy import default
from itertools import product
from typing import Optional
import mlflow
import numpy as np
import pandas as pd
import openml
from cohirf.experiment.clustering_experiment import ClusteringExperiment
from cohirf.experiment.tested_models import models_dict as default_models_dict
from cohirf.models.cohirf import CoHiRF, BaseCoHiRF
from cohirf.models.batch_cohirf import BatchCoHiRF
import optuna
from sklearn.cluster import DBSCAN, KMeans
from sklearn.kernel_approximation import RBFSampler


models_dict = default_models_dict.copy()
# n_features can be 1, so we re-declare here to change the upper limit from 0.6 to 1
models_dict.update(
    {
        CoHiRF.__name__: (
            CoHiRF,
            dict(),
            dict(
                n_features=optuna.distributions.FloatDistribution(0.1, 1),
                repetitions=optuna.distributions.IntDistribution(2, 10),
                kmeans_n_clusters=optuna.distributions.IntDistribution(2, 5),
            ),
            [
                dict(
                    n_features=0.3,
                    repetitions=5,
                    kmeans_n_clusters=3,
                ),
            ],
        ),
        "CoHiRF-KernelRBF": (
            BaseCoHiRF,
            dict(
                base_model=KMeans,
                transform_method=RBFSampler,
                transform_kwargs=dict(n_components=500),
                representative_method="rbf",
            ),
            dict(
                n_features=optuna.distributions.FloatDistribution(0.1, 1),
                repetitions=optuna.distributions.IntDistribution(2, 10),
                base_model_kwargs=dict(
                    n_clusters=optuna.distributions.IntDistribution(2, 5),
                ),
                transform_kwargs=dict(
                    gamma=optuna.distributions.FloatDistribution(0.1, 30),
                ),
            ),
            [
                dict(
                    n_features=0.3,
                    repetitions=5,
                    base_model_kwargs=dict(
                        n_clusters=3,
                    ),
                    transform_kwargs=dict(
                        gamma=1.0,
                    ),
                )
            ],
        ),
        "CoHiRF-DBSCAN": (
            BaseCoHiRF,
            dict(base_model=DBSCAN),
            dict(
                n_features=optuna.distributions.FloatDistribution(0.1, 1),
                repetitions=optuna.distributions.IntDistribution(1, 10),
                base_model_kwargs=dict(
                    eps=optuna.distributions.FloatDistribution(1e-1, 10),
                    min_samples=optuna.distributions.IntDistribution(2, 50),
                ),
            ),
            [
                dict(
                    n_features=0.3,
                    repetitions=5,
                    base_model_kwargs=dict(
                        eps=0.5,
                        min_samples=5,
                    ),
                ),
            ],
        ),
        "BatchCoHiRF": (
            BatchCoHiRF,
            dict(),
            dict(
                cohirf_kwargs=dict(
                    n_features=optuna.distributions.FloatDistribution(0.1, 1),
                    repetitions=optuna.distributions.IntDistribution(2, 10),
                    kmeans_n_clusters=optuna.distributions.IntDistribution(2, 5),
                )
            ),
            [
                dict(
                    cohirf_kwargs=dict(
                        n_features=0.3,
                        repetitions=5,
                        kmeans_n_clusters=3,
                    )
                ),
            ],
        ),
        "BatchCoHiRF-1iter": (
            BatchCoHiRF,
            dict(cohirf_kwargs=dict(max_iter=1)),
            dict(
                cohirf_kwargs=dict(
                    n_features=optuna.distributions.FloatDistribution(0.1, 1),
                    repetitions=optuna.distributions.IntDistribution(2, 10),
                    kmeans_n_clusters=optuna.distributions.IntDistribution(2, 5),
                )
            ),
            [
                dict(
                    cohirf_kwargs=dict(
                        n_features=0.3,
                        repetitions=5,
                        kmeans_n_clusters=3,
                    )
                ),
            ],
        ),
        "BatchCoHiRF-DBSCAN": (
            BatchCoHiRF,
            dict(
                cohirf_model=BaseCoHiRF,
                cohirf_kwargs=dict(base_model=DBSCAN),
            ),
            dict(
                cohirf_kwargs=dict(
                    n_features=optuna.distributions.FloatDistribution(0.1, 1),
                    repetitions=optuna.distributions.IntDistribution(1, 10),
                    base_model_kwargs=dict(
                        eps=optuna.distributions.FloatDistribution(1e-1, 10),
                        min_samples=optuna.distributions.IntDistribution(2, 50),
                    ),
                )
            ),
            [
                dict(
                    cohirf_kwargs=dict(
                        n_features=0.3,
                        repetitions=5,
                        base_model_kwargs=dict(
                            eps=0.5,
                            min_samples=5,
                        ),
                    )
                ),
            ],
        ),
        "BatchCoHiRF-DBSCAN-1iter": (
            BatchCoHiRF,
            dict(
                cohirf_model=BaseCoHiRF,
                cohirf_kwargs=dict(base_model=DBSCAN, max_iter=1),
            ),
            dict(
                cohirf_kwargs=dict(
                    n_features=optuna.distributions.FloatDistribution(0.1, 1),
                    repetitions=optuna.distributions.IntDistribution(1, 10),
                    base_model_kwargs=dict(
                        eps=optuna.distributions.FloatDistribution(1e-1, 10),
                        min_samples=optuna.distributions.IntDistribution(2, 50),
                    ),
                )
            ),
            [
                dict(
                    cohirf_kwargs=dict(
                        n_features=0.3,
                        repetitions=5,
                        base_model_kwargs=dict(
                            eps=0.5,
                            min_samples=5,
                        ),
                    )
                ),
            ],
        ),
        "BatchCoHiRF-KernelRBF-1iter": (
            BatchCoHiRF,
            dict(
                cohirf_model=BaseCoHiRF,
                cohirf_kwargs=dict(
                    base_model=KMeans,
                    transform_method=RBFSampler,
                    transform_kwargs=dict(n_components=500),
                    representative_method="rbf",
                    max_iter=1,
                ),
            ),
            dict(
                cohirf_kwargs=dict(
                    n_features=optuna.distributions.FloatDistribution(0.1, 1),
                    repetitions=optuna.distributions.IntDistribution(1, 10),
                    base_model_kwargs=dict(
                        n_clusters=optuna.distributions.IntDistribution(2, 5),
                    ),
                    transform_kwargs=dict(
                        gamma=optuna.distributions.FloatDistribution(0.1, 30),
                    ),
                )
            ),
            [
                dict(
                    cohirf_kwargs=dict(
                        n_features=0.3,
                        repetitions=5,
                        base_model_kwargs=dict(
                            n_clusters=3,
                        ),
                        transform_kwargs=dict(
                            gamma=1.0,
                        ),
                    ),
                )
            ],
        ),
    }
)


class OpenmlClusteringExperiment(ClusteringExperiment):
    """
    Experiment class for clustering real-world datasets from OpenML.
    
    This experiment loads datasets or tasks from the OpenML platform and evaluates
    clustering algorithms on them. It supports both direct dataset loading and task-based
    loading with proper train/test splits. The experiment includes automatic preprocessing
    for categorical features and optional standardization. It's designed to work with
    the diverse collection of real-world datasets available on OpenML for comprehensive
    clustering performance evaluation.
    """
    
    def __init__(
            self,
            dataset_id: Optional[int | list[int]] = None,
            task_id: Optional[int | list[int]] = None,
            task_repeat: int | list[int] = 0,
            task_fold: int | list[int] = 0,
            task_sample: int | list[int] = 0,
            standardize: bool = False,
            **kwargs
    ):
        """
        Initialize the OpenmlClusteringExperiment.

        Args:
            dataset_id (Optional[int | list[int]], optional): OpenML dataset ID(s) to load directly.
                Cannot be used together with task_id. If list, creates multiple experiments
                with different datasets. Defaults to None.
            task_id (Optional[int | list[int]], optional): OpenML task ID(s) to load with proper
                train/test splits. Cannot be used together with dataset_id. If list, creates
                multiple experiments with different tasks. Defaults to None.
            task_repeat (int | list[int], optional): Task repeat index(es) for cross-validation
                when using task_id. Used to select specific repetitions of the experimental setup.
                If list, creates multiple experiments. Defaults to 0.
            task_fold (int | list[int], optional): Task fold index(es) for cross-validation
                when using task_id. Used to select specific folds within a repetition.
                If list, creates multiple experiments. Defaults to 0.
            task_sample (int | list[int], optional): Task sample index(es) for stratified sampling
                when using task_id. Used for datasets with stratified sampling strategies.
                If list, creates multiple experiments. Defaults to 0.
            standardize (bool, optional): Whether to standardize features (zero mean, unit variance)
                before clustering. Recommended for algorithms sensitive to feature scales.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.dataset_id = dataset_id
        self.task_id = task_id
        self.task_repeat = task_repeat if task_repeat else [0]
        self.task_fold = task_fold if task_fold else [0]
        self.task_sample = task_sample if task_sample else [0]
        self.standardize = standardize

    @property
    def models_dict(self):
        return models_dict.copy()

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        if self.parser is None:
            raise ValueError('Parser must be set before calling _add_arguments_to_parser')
        self.parser.add_argument('--dataset_id', type=int, nargs='*', default=self.dataset_id)
        self.parser.add_argument('--task_id', type=int, nargs='*', default=self.task_id)
        self.parser.add_argument('--task_repeat', type=int, nargs='*', default=self.task_repeat)
        self.parser.add_argument('--task_fold', type=int, nargs='*', default=self.task_fold)
        self.parser.add_argument('--task_sample', type=int, nargs='*', default=self.task_sample)
        self.parser.add_argument('--standardize', action='store_true', default=self.standardize)

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.dataset_id = args.dataset_id
        self.task_id = args.task_id
        self.task_repeat = args.task_repeat
        self.task_fold = args.task_fold
        self.task_sample = args.task_sample
        self.standardize = args.standardize
        return args

    def _get_combinations_names(self) -> list[str]:
        combination_names = super()._get_combinations_names()
        combination_names.extend([
            'dataset_id',
            'task_id',
            'task_repeat',
            'task_fold',
            'task_sample',
        ])
        return combination_names

    def _get_unique_params(self):
        unique_params = super()._get_unique_params()
        unique_params['standardize'] = self.standardize
        return unique_params

    def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        dataset_id = combination['dataset_id']
        task_id = combination['task_id']
        task_repeat = combination['task_repeat']
        task_fold = combination['task_fold']
        task_sample = combination['task_sample']
        standardize = unique_params['standardize']
        if task_id is not None:
            if dataset_id is not None:
                raise ValueError('You cannot specify both dataset_id and task_id')
            task = openml.tasks.get_task(task_id)
            split = task.get_train_test_split_indices(task_fold, task_repeat, task_sample)
            dataset = task.get_dataset()
            X, y, cat_ind, att_names = dataset.get_data(target=task.target_name)  # type: ignore
            train_indices = split.train  # type: ignore
            # we will use only the training data
            if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
                raise ValueError('X and y must be pandas DataFrame and Series respectively')
            X = X.iloc[train_indices]
            y = y.iloc[train_indices]
        elif dataset_id is not None:
            dataset = openml.datasets.get_dataset(dataset_id)
            target = dataset.default_target_attribute
            X, y, cat_ind, att_names = dataset.get_data(target=target)
            if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
                raise ValueError('X and y must be pandas DataFrame and Series respectively')
            if dataset_id == 46785:
                X = X.drop('cell_type1', axis=1)
                att_names = att_names[:-1]
                cat_ind = cat_ind[:-1]
        else:
            raise ValueError('You must specify either dataset_id or task_id')
        cat_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is True]
        cont_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is False]
        cat_dims = [len(X[cat_feature].cat.categories) for cat_feature in cat_features_names]
        n_classes = len(y.unique())
        dataset_name = dataset.name
        # we will preprocess the data always in the same way
        # we first delete duplicated rows
        y = y[~X.duplicated()]
        X = X[~X.duplicated()]
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
            X = pd.get_dummies(X, columns=cat_features_names_more_10, drop_first=True, dtype=float)
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
        if mlflow_run_id is not None:
            mlflow.log_params({
                'n_samples': X.shape[0],
                'n_features_dataset': X.shape[1],
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


if __name__ == '__main__':
    experiment = OpenmlClusteringExperiment()
    experiment.run_from_cli()
