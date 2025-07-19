import argparse
from itertools import product
from typing import Optional
import mlflow
import numpy as np
import pandas as pd

from cohirf.experiment.clustering_experiment import ClusteringExperiment


class CSVClusteringExperiment(ClusteringExperiment):
    """
    Experiment class for clustering real datasets loaded from CSV files.
    
    This experiment loads real-world datasets from CSV files using a metadata CSV that specifies
    dataset paths and categorical feature information. It performs automatic preprocessing including
    handling categorical features (one-hot encoding), filling missing values, standardizing 
    continuous features, and removing zero-variance features. The experiment is designed to work
    with diverse real-world datasets that require preprocessing before clustering.
    """
    
    def __init__(
            self,
            datasets_names: Optional[list[str]] = None,
            **kwargs
    ):
        """
        Initialize the CSVClusteringExperiment.

        Args:
            datasets_names (Optional[list[str]], optional): List of dataset names to process.
                These names should correspond to entries in the 'csv_data.csv' metadata file
                that contains dataset paths and feature information. If None, must be specified
                via command line arguments or before running experiments. Defaults to None.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.datasets_names = datasets_names

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--datasets_names', type=str, nargs='*')

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.datasets_names = args.datasets_names
        return args

    def _load_data(self, combination: dict, unique_params: dict, extra_params: dict, **kwargs):
        csv_data = pd.read_csv('csv_data.csv', sep=';',
                               dtype={'dataset_name': str, 'X_path': str, 'y_path': str, 'cat_features': str})
        dataset_name = combination['dataset_name']
        dataset_row = csv_data[csv_data['dataset_name'] == dataset_name]
        X_path = dataset_row['X_path'].values[0]
        y_path = dataset_row['y_path'].values[0]
        cat_features = dataset_row['cat_features'].values[0]
        # transform nan to 'None'
        if pd.isna(cat_features):
            cat_features = 'None'
        if cat_features != 'None':
            # cat features is a string with the format 'feature1,feature2,feature3' transform it to list of integers
            cat_features = [int(feature) for feature in cat_features.split(',')]
        else:
            cat_features = []
        X = pd.read_csv(X_path, index_col=0)
        y = pd.read_csv(y_path, index_col=0)
        y = y.iloc[:, 0]
        cat_features_names = X.columns[cat_features].tolist()
        cont_features_names = [feature for feature in X.columns if feature not in cat_features_names]
        n_classes = len(y.unique())
        # we will preprocess the data always in the same way
        # categorical features
        if cat_features_names:
            # we will convert categorical features to codes
            for cat_feature in cat_features_names:
                X[cat_feature] = X[cat_feature].astype('category').cat.codes
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
            X[cont_features_names] = (X[cont_features_names] - X[cont_features_names].mean()) / X[
                cont_features_names].std()
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
            'cat_features_names': cat_features_names,
            'n_classes': n_classes,
            'dataset_name': dataset_name
        }

    def _get_combinations(self):
        if self.datasets_names is None:
            raise ValueError('datasets_names must be specified')
        combinations = list(product(self.models_nickname, self.seeds_models, self.datasets_names))
        combination_names = ['model_nickname', 'seed_model', 'dataset_name']
        combinations = [list(combination) + [self.models_params[combination[0]]] + [self.fits_params[combination[0]]]
                        for combination in combinations]
        combination_names += ['model_params', 'fit_params']
        unique_params = dict()
        extra_params = dict(n_jobs=self.n_jobs, return_results=False, timeout_combination=self.timeout_combination,
                            timeout_fit=self.timeout_fit)
        return combinations, combination_names, unique_params, extra_params

    def run_csv_experiment_combination(
            self, model_nickname: str, dataset_name: str, seed_model: int = 0, model_params: Optional[dict] = None,
            fit_params: Optional[dict] = None,
            n_jobs: int = 1, return_results: bool = True,
            log_to_mlflow: bool = False,
            timeout_combination: Optional[int] = None, timeout_fit: Optional[int] = None,
    ):

        combination = {
            'model_nickname': model_nickname,
            'seed_model': seed_model,
            'dataset_name': dataset_name,
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
    experiment = CSVClusteringExperiment(parser=parser)
    experiment.run()
