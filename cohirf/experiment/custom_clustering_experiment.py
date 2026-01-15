from typing import Optional
import mlflow
import pandas as pd
from cohirf.experiment.open_ml_clustering_experiment import preprocess, models_dict
from cohirf.experiment.clustering_experiment import ClusteringExperiment
from pathlib import Path


class CustomClusteringExperiment(ClusteringExperiment):
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
            X: pd.DataFrame,
            y: pd.Series,
            dataset_name: Optional[str] = None,
            standardize: bool = False,
            seed_dataset_order: Optional[int | list[int]] = None,
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
        self.X = X
        self.y = y
        self.dataset_name = dataset_name
        self.standardize = standardize
        self.seed_dataset_order = seed_dataset_order

    @property
    def models_dict(self):
        return models_dict.copy()

    # def _add_arguments_to_parser(self):
    #     super()._add_arguments_to_parser()
    #     self.parser.add_argument("--dataset_name", type=str, nargs="*")
    #     self.parser.add_argument('--standardize', action='store_true')
    #     self.parser.add_argument('--seed_dataset_order', type=int, nargs="*")

    # def _unpack_parser(self):
    #     args = super()._unpack_parser()
    #     self.dataset_name = args.dataset_name
    #     self.standardize = args.standardize
    #     self.seed_dataset_order = args.seed_dataset_order
    #     return args

    def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        dataset_name = unique_params["dataset_name"]
        standardize = unique_params["standardize"]
        seed_dataset_order = combination["seed_dataset_order"]
        X = self.X
        y = self.y
        # try to infer categorical features from data
        cat_features_names = X.select_dtypes(include=['object', 'category']).columns.tolist()
        cont_features_names = X.select_dtypes(include=['number', 'bool']).columns.tolist()
        
        # we will preprocess the data always in the same way
        X, y = preprocess(X, y, cat_features_names, cont_features_names, standardize, seed_dataset_order)
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

    def _get_combinations_names(self) -> list[str]:
        combination_names = super()._get_combinations_names()
        combination_names.extend(
            [
                "seed_dataset_order",
            ]
        )
        return combination_names

    def _get_unique_params(self):
        unique_params = super()._get_unique_params()
        unique_params["dataset_name"] = self.dataset_name
        unique_params["standardize"] = self.standardize
        return unique_params

    def _get_extra_params(self):
        extra_params = super()._get_extra_params()
        extra_params["X"] = self.X
        extra_params["y"] = self.y
        return extra_params


# if __name__ == '__main__':
#     experiment = CustomClusteringExperiment()
#     experiment.run_from_cli()
