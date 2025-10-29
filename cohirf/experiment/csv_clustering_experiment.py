from typing import Optional
import mlflow
import pandas as pd
from cohirf.experiment.open_ml_clustering_experiment import preprocess, models_dict
from cohirf.experiment.clustering_experiment import ClusteringExperiment
from pathlib import Path


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
            dataset_name: Optional[str | list[str]] = None,
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
        self.dataset_name = dataset_name
        self.standardize = standardize
        self.seed_dataset_order = seed_dataset_order

    @property
    def models_dict(self):
        return models_dict.copy()

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument("--dataset_name", type=str, nargs="*")
        self.parser.add_argument('--standardize', action='store_true')
        self.parser.add_argument('--seed_dataset_order', type=int, nargs="*")

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.dataset_name = args.dataset_name
        self.standardize = args.standardize
        self.seed_dataset_order = args.seed_dataset_order
        return args

    def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        csv_data = pd.read_csv(Path(__file__).parent / 'csv_data.csv', sep=';',
                               dtype={'dataset_name': str, 'X_path': str, 'y_path': str, 'cat_features': str})
        dataset_name = combination['dataset_name']
        dataset_row = csv_data[csv_data['dataset_name'] == dataset_name]
        cat_features = dataset_row['cat_features'].values[0]
        standardize = unique_params["standardize"]
        seed_dataset_order = combination["seed_dataset_order"]
        # transform nan to 'None'
        if pd.isna(cat_features):
            cat_features = 'None'
        if cat_features != 'None':
            # cat features is a string with the format 'feature1,feature2,feature3' transform it to list of integers
            cat_features = [int(feature) for feature in cat_features.split(',')]
        else:
            cat_features = []
        X = pd.read_csv(Path(__file__).parent.parent.parent / 'data' / dataset_name / 'X.csv', index_col=0)
        y = pd.read_csv(Path(__file__).parent.parent.parent / 'data' / dataset_name / 'y.csv', index_col=0)
        y = y.iloc[:, 0]
        cat_features_names = X.columns[cat_features].tolist()
        cont_features_names = [feature for feature in X.columns if feature not in cat_features_names]
        n_classes = len(y.unique())
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
                "dataset_name",
                "seed_dataset_order",
            ]
        )
        return combination_names

    def _get_unique_params(self):
        unique_params = super()._get_unique_params()
        unique_params["standardize"] = self.standardize
        return unique_params


if __name__ == '__main__':
    experiment = CSVClusteringExperiment()
    experiment.run_from_cli()
