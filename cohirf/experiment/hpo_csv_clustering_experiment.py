import argparse
from typing import Optional
from cohirf.experiment.hpo_clustering_experiment import HPOClusteringExperiment
from cohirf.experiment.csv_clustering_experiment import CSVClusteringExperiment


class HPOCSVClusteringExperiment(HPOClusteringExperiment, CSVClusteringExperiment):

    def _load_simple_experiment(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        experiment = CSVClusteringExperiment(
            # experiment parameters
            experiment_name=self.experiment_name,
            log_dir=self.log_dir,
            log_file_name=self.log_file_name,
            work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir,
            clean_work_dir=self.clean_work_dir,
            clean_data_dir=False,
            raise_on_error=self.raise_on_error,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            check_if_exists=self.check_if_exists,
            profile_memory=self.profile_memory,
            profile_time=self.profile_time,
            verbose=0,
        )
        return experiment


def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        dataset_name = combination['dataset_name']
        standardize = unique_params['standardize']
        return {
            'dataset_name': dataset_name,
            'standardize': standardize
        }


if __name__ == '__main__':
    experiment = HPOCSVClusteringExperiment()
    experiment.run_from_cli()
