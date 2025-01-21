import argparse
from typing import Optional
from ml_experiments.hpo_experiment import HPOExperiment
from recursive_clustering.experiment.csv_clustering_experiment import CSVClusteringExperiment


class HPOCSVClusteringExperiment(HPOExperiment, CSVClusteringExperiment):
    def get_hyperband_max_resources(self, combination: dict, unique_params: Optional[dict] = None,
                                    extra_params: Optional[dict] = None, **kwargs):
        raise NotImplementedError('Hyperband is not available for this experiment')

    def _load_single_experiment(self, combination: dict, unique_params: Optional[dict] = None,
                                extra_params: Optional[dict] = None, **kwargs):
        openml_clustering_experiment = CSVClusteringExperiment(
            # experiment parameters
            experiment_name=self.experiment_name, create_validation_set=self.create_validation_set,
            log_dir=self.log_dir, log_file_name=self.log_file_name, work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir, clean_work_dir=self.clean_work_dir,
            raise_on_fit_error=self.raise_on_fit_error, error_score=self.error_score, log_to_mlflow=self.log_to_mlflow,
            mlflow_tracking_uri=self.mlflow_tracking_uri, check_if_exists=self.check_if_exists
        )
        return openml_clustering_experiment

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
        dataset_name = combination['dataset_name']
        return {
            'dataset_name': dataset_name
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOCSVClusteringExperiment(parser=parser)
    experiment.run()
    