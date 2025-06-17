from typing import Optional
from cohirf.experiment.spherical_clustering_experiment import SphericalClusteringExperiment
from cohirf.experiment.hpo_clustering_experiment import HPOClusteringExperiment


class HPOSphericalClusteringExperiment(HPOClusteringExperiment, SphericalClusteringExperiment):
    def get_hyperband_max_resources(self, combination: dict, unique_params: dict, extra_params: dict, **kwargs):
        raise NotImplementedError('Hyperband is not available for this experiment')

    def _load_simple_experiment(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        experiment = SphericalClusteringExperiment(
            # experiment parameters
            experiment_name=self.experiment_name,
            log_dir=self.log_dir,
            log_file_name=self.log_file_name,
            work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir,
            clean_work_dir=self.clean_work_dir,
            clean_data_dir=False,
            raise_on_error=self.raise_on_error,
            log_to_mlflow=self.log_to_mlflow,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            check_if_exists=self.check_if_exists,
            verbose=0,
        )
        return experiment

    def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        # load the data and save it to disk, but do not return it here
        load_data_return = super()._load_data(combination=combination, unique_params=unique_params,
                                              extra_params=extra_params, mlflow_run_id=mlflow_run_id, **kwargs)
        return {}


if __name__ == '__main__':
    experiment = HPOSphericalClusteringExperiment()
    experiment.run_from_cli()
