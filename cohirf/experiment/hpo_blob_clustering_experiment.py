import argparse
from typing import Optional
from ml_experiments.hpo_experiment import HPOExperiment
from cohirf.experiment.blob_clustering_experiment import BlobClusteringExperiment


class HPOBlobClusteringExperiment(HPOExperiment, BlobClusteringExperiment):
    def get_hyperband_max_resources(self, combination: dict, unique_params: Optional[dict] = None,
                                    extra_params: Optional[dict] = None, **kwargs):
        raise NotImplementedError('Hyperband is not available for this experiment')

    def _load_single_experiment(self, combination: dict, unique_params: Optional[dict] = None,
                                extra_params: Optional[dict] = None, **kwargs):
        blob_clustering_experiment = BlobClusteringExperiment(
            n_samples=self.n_samples, n_features=self.n_features, centers=self.centers, cluster_std=self.cluster_std,
            center_box=self.center_box, shuffle=self.shuffle,
            # experiment parameters
            experiment_name=self.experiment_name, create_validation_set=self.create_validation_set,
            log_dir=self.log_dir, log_file_name=self.log_file_name, work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir, clean_work_dir=self.clean_work_dir, clean_data_dir=False,
            raise_on_fit_error=self.raise_on_fit_error, error_score=self.error_score, log_to_mlflow=self.log_to_mlflow,
            mlflow_tracking_uri=self.mlflow_tracking_uri, check_if_exists=self.check_if_exists, verbose=0
        )
        return blob_clustering_experiment

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
        # load the data and save it to disk, but do not return it here
        load_data_return = super()._load_data(combination=combination, unique_params=unique_params,
                                              extra_params=extra_params, **kwargs)
        return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOBlobClusteringExperiment(parser=parser)
    experiment.run()
