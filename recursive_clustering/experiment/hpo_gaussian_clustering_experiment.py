import argparse
from typing import Optional
from ml_experiments.hpo_experiment import HPOExperiment
from recursive_clustering.experiment.gaussian_clustering_experiment import GaussianClusteringExperiment


class HPOGaussianClusteringExperiment(HPOExperiment, GaussianClusteringExperiment):
    def get_hyperband_max_resources(self, combination: dict, unique_params: Optional[dict] = None,
                                    extra_params: Optional[dict] = None, **kwargs):
        raise NotImplementedError('Hyperband is not available for this experiment')

    def _load_single_experiment(self, combination: dict, unique_params: Optional[dict] = None,
                                extra_params: Optional[dict] = None, **kwargs):
        blob_clustering_experiment = GaussianClusteringExperiment(
            n_samples=self.n_samples, n_features=self.n_features, n_centers=self.n_centers, distances=self.distances,
            # experiment parameters
            experiment_name=self.experiment_name, create_validation_set=self.create_validation_set,
            log_dir=self.log_dir, log_file_name=self.log_file_name, work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir, clean_work_dir=self.clean_work_dir, clean_data_dir=False,
            raise_on_fit_error=self.raise_on_fit_error, error_score=self.error_score, log_to_mlflow=self.log_to_mlflow,
            mlflow_tracking_uri=self.mlflow_tracking_uri, check_if_exists=self.check_if_exists
        )
        return blob_clustering_experiment

    def _get_tell_metric_from_results(self, results):
        evaluate_model_return = results.get('evaluate_model_return', {})
        if not evaluate_model_return:
            if self.direction == 'maximize':
                return -float('inf')
            else:
                return float('inf')
        return evaluate_model_return['silhouette']

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
        # load the data and save it to disk, but do not return it here
        load_data_return = super()._load_data(combination=combination, unique_params=unique_params,
                                              extra_params=extra_params, **kwargs)
        if 'dataset_name' in load_data_return:
            dataset_name = load_data_return['dataset_name']
        else:
            dataset_name = 'gaussian'
        return {
            'dataset_name': dataset_name
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOGaussianClusteringExperiment(parser=parser)
    experiment.run()
