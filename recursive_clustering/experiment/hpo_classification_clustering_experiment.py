import argparse
from typing import Optional

from ml_experiments.hpo_experiment import HPOExperiment

from recursive_clustering.experiment.classification_clustering_experiment import ClassificationClusteringExperiment


class HPOClassificationClusteringExperiment(HPOExperiment, ClassificationClusteringExperiment):
    def get_hyperband_max_resources(self, combination: dict, unique_params: Optional[dict] = None,
                                    extra_params: Optional[dict] = None, **kwargs):
        raise NotImplementedError('Hyperband is not available for this experiment')

    def _load_single_experiment(self, combination: dict, unique_params: Optional[dict] = None,
                                extra_params: Optional[dict] = None, **kwargs):
        classification_clustering_experiment = ClassificationClusteringExperiment(
            n_samples=self.n_samples, n_random=self.n_random, n_informative=self.n_informative,
            n_redundant=self.n_redundant, n_repeated=self.n_repeated, n_classes=self.n_classes,
            n_clusters_per_class=self.n_clusters_per_class, weights=self.weights, flip_y=self.flip_y,
            class_sep=self.class_sep, hypercube=self.hypercube, shift=self.shift, scale=self.scale,
            shuffle=self.shuffle,
            # experiment parameters
            experiment_name=self.experiment_name, create_validation_set=self.create_validation_set,
            log_dir=self.log_dir, log_file_name=self.log_file_name, work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir, clean_work_dir=self.clean_work_dir, clean_data_dir=False,
            raise_on_fit_error=self.raise_on_fit_error, error_score=self.error_score, log_to_mlflow=self.log_to_mlflow,
            mlflow_tracking_uri=self.mlflow_tracking_uri, check_if_exists=self.check_if_exists
        )
        return classification_clustering_experiment

    def _get_tell_metric_from_results(self, results):
        if not results:
            if self.direction == 'maximize':
                return -float('inf')
            else:
                return float('inf')
        return results['evaluate_model_return']['silhouette']

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
        load_data_return = super()._load_data(combination=combination, unique_params=unique_params,
                                              extra_params=extra_params, **kwargs)
        return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOClassificationClusteringExperiment(parser=parser)
    experiment.run()
    