import argparse
from itertools import product
from typing import Optional

import openml
from ml_experiments.hpo_experiment import HPOExperiment
from recursive_clustering.experiment.open_ml_clustering_experiment import OpenmlClusteringExperiment


class HPOOpenmlClusteringExperiment(HPOExperiment, OpenmlClusteringExperiment):
    def get_hyperband_max_resources(self, combination: dict, unique_params: Optional[dict] = None,
                                    extra_params: Optional[dict] = None, **kwargs):
        raise NotImplementedError('Hyperband is not available for this experiment')

    def _load_single_experiment(self, combination: dict, unique_params: Optional[dict] = None,
                                extra_params: Optional[dict] = None, **kwargs):
        openml_clustering_experiment = OpenmlClusteringExperiment(
            # experiment parameters
            experiment_name=self.experiment_name, create_validation_set=self.create_validation_set,
            log_dir=self.log_dir, log_file_name=self.log_file_name, work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir, clean_work_dir=self.clean_work_dir,
            raise_on_fit_error=self.raise_on_fit_error, error_score=self.error_score, log_to_mlflow=self.log_to_mlflow,
            mlflow_tracking_uri=self.mlflow_tracking_uri, check_if_exists=self.check_if_exists, verbose=0
        )
        return openml_clustering_experiment

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
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
            X, y, cat_ind, att_names = dataset.get_data(target=task.target_name)
            train_indices = split.train
            # we will use only the training data
            X = X.iloc[train_indices]
            y = y.iloc[train_indices]
        elif dataset_id is not None:
            dataset = openml.datasets.get_dataset(dataset_id)
            target = dataset.default_target_attribute
            X, y, cat_ind, att_names = dataset.get_data(target=target)
        else:
            raise ValueError('You must specify either dataset_id or task_id')
        cat_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is True]
        cat_dims = [len(X[cat_feature].cat.categories) for cat_feature in cat_features_names]
        n_classes = len(y.unique())
        dataset_name = dataset.name
        return {
            'dataset_name': dataset_name
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOOpenmlClusteringExperiment(parser=parser)
    experiment.run()
    