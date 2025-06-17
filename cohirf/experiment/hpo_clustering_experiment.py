from ml_experiments.hpo_experiment import HPOExperiment
from ml_experiments.base_experiment import BaseExperiment
from ml_experiments.utils import flatten_dict, unflatten_dict, update_recursively
from optuna import Study, Trial
from abc import abstractmethod
from typing import Optional
import numpy as np
import mlflow


class HPOClusteringExperiment(HPOExperiment):
    def __init__(
			self,
			*args,
			search_space: Optional[dict] = None,
			default_values: Optional[list] = None,
			**kwargs,
	):
        super().__init__(*args, **kwargs)
        self.search_space = search_space if search_space is not None else {}
        self.default_values = default_values if default_values is not None else []

    @abstractmethod
    def _load_simple_experiment(
		self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
	):
        raise NotImplementedError

    def _before_fit_model(
		self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
	):
        hpo_seed = unique_params["hpo_seed"]
        ret = super()._before_fit_model(combination, unique_params, extra_params, mlflow_run_id, **kwargs)
        simple_experiment = self._load_simple_experiment(
			combination, unique_params, extra_params, mlflow_run_id, **kwargs
		)
        random_generator = np.random.default_rng(hpo_seed)
        ret["simple_experiment"] = simple_experiment
        ret["random_generator"] = random_generator
        return ret

    def get_trial_fn(
		self,
		study: Study,
		search_space: dict, 
		combination: dict,
		unique_params: dict,
		extra_params: dict,
		mlflow_run_id: Optional[str] = None,
		child_runs_ids: Optional[list] = None,
		**kwargs,
	) -> dict:
        random_generator = kwargs["before_fit_model_return"]["random_generator"]
        seed_model = int(random_generator.integers(0, 2**31 - 1))
        flatten_search_space = flatten_dict(search_space)
        trial = study.ask(flatten_search_space)
        trial_number = trial.number
        trial_key = "_".join([str(value) for value in combination.values()])
        trial_key = trial_key + f"-{trial_number}"  # unique key (trial number)
        child_run_id = child_runs_ids[trial_number] if child_runs_ids else None
        trial.set_user_attr("child_run_id", child_run_id)
        return dict(trial=trial, trial_key=trial_key, child_run_id=child_run_id, seed_model=seed_model)

    def training_fn(
		self,
		trial_dict: dict,
		combination: dict,
		unique_params: dict,
		extra_params: dict,
		mlflow_run_id: str | None = None,
		**kwargs,
	) -> dict:
        trial: Trial = trial_dict["trial"]
        child_run_id = trial_dict["child_run_id"]
        seed_model = trial_dict["seed_model"]
        simple_experiment: BaseExperiment = kwargs["before_fit_model_return"]["simple_experiment"]

        # update the model parameters in unique_params
        trial_params = trial.params.copy()
        trial_params = unflatten_dict(trial_params)
        unique_params = unique_params.copy()
        model_params = unique_params["model_params"]
        model_params = update_recursively(model_params, trial_params)
        unique_params["model_params"] = model_params

        # update the seed_model in combination
        combination = combination.copy()
        combination["seed_model"] = seed_model

        if mlflow_run_id is not None:
            results = simple_experiment._run_mlflow_and_train_model(
				combination=combination,
				unique_params=unique_params,
				extra_params=extra_params,
				mlflow_run_id=child_run_id,
				return_results=True,
			)
        else:
            results = simple_experiment._train_model(
				combination=combination,
				unique_params=unique_params,
				extra_params=extra_params,
				mlflow_run_id=child_run_id,
				return_results=True,
			)

        if not isinstance(results, dict):
            results = dict()

        keep_results = results.get("evaluate_model_return", {})

        if mlflow_run_id is not None:
            log_metrics = keep_results.copy()
            log_metrics.pop("elapsed_time", None)
            log_metrics.pop("max_memory_used", None)
            mlflow.log_metrics(log_metrics, run_id=mlflow_run_id, step=trial.number)
        return keep_results

    def get_search_space(
		self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
	) -> dict:
        model = combination["model"]
        if isinstance(model, str):
            model_class, model_default_params, search_space, default_values = self.models_dict[model]
        else:
            search_space = self.search_space
            if search_space is None:
                raise ValueError("Search space must be defined if model is not defined as a string.")
        return search_space

    def get_default_values(
		self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
	) -> list:
        model = combination["model"]
        if isinstance(model, str):
            model_class, model_default_params, search_space, default_values = self.models_dict[model]
        else:
            default_values = self.default_values
            if default_values is None:
                raise ValueError("Default values must be defined if model is not defined as a string.")
        return default_values
